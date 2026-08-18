"""EveryAyah streaming access, disjoint-reciter splits, and NeMo manifest materialization."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from datasets import Audio, IterableDataset, load_dataset

from src.common import utc_now, write_json
from src.metrics import normalize_text, text_group_key


REQUIRED_COLUMNS = {"audio", "text", "duration", "reciter"}


def _base_stream(config: Mapping[str, Any]) -> IterableDataset:
    """Open the source split as a stream; no corpus-wide download is performed."""
    dataset_config = config["dataset"]
    return load_dataset(
        dataset_config["name"],
        name=dataset_config.get("config"),
        split=dataset_config["source_split"],
        streaming=bool(dataset_config["streaming"]),
    )


def _usable(row: Mapping[str, Any], config: Mapping[str, Any]) -> bool:
    """Keep only valid audio/text rows within the NeMo experiment duration window."""
    duration, text, reciter = row.get("duration"), row.get("text"), row.get("reciter")
    if duration is None or not isinstance(text, str) or not text.strip() or not reciter:
        return False
    dataset_config = config["dataset"]
    return dataset_config["min_duration_seconds"] <= float(duration) <= dataset_config["max_duration_seconds"]


def _stable_key(row: Mapping[str, Any], stream_position: int) -> str:
    """Make a replay key that changes if source ordering or row metadata changes."""
    payload = "|".join(
        [
            str(stream_position),
            str(row.get("reciter", "")),
            f"{float(row.get('duration', 0.0)):.6f}",
            " ".join(str(row.get("text", "")).split()),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _candidate_row(row: Mapping[str, Any], stream_position: int) -> dict[str, Any]:
    """Store only metadata needed to recreate audio later from the deterministic stream."""
    return {
        "stream_position": stream_position,
        "key": _stable_key(row, stream_position),
        "reciter": str(row["reciter"]),
        "duration": float(row["duration"]),
        "text": str(row["text"]),
        "text_group_id": text_group_key(str(row["text"])),
    }


def inspect_schema(config: Mapping[str, Any]) -> dict[str, Any]:
    """Inspect a real source sample before selection or audio materialization."""
    sample = next(iter(_base_stream(config)))
    audio = sample.get("audio", {})
    return {
        "observed_columns": sorted(sample),
        "missing_expected_columns": sorted(REQUIRED_COLUMNS - set(sample)),
        "audio_fields": sorted(audio) if isinstance(audio, Mapping) else [],
        "audio_sampling_rate": audio.get("sampling_rate") if isinstance(audio, Mapping) else None,
        "duration_seconds": sample.get("duration"),
        "reciter": sample.get("reciter"),
        "text_preview": str(sample.get("text", ""))[:180],
    }


def _seeded_order(items: list[str], seed: int) -> list[str]:
    """Create deterministic random-like ordering independent of source dict order."""
    return sorted(items, key=lambda item: hashlib.sha256(f"{seed}:{item}".encode()).hexdigest())


def _collect_candidates(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], int]:
    """Read only a finite metadata candidate pool from the shuffled source stream."""
    dataset_config = config["dataset"]
    stream = _base_stream(config).shuffle(
        seed=int(config["project"]["seed"]),
        buffer_size=int(dataset_config["shuffle_buffer"]),
    )
    target = int(dataset_config["candidate_pool_size"])
    candidates: list[dict[str, Any]] = []
    inspected = 0
    for position, row in enumerate(stream):
        inspected += 1
        if _usable(row, config):
            candidates.append(_candidate_row(row, position))
        if len(candidates) >= target:
            break
    if len(candidates) < target:
        raise RuntimeError(f"Collected {len(candidates)} usable rows; expected candidate pool of {target}.")
    return candidates, inspected


def _take_cap(rows: list[dict[str, Any]], cap: int, split_name: str) -> list[dict[str, Any]]:
    """Select an exact cap while preserving the stream-derived deterministic order."""
    if len(rows) < cap:
        raise RuntimeError(f"Split '{split_name}' has {len(rows)} usable rows but needs {cap}.")
    return sorted(rows, key=lambda row: int(row["stream_position"]))[:cap]


def build_manifest(config: Mapping[str, Any], destination: str | Path) -> dict[str, Any]:
    """Create an immutable experiment manifest with no reciter crossing split boundaries."""
    candidates, inspected_rows = _collect_candidates(config)
    by_reciter: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        by_reciter[row["reciter"]].append(row)

    split_config = config["dataset"]["reciter_split"]
    min_holdout = int(split_config["minimum_samples_per_holdout_reciter"])
    eligible = [reciter for reciter, rows in by_reciter.items() if len(rows) >= min_holdout]
    needed = int(split_config["validation_reciters"]) + int(split_config["test_reciters"])
    if len(eligible) < needed:
        counts = {reciter: len(rows) for reciter, rows in sorted(by_reciter.items())}
        raise RuntimeError(
            f"Only {len(eligible)} reciters meet the holdout threshold of {min_holdout}; need {needed}. "
            f"Increase candidate_pool_size or lower the threshold. Candidate counts: {counts}"
        )

    ordered_eligible = _seeded_order(eligible, int(config["project"]["seed"]))
    test_reciters = ordered_eligible[: int(split_config["test_reciters"])]
    validation_reciters = ordered_eligible[
        int(split_config["test_reciters"]): needed
    ]
    train_reciters = sorted(set(by_reciter) - set(test_reciters) - set(validation_reciters))
    if not train_reciters:
        raise RuntimeError("No reciters remain for training after reciter-level holdout selection.")

    caps = config["dataset"]["sample_caps"]
    selections = {
        "train": _take_cap([row for reciter in train_reciters for row in by_reciter[reciter]], int(caps["train"]), "train"),
        "validation": _take_cap([row for reciter in validation_reciters for row in by_reciter[reciter]], int(caps["validation"]), "validation"),
        "test": _take_cap([row for reciter in test_reciters for row in by_reciter[reciter]], int(caps["test"]), "test"),
    }
    owners = {row["reciter"]: split for split, rows in selections.items() for row in rows}
    if set(owners) != set(train_reciters) | set(validation_reciters) | set(test_reciters):
        raise RuntimeError("Reciter assignment became inconsistent while creating the manifest.")

    manifest = {
        "format_version": 3,
        "created_at": utc_now(),
        "dataset": {
            "name": config["dataset"]["name"],
            "config": config["dataset"].get("config"),
            "source_split": config["dataset"]["source_split"],
            "streaming": True,
        },
        "selection": {
            "strategy": "disjoint_reciters",
            "seed": int(config["project"]["seed"]),
            "shuffle_buffer": int(config["dataset"]["shuffle_buffer"]),
            "candidate_pool_size": int(config["dataset"]["candidate_pool_size"]),
            "candidate_rows_collected": len(candidates),
            "inspected_rows": inspected_rows,
            "duration_range_seconds": [config["dataset"]["min_duration_seconds"], config["dataset"]["max_duration_seconds"]],
            "train_reciters": train_reciters,
            "validation_reciters": validation_reciters,
            "test_reciters": test_reciters,
        },
        "splits": selections,
    }
    write_json(manifest, destination)
    return manifest


def manifest_summary(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """Summarize counts and prove that no reciter is shared by two partitions."""
    reciter_owners: dict[str, set[str]] = defaultdict(set)
    result: dict[str, Any] = {}
    for split_name, rows in manifest["splits"].items():
        reciters = sorted({str(row["reciter"]) for row in rows})
        for reciter in reciters:
            reciter_owners[reciter].add(split_name)
        result[split_name] = {
            "count": len(rows),
            "reciters": reciters,
            "distinct_text_groups": len({row["text_group_id"] for row in rows}),
        }
    leakage = {reciter: sorted(splits) for reciter, splits in reciter_owners.items() if len(splits) > 1}
    result["integrity"] = {"reciter_leakage_count": len(leakage), "reciter_leakage": leakage}
    return result


def _replay_selected_rows(config: Mapping[str, Any], manifest: Mapping[str, Any]) -> Iterator[tuple[str, dict[str, Any], Mapping[str, Any]]]:
    """Replay all selected positions once and yield their decoded audio records by split."""
    expected: dict[int, tuple[str, dict[str, Any]]] = {}
    for split_name, rows in manifest["splits"].items():
        for metadata in rows:
            expected[int(metadata["stream_position"])] = (split_name, metadata)
    stream = _base_stream(config).shuffle(
        seed=int(manifest["selection"]["seed"]),
        buffer_size=int(manifest["selection"]["shuffle_buffer"]),
    )
    highest = max(expected)
    seen: set[int] = set()
    for position, row in enumerate(stream):
        if position > highest:
            break
        if position not in expected:
            continue
        split_name, metadata = expected[position]
        if _stable_key(row, position) != metadata["key"]:
            raise RuntimeError("The streamed source changed after manifest creation; rebuild the experiment before materializing audio.")
        seen.add(position)
        yield split_name, metadata, row
    missing = set(expected) - seen
    if missing:
        raise RuntimeError(f"Could not replay {len(missing)} selected source rows.")


def materialize_nemo_manifests(config: Mapping[str, Any], manifest: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    """Download exactly the chosen clips, write WAV files, and produce NeMo JSONL manifests."""
    artifacts = project_root / config["project"]["artifacts_dir"]
    audio_root = artifacts / "nemo" / "audio"
    manifest_root = artifacts / "nemo" / "manifests"
    manifest_root.mkdir(parents=True, exist_ok=True)
    handles: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    try:
        for split_name in ("train", "validation", "test"):
            split_dir = audio_root / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            path = manifest_root / f"{split_name}.jsonl"
            paths[split_name] = path
            handles[split_name] = path.open("w", encoding="utf-8")

        target_rate = int(config["dataset"]["target_sampling_rate"])
        target_normalization = config["text"]["target_normalization"]
        for split_name, metadata, row in _replay_selected_rows(config, manifest):
            audio = row["audio"]
            if int(audio["sampling_rate"]) != target_rate:
                raise RuntimeError(
                    f"Expected {target_rate} Hz audio after dataset decoding, received {audio['sampling_rate']} Hz."
                )
            filename = f"{int(metadata['stream_position']):08d}.wav"
            audio_path = audio_root / split_name / filename
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sf.write(audio_path, waveform, target_rate, subtype="PCM_16")
            record = {
                "audio_filepath": str(audio_path.resolve()),
                "duration": float(metadata["duration"]),
                "text": normalize_text(metadata["text"], target_normalization),
                "reciter": metadata["reciter"],
                "manifest_key": metadata["key"],
                "text_group_id": metadata["text_group_id"],
            }
            handles[split_name].write(json.dumps(record, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return paths


def load_nemo_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a materialized NeMo JSONL file for offline evaluation bookkeeping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
