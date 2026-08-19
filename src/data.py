"""EveryAyah streaming access, disjoint-reciter splits, and NeMo manifest materialization."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
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


def _base_stream(config: Mapping[str, Any], decode_audio: bool = False) -> IterableDataset:
    """Open the source split without decoding audio into the shuffle buffer by default."""
    dataset_config = config["dataset"]
    stream = load_dataset(
        dataset_config["name"],
        name=dataset_config.get("config"),
        split=dataset_config["source_split"],
        streaming=bool(dataset_config["streaming"]),
    )
    # The metadata-selection pass must not retain decoded waveforms in the
    # streaming shuffle buffer. Audio is decoded only in the materialization pass.
    return stream.cast_column(
        "audio",
        Audio(sampling_rate=int(dataset_config["target_sampling_rate"]), decode=decode_audio),
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

    fixed_validation = [str(value) for value in split_config.get("validation_reciter_names", [])]
    fixed_test = [str(value) for value in split_config.get("test_reciter_names", [])]
    if fixed_validation or fixed_test:
        if len(fixed_validation) != int(split_config["validation_reciters"]):
            raise RuntimeError("validation_reciter_names count does not match validation_reciters.")
        if len(fixed_test) != int(split_config["test_reciters"]):
            raise RuntimeError("test_reciter_names count does not match test_reciters.")
        if set(fixed_validation) & set(fixed_test):
            raise RuntimeError("A fixed reciter was assigned to both Validation and Test.")
        requested = set(fixed_validation) | set(fixed_test)
        unavailable = sorted(requested - set(eligible))
        if unavailable:
            raise RuntimeError(
                "Configured held-out reciters are unavailable or below the sample threshold: "
                f"{unavailable}."
            )
        validation_reciters = fixed_validation
        test_reciters = fixed_test
    else:
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
    selected_reciters = {
        split_name: {str(row["reciter"]) for row in rows}
        for split_name, rows in selections.items()
    }
    expected_reciters = {
        "train": set(train_reciters),
        "validation": set(validation_reciters),
        "test": set(test_reciters),
    }
    if any(not selected_reciters[name] for name in selected_reciters):
        raise RuntimeError("At least one split is empty after reciter-level selection.")
    if any(not selected_reciters[name].issubset(expected_reciters[name]) for name in selected_reciters):
        raise RuntimeError("A selected row was assigned to the wrong reciter split.")
    if (
        selected_reciters["train"] & selected_reciters["validation"]
        or selected_reciters["train"] & selected_reciters["test"]
        or selected_reciters["validation"] & selected_reciters["test"]
    ):
        raise RuntimeError("Reciter leakage detected while creating the manifest.")

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
    stream = _base_stream(config, decode_audio=False).shuffle(
        seed=int(manifest["selection"]["seed"]),
        buffer_size=int(manifest["selection"]["shuffle_buffer"]),
    ).cast_column(
        "audio",
        Audio(sampling_rate=int(config["dataset"]["target_sampling_rate"]), decode=True),
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


def _write_wav_atomically(destination: Path, waveform: np.ndarray, sample_rate: int) -> None:
    """Write through local disk, then copy to Drive and confirm the file is readable."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.gettempdir()) / "quran_fastconformer_wav_staging"
    temporary_root.mkdir(parents=True, exist_ok=True)
    temporary_path = temporary_root / f"{destination.stem}.{hashlib.sha256(str(destination).encode()).hexdigest()[:10]}.wav"
    try:
        sf.write(temporary_path, waveform, sample_rate, subtype="PCM_16")
        shutil.copyfile(temporary_path, destination)
        if not destination.is_file() or destination.stat().st_size < 44:
            raise RuntimeError(f"WAV write did not persist at {destination}.")
    finally:
        temporary_path.unlink(missing_ok=True)


def _validate_materialized_manifests(paths: Mapping[str, Path], manifest: Mapping[str, Any]) -> None:
    """Reject JSONL files that reference missing or empty WAV artifacts."""
    for split_name, path in paths.items():
        records = load_nemo_manifest(path)
        expected_count = len(manifest["splits"][split_name])
        if len(records) != expected_count:
            raise RuntimeError(
                f"NeMo {split_name} manifest has {len(records)} records; expected {expected_count}."
            )
        missing = [
            row["audio_filepath"]
            for row in records
            if not Path(row["audio_filepath"]).is_file() or Path(row["audio_filepath"]).stat().st_size < 44
        ]
        if missing:
            preview = ", ".join(missing[:3])
            raise RuntimeError(
                f"NeMo {split_name} manifest references {len(missing)} unavailable WAV files. "
                f"Examples: {preview}"
            )


def materialize_nemo_manifests(config: Mapping[str, Any], manifest: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    """Download exactly the chosen clips, write verified WAV files, and produce NeMo JSONL manifests."""
    artifacts = project_root / config["project"]["artifacts_dir"]
    # Drive remains the durable home for JSONL, checkpoints, and reports. WAVs
    # stay on Colab's local disk because NeMo must open thousands of them quickly.
    audio_root = Path(config["project"].get("runtime_audio_dir", "/content/quran-fastconformer-audio"))
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
            if not audio_path.is_file() or audio_path.stat().st_size < 44:
                _write_wav_atomically(audio_path, waveform, target_rate)
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
    _validate_materialized_manifests(paths, manifest)
    return paths


def ensure_nemo_audio_cache(config: Mapping[str, Any], manifest: Mapping[str, Any], project_root: Path) -> dict[str, Path]:
    """Reuse a valid local WAV cache or rebuild it after a Colab runtime reset."""
    artifacts = project_root / config["project"]["artifacts_dir"]
    paths = {split: artifacts / "nemo" / "manifests" / f"{split}.jsonl" for split in ("train", "validation", "test")}
    try:
        _validate_materialized_manifests(paths, manifest)
    except (FileNotFoundError, RuntimeError) as error:
        print(f"Rebuilding local NeMo audio cache: {error}")
        return materialize_nemo_manifests(config, manifest, project_root)
    return paths


def load_nemo_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a materialized NeMo JSONL file for offline evaluation bookkeeping."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
