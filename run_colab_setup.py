#!/usr/bin/env python3
"""Prepare the Colab environment for the FastConformer Quran ASR experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import torch

from src.common import ensure_project_dirs, load_config, write_json
from src.data import build_manifest, inspect_schema, manifest_summary, materialize_nemo_manifests


def _manifest_fingerprint(manifest: object) -> str:
    """Return a content hash used to identify an immutable split artifact."""
    encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _backup_manifest(manifest: dict, manifest_path: Path, config: dict, artifacts: Path) -> list[Path]:
    """Write immutable split copies both inside the experiment and outside the project tree."""
    fingerprint = _manifest_fingerprint(manifest)
    filename = f"experiment_manifest_{fingerprint[:16]}.json"
    backup_dir_value = config["project"].get(
        "manifest_backup_dir",
        "artifacts/manifest_backups",
    )
    backup_dir = Path(backup_dir_value).expanduser()
    if not backup_dir.is_absolute():
        backup_dir = Path(config["_project_root"]) / backup_dir
    destinations = [
        artifacts / "manifests" / "archive" / filename,
        backup_dir / str(config["project"]["name"]) / filename,
    ]
    written: list[Path] = []
    for destination in destinations:
        write_json(manifest, destination)
        written.append(destination)
    write_json(
        {
            "manifest_path": str(manifest_path.resolve()),
            "sha256": fingerprint,
            "backup_paths": [str(path.resolve()) for path in written],
        },
        artifacts / "manifests" / "manifest_backup_receipt.json",
    )
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a disjoint-reciter manifest and local NeMo JSONL files.")
    parser.add_argument("--config", default="configs/fastconformer_quran.yaml")
    parser.add_argument(
        "--manifest",
        help="Reuse an existing locked manifest instead of creating an experiment-local split.",
    )
    parser.add_argument("--rebuild-manifest", action="store_true", help="Intentionally create a new split and invalidate prior results.")
    parser.add_argument("--materialize", action="store_true", help="Download only selected clips and write NeMo WAV/JSONL artifacts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    paths = ensure_project_dirs(config)
    artifacts = paths["artifacts"]
    runtime = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    schema = inspect_schema(config)
    write_json(runtime, artifacts / "runtime.json")
    write_json(schema, paths["manifests"] / "dataset_schema.json")
    if schema["missing_expected_columns"]:
        raise RuntimeError(f"Dataset is missing required columns: {schema['missing_expected_columns']}")

    if args.manifest and args.rebuild_manifest:
        raise ValueError("--manifest and --rebuild-manifest cannot be used together.")
    manifest_path = Path(args.manifest).resolve() if args.manifest else paths["manifests"] / "experiment_manifest.json"
    if args.manifest:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Locked manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"Reusing locked manifest: {manifest_path}")
    elif manifest_path.exists() and not args.rebuild_manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"Keeping existing manifest: {manifest_path}")
    else:
        manifest = build_manifest(config, manifest_path)
        print(f"Created disjoint-reciter manifest: {manifest_path}")

    backups = _backup_manifest(manifest, manifest_path, config, artifacts)
    print("Manifest backup copies:")
    for backup in backups:
        print(f" - {backup}")

    summary = manifest_summary(manifest)
    if summary["integrity"]["reciter_leakage_count"]:
        raise RuntimeError(f"Reciter leakage found: {summary['integrity']['reciter_leakage']}")
    for split_name in ("train", "validation", "test"):
        item = summary[split_name]
        print(
            f"{split_name}: {item['count']} clips | reciters={item['reciters']} | "
            f"distinct verse texts={item['distinct_text_groups']}"
        )
    print("Integrity: no reciter is shared by Train, Validation, and Test.")

    if args.materialize:
        generated = materialize_nemo_manifests(config, manifest, Path(config["_project_root"]))
        for split_name, path in generated.items():
            print(f"NeMo {split_name} manifest: {path}")


if __name__ == "__main__":
    main()
