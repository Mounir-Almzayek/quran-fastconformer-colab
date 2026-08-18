"""Evaluate the untouched Arabic FastConformer on the held-out reciter set."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.common import ensure_project_dirs, load_config, set_seed, utc_now, write_json
from src.data import load_nemo_manifest
from src.metrics import build_metrics, save_predictions
from src.nemo_utils import load_pretrained_ctc_model, normalize_transcriptions


def _predict(model: Any, records: list[dict[str, Any]], batch_size: int) -> list[dict[str, Any]]:
    """Run CTC transcription and preserve metadata needed for reproducible analysis."""
    audio_paths = [record["audio_filepath"] for record in records]
    raw_outputs = model.transcribe(audio_paths, batch_size=batch_size)
    predictions = normalize_transcriptions(raw_outputs)
    if len(predictions) != len(records):
        raise RuntimeError("NeMo returned a different number of predictions than manifest rows.")
    return [{**record, "reference": record["text"], "prediction": prediction} for record, prediction in zip(records, predictions)]


def run(config_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Measure pretrained FastConformer before any Quran-specific adaptation in this project."""
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    paths = ensure_project_dirs(config)
    import json
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    test_manifest = paths["nemo"] / "manifests" / "test.jsonl"
    if not test_manifest.exists():
        raise FileNotFoundError("NeMo test manifest is missing. Run run_colab_setup.py --materialize first.")

    model = load_pretrained_ctc_model(config["model"]["pretrained_name"])
    records = load_nemo_manifest(test_manifest)
    rows = _predict(model, records, int(config["training"]["validation_batch_size"]))
    metrics, normalized_rows = build_metrics(
        [row["reference"] for row in rows],
        [row["prediction"] for row in rows],
        config["text"]["primary_normalization"],
        config["text"]["diagnostic_normalization"],
    )
    enriched = [{**row, **normalized} for row, normalized in zip(rows, normalized_rows)]
    metrics.update(
        {
            "stage": "baseline",
            "model": config["model"]["pretrained_name"],
            "decoder": config["model"]["decoder"],
            "metric_protocol": {
                "strict": config["text"]["primary_normalization"],
                "diagnostic": config["text"]["diagnostic_normalization"],
            },
            "manifest_path": str(Path(manifest_path).resolve()),
            "held_out_test_reciters": manifest["selection"]["test_reciters"],
            "evaluated_at": utc_now(),
        }
    )
    save_predictions(enriched, paths["baseline"] / "predictions.csv", paths["baseline"] / "predictions.json")
    write_json(metrics, paths["baseline"] / "metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate pretrained FastConformer on held-out reciters.")
    parser.add_argument("--config", default="configs/fastconformer_quran.yaml")
    parser.add_argument("--manifest", default="artifacts/manifests/experiment_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run(args.config, args.manifest)
    print(f"Baseline complete | strict WER={result['strict']['wer_percent']:.2f}% | strict CER={result['strict']['cer_percent']:.2f}%")
