"""Evaluate the fine-tuned FastConformer on the same held-out reciter cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.baseline import _predict
from src.common import ensure_project_dirs, load_config, set_seed, utc_now, write_json
from src.data import ensure_nemo_audio_cache, load_nemo_manifest
from src.metrics import build_metrics, save_predictions
from src.nemo_utils import import_nemo, normalize_transcriptions


def _restore_model(model_path: Path) -> Any:
    """Restore a local NeMo artifact and select CTC decoding for comparable evaluation."""
    _, nemo_asr, _ = import_nemo()
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.restore_from(str(model_path))
    model.change_decoding_strategy(decoder_type="ctc")
    return model


def run(config_path: str | Path, manifest_path: str | Path, model_path: str | Path | None = None) -> dict[str, Any]:
    """Score a trained `.nemo` checkpoint with the identical test JSONL used by baseline."""
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    paths = ensure_project_dirs(config)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    resolved_model = Path(model_path) if model_path else paths["model"]
    if not resolved_model.is_file():
        raise FileNotFoundError(f"Fine-tuned NeMo file not found: {resolved_model}")
    local_manifests = ensure_nemo_audio_cache(config, manifest, Path(config["_project_root"]))
    records = load_nemo_manifest(local_manifests["test"])
    model = _restore_model(resolved_model)
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
            "stage": "finetuned",
            "model": str(resolved_model.resolve()),
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
    save_predictions(enriched, paths["finetuned"] / "predictions.csv", paths["finetuned"] / "predictions.json")
    write_json(metrics, paths["finetuned"] / "metrics.json")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate final FastConformer against the reciter-held-out test cohort.")
    parser.add_argument("--config", default="configs/fastconformer_quran.yaml")
    parser.add_argument("--manifest", default="artifacts/manifests/experiment_manifest.json")
    parser.add_argument("--model", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run(args.config, args.manifest, args.model)
    print(f"Final evaluation complete | strict WER={output['strict']['wer_percent']:.2f}% | strict CER={output['strict']['cer_percent']:.2f}%")
