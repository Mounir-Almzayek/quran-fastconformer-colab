"""Compare pretrained and saved CTC-stage outputs without changing model artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.baseline import _predict
from src.common import ensure_project_dirs, load_config, set_seed, utc_now, write_json
from src.data import ensure_nemo_audio_cache, load_nemo_manifest
from src.metrics import build_metrics
from src.nemo_utils import load_exported_ctc_model, load_pretrained_ctc_model


def _describe_predictions(rows: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    predictions = [str(row["prediction"]).strip() for row in rows]
    metrics, _ = build_metrics(
        [row["reference"] for row in rows],
        predictions,
        config["text"]["primary_normalization"],
        config["text"]["diagnostic_normalization"],
    )
    return {
        "examples": len(rows),
        "empty_predictions": sum(not prediction for prediction in predictions),
        "unknown_symbol_predictions": sum(prediction == "⁇" for prediction in predictions),
        "unique_predictions": len(set(predictions)),
        "strict_wer_percent": metrics["strict"]["wer_percent"],
        "diagnostic_wer_percent": metrics["diagnostic"]["wer_percent"],
        "examples_preview": [
            {"reference": row["reference"], "prediction": row["prediction"]} for row in rows[:5]
        ],
    }


def _model_state(model: Any) -> dict[str, Any]:
    return {
        "ctc_loss_weight": float(model.ctc_loss_weight),
        "cfg_ctc_loss_weight": float(model.cfg.aux_ctc.get("ctc_loss_weight", -1)),
        "ctc_decoder_classes": int(model.ctc_decoder.num_classes_with_blank),
        "ctc_vocabulary_size": len(model.ctc_decoder.vocabulary),
    }


def run(config_path: str | Path, manifest_path: str | Path, limit: int) -> dict[str, Any]:
    """Evaluate a small fixed held-out slice on the pretrained model and each saved stage export."""
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    paths = ensure_project_dirs(config)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    local_manifests = ensure_nemo_audio_cache(config, manifest, Path(config["_project_root"]))
    records = load_nemo_manifest(local_manifests["test"])[:limit]
    if not records:
        raise RuntimeError("The held-out test manifest is empty.")

    candidates: list[tuple[str, Path | None]] = [("pretrained", None)]
    for index, stage in enumerate(config["training"]["stages"], start=1):
        stage_path = paths["nemo"] / "checkpoints" / f"{index:02d}_{stage['name']}" / "stage_model.nemo"
        candidates.append((f"stage_{index}_{stage['name']}", stage_path))
    candidates.append(("final", paths["model"]))

    report: dict[str, Any] = {
        "created_at": utc_now(),
        "config_path": str(Path(config_path).resolve()),
        "manifest_path": str(Path(manifest_path).resolve()),
        "limit": limit,
        "stages": {},
    }
    for label, model_path in candidates:
        if model_path is not None and not model_path.is_file():
            report["stages"][label] = {"status": "missing", "model_path": str(model_path)}
            continue
        model = load_pretrained_ctc_model(config["model"]["pretrained_name"]) if model_path is None else load_exported_ctc_model(model_path)
        rows = _predict(model, records, int(config["training"]["validation_batch_size"]))
        report["stages"][label] = {
            "status": "evaluated",
            "model_path": config["model"]["pretrained_name"] if model_path is None else str(model_path.resolve()),
            "model_state": _model_state(model),
            **_describe_predictions(rows, config),
        }
        del model

    destination = paths["results"] / "diagnostics" / "ctc_stage_probe.json"
    write_json(report, destination)
    report["report_path"] = str(destination.resolve())
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose when CTC output collapses across saved fine-tuning stages.")
    parser.add_argument("--config", default="configs/fastconformer_quran_pc_v3_ctc_only.yaml")
    parser.add_argument(
        "--manifest", default="artifacts/experiments/fastconformer_pc_v2/manifests/experiment_manifest.json"
    )
    parser.add_argument("--limit", type=int, default=32)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run(args.config, args.manifest, args.limit)
    print(json.dumps(output, ensure_ascii=False, indent=2))
