"""Fine-tune Arabic FastConformer for Quran ASR with staged unfreezing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.common import ensure_project_dirs, load_config, set_seed, utc_now, write_json
from src.data import ensure_nemo_audio_cache
from src.nemo_utils import attach_manifests, configure_trainable_layers, import_nemo, load_pretrained_ctc_model, set_learning_rate


def _manifest_paths(nemo_dir: Path) -> dict[str, Path]:
    """Return the local JSONL manifests generated during setup."""
    manifest_dir = nemo_dir / "manifests"
    result = {split: manifest_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    missing = [str(path) for path in result.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing NeMo manifests. Run setup with --materialize first: " + ", ".join(missing))
    return result


def _stage_trainer(stage_dir: Path, stage: dict[str, Any], config: dict[str, Any]) -> tuple[Any, Any]:
    """Create a fresh Lightning trainer and best-WER checkpoint callback for one stage."""
    pl, _, _ = import_nemo()
    checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(stage_dir),
        filename="{epoch:02d}-{val_wer:.4f}",
        monitor="val_wer",
        mode="min",
        save_top_k=1,
        save_last=True,
    )
    use_gpu = __import__("torch").cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        max_epochs=int(stage["max_epochs"]),
        precision=config["training"]["precision"] if use_gpu else "32-true",
        gradient_clip_val=float(config["training"]["gradient_clip_val"]),
        callbacks=[checkpoint],
        log_every_n_steps=10,
        enable_progress_bar=True,
        num_sanity_val_steps=0,
        default_root_dir=str(stage_dir),
    )
    return trainer, checkpoint


def run(config_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Run the declared staged adaptation policy and export one final `.nemo` checkpoint."""
    config = load_config(config_path)
    set_seed(int(config["project"]["seed"]))
    paths = ensure_project_dirs(config)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    local_manifests = ensure_nemo_audio_cache(config, manifest, Path(config["_project_root"]))

    model = load_pretrained_ctc_model(config["model"]["pretrained_name"])
    attach_manifests(model, local_manifests["train"], local_manifests["validation"], local_manifests["test"], config)
    stage_summaries: list[dict[str, Any]] = []

    for index, stage in enumerate(config["training"]["stages"], start=1):
        stage_dir = paths["nemo"] / "checkpoints" / f"{index:02d}_{stage['name']}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        configure_trainable_layers(model, str(stage["encoder_layers"]))
        set_learning_rate(model, float(stage["learning_rate"]))
        trainer, checkpoint = _stage_trainer(stage_dir, stage, config)
        trainer.fit(model)
        stage_export = stage_dir / "stage_model.nemo"
        model.save_to(str(stage_export))
        stage_summaries.append(
            {
                "name": stage["name"],
                "encoder_layers": stage["encoder_layers"],
                "max_epochs": int(stage["max_epochs"]),
                "learning_rate": float(stage["learning_rate"]),
                "best_checkpoint": checkpoint.best_model_path or None,
                "stage_export": str(stage_export.resolve()),
            }
        )

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(paths["model"]))
    summary = {
        "created_at": utc_now(),
        "model_source": config["model"]["pretrained_name"],
        "decoder": config["model"]["decoder"],
        "manifest_path": str(Path(manifest_path).resolve()),
        "train_reciters": manifest["selection"]["train_reciters"],
        "validation_reciters": manifest["selection"]["validation_reciters"],
        "test_reciters": manifest["selection"]["test_reciters"],
        "stages": stage_summaries,
        "final_model": str(paths["model"].resolve()),
    }
    write_json(summary, paths["model"].parent / "training_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Arabic FastConformer with a reciter-held-out Quran benchmark.")
    parser.add_argument("--config", default="configs/fastconformer_quran.yaml")
    parser.add_argument("--manifest", default="artifacts/manifests/experiment_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run(args.config, args.manifest)
    print(f"Training complete. Final model: {output['final_model']}")
