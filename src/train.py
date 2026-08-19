"""Fine-tune Arabic FastConformer for Quran ASR with resumable staged unfreezing."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping

from src.common import ensure_project_dirs, load_config, set_seed, utc_now, write_json
from src.data import ensure_nemo_audio_cache
from src.nemo_utils import (
    attach_manifests,
    configure_compact_nemo_logging,
    configure_trainable_layers,
    import_nemo,
    load_exported_ctc_model,
    load_pretrained_ctc_model,
    set_learning_rate,
)


def _manifest_paths(nemo_dir: Path) -> dict[str, Path]:
    """Return the local JSONL manifests generated during setup."""
    manifest_dir = nemo_dir / "manifests"
    result = {split: manifest_dir / f"{split}.jsonl" for split in ("train", "validation", "test")}
    missing = [str(path) for path in result.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing NeMo manifests. Run setup with --materialize first: " + ", ".join(missing))
    return result


def _checkpoint_interval(config: Mapping[str, Any]) -> int:
    """Return a bounded interval for Drive-backed recovery checkpoints."""
    interval = int(config["training"].get("checkpoint_every_n_train_steps", 500))
    if interval < 1:
        raise ValueError("training.checkpoint_every_n_train_steps must be at least 1.")
    return interval


def _stage_dir(nemo_dir: Path, index: int, stage: Mapping[str, Any]) -> Path:
    """Build the durable Drive location for one progressive-unfreezing stage."""
    return nemo_dir / "checkpoints" / f"{index:02d}_{stage['name']}"


def _stage_export_path(stage_dir: Path) -> Path:
    return stage_dir / "stage_model.nemo"


def _stage_record_path(stage_dir: Path) -> Path:
    return stage_dir / "stage_complete.json"


def _read_json_if_present(path: Path) -> dict[str, Any] | None:
    """Read one project JSON artifact without treating an interrupted write as completion."""
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _completed_stage_record(stage_dir: Path) -> dict[str, Any] | None:
    """Return a stage completion record only when both metadata and exported model exist."""
    record = _read_json_if_present(_stage_record_path(stage_dir))
    if record is None or not _stage_export_path(stage_dir).exists():
        return None
    return record


def _latest_recovery_checkpoint(stage_dir: Path) -> Path | None:
    """Locate the newest valid recovery checkpoint written during an unfinished stage."""
    candidates = [stage_dir / "last.ckpt"]
    candidates.extend(sorted(stage_dir.glob("*.ckpt"), key=lambda item: item.stat().st_mtime, reverse=True))
    for candidate in candidates:
        if candidate.exists() and candidate.is_file() and candidate.stat().st_size > 0:
            return candidate
    return None


def _write_run_state(
    nemo_dir: Path,
    *,
    status: str,
    active_stage: Mapping[str, Any] | None = None,
    checkpoint_path: Path | None = None,
    completed_stages: list[dict[str, Any]] | None = None,
) -> Path:
    """Persist compact state for humans; Lightning checkpoints preserve optimizer state."""
    payload: dict[str, Any] = {
        "updated_at": utc_now(),
        "status": status,
        "completed_stages": completed_stages or [],
    }
    if active_stage is not None:
        payload["active_stage"] = {
            "name": active_stage["name"],
            "encoder_layers": active_stage["encoder_layers"],
        }
    if checkpoint_path is not None:
        payload["resume_checkpoint"] = str(checkpoint_path.resolve())
    return write_json(payload, nemo_dir / "training_state.json")


def _metric_text(metrics: Mapping[str, Any], *names: str) -> str:
    """Render the first available Lightning metric without importing torch at module load."""
    for name in names:
        value = metrics.get(name)
        if value is None:
            continue
        try:
            return f"{float(value.detach().cpu() if hasattr(value, 'detach') else value):.4f}"
        except (TypeError, ValueError):
            continue
    return "-"


def _duration_text(seconds: float) -> str:
    seconds = max(0, round(seconds))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def _compact_console_callback(pl: Any, stage_dir: Path, stage_index: int, total_stages: int, stage: Mapping[str, Any], config: Mapping[str, Any]) -> Any:
    """Create a compact console reporter independent of NeMo's per-sample logging."""
    progress_interval = max(1, int(config["training"].get("progress_every_n_train_steps", 100)))
    checkpoint_interval = _checkpoint_interval(config)
    stage_epochs = int(stage["max_epochs"])

    class CompactConsoleReporter(pl.Callback):
        def __init__(self) -> None:
            self.started_at = 0.0
            self.total_steps = 0

        def on_train_start(self, trainer: Any, pl_module: Any) -> None:
            self.started_at = time.perf_counter()
            self.total_steps = max(1, int(trainer.estimated_stepping_batches))
            print("─" * 88)
            print(
                f"[stage {stage_index}/{total_stages}] {stage['name']} | encoder={stage['encoder_layers']} | "
                f"epochs={stage_epochs} | lr={float(stage['learning_rate']):.2e}"
            )
            print(
                f"[data] batch={config['training']['batch_size']} | workers={config['training']['num_workers']} | "
                f"recovery=every {checkpoint_interval} steps → {stage_dir / 'last.ckpt'}"
            )
            print("[progress] step/total | epoch | train_loss | speed | elapsed | ETA | GPU memory")

        def on_train_batch_end(self, trainer: Any, pl_module: Any, outputs: Any, batch: Any, batch_idx: int) -> None:
            step = min(max(int(trainer.global_step), 1), self.total_steps)
            if step % progress_interval != 0 and step != self.total_steps:
                return

            elapsed = max(time.perf_counter() - self.started_at, 0.001)
            rate = step / elapsed
            eta = (self.total_steps - step) / rate if rate else 0.0
            metrics = trainer.callback_metrics
            try:
                import torch

                gpu_memory = f"{torch.cuda.memory_allocated() / 1024**3:.1f} GB" if torch.cuda.is_available() else "CPU"
            except ImportError:
                gpu_memory = "-"
            epoch_label = f"{int(trainer.current_epoch) + 1}/{stage_epochs}"
            print(
                f"[progress] {step:>5}/{self.total_steps:<5} | {epoch_label:^5} | "
                f"{_metric_text(metrics, 'train_loss', 'loss'):>10} | {rate:>5.2f} step/s | "
                f"{_duration_text(elapsed):>7} | {_duration_text(eta):>7} | {gpu_memory}"
            )
            if step % checkpoint_interval == 0 or step == self.total_steps:
                checkpoint_path = stage_dir / "last.ckpt"
                status = "saved" if checkpoint_path.exists() else "pending"
                print(f"[checkpoint] step {step}: {status} → {checkpoint_path}")

        def on_validation_end(self, trainer: Any, pl_module: Any) -> None:
            metrics = trainer.callback_metrics
            print(
                f"[validation] stage {stage_index}/{total_stages} | "
                f"loss={_metric_text(metrics, 'val_loss')} | "
                f"rnnt_wer={_metric_text(metrics, 'val_wer')} | "
                f"ctc_wer={_metric_text(metrics, 'val_wer_ctc')}"
            )

        def on_exception(self, trainer: Any, pl_module: Any, exception: BaseException) -> None:
            checkpoint_path = stage_dir / "last.ckpt"
            try:
                trainer.save_checkpoint(str(checkpoint_path))
                print(f"[recovery] saved interruption checkpoint → {checkpoint_path}")
            except Exception as save_error:
                print(f"[recovery] checkpoint-on-exception unavailable: {type(save_error).__name__}")
            print(f"[recovery] training interrupted: {type(exception).__name__}. Re-run the same command to resume.")

    return CompactConsoleReporter()


def _stage_trainer(
    stage_dir: Path,
    stage_index: int,
    total_stages: int,
    stage: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[Any, Any, Any]:
    """Create one trainer with clean progress output and frequent Drive recovery checkpoints."""
    pl, _, _ = import_nemo()
    monitor_metric = str(config["model"].get("monitor_metric", "val_wer"))
    best_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(stage_dir),
        filename="best-{epoch:02d}-{step:06d}",
        monitor=monitor_metric,
        mode="min",
        save_top_k=1,
    )
    recovery_checkpoint = pl.callbacks.ModelCheckpoint(
        dirpath=str(stage_dir),
        filename="recovery-{step:06d}",
        monitor=None,
        save_top_k=0,
        save_last=True,
        every_n_train_steps=_checkpoint_interval(config),
    )
    console = _compact_console_callback(pl, stage_dir, stage_index, total_stages, stage, config)
    use_gpu = __import__("torch").cuda.is_available()
    trainer = pl.Trainer(
        accelerator="gpu" if use_gpu else "cpu",
        devices=1,
        max_epochs=int(stage["max_epochs"]),
        max_steps=int(stage.get("max_steps", -1)),
        val_check_interval=stage.get("validation_every_n_steps", 1.0),
        precision=config["training"]["precision"] if use_gpu else "32-true",
        gradient_clip_val=float(config["training"]["gradient_clip_val"]),
        callbacks=[best_checkpoint, recovery_checkpoint, console],
        logger=False,
        log_every_n_steps=max(1, int(config["training"].get("progress_every_n_train_steps", 100))),
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        default_root_dir=str(stage_dir),
    )
    return trainer, best_checkpoint, recovery_checkpoint


def _build_stage_record(
    stage: Mapping[str, Any],
    stage_export: Path,
    best_checkpoint: Any,
) -> dict[str, Any]:
    """Build the immutable record that permits later stages to be skipped safely."""
    return {
        "completed_at": utc_now(),
        "name": stage["name"],
        "encoder_layers": stage["encoder_layers"],
        "max_epochs": int(stage["max_epochs"]),
        "learning_rate": float(stage["learning_rate"]),
        "max_steps": int(stage.get("max_steps", -1)),
        "maximum_ctc_wer": stage.get("maximum_ctc_wer"),
        "best_checkpoint": best_checkpoint.best_model_path or None,
        "stage_export": str(stage_export.resolve()),
    }


def run(config_path: str | Path, manifest_path: str | Path) -> dict[str, Any]:
    """Run staged adaptation, automatically resuming Drive-backed interrupted stages."""
    config = load_config(config_path)
    configure_compact_nemo_logging()
    set_seed(int(config["project"]["seed"]))
    paths = ensure_project_dirs(config)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    local_manifests = ensure_nemo_audio_cache(config, manifest, Path(config["_project_root"]))

    model = load_pretrained_ctc_model(config["model"]["pretrained_name"])
    attach_manifests(model, local_manifests["train"], local_manifests["validation"], local_manifests["test"], config)
    stages = config["training"]["stages"]
    print("=" * 88)
    print("Quran FastConformer | resumable Colab fine-tuning")
    print(
        f"[runtime] decoder={config['model']['decoder']} | precision={config['training']['precision']} | "
        f"stages={len(stages)} | checkpoint_interval={_checkpoint_interval(config)}"
    )
    stage_summaries: list[dict[str, Any]] = []

    for index, stage in enumerate(stages, start=1):
        stage_dir = _stage_dir(paths["nemo"], index, stage)
        stage_dir.mkdir(parents=True, exist_ok=True)
        stage_export = _stage_export_path(stage_dir)
        completed_record = _completed_stage_record(stage_dir)

        if completed_record is not None:
            print(f"Stage {index} already completed; loading {stage_export.name} and continuing.")
            model = load_exported_ctc_model(stage_export)
            attach_manifests(model, local_manifests["train"], local_manifests["validation"], local_manifests["test"], config)
            stage_summaries.append(completed_record)
            continue

        configure_trainable_layers(model, str(stage["encoder_layers"]))
        set_learning_rate(model, float(stage["learning_rate"]))
        resume_checkpoint = _latest_recovery_checkpoint(stage_dir)
        _write_run_state(
            paths["nemo"],
            status="running",
            active_stage=stage,
            checkpoint_path=resume_checkpoint,
            completed_stages=stage_summaries,
        )
        if resume_checkpoint is not None:
            print(f"Resuming stage {index} from recovery checkpoint: {resume_checkpoint}")
        else:
            print(f"Starting stage {index} from its declared pretrained/stage-export model.")

        trainer, best_checkpoint, recovery_checkpoint = _stage_trainer(stage_dir, index, len(stages), stage, config)
        trainer.fit(model, ckpt_path=str(resume_checkpoint) if resume_checkpoint is not None else None)

        if trainer.interrupted:
            last_checkpoint = _latest_recovery_checkpoint(stage_dir)
            _write_run_state(
                paths["nemo"],
                status="interrupted",
                active_stage=stage,
                checkpoint_path=last_checkpoint,
                completed_stages=stage_summaries,
            )
            return {
                "status": "interrupted",
                "message": "Training stopped safely. Re-run the same command to resume from the latest recovery checkpoint.",
                "resume_checkpoint": str(last_checkpoint.resolve()) if last_checkpoint is not None else None,
                "stages": stage_summaries,
            }

        maximum_ctc_wer = stage.get("maximum_ctc_wer")
        if maximum_ctc_wer is not None:
            observed = trainer.callback_metrics.get("val_wer_ctc")
            if observed is None:
                raise RuntimeError(
                    "CTC safety guard could not find val_wer_ctc. Configure a validation interval before exporting this stage."
                )
            observed_value = float(observed.detach().cpu() if hasattr(observed, "detach") else observed)
            if observed_value > float(maximum_ctc_wer):
                raise RuntimeError(
                    f"CTC safety guard stopped stage {index}: val_wer_ctc={observed_value:.4f} exceeds "
                    f"the allowed {float(maximum_ctc_wer):.4f}. No stage model was exported."
                )

        model.save_to(str(stage_export))
        stage_record = _build_stage_record(stage, stage_export, best_checkpoint)
        write_json(stage_record, _stage_record_path(stage_dir))
        stage_summaries.append(stage_record)
        _write_run_state(paths["nemo"], status="running", completed_stages=stage_summaries)
        print(f"Stage {index} completed and exported to {stage_export}")

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(paths["model"]))
    summary = {
        "created_at": utc_now(),
        "status": "complete",
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
    _write_run_state(paths["nemo"], status="complete", completed_stages=stage_summaries)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Arabic FastConformer with a reciter-held-out Quran benchmark.")
    parser.add_argument("--config", default="configs/fastconformer_quran.yaml")
    parser.add_argument("--manifest", default="artifacts/manifests/experiment_manifest.json")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    output = run(args.config, args.manifest)
    if output.get("status") == "interrupted":
        print(output["message"])
        if output.get("resume_checkpoint"):
            print(f"Latest checkpoint: {output['resume_checkpoint']}")
    else:
        print(f"Training complete. Final model: {output['final_model']}")
