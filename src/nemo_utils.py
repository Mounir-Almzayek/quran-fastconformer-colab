"""NeMo-specific helpers kept separate from data selection and report logic."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping


def import_nemo() -> tuple[Any, Any, Any]:
    """Import heavy NeMo dependencies only in Colab execution stages."""
    import lightning.pytorch as pl
    import nemo.collections.asr as nemo_asr
    from omegaconf import open_dict

    return pl, nemo_asr, open_dict


def load_pretrained_ctc_model(model_name: str) -> Any:
    """Load the Arabic hybrid FastConformer and explicitly select its CTC decoder."""
    _, nemo_asr, _ = import_nemo()
    model = nemo_asr.models.EncDecHybridRNNTCTCBPEModel.from_pretrained(model_name=model_name)
    model.change_decoding_strategy(decoder_type="ctc")
    return model


def attach_manifests(model: Any, train_manifest: Path, validation_manifest: Path, test_manifest: Path, config: Mapping[str, Any]) -> None:
    """Point the pretrained model at local WAV/JSONL files while retaining its tokenizer."""
    _, _, open_dict = import_nemo()
    batch_size = int(config["training"]["batch_size"])
    validation_batch_size = int(config["training"]["validation_batch_size"])
    workers = int(config["training"]["num_workers"])
    sample_rate = int(config["model"]["sample_rate"])

    with open_dict(model.cfg):
        model.cfg.train_ds.manifest_filepath = str(train_manifest)
        model.cfg.train_ds.batch_size = batch_size
        model.cfg.train_ds.num_workers = workers
        model.cfg.train_ds.shuffle = True
        model.cfg.train_ds.sample_rate = sample_rate
        model.cfg.train_ds.max_duration = float(config["dataset"]["max_duration_seconds"])
        model.cfg.train_ds.min_duration = float(config["dataset"]["min_duration_seconds"])

        model.cfg.validation_ds.manifest_filepath = str(validation_manifest)
        model.cfg.validation_ds.batch_size = validation_batch_size
        model.cfg.validation_ds.num_workers = workers
        model.cfg.validation_ds.shuffle = False
        model.cfg.validation_ds.sample_rate = sample_rate

        model.cfg.test_ds.manifest_filepath = str(test_manifest)
        model.cfg.test_ds.batch_size = validation_batch_size
        model.cfg.test_ds.num_workers = workers
        model.cfg.test_ds.shuffle = False
        model.cfg.test_ds.sample_rate = sample_rate

    model.setup_training_data(model.cfg.train_ds)
    model.setup_validation_data(model.cfg.validation_ds)
    model.setup_test_data(model.cfg.test_ds)


def configure_trainable_layers(model: Any, policy: str) -> None:
    """Freeze/unfreeze encoder regions while always adapting output-decoder parameters.

    The helper intentionally works by named modules rather than assuming a copied upstream
    training script. It supports the documented three-stage adaptation policy in this project.
    """
    for parameter in model.parameters():
        parameter.requires_grad = False

    encoder_layers = list(getattr(getattr(model, "encoder", None), "layers", []))
    if not encoder_layers:
        raise RuntimeError("The loaded model does not expose encoder.layers required for progressive fine-tuning.")
    if policy == "top_3":
        selected_layers = encoder_layers[-3:]
    elif policy == "upper_half":
        selected_layers = encoder_layers[len(encoder_layers) // 2 :]
    elif policy == "all":
        selected_layers = encoder_layers
    else:
        raise ValueError(f"Unsupported encoder layer policy: {policy}")
    for layer in selected_layers:
        for parameter in layer.parameters():
            parameter.requires_grad = True

    # Adapt decoder-side components so Quranic orthography can move even in the first stage.
    decoder_terms = ("decoder", "joint", "ctc")
    for name, parameter in model.named_parameters():
        if any(term in name.lower() for term in decoder_terms):
            parameter.requires_grad = True


def set_learning_rate(model: Any, learning_rate: float) -> None:
    """Set a conservative AdamW optimizer for an already pretrained ASR model."""
    _, _, open_dict = import_nemo()
    with open_dict(model.cfg):
        model.cfg.optim.name = "adamw"
        model.cfg.optim.lr = float(learning_rate)
        model.cfg.optim.weight_decay = 1.0e-4
    model.setup_optimization(model.cfg.optim)


def normalize_transcriptions(outputs: list[Any]) -> list[str]:
    """Coerce NeMo string/Hypothesis outputs to a plain transcription list."""
    return [str(getattr(item, "text", item)).strip() for item in outputs]
