#!/usr/bin/env python3
"""Check local-WAV overrides without loading the heavyweight NeMo runtime."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import src.nemo_utils as nemo_utils


def main() -> None:
    cfg = SimpleNamespace(
        tokenizer=SimpleNamespace(dir="???"),
        train_ds=SimpleNamespace(
            is_tarred=True,
            tarred_audio_filepaths="???",
            shuffle_n=2048,
            manifest_filepath="???",
        ),
        validation_ds=SimpleNamespace(is_tarred=True, manifest_filepath="???"),
        test_ds=SimpleNamespace(is_tarred=True, manifest_filepath="???"),
    )
    calls = []
    model = SimpleNamespace(
        cfg=cfg,
        setup_training_data=lambda value: calls.append(("train", value)),
        setup_validation_data=lambda value: calls.append(("validation", value)),
        setup_test_data=lambda value: calls.append(("test", value)),
    )
    original = nemo_utils.import_nemo
    nemo_utils.import_nemo = lambda: (None, None, lambda _: nullcontext())
    try:
        nemo_utils.attach_manifests(
            model,
            Path("/tmp/train.jsonl"),
            Path("/tmp/validation.jsonl"),
            Path("/tmp/test.jsonl"),
            {
                "training": {"batch_size": 4, "validation_batch_size": 4, "num_workers": 2},
                "model": {"sample_rate": 16000},
                "dataset": {"min_duration_seconds": 0.3, "max_duration_seconds": 25.0},
            },
        )
    finally:
        nemo_utils.import_nemo = original
    assert cfg.tokenizer.dir is None
    assert cfg.train_ds.is_tarred is False
    assert cfg.train_ds.tarred_audio_filepaths is None
    assert cfg.train_ds.shuffle_n == 0
    assert cfg.train_ds.manifest_filepath == "/tmp/train.jsonl"
    assert cfg.validation_ds.is_tarred is False
    assert cfg.test_ds.is_tarred is False
    assert [name for name, _ in calls] == ["train", "validation", "test"]
    print("Non-tarred NeMo manifest configuration checks passed.")


if __name__ == "__main__":
    main()
