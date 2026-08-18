#!/usr/bin/env python3
"""Regression test for disjoint-reciter manifest construction."""

from __future__ import annotations

from pathlib import Path

from src.common import load_config
from src.data import build_manifest


def fake_candidates(config):
    rows = []
    position = 0
    # Train owns two reciters, but the train cap intentionally selects only the
    # first one. That is valid and previously triggered the faulty check.
    for reciter, count in (("train_a", 8), ("train_b", 8), ("validation_a", 6), ("test_a", 6)):
        for item in range(count):
            rows.append(
                {
                    "stream_position": position,
                    "key": f"{reciter}-{item}",
                    "reciter": reciter,
                    "duration": 2.0,
                    "text": f"نص {reciter} {item}",
                    "text_group_id": f"group-{reciter}-{item}",
                }
            )
            position += 1
    return rows, position


def main() -> None:
    config = load_config("configs/fastconformer_quran.yaml")
    config["dataset"]["sample_caps"] = {"train": 6, "validation": 4, "test": 4}
    config["dataset"]["reciter_split"]["minimum_samples_per_holdout_reciter"] = 4
    config["dataset"]["reciter_split"]["validation_reciters"] = 1
    config["dataset"]["reciter_split"]["test_reciters"] = 1

    import src.data as data_module
    original = data_module._collect_candidates
    data_module._collect_candidates = fake_candidates
    try:
        manifest = build_manifest(config, Path("/tmp/reciter_split_manifest.json"))
    finally:
        data_module._collect_candidates = original
    split_reciters = {split: {row["reciter"] for row in rows} for split, rows in manifest["splits"].items()}
    assert not (split_reciters["train"] & split_reciters["validation"])
    assert not (split_reciters["train"] & split_reciters["test"])
    assert not (split_reciters["validation"] & split_reciters["test"])
    print("Reciter-subset regression check passed.")


if __name__ == "__main__":
    main()
