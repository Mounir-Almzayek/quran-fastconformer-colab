#!/usr/bin/env python3
"""Dependency-light integrity checks for the FastConformer Quran ASR project."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent


def main() -> None:
    config = yaml.safe_load((ROOT / "configs" / "fastconformer_quran.yaml").read_text(encoding="utf-8"))
    assert config["model"]["pretrained_name"] == "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    assert config["model"]["decoder"] == "ctc"
    assert config["dataset"]["reciter_split"]["strategy"] == "disjoint_reciters"
    assert config["dataset"]["sample_caps"] == {"train": 8000, "validation": 1000, "test": 1000}
    assert config["dataset"]["shuffle_buffer"] <= 1024
    assert [stage["encoder_layers"] for stage in config["training"]["stages"]] == ["top_3", "upper_half", "all"]

    contract = yaml.safe_load((ROOT / "configs" / "evaluation_matrix.yaml").read_text(encoding="utf-8"))
    assert contract["scope"]["current_pipeline"] == "fastconformer_quran_text_asr"
    assert contract["scope"]["primary_test_policy"] == "locked_disjoint_reciter_manifest"

    for source in sorted((ROOT / "src").glob("*.py")):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))

    expected_notebooks = [
        "01_setup.ipynb",
        "02_inspect_everyayah.ipynb",
        "03_prepare_nemo_manifests.ipynb",
        "04_baseline_fastconformer.ipynb",
        "05_finetune_fastconformer.ipynb",
        "06_evaluate_fastconformer.ipynb",
        "07_compare_before_after.ipynb",
    ]
    observed_notebooks = [path.name for path in sorted((ROOT / "notebooks").glob("*.ipynb"))]
    assert observed_notebooks == expected_notebooks
    for notebook_name in observed_notebooks:
        notebook = json.loads((ROOT / "notebooks" / notebook_name).read_text(encoding="utf-8"))
        assert notebook["nbformat"] == 4 and notebook["cells"]

    assert (ROOT / "docs" / "EVALUATION_MATRIX.md").is_file()
    assert not (ROOT / "configs" / "whisper_base.yaml").exists()
    assert not (ROOT / "src" / "modeling.py").exists()
    print("FastConformer project integrity checks passed.")


if __name__ == "__main__":
    main()
