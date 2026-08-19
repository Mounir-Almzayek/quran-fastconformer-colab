#!/usr/bin/env python3
"""Dependency-light integrity checks for the FastConformer Quran ASR project."""

from __future__ import annotations

import ast
import json
from pathlib import Path



ROOT = Path(__file__).resolve().parent


def main() -> None:
    config_source = (ROOT / "configs" / "fastconformer_quran.yaml").read_text(encoding="utf-8")
    legacy_config_source = (ROOT / "configs" / "fastconformer_quran_pcd_legacy.yaml").read_text(encoding="utf-8")
    for expected in (
        "pretrained_name: nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0",
        "artifacts_dir: artifacts/experiments/fastconformer_pc_v2",
        "manifest_backup_dir: /content/drive/MyDrive/quran-fastconformer-manifest-backups",
        "validation_reciter_names: [parhizgar]",
        "test_reciter_names: [fares_abbad]",
        "decoder: ctc",
        "strategy: disjoint_reciters",
        "train: 8000",
        "validation: 1000",
        "test: 1000",
        "shuffle_buffer: 1024",
        "runtime_audio_dir: /content/quran-fastconformer-audio",
        "encoder_layers: top_3",
        "encoder_layers: upper_half",
        "encoder_layers: all",
    ):
        assert expected in config_source, f"Missing PC experiment setting: {expected}"
    assert "pretrained_name: nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0" in legacy_config_source
    assert (ROOT / "configs" / "fastconformer_quran_pc_v1_archived.yaml").is_file()

    contract_source = (ROOT / "configs" / "evaluation_matrix.yaml").read_text(encoding="utf-8")
    assert "current_pipeline: fastconformer_quran_text_asr" in contract_source
    assert "primary_test_policy: locked_disjoint_reciter_manifest" in contract_source

    for source in sorted((ROOT / "src").glob("*.py")):
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for script_name in ("run_colab_setup.py", "make_notebooks.py", "make_master_notebook.py", "update_colab_dependency_guards.py"):
        ast.parse((ROOT / script_name).read_text(encoding="utf-8"), filename=script_name)
    notebook_generator = (ROOT / "make_notebooks.py").read_text(encoding="utf-8")
    assert "Create the PC v2 split once" in notebook_generator
    assert "artifacts/experiments/fastconformer_pc_v2/manifests/experiment_manifest.json" in notebook_generator
    assert "artifacts/experiments/fastconformer_pc_v2" in notebook_generator

    requirements = (ROOT / "requirements.txt").read_text(encoding="utf-8")
    assert "numpy==1.26.4" in requirements

    expected_notebooks = [
        "00_run_pc_v2_end_to_end.ipynb",
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

    master = json.loads((ROOT / "notebooks" / "00_run_pc_v2_end_to_end.ipynb").read_text(encoding="utf-8"))
    master_text = "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list) else cell.get("source", "")
        for cell in master["cells"]
    )
    for expected in (
        "only notebook needed for normal execution",
        "QURAN_COLAB_RESTART_REQUIRED",
        "run_colab_setup.py --config configs/fastconformer_quran.yaml",
        "--materialize",
        "python -m src.baseline",
        "python -m src.train",
        "python -m src.evaluate",
        "python -m src.compare",
    ):
        assert expected in master_text, f"Master notebook is missing: {expected}"

    assert (ROOT / "docs" / "EVALUATION_MATRIX.md").is_file()
    assert not (ROOT / "configs" / "whisper_base.yaml").exists()
    assert not (ROOT / "src" / "modeling.py").exists()
    print("FastConformer project integrity checks passed.")


if __name__ == "__main__":
    main()
