#!/usr/bin/env python3
"""Generate the ordered Google Colab notebooks for FastConformer Quran ASR."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def save(filename: str, cells: list[dict]) -> None:
    payload = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "include_colab_link": True},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    (NOTEBOOKS / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


PROJECT_CELL = '''from pathlib import Path
import os

# Each notebook may open in a fresh Colab runtime, so mount Drive before any
# path check rather than relying on a previous notebook's session.
from google.colab import drive
DRIVE_ROOT = Path("/content/drive/MyDrive")
if not DRIVE_ROOT.exists():
    drive.mount("/content/drive")

# Place the *contents* of this repository in this Google Drive folder, or edit
# this one variable to match the folder you chose.
PROJECT_DIR = DRIVE_ROOT / "quran-fastconformer-colab"
assert PROJECT_DIR.exists(), f"Project directory not found: {PROJECT_DIR}"
os.chdir(PROJECT_DIR)
print("Working directory:", Path.cwd())
'''

DEPENDENCY_GUARD_CELL = '''import importlib.util
import subprocess
import sys

# A notebook can be opened after a runtime restart. Install project dependencies
# only when a required module is missing, rather than assuming notebook 01 ran.
required_modules = ("datasets", "jiwer", "soundfile", "yaml", "nemo", "pandas", "matplotlib")

missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]
needs_numpy_downgrade = False
if not missing_modules:
    import numpy as np
    needs_numpy_downgrade = int(np.__version__.split(".")[0]) >= 2

if missing_modules or needs_numpy_downgrade:
    reason = missing_modules or ["numpy<2 required by the current NeMo audio loader"]
    print("Installing compatible runtime dependencies:", reason)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "-r", "requirements.txt"])
    print("Dependencies updated. Restart the runtime once before launching a NeMo stage.")
else:
    print("Core project dependencies are available.")
'''


def main() -> None:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)

    save("01_setup.ipynb", [
        markdown("""# 01 — FastConformer environment and reciter-held-out manifest

This notebook prepares a single-GPU Colab runtime, validates the current EveryAyah schema, and creates the one immutable experiment manifest. The split is by **reciter**, not random clip: a reciter assigned to Test cannot appear in Train or Validation."""),
        markdown("""> **Do not recreate the manifest after the baseline.** A new manifest changes test reciters and invalidates Before/After comparison."""),
        code("""from google.colab import drive
drive.mount('/content/drive')
"""),
        code(PROJECT_CELL),
        code(DEPENDENCY_GUARD_CELL),
        code("""!nvidia-smi
"""),
        code("""# Create and validate the disjoint-reciter split. No audio files are downloaded in this first step.
!python run_colab_setup.py --config configs/fastconformer_quran.yaml
"""),
    ])

    save("02_inspect_everyayah.ipynb", [
        markdown("""# 02 — Inspect source audio, text, and reciter metadata

The source is read with streaming. This notebook is a schema sanity check; it does not download the full dataset."""),
        code(PROJECT_CELL),
        code(DEPENDENCY_GUARD_CELL),
        code("""from datasets import load_dataset

stream = load_dataset("tarteel-ai/everyayah", name="default", split="train", streaming=True)
for index, item in zip(range(3), stream):
    print(f"Sample {index}")
    print("  reciter:", item["reciter"])
    print("  duration:", item["duration"])
    print("  text:", item["text"][:160])
    print("  sample rate:", item["audio"]["sampling_rate"])
"""),
    ])

    save("03_prepare_nemo_manifests.ipynb", [
        markdown("""# 03 — Materialize only the selected NeMo data

NeMo training requires local audio paths. This stage downloads **only** the rows selected in the manifest and writes WAV files plus JSONL manifests for Train, Validation, and Test. The rest of EveryAyah remains undownloaded."""),
        code(PROJECT_CELL),
        code(DEPENDENCY_GUARD_CELL),
        code("""!python run_colab_setup.py --config configs/fastconformer_quran.yaml --materialize
"""),
        code("""import json
from pathlib import Path
from src.data import manifest_summary

manifest = json.loads(Path("artifacts/manifests/experiment_manifest.json").read_text(encoding="utf-8"))
summary = manifest_summary(manifest)
summary
"""),
        code("""# Scientific integrity guard: every reciter must be owned by one split only.
reciter_sets = {split: set(info["reciters"]) for split, info in summary.items() if split != "integrity"}
for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
    assert not (reciter_sets[left] & reciter_sets[right]), f"Reciter leakage: {left}/{right}"
assert summary["integrity"]["reciter_leakage_count"] == 0
print("Passed: all validation/test reciters are unseen during training.")
"""),
    ])

    save("04_baseline_fastconformer.ipynb", [
        markdown("""# 04 — Baseline: pretrained Arabic FastConformer

This measures the untouched NVIDIA Arabic FastConformer on the held-out test reciter(s). The result is the only valid Before value for this manifest."""),
        code(PROJECT_CELL),
        code(DEPENDENCY_GUARD_CELL),
        code("""!python -m src.baseline --config configs/fastconformer_quran.yaml --manifest artifacts/manifests/experiment_manifest.json
"""),
        code("""import json
from pathlib import Path
metrics = json.loads(Path("artifacts/results/baseline/metrics.json").read_text(encoding="utf-8"))
print("Held-out reciter(s):", metrics["held_out_test_reciters"])
print(f"Strict WER / CER: {metrics['strict']['wer_percent']:.2f}% / {metrics['strict']['cer_percent']:.2f}%")
print(f"Diagnostic WER / CER: {metrics['diagnostic']['wer_percent']:.2f}% / {metrics['diagnostic']['cer_percent']:.2f}%")
"""),
    ])

    save("05_finetune_fastconformer.ipynb", [
        markdown("""# 05 — Fine-tune with progressive unfreezing

The workflow keeps the pretrained Arabic tokenizer, selects CTC decoding, and performs conservative staged adaptation: top three encoder layers, upper half, then the full model. Every stage stores checkpoints under the Drive-backed project folder."""),
        code(PROJECT_CELL),
        code(DEPENDENCY_GUARD_CELL),
        code("""import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
"""),
        code("""!python -m src.train --config configs/fastconformer_quran.yaml --manifest artifacts/manifests/experiment_manifest.json
"""),
        code("""import json
from pathlib import Path
summary = json.loads(Path("artifacts/models/training_summary.json").read_text(encoding="utf-8"))
summary
"""),
    ])

    save("06_evaluate_fastconformer.ipynb", [
        markdown("""# 06 — Final FastConformer evaluation

This evaluates the exported `.nemo` file on the same held-out reciter set used in notebook 04. It does not select checkpoints or change data membership."""),
        code(PROJECT_CELL),
        code(DEPENDENCY_GUARD_CELL),
        code("""!python -m src.evaluate --config configs/fastconformer_quran.yaml --manifest artifacts/manifests/experiment_manifest.json
"""),
        code("""import json
from pathlib import Path
metrics = json.loads(Path("artifacts/results/finetuned/metrics.json").read_text(encoding="utf-8"))
print("Held-out reciter(s):", metrics["held_out_test_reciters"])
print(f"Strict WER / CER: {metrics['strict']['wer_percent']:.2f}% / {metrics['strict']['cer_percent']:.2f}%")
print(f"Diagnostic WER / CER: {metrics['diagnostic']['wer_percent']:.2f}% / {metrics['diagnostic']['cer_percent']:.2f}%")
"""),
    ])

    save("07_compare_before_after.ipynb", [
        markdown("""# 07 — Before/After analysis

This final notebook verifies that baseline and final metrics use the same manifest, then creates strict/diagnostic global scores plus breakouts by held-out reciter and recording duration."""),
        code(PROJECT_CELL),
        code(DEPENDENCY_GUARD_CELL),
        code("""!python -m src.compare --config configs/fastconformer_quran.yaml
"""),
        code("""import pandas as pd
from IPython.display import Image, display

display(pd.read_csv("artifacts/results/comparison/metrics_comparison.csv"))
display(Image("artifacts/results/comparison/metrics_comparison.png"))
"""),
        code("""print("Performance by reciter")
display(pd.read_csv("artifacts/results/comparison/metrics_by_reciter.csv"))
print("Performance by duration")
display(pd.read_csv("artifacts/results/comparison/metrics_by_duration.csv"))
print("Representative predictions")
display(pd.read_csv("artifacts/results/comparison/prediction_examples.csv").head(12))
"""),
    ])


if __name__ == "__main__":
    main()
