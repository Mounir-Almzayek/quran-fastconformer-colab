"""Generate the single-session Colab notebook for the FastConformer PC v2 workflow."""

from __future__ import annotations

from make_notebooks import DEPENDENCY_GUARD_CELL, PROJECT_CELL, code, markdown, save


MANIFEST = "artifacts/experiments/fastconformer_pc_v2/manifests/experiment_manifest.json"


def main() -> None:
    save(
        "00_run_pc_v2_end_to_end.ipynb",
        [
            markdown(
                """# Quran FastConformer PC v2 — complete Colab workflow

This is the **only notebook needed for normal execution**. It keeps setup, immutable split creation, audio preparation, baseline, restart-safe fine-tuning, final evaluation, and comparison in **one GPU runtime**.

> **First use only:** run the setup cells through **Dependency guard**. If it asks for a runtime restart, restart once and then use **Runtime → Run all** from the top. After the guard reports compatibility, do not move to the old numbered notebooks."""
            ),
            code(
                """from google.colab import drive
drive.mount(\"/content/drive\")
"""
            ),
            code(PROJECT_CELL),
            code(
                """# Keep this Drive copy synchronized with the tested GitHub revision.
# artifacts/ is excluded so manifests, checkpoints, models, and results remain intact.
!git fetch origin main
!git reset --hard origin/main
!git clean -fd -e artifacts/
!git status --short
"""
            ),
            markdown(
                """## 1. Environment readiness

Run the next cell before any dataset or NeMo command. On a new runtime it may install or repair dependencies. **If it prints a restart instruction, restart the runtime immediately, then run this notebook from the top.**"""
            ),
            code(DEPENDENCY_GUARD_CELL),
            code(
                """import os

if os.environ.get(\"QURAN_COLAB_RESTART_REQUIRED\") == \"1\":
    raise RuntimeError(
        \"Restart is required after dependency installation. Use Runtime → Restart session, then Runtime → Run all.\"
    )

import jiwer
import numpy as np
import pandas as pd
from datasets import load_dataset  # noqa: F401

print(\"Environment verified\")
print(\"NumPy:\", np.__version__)
print(\"pandas:\", pd.__version__)
print(\"jiwer: OK | datasets: OK\")
"""
            ),
            code(
                """import torch

os.environ[\"PYTORCH_CUDA_ALLOC_CONF\"] = \"expandable_segments:True\"
assert torch.cuda.is_available(), \"No GPU. Choose Runtime → Change runtime type → T4 GPU or L4 GPU, then restart this notebook.\"
print(\"GPU:\", torch.cuda.get_device_name(0))
!nvidia-smi
"""
            ),
            markdown(
                """## 2. Immutable PC v2 manifest

This cell creates the manifest only if it does not already exist. The held-out reciters are fixed to **parhizgar** for validation and **fares_abbad** for test. Two immutable backup copies are written automatically."""
            ),
            code(
                """!python run_colab_setup.py --config configs/fastconformer_quran.yaml
"""
            ),
            code(
                f"""import json
from pathlib import Path
from src.data import manifest_summary

manifest_path = Path(\"{MANIFEST}\")
assert manifest_path.exists(), f\"Manifest missing: {{manifest_path}}\"
manifest = json.loads(manifest_path.read_text(encoding=\"utf-8\"))
summary = manifest_summary(manifest)
print(summary)
assert summary[\"integrity\"][\"reciter_leakage_count\"] == 0
print(\"Manifest integrity: OK\")
"""
            ),
            markdown(
                """## 3. Selected audio and NeMo JSONL manifests

Only the 8,000/1,000/1,000 selected clips are materialized. The local WAV cache is rebuilt automatically after a Colab reset; the JSONL manifests remain on Drive."""
            ),
            code(
                f"""!python run_colab_setup.py --config configs/fastconformer_quran.yaml --manifest {MANIFEST} --materialize
"""
            ),
            markdown(
                """## 4. Baseline before adaptation

The untouched PC model is evaluated before fine-tuning. Canonical metrics retain diacritics; lexical metrics are the primary fair measure for this non-diacritized backbone."""
            ),
            code(
                f"""!python -m src.baseline --config configs/fastconformer_quran.yaml --manifest {MANIFEST}
"""
            ),
            code(
                """import json
from pathlib import Path

baseline_path = Path(\"artifacts/experiments/fastconformer_pc_v2/results/baseline/metrics.json\")
baseline = json.loads(baseline_path.read_text(encoding=\"utf-8\"))
print(\"Baseline test reciter(s):\", baseline[\"held_out_test_reciters\"])
print(f\"Canonical WER / CER: {baseline['strict']['wer_percent']:.2f}% / {baseline['strict']['cer_percent']:.2f}%\")
print(f\"Lexical WER / CER: {baseline['diagnostic']['wer_percent']:.2f}% / {baseline['diagnostic']['cer_percent']:.2f}%\")
"""
            ),
            markdown(
                """## 5. Fine-tuning with recovery checkpoints

Training is restart-safe. A `last.ckpt` is saved every 500 steps on Drive. If Colab disconnects later, reopen this notebook, run the environment section, then rerun **this training cell**; the unfinished stage resumes automatically."""
            ),
            code(
                """import json
from pathlib import Path

state_path = Path(\"artifacts/experiments/fastconformer_pc_v2/nemo/training_state.json\")
if state_path.exists():
    print(\"Existing recovery state:\")
    print(json.loads(state_path.read_text(encoding=\"utf-8\")))
else:
    print(\"No recovery checkpoint yet. Training will start from stage 1.\")
"""
            ),
            code(
                f"""!python -m src.train --config configs/fastconformer_quran.yaml --manifest {MANIFEST}
"""
            ),
            code(
                """summary_path = Path(\"artifacts/experiments/fastconformer_pc_v2/models/training_summary.json\")
if summary_path.exists():
    print(json.loads(summary_path.read_text(encoding=\"utf-8\")))
else:
    print(\"Training is incomplete. Re-run the training cell after any interruption to resume from the latest checkpoint.\")
"""
            ),
            markdown(
                """## 6. Final evaluation and report

These final cells run only after training completes and produce the Before/After tables and chart."""
            ),
            code(
                f"""!python -m src.evaluate --config configs/fastconformer_quran.yaml --manifest {MANIFEST}
"""
            ),
            code(
                """final_path = Path(\"artifacts/experiments/fastconformer_pc_v2/results/finetuned/metrics.json\")
final_metrics = json.loads(final_path.read_text(encoding=\"utf-8\"))
print(\"Final test reciter(s):\", final_metrics[\"held_out_test_reciters\"])
print(f\"Canonical WER / CER: {final_metrics['strict']['wer_percent']:.2f}% / {final_metrics['strict']['cer_percent']:.2f}%\")
print(f\"Lexical WER / CER: {final_metrics['diagnostic']['wer_percent']:.2f}% / {final_metrics['diagnostic']['cer_percent']:.2f}%\")
"""
            ),
            code(
                """!python -m src.compare --config configs/fastconformer_quran.yaml
"""
            ),
            code(
                """from IPython.display import Image, display

comparison_dir = Path(\"artifacts/experiments/fastconformer_pc_v2/results/comparison\")
display(pd.read_csv(comparison_dir / \"metrics_comparison.csv\"))
display(Image(comparison_dir / \"metrics_comparison.png\"))
print(\"PC v2 workflow completed.\")
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
