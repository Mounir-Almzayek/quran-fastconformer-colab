"""Generate the restart-free single-session Colab notebook for FastConformer PC v2."""

from __future__ import annotations


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text}


def save(filename: str, cells: list[dict]) -> None:
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent
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
    (root / "notebooks" / filename).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


MANIFEST = "artifacts/experiments/fastconformer_pc_v2/manifests/experiment_manifest.json"
CONFIG = "configs/fastconformer_quran_pc_v4_encoder_probe.yaml"
EXPERIMENT_ROOT = "artifacts/experiments/fastconformer_pc_v4_encoder_probe"


PROJECT_CELL = '''from pathlib import Path
import os

from google.colab import drive
DRIVE_ROOT = Path("/content/drive/MyDrive")
if not DRIVE_ROOT.exists():
    drive.mount("/content/drive")

PROJECT_DIR = DRIVE_ROOT / "quran-fastconformer-colab"
assert PROJECT_DIR.exists(), f"Project directory not found: {PROJECT_DIR}"
os.chdir(PROJECT_DIR)
print("Working directory:", PROJECT_DIR)
'''


VENV_CELL = '''from pathlib import Path
import hashlib
import os
import shutil
import subprocess
import sys

# Keep project dependencies separate from Colab's preinstalled Python packages.
# This prevents NumPy 1.26.4 (required by NeMo) from replacing Colab's own
# NumPy stack, so no runtime restart is needed after package installation.
VENV_DIR = Path("/content/quran-fastconformer-venv")
VENV_PYTHON = VENV_DIR / "bin" / "python"
VENV_READY = VENV_DIR / ".virtualenv_ready"
REQUIREMENTS = PROJECT_DIR / "requirements.txt"
MARKER = VENV_DIR / ".requirements_sha256"
requirements_hash = hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()

# Google Colab images can disable ensurepip, so use virtualenv's bundled seeder
# rather than the standard-library venv module. If a previous creation failed,
# the missing readiness marker causes the partial directory to be rebuilt.
if not VENV_READY.exists():
    print("Creating isolated project environment with virtualenv...")
    shutil.rmtree(VENV_DIR, ignore_errors=True)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "virtualenv>=20.26,<21"])
    subprocess.check_call([sys.executable, "-m", "virtualenv", "--system-site-packages", str(VENV_DIR)])
    assert VENV_PYTHON.exists(), f"virtualenv creation failed: {VENV_PYTHON}"
    VENV_READY.write_text("ready\\n", encoding="utf-8")

if not MARKER.exists() or MARKER.read_text(encoding="utf-8").strip() != requirements_hash:
    print("Installing project dependencies into the isolated environment. This runs once per fresh Colab runtime...")
    subprocess.check_call([
        str(VENV_PYTHON), "-m", "pip", "install", "--no-cache-dir", "-r", str(REQUIREMENTS)
    ])
    MARKER.write_text(requirements_hash + "\\n", encoding="utf-8")
else:
    print("Isolated project environment is already ready.")


def run_project_script(script: str, *arguments: str) -> None:
    subprocess.check_call(
        [str(VENV_PYTHON), str(PROJECT_DIR / script), *arguments],
        cwd=PROJECT_DIR,
        env=os.environ.copy(),
    )


def run_project_module(module: str, *arguments: str) -> None:
    subprocess.check_call(
        [str(VENV_PYTHON), "-m", module, *arguments],
        cwd=PROJECT_DIR,
        env=os.environ.copy(),
    )

probe = "import jiwer, nemo, numpy, pandas, datasets; print('VENV READY', numpy.__version__, pandas.__version__)"
subprocess.check_call([str(VENV_PYTHON), "-c", probe], cwd=PROJECT_DIR)
print("No Colab restart is required. All project commands below use:", VENV_PYTHON)
'''


def main() -> None:
    save(
        "00_run_pc_v4_encoder_probe_end_to_end.ipynb",
        [
            markdown(
                """# Quran FastConformer PC v4 encoder-only stability probe — one-session Colab workflow

This notebook runs the **PC v4 safety probe**. It reuses the locked PC v2 reciter-held-out manifest and valid baseline, then performs only 100 guarded encoder-only training steps. It stops automatically before exporting a model if CTC WER exceeds the safety limit.

> Use **Runtime → Run all**. The first run may spend time installing packages into the isolated environment, but it does **not** change Colab's own packages and does **not** require a runtime restart."""
            ),
            code(PROJECT_CELL),
            code(
                """# Sync code only. artifacts/ remains untouched.
!git fetch origin main
!git reset --hard origin/main
!git clean -fd -e artifacts/
!git status --short
"""
            ),
            markdown("""## 1. Isolated environment and GPU check

All later project commands run through the local virtual environment, not the notebook kernel's Python."""),
            code(VENV_CELL),
            code(
                """os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
gpu_probe = "import torch; assert torch.cuda.is_available(), 'No GPU'; print('GPU:', torch.cuda.get_device_name(0))"
subprocess.check_call([str(VENV_PYTHON), "-c", gpu_probe], cwd=PROJECT_DIR)
!nvidia-smi
"""
            ),
            markdown(
                """## 2. Locked PC v2 manifest reused by PC v4

PC v4 deliberately reuses the immutable PC v2 split: validation is **parhizgar** and test is **fares_abbad**. It never creates a new split or changes the held-out reciters."""
            ),
            code(
                """manifest_path = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v2/manifests/experiment_manifest.json"
assert manifest_path.exists(), f"Locked PC v2 manifest missing: {manifest_path}"
print("Reusing locked manifest:", manifest_path)
"""
            ),
            code(
                f"""import json
from pathlib import Path

manifest_path = PROJECT_DIR / "{MANIFEST}"
assert manifest_path.exists(), f"Manifest missing: {{manifest_path}}"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
reciter_sets = {{name: {{row["reciter"] for row in manifest["splits"][name]}} for name in ("train", "validation", "test")}}
for name in ("train", "validation", "test"):
    print(f"{{name}}: {{len(manifest['splits'][name])}} clips | reciters={{sorted(reciter_sets[name])}}")
assert not (reciter_sets["train"] & reciter_sets["validation"])
assert not (reciter_sets["train"] & reciter_sets["test"])
assert not (reciter_sets["validation"] & reciter_sets["test"])
receipt = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v2/manifests/manifest_backup_receipt.json"
print("Manifest backups:", json.loads(receipt.read_text(encoding="utf-8"))["backup_paths"])
"""
            ),
            markdown(
                """## 3. Selected audio and NeMo JSONL manifests

Only the 8,000/1,000/1,000 selected clips are materialized. Audio cache stays local for speed; durable JSONL manifests remain on Drive."""
            ),
            code(
                f"""run_project_script(
    "run_colab_setup.py",
    "--config", "{CONFIG}",
    "--manifest", "{MANIFEST}",
    "--materialize",
)
"""
            ),
            markdown(
                """## 4. Fixed baseline before adaptation

The valid PC v2 baseline is copied into the PC v3 report area without rerunning inference. Canonical metrics retain diacritics; lexical metrics are the primary fair measure for this non-diacritized backbone."""
            ),
            code(
                """import shutil

source_baseline = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v2/results/baseline"
target_baseline = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v4_encoder_probe/results/baseline"
assert (source_baseline / "metrics.json").exists(), f"Valid PC v2 baseline is missing: {source_baseline}"
target_baseline.parent.mkdir(parents=True, exist_ok=True)
shutil.copytree(source_baseline, target_baseline, dirs_exist_ok=True)
print("Reused fixed PC v2 baseline in:", target_baseline)
"""
            ),
            code(
                """baseline_path = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v4_encoder_probe/results/baseline/metrics.json"
baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
print("Baseline test reciter(s):", baseline["held_out_test_reciters"])
print(f"Canonical WER / CER: {baseline['strict']['wer_percent']:.2f}% / {baseline['strict']['cer_percent']:.2f}%")
print(f"Lexical WER / CER: {baseline['diagnostic']['wer_percent']:.2f}% / {baseline['diagnostic']['cer_percent']:.2f}%")
"""
            ),
            markdown(
                """## 5. Guarded encoder-only stability probe

A Drive-backed `last.ckpt` is saved every 50 steps. The probe runs only 100 steps and checks held-out CTC WER before exporting its model. If the safety guard stops, do not continue to final evaluation; inspect the printed CTC WER first."""
            ),
            code(
                """state_path = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v4_encoder_probe/nemo/training_state.json"
if state_path.exists():
    print("Existing recovery state:")
    print(json.loads(state_path.read_text(encoding="utf-8")))
else:
    print("No recovery checkpoint yet. Training will start from stage 1.")
"""
            ),
            code(
                f"""run_project_module(
    "src.train",
    "--config", "{CONFIG}",
    "--manifest", "{MANIFEST}",
)
"""
            ),
            code(
                """summary_path = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v4_encoder_probe/models/training_summary.json"
if summary_path.exists():
    print(json.loads(summary_path.read_text(encoding="utf-8")))
else:
    print("Training is incomplete. Re-run the training cell after an interruption to resume from the latest checkpoint.")
"""
            ),
            markdown("""## 6. Final evaluation and Before/After report"""),
            code(
                f"""run_project_module(
    "src.evaluate",
    "--config", "{CONFIG}",
    "--manifest", "{MANIFEST}",
)
"""
            ),
            code(
                """final_path = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v4_encoder_probe/results/finetuned/metrics.json"
final_metrics = json.loads(final_path.read_text(encoding="utf-8"))
print("Final test reciter(s):", final_metrics["held_out_test_reciters"])
print(f"Canonical WER / CER: {final_metrics['strict']['wer_percent']:.2f}% / {final_metrics['strict']['cer_percent']:.2f}%")
print(f"Lexical WER / CER: {final_metrics['diagnostic']['wer_percent']:.2f}% / {final_metrics['diagnostic']['cer_percent']:.2f}%")
"""
            ),
            code(f"""run_project_module("src.compare", "--config", "{CONFIG}")
"""),
            code(
                """from IPython.display import Image, display

comparison_dir = PROJECT_DIR / "artifacts/experiments/fastconformer_pc_v4_encoder_probe/results/comparison"
print((comparison_dir / "metrics_comparison.csv").read_text(encoding="utf-8"))
display(Image(comparison_dir / "metrics_comparison.png"))
print("PC v4 encoder-only stability probe completed.")
"""
            ),
        ],
    )


if __name__ == "__main__":
    main()
