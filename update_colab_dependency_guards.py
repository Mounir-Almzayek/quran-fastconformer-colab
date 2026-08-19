"""Synchronize all Colab notebooks with a NumPy-safe dependency guard.

The guard verifies actual imports rather than only package metadata. If Colab has
mixed NumPy binary files after a package change, it removes the stale NumPy and
pandas package trees, installs the pinned compatible wheels, and explicitly
requires a runtime restart before any dataset or NeMo import.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NOTEBOOKS = ROOT / "notebooks"
TARGETS = (
    "01_setup.ipynb",
    "02_inspect_everyayah.ipynb",
    "03_prepare_nemo_manifests.ipynb",
    "04_baseline_fastconformer.ipynb",
    "05_finetune_fastconformer.ipynb",
    "06_evaluate_fastconformer.ipynb",
    "07_compare_before_after.ipynb",
)

GUARD = '''from pathlib import Path
import importlib.util
import shutil
import site
import subprocess
import sys

# Validate real imports, not merely package metadata. Recent Colab images can
# leave NumPy 2.x binary extensions behind after NeMo pins NumPy 1.26.4.
def _numpy_compatible_error():
    try:
        import numpy as np
        if np.__version__ != "1.26.4":
            return f"NumPy {np.__version__} is installed; this project requires 1.26.4"
        import pandas as pd
        from datasets import load_dataset  # noqa: F401
        return None
    except Exception as error:
        return f"binary/import compatibility check failed: {type(error).__name__}: {error}"


def _remove_stale_binary_packages():
    patterns = ("numpy", "numpy-*.dist-info", "pandas", "pandas-*.dist-info")
    for package_dir in site.getsitepackages():
        root = Path(package_dir)
        for pattern in patterns:
            for target in root.glob(pattern):
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    target.unlink(missing_ok=True)


compatibility_error = _numpy_compatible_error()
required_modules = ("datasets", "jiwer", "soundfile", "yaml", "nemo", "matplotlib")
missing_modules = [name for name in required_modules if importlib.util.find_spec(name) is None]

if compatibility_error:
    print("Repairing incompatible NumPy/pandas binaries:", compatibility_error)
    subprocess.call([sys.executable, "-m", "pip", "uninstall", "-y", "numpy", "pandas"])
    _remove_stale_binary_packages()
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--force-reinstall",
        "numpy==1.26.4",
        "pandas==2.2.2",
    ])
    print("Clean NumPy repair completed. Use Runtime → Restart session before running any other cell.")
elif missing_modules:
    print("Installing missing project dependencies:", missing_modules)
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--no-cache-dir", "-r", "requirements.txt"])
    print("Dependencies updated. Use Runtime → Restart session before launching a NeMo stage.")
else:
    print("Core project dependencies and NumPy binary compatibility are available.")
'''


def _set_source(cell: dict, source: str) -> None:
    existing = cell.get("source", "")
    cell["source"] = source.splitlines(keepends=True) if isinstance(existing, list) else source


def _update_notebook_generator() -> None:
    """Keep make_notebooks.py aligned with the hardened notebook dependency guard."""
    path = ROOT / "make_notebooks.py"
    source = path.read_text(encoding="utf-8")
    marker = "DEPENDENCY_GUARD_CELL = '''"
    start = source.index(marker) + len(marker)
    end_marker = "'''\n\n\ndef main"
    end = source.index(end_marker, start)
    updated = source[:start] + "\n" + GUARD + source[end:]
    path.write_text(updated, encoding="utf-8")
    print(f"Updated NumPy-safe dependency guard: {path.relative_to(ROOT)}")


def main() -> None:
    _update_notebook_generator()
    for name in TARGETS:
        path = NOTEBOOKS / name
        notebook = json.loads(path.read_text(encoding="utf-8"))
        replaced = False
        for cell in notebook.get("cells", []):
            source = cell.get("source", "")
            text = "".join(source) if isinstance(source, list) else source
            if "required_modules =" in text and "subprocess.check_call" in text:
                _set_source(cell, GUARD)
                replaced = True
                break
        if not replaced:
            raise RuntimeError(f"Dependency guard cell not found in {name}.")
        path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Updated NumPy-safe dependency guard: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
