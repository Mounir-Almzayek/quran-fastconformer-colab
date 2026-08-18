"""Shared configuration, filesystem, and reproducibility helpers."""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]


def resolve_path(value: str | Path, project_root: Path = ROOT_DIR) -> Path:
    """Resolve a config path relative to the project root unless it is absolute."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else project_root / path


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration and attach the project root for downstream modules."""
    path = resolve_path(config_path)
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_project_root"] = str(ROOT_DIR)
    return config


def ensure_project_dirs(config: Mapping[str, Any]) -> dict[str, Path]:
    """Create the small metadata/result folders; model files are created by training."""
    root = Path(config["_project_root"])
    artifacts = resolve_path(config["project"]["artifacts_dir"], root)
    training = config.get("training", {})
    paths = {
        "artifacts": artifacts,
        "manifests": artifacts / "manifests",
        "baseline": artifacts / "results" / "baseline",
        "finetuned": artifacts / "results" / "finetuned",
        "comparison": artifacts / "results" / "comparison",
        "nemo": resolve_path(training.get("work_dir", artifacts / "nemo"), root),
        "model": resolve_path(training.get("final_model_path", artifacts / "models" / "fastconformer-quran.nemo"), root),
    }
    for name, path in paths.items():
        # `model` is a `.nemo` file destination; create its parent rather than a
        # directory with the filename.
        (path.parent if name == "model" else path).mkdir(parents=True, exist_ok=True)
    return paths


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch for repeatable sample manifests and training runs."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(payload: Any, path: str | Path) -> Path:
    """Persist structured metadata with UTF-8 Arabic text preserved."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    return target


def read_json(path: str | Path) -> Any:
    """Read a UTF-8 JSON artifact."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def utc_now() -> str:
    """Return an ISO-8601 timestamp for artifact provenance."""
    return datetime.now(timezone.utc).isoformat()


def choose_precision_mode(config: Mapping[str, Any]) -> dict[str, bool]:
    """Choose a safe mixed-precision mode from runtime capability and config intent."""
    import torch

    cuda_available = torch.cuda.is_available()
    cuda_major = torch.cuda.get_device_capability()[0] if cuda_available else 0
    fp16_request = config["training"].get("fp16", "auto")
    bf16_request = config["training"].get("bf16", "auto")

    fp16 = cuda_available if fp16_request == "auto" else bool(fp16_request)
    bf16 = (cuda_available and cuda_major >= 8) if bf16_request == "auto" else bool(bf16_request)
    if bf16:
        fp16 = False
    return {"fp16": fp16, "bf16": bf16}
