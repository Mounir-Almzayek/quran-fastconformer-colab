"""Runtime guard for the project’s declared evaluation scope."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = PROJECT_ROOT / "configs" / "evaluation_matrix.yaml"


def load_evaluation_contract(path: str | Path = CONTRACT_PATH) -> dict[str, Any]:
    """Load the machine-readable metric contract stored with the experiment configuration."""
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def active_metric_ids(contract: dict[str, Any] | None = None) -> set[str]:
    """Return only identifiers approved for emission by the current text-ASR pipeline."""
    resolved = contract or load_evaluation_contract()
    return {str(metric["id"]) for metric in resolved.get("active_metrics", [])}


def require_active_metrics(metric_ids: Iterable[str]) -> None:
    """Prevent a report from silently claiming a metric outside the active contract."""
    active = active_metric_ids()
    requested = set(metric_ids)
    unavailable = requested - active
    if unavailable:
        raise RuntimeError(
            "The report requested metrics that are not active for this project: "
            f"{sorted(unavailable)}. Check configs/evaluation_matrix.yaml and its prerequisites."
        )
