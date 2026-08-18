"""Create a locked-test comparison with strict, diagnostic, and subgroup analyses."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import pandas as pd

from src.common import ensure_project_dirs, load_config, read_json, write_json
from src.evaluation_contract import require_active_metrics
from src.metrics import build_metrics, relative_reduction


_PROTOCOLS = (
    ("strict", "Strict (diacritics retained)"),
    ("diagnostic", "Diagnostic (diacritics-insensitive)"),
)


def _metric_rows(baseline: dict[str, Any], finetuned: dict[str, Any]) -> list[dict[str, Any]]:
    """Build main-table rows for WER and CER under both explicitly named protocols."""
    rows: list[dict[str, Any]] = []
    for protocol_key, protocol_label in _PROTOCOLS:
        for metric_key, metric_label in (("wer", "Word error rate"), ("cer", "Character error rate")):
            before = float(baseline[protocol_key][metric_key])
            after = float(finetuned[protocol_key][metric_key])
            reduction = relative_reduction(before, after)
            rows.append(
                {
                    "protocol": protocol_key,
                    "protocol_label": protocol_label,
                    "metric": metric_key.upper(),
                    "label": f"{protocol_label}: {metric_label}",
                    "baseline_percent": before * 100,
                    "finetuned_percent": after * 100,
                    "absolute_change_points": (after - before) * 100,
                    "relative_error_reduction_percent": None if reduction is None else reduction * 100,
                }
            )
    return rows


def _draw_main_chart(rows: list[dict[str, Any]], destination: Path) -> None:
    """Render the four global error rates with legible protocol labels."""
    labels = [f"{row['protocol']}\n{row['metric']}" for row in rows]
    baseline = [row["baseline_percent"] for row in rows]
    finetuned = [row["finetuned_percent"] for row in rows]
    positions = list(range(len(labels)))
    width = 0.34

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axis = plt.subplots(figsize=(9.0, 4.8), dpi=160)
    baseline_bars = axis.bar([position - width / 2 for position in positions], baseline, width, label="Baseline", color="#7b8794")
    finetuned_bars = axis.bar([position + width / 2 for position in positions], finetuned, width, label="Fine-tuned", color="#126e82")
    axis.set_xticks(positions, labels)
    axis.set_ylabel("Error rate (%)")
    axis.set_title("FastConformer: held-out-reciter before/after comparison")
    axis.legend(frameon=False)
    axis.bar_label(baseline_bars, fmt="%.2f%%", padding=3, fontsize=8)
    axis.bar_label(finetuned_bars, fmt="%.2f%%", padding=3, fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, bbox_inches="tight")
    plt.close(figure)


def _load_joined_predictions(baseline_file: Path, finetuned_file: Path) -> pd.DataFrame:
    """Join stage predictions one-to-one and verify the locked cohort has not changed."""
    baseline = pd.read_json(baseline_file)
    finetuned = pd.read_json(finetuned_file)
    merged = baseline.merge(
        finetuned,
        on="manifest_key",
        suffixes=("_baseline", "_finetuned"),
        validate="one_to_one",
    )
    if not (merged["text_group_id_baseline"] == merged["text_group_id_finetuned"]).all():
        raise RuntimeError("Text-group IDs differ between the baseline and fine-tuned prediction files.")
    return merged


def _protocol_metrics(frame: pd.DataFrame, prediction_column: str, primary: str, diagnostic: str) -> dict[str, Any]:
    """Score one prediction column with exactly the same protocol as the global report."""
    metrics, _ = build_metrics(frame["reference_baseline"], frame[prediction_column], primary, diagnostic)
    return metrics


def _subgroup_rows(
    frame: pd.DataFrame,
    group_type: str,
    group_values: Iterable[tuple[str, pd.DataFrame]],
    primary: str,
    diagnostic: str,
    minimum_examples: int,
) -> list[dict[str, Any]]:
    """Calculate baseline/fine-tuned scores for groups that have enough observations."""
    rows: list[dict[str, Any]] = []
    for group_label, group in group_values:
        if len(group) < minimum_examples:
            continue
        baseline_scores = _protocol_metrics(group, "prediction_baseline", primary, diagnostic)
        finetuned_scores = _protocol_metrics(group, "prediction_finetuned", primary, diagnostic)
        for protocol_key, _ in _PROTOCOLS:
            for metric_key in ("wer", "cer"):
                before = float(baseline_scores[protocol_key][metric_key])
                after = float(finetuned_scores[protocol_key][metric_key])
                reduction = relative_reduction(before, after)
                rows.append(
                    {
                        "group_type": group_type,
                        "group": group_label,
                        "examples": len(group),
                        "protocol": protocol_key,
                        "metric": metric_key.upper(),
                        "baseline_percent": before * 100,
                        "finetuned_percent": after * 100,
                        "absolute_change_points": (after - before) * 100,
                        "relative_error_reduction_percent": None if reduction is None else reduction * 100,
                    }
                )
    return rows


def _duration_groups(frame: pd.DataFrame, bins: list[dict[str, Any]]) -> list[tuple[str, pd.DataFrame]]:
    """Map each test record to one configured half-open duration interval."""
    groups: list[tuple[str, pd.DataFrame]] = []
    duration = pd.to_numeric(frame["duration_baseline"], errors="coerce")
    for item in bins:
        lower, upper = float(item["lower"]), float(item["upper"])
        mask = (duration >= lower) & (duration < upper)
        groups.append((str(item["label"]), frame.loc[mask]))
    return groups


def _representative_examples(merged: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Keep a readable before/after sample with the locked verse-group identifier."""
    columns = [
        "manifest_key",
        "text_group_id_baseline",
        "reciter_baseline",
        "duration_baseline",
        "reference_baseline",
        "prediction_baseline",
        "prediction_finetuned",
    ]
    return merged.loc[:, columns].rename(
        columns={
            "text_group_id_baseline": "text_group_id",
            "reciter_baseline": "reciter",
            "duration_baseline": "duration_seconds",
            "reference_baseline": "reference",
            "prediction_baseline": "baseline_prediction",
            "prediction_finetuned": "finetuned_prediction",
        }
    ).head(limit)


def _write_dataframe(frame: pd.DataFrame, csv_path: Path, json_path: Path) -> None:
    """Write subgroup artifacts even when no group meets the reporting threshold."""
    frame.to_csv(csv_path, index=False)
    frame.to_json(json_path, orient="records", force_ascii=False, indent=2)


def run(config_path: str | Path) -> dict[str, Any]:
    """Create global and subgroup comparisons from matched baseline/fine-tuned predictions."""
    require_active_metrics(
        {
            "strict_wer",
            "strict_cer",
            "diagnostic_wer",
            "diagnostic_cer",
            "per_reciter_breakdown",
            "per_duration_breakdown",
        }
    )
    config = load_config(config_path)
    paths = ensure_project_dirs(config)
    reporting = config["reporting"]
    text_config = config["text"]
    baseline_path = Path(config["_project_root"]) / reporting["baseline_metrics_file"]
    finetuned_path = Path(config["_project_root"]) / reporting["finetuned_metrics_file"]
    baseline = read_json(baseline_path)
    finetuned = read_json(finetuned_path)

    if baseline["manifest_path"] != finetuned["manifest_path"]:
        raise RuntimeError("The baseline and fine-tuned results were produced from different manifests.")
    if baseline.get("metric_protocol") != finetuned.get("metric_protocol"):
        raise RuntimeError("The baseline and fine-tuned results use different metric protocols.")
    expected_protocol = {
        "strict": text_config["primary_normalization"],
        "diagnostic": text_config["diagnostic_normalization"],
    }
    if baseline.get("metric_protocol") != expected_protocol:
        raise RuntimeError("The saved metrics do not match the configuration's strict/diagnostic protocol.")

    comparison_dir = paths["comparison"]
    metric_rows = _metric_rows(baseline, finetuned)
    metrics_frame = pd.DataFrame(metric_rows)
    _write_dataframe(metrics_frame, comparison_dir / "metrics_comparison.csv", comparison_dir / "metrics_comparison.json")
    _draw_main_chart(metric_rows, comparison_dir / "metrics_comparison.png")

    merged = _load_joined_predictions(paths["baseline"] / "predictions.json", paths["finetuned"] / "predictions.json")
    minimum_examples = int(reporting["minimum_group_examples"])
    reciter_groups = [(str(name), group) for name, group in merged.groupby("reciter_baseline", dropna=False)]
    reciter_frame = pd.DataFrame(
        _subgroup_rows(
            merged,
            "reciter",
            reciter_groups,
            text_config["primary_normalization"],
            text_config["diagnostic_normalization"],
            minimum_examples,
        )
    )
    _write_dataframe(reciter_frame, comparison_dir / "metrics_by_reciter.csv", comparison_dir / "metrics_by_reciter.json")

    duration_frame = pd.DataFrame(
        _subgroup_rows(
            merged,
            "duration",
            _duration_groups(merged, reporting["duration_bins_seconds"]),
            text_config["primary_normalization"],
            text_config["diagnostic_normalization"],
            minimum_examples,
        )
    )
    _write_dataframe(duration_frame, comparison_dir / "metrics_by_duration.csv", comparison_dir / "metrics_by_duration.json")

    examples = _representative_examples(merged, int(reporting["representative_examples"]))
    _write_dataframe(examples, comparison_dir / "prediction_examples.csv", comparison_dir / "prediction_examples.json")

    summary = {
        "manifest_path": baseline["manifest_path"],
        "metric_protocol": expected_protocol,
        "metrics": metric_rows,
        "subgroup_policy": {"minimum_examples": minimum_examples, "duration_bins_seconds": reporting["duration_bins_seconds"]},
        "artifacts": {
            "global_table": str((comparison_dir / "metrics_comparison.csv").resolve()),
            "global_chart": str((comparison_dir / "metrics_comparison.png").resolve()),
            "reciter_table": str((comparison_dir / "metrics_by_reciter.csv").resolve()),
            "duration_table": str((comparison_dir / "metrics_by_duration.csv").resolve()),
            "examples": str((comparison_dir / "prediction_examples.csv").resolve()),
        },
    }
    write_json(summary, comparison_dir / "comparison_summary.json")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create strict, diagnostic, and subgroup reports from locked-test evaluations.")
    parser.add_argument("--config", default="configs/whisper_base.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    output = run(arguments.config)
    print(f"Comparison created: {output['artifacts']['global_chart']}")
