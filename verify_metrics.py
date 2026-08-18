#!/usr/bin/env python3
"""Validate text-group identity and strict/diagnostic metric semantics."""

from __future__ import annotations

from src.metrics import build_metrics, text_group_key


def main() -> None:
    fully_marked = "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ"
    unmarked = "بسم الله الرحمن الرحيم"
    alternate_alef = "بسم ٱلله الرحمن الرحيم"

    assert text_group_key(fully_marked) == text_group_key(unmarked)
    assert text_group_key(unmarked) == text_group_key(alternate_alef)

    metrics, rows = build_metrics([fully_marked], [unmarked], "canonical", "quranic_light")
    assert metrics["strict"]["cer"] > 0
    assert metrics["diagnostic"]["wer"] == 0
    assert rows[0]["strict_reference"] != rows[0]["strict_prediction"]
    assert rows[0]["diagnostic_reference"] == rows[0]["diagnostic_prediction"]
    print("Metric and text-group checks passed.")


if __name__ == "__main__":
    main()
