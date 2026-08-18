"""Arabic transcript normalization, grouped evaluation metrics, and prediction persistence."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Mapping

import jiwer



_TATWEEL = "ـ"
_WHITESPACE = re.compile(r"\s+")
_ARABIC_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCTUATION = re.compile(r"[^\w\s\u0600-\u06FF]")
_COMMON_ALEF = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ؤ": "و", "ئ": "ي"})


def normalize_text(text: str, mode: str = "canonical") -> str:
    """Normalize Arabic transcription according to a named, auditable comparison policy.

    ``canonical`` is the primary scoring mode: it retains Quranic diacritics and only
    applies Unicode NFC, tatweel removal, and whitespace cleanup. ``quranic_light`` is
    diagnostic only: it removes diacritics and punctuation while normalizing common Arabic
    letter variants, helping distinguish lexical mistakes from surface-form differences.
    """
    normalized = unicodedata.normalize("NFC", str(text)).replace(_TATWEEL, "")
    normalized = _WHITESPACE.sub(" ", normalized).strip()
    if mode == "canonical":
        return normalized
    if mode in {"light", "quranic_light"}:
        normalized = _ARABIC_DIACRITICS.sub("", normalized)
        normalized = _PUNCTUATION.sub(" ", normalized).translate(_COMMON_ALEF)
        return _WHITESPACE.sub(" ", normalized).strip()
    raise ValueError(f"Unsupported normalization mode: {mode}")


def text_group_key(text: str) -> str:
    """Return a deterministic identifier for a verse-text group, ignoring surface variants."""
    normalized = normalize_text(text, "quranic_light")
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"verse_{digest}"


def _corpus_error_rates(references: list[str], predictions: list[str]) -> dict[str, float]:
    """Compute corpus-level WER and CER on aligned, already normalized strings."""
    if not references:
        raise ValueError("No references were provided for metric computation.")
    return {
        "wer": float(jiwer.wer(references, predictions)),
        "cer": float(jiwer.cer(references, predictions)),
    }


def _named_scores(references: list[str], predictions: list[str], normalization: str) -> dict[str, Any]:
    """Calculate one named metric protocol and include both proportions and percentages."""
    normalized_references = [normalize_text(value, normalization) for value in references]
    normalized_predictions = [normalize_text(value, normalization) for value in predictions]
    scores = _corpus_error_rates(normalized_references, normalized_predictions)
    return {
        "normalization": normalization,
        "wer": scores["wer"],
        "cer": scores["cer"],
        "wer_percent": scores["wer"] * 100,
        "cer_percent": scores["cer"] * 100,
        "references": normalized_references,
        "predictions": normalized_predictions,
    }


def build_metrics(
    references: Iterable[str],
    predictions: Iterable[str],
    primary_normalization: str = "canonical",
    diagnostic_normalization: str = "quranic_light",
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    """Return strict primary scores, diagnostic scores, and row-level normalization evidence."""
    references_list = list(references)
    predictions_list = list(predictions)
    if len(references_list) != len(predictions_list):
        raise ValueError("Reference and prediction counts differ.")

    primary = _named_scores(references_list, predictions_list, primary_normalization)
    diagnostic = _named_scores(references_list, predictions_list, diagnostic_normalization)
    rows = [
        {
            "reference": reference,
            "prediction": prediction,
            "strict_reference": strict_reference,
            "strict_prediction": strict_prediction,
            "diagnostic_reference": diagnostic_reference,
            "diagnostic_prediction": diagnostic_prediction,
        }
        for reference, prediction, strict_reference, strict_prediction, diagnostic_reference, diagnostic_prediction in zip(
            references_list,
            predictions_list,
            primary.pop("references"),
            primary.pop("predictions"),
            diagnostic.pop("references"),
            diagnostic.pop("predictions"),
        )
    ]
    metrics = {
        # The top-level values intentionally mirror the strict primary protocol for a
        # simple, backwards-compatible main score.
        "normalization": primary_normalization,
        "examples_evaluated": len(rows),
        "wer": primary["wer"],
        "cer": primary["cer"],
        "wer_percent": primary["wer_percent"],
        "cer_percent": primary["cer_percent"],
        "strict": primary,
        "diagnostic": diagnostic,
    }
    return metrics, rows


def save_predictions(rows: list[Mapping[str, Any]], csv_path: str | Path, json_path: str | Path) -> None:
    """Save prediction rows in both portable CSV and lossless UTF-8 JSON formats."""
    csv_target = Path(csv_path)
    csv_target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["reference", "prediction"]
    with csv_target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    # Import lazily so pure text metrics remain usable without a Torch runtime.
    from src.common import write_json

    write_json(rows, json_path)


def relative_reduction(before: float, after: float) -> float | None:
    """Return relative error reduction, or None when the baseline error is exactly zero."""
    if before == 0:
        return None
    return (before - after) / before
