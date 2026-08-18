#!/usr/bin/env python3
"""Exercise comparison outputs without loading a NeMo model or downloading audio."""

from __future__ import annotations

import shutil
from pathlib import Path

from src.common import ROOT_DIR, write_json
from src.compare import run
from src.metrics import build_metrics, save_predictions


def write_stage(stage: str, predictions: list[str]) -> None:
    references = [
        "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
        "الْحَمْدُ لِلَّهِ رَبِّ الْعَالَمِينَ",
        "مَالِكِ يَوْمِ الدِّينِ",
        "إِيَّاكَ نَعْبُدُ وَإِيَّاكَ نَسْتَعِينُ",
        "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
        "قُلْ هُوَ اللَّهُ أَحَدٌ",
    ]
    metrics, normalized_rows = build_metrics(references, predictions, "canonical", "quranic_light")
    rows = []
    for index, (reference, prediction, normalized) in enumerate(zip(references, predictions, normalized_rows)):
        rows.append(
            {
                "audio_filepath": f"/tmp/{index}.wav",
                "duration": 5.0 if index < 3 else 15.0,
                "text": reference,
                "reciter": "heldout_reciter_a" if index < 3 else "heldout_reciter_b",
                "manifest_key": f"sample-{index}",
                "text_group_id": f"verse-{index}",
                "reference": reference,
                "prediction": prediction,
                **normalized,
            }
        )
    metrics.update(
        {
            "stage": stage,
            "model": "test-model",
            "manifest_path": "/tmp/disjoint-reciter-manifest.json",
            "metric_protocol": {"strict": "canonical", "diagnostic": "quranic_light"},
            "held_out_test_reciters": ["heldout_reciter_a", "heldout_reciter_b"],
        }
    )
    folder = ROOT_DIR / "artifacts" / "results" / stage
    save_predictions(rows, folder / "predictions.csv", folder / "predictions.json")
    write_json(metrics, folder / "metrics.json")


def main() -> None:
    try:
        write_stage("baseline", ["بسم الله الرحمن الرحيم"] * 6)
        write_stage(
            "finetuned",
            [
                "بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ",
                "الحمد لله رب العالمين",
                "مَالِكِ يَوْمِ الدِّينِ",
                "اياك نعبد واياك نستعين",
                "اهْدِنَا الصِّرَاطَ الْمُسْتَقِيمَ",
                "قُلْ هُوَ اللَّهُ أَحَدٌ",
            ],
        )
        result = run("configs/fastconformer_quran.yaml")
        required = {"global_table", "global_chart", "reciter_table", "duration_table", "examples"}
        assert set(result["artifacts"]) == required
        assert all(Path(path).exists() for path in result["artifacts"].values())
        print("FastConformer comparison checks passed.")
    finally:
        shutil.rmtree(ROOT_DIR / "artifacts", ignore_errors=True)


if __name__ == "__main__":
    main()
