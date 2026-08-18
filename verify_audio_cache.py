#!/usr/bin/env python3
"""Check the local-audio manifest contract without downloading EveryAyah."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from src.data import _validate_materialized_manifests, _write_wav_atomically


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        audio = root / "audio" / "sample.wav"
        _write_wav_atomically(audio, np.zeros(1600, dtype=np.float32), 16000)
        manifest_path = root / "test.jsonl"
        manifest_path.write_text(
            json.dumps({"audio_filepath": str(audio), "text": "بِسْمِ اللَّهِ", "duration": 0.1}) + "\n",
            encoding="utf-8",
        )
        manifest = {"splits": {"test": [{}]}}
        _validate_materialized_manifests({"test": manifest_path}, manifest)
        audio.unlink()
        try:
            _validate_materialized_manifests({"test": manifest_path}, manifest)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Missing WAV file was not detected")
    print("Local audio cache checks passed.")


if __name__ == "__main__":
    main()
