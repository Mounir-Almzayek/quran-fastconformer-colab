"""Static regression checks for Colab recovery and low-memory training defaults."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main() -> None:
    config_source = (ROOT / "configs" / "fastconformer_quran.yaml").read_text(encoding="utf-8")
    for expected in (
        "pretrained_name: nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0",
        "artifacts_dir: artifacts/experiments/fastconformer_pc",
        "num_workers: 0",
        "pin_memory: false",
        "checkpoint_every_n_train_steps: 500",
    ):
        assert expected in config_source, f"Missing low-RAM configuration: {expected}"

    setup_source = (ROOT / "run_colab_setup.py").read_text(encoding="utf-8")
    ast.parse(setup_source)
    assert '"--manifest"' in setup_source
    assert "Reusing locked manifest" in setup_source
    assert "--manifest and --rebuild-manifest cannot be used together" in setup_source

    train_path = ROOT / "src" / "train.py"
    train_source = train_path.read_text(encoding="utf-8")
    ast.parse(train_source)
    for expected in (
        "save_on_exception=True",
        "every_n_train_steps=_checkpoint_interval(config)",
        "trainer.fit(model, ckpt_path=",
        "training_state.json",
        "stage_complete.json",
        "load_exported_ctc_model",
        "CompactConsoleReporter",
        "enable_progress_bar=False",
        "[progress] step/total",
        "[checkpoint] step",
        "[validation] stage",
    ):
        assert expected in train_source, f"Missing recovery behavior: {expected}"

    nemo_source = (ROOT / "src" / "nemo_utils.py").read_text(encoding="utf-8")
    ast.parse(nemo_source)
    assert "load_exported_ctc_model" in nemo_source
    assert "model.cfg.train_ds.pin_memory = pin_memory" in nemo_source
    assert "configure_compact_nemo_logging" in nemo_source
    assert "metric.log_prediction = False" in nemo_source

    notebook = json.loads((ROOT / "notebooks" / "05_finetune_fastconformer.ipynb").read_text(encoding="utf-8"))
    notebook_sources = []
    for cell in notebook["cells"]:
        source = cell.get("source", "")
        notebook_sources.append("".join(source) if isinstance(source, list) else source)
    notebook_text = "\n".join(notebook_sources)
    assert "restart-safe" in notebook_text
    assert "training_state.json" in notebook_text
    assert "console is intentionally compact" in notebook_text
    assert "artifacts/experiments/fastconformer_pc" in notebook_text

    setup_notebook = json.loads((ROOT / "notebooks" / "01_setup.ipynb").read_text(encoding="utf-8"))
    setup_text = "\n".join(
        "".join(cell.get("source", "")) if isinstance(cell.get("source", ""), list) else cell.get("source", "")
        for cell in setup_notebook["cells"]
    )
    assert "--manifest artifacts/manifests/experiment_manifest.json" in setup_text

    print("Passed: PC isolation, locked-manifest reuse, recovery checkpoints, low-RAM defaults, and compact console logging are configured.")


if __name__ == "__main__":
    main()
