# Quran FastConformer ASR Fine-tuning on Google Colab

This project is an **independent FastConformer Quran ASR experiment** built for a single Google Colab GPU. It starts from NVIDIA’s Arabic hybrid FastConformer checkpoint, retains its pretrained BPE tokenizer, selects CTC decoding, and adapts the model to Quranic recitation with NVIDIA NeMo.

> This repository deliberately implements only **Stage 1: Quran ASR domain adaptation**. It does not claim to measure phoneme, tajweed, or mispronunciation metrics that lack the required labels. Those later capabilities need a separate alignment/pronunciation pipeline.

| Design choice | Implementation |
|---|---|
| Backbone | `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` loaded through NVIDIA NeMo |
| Decoder | CTC, chosen for deterministic token timing/alignment compatibility in later work |
| Tokenizer | The pretrained model’s BPE tokenizer is retained; this experiment does not invent or copy a tokenizer |
| Dataset | `tarteel-ai/everyayah`, read with streaming during selection |
| Split | **Disjoint-reciter** Train/Validation/Test partitions; no reciter crosses a boundary |
| Cohort size | Capped to 8,000 train, 1,000 validation, and 1,000 test clips for a first Colab-scale experiment |
| Baseline | The untouched NVIDIA model is measured before fine-tuning on the held-out reciter set |
| Adaptation | Conservative staged unfreezing: top encoder layers, upper encoder, then all layers |
| Final evidence | Exact same held-out reciter manifest is evaluated before and after training |

The official EveryAyah card provides Arabic recitation audio with diacritized `text`, `duration`, and `reciter` fields. The project checks these fields at runtime rather than assuming that their schema is permanent.[1]

## Why the split is by reciter

A random clip split can place one reciter’s voice in both Train and Test, causing a speech model to look better because it has already learned the speaker rather than because it generalizes to Quranic recitation. This project assigns entire reciters to one partition.

```text
Train reciters        Validation reciter       Test reciter
A, B, C, ...                 D                      E
        │                    │                      │
     Fine-tuning       Checkpoint selection    Before/After evidence
```

The created manifest records the held-out identities and fails if a reciter crosses a split boundary. It is the evidence behind the final comparison; do not recreate it after running the baseline.

## Folder layout

```text
quran-fastconformer-colab/
├── configs/
│   ├── fastconformer_quran.yaml      # Data, split, NeMo, and training policy
│   └── evaluation_matrix.yaml        # Reportable metric contract
├── docs/
│   └── EVALUATION_MATRIX.md          # Active and deferred metric definitions
├── notebooks/
│   ├── 01_setup.ipynb
│   ├── 02_inspect_everyayah.ipynb
│   ├── 03_prepare_nemo_manifests.ipynb
│   ├── 04_baseline_fastconformer.ipynb
│   ├── 05_finetune_fastconformer.ipynb
│   ├── 06_evaluate_fastconformer.ipynb
│   └── 07_compare_before_after.ipynb
├── src/
│   ├── data.py                       # Streaming selection, WAV/JSONL materialization
│   ├── nemo_utils.py                 # NeMo model loading and staged unfreezing
│   ├── baseline.py                   # Pre-fine-tuning CTC evaluation
│   ├── train.py                      # Progressive NeMo adaptation
│   ├── evaluate.py                   # Final `.nemo` evaluation
│   ├── metrics.py                    # Strict/diagnostic WER and CER
│   └── compare.py                    # Before/After reports
├── run_colab_setup.py
└── requirements.txt
```

## Run order in Colab

Copy the project contents to a Google Drive folder such as `MyDrive/quran-fastconformer-colab`. If you choose another name, change `PROJECT_DIR` in the first code cell of each notebook. Select **Runtime → Change runtime type → GPU** in Colab, then execute the notebooks in this order.

| Notebook | Purpose | Main output |
|---|---|---|
| `01_setup` | Mounts Drive, installs NeMo, checks the schema, and creates the immutable disjoint-reciter manifest | `artifacts/manifests/experiment_manifest.json` |
| `02_inspect_everyayah` | Inspects a few real streamed rows | Schema confirmation |
| `03_prepare_nemo_manifests` | Downloads only the selected clips and creates local WAV/NeMo JSONL files | `artifacts/nemo/{audio,manifests}/` |
| `04_baseline_fastconformer` | Measures untouched Arabic FastConformer on held-out test reciter(s) | Baseline WER/CER and predictions |
| `05_finetune_fastconformer` | Runs progressive unfreezing stages through NeMo | Stage checkpoints and `fastconformer-quran.nemo` |
| `06_evaluate_fastconformer` | Measures final model on the unchanged test reciter(s) | Final WER/CER and predictions |
| `07_compare_before_after` | Builds global and subgroup comparison reports | CSV, JSON, PNG, examples |

## Training policy

This is a **fine-tuning**, not training-from-scratch, experiment. The pretrained Arabic FastConformer already contains general Arabic acoustic/language representations. The project uses a small learning rate and stages unfreezing to reduce the risk of catastrophic forgetting while adapting the model to Quranic pronunciation and orthography.

| Stage | Trainable encoder region | Epochs | Learning rate |
|---|---|---:|---:|
| 1 | Top 3 encoder layers + decoder-side parameters | 1 | `5e-5` |
| 2 | Upper half of encoder + decoder-side parameters | 1 | `2e-5` |
| 3 | Full encoder + decoder-side parameters | 2 | `1e-5` |

These are conservative initial settings for a Colab experiment, not claims about the hyperparameters of any external project. NeMo supports pretrained ASR checkpoints, BPE tokenizers, JSONL manifests, CTC models, and FastConformer configuration/training workflows.[2]

## Evaluation policy

The active metrics are strict and diagnostic WER/CER, with optional analysis by held-out reciter and recording duration. Strict scores retain diacritics after safe Unicode cleanup; diagnostic scores remove diacritics and normalize selected Arabic variants only to explain error type.

The detailed contract appears in [`docs/EVALUATION_MATRIX.md`](docs/EVALUATION_MATRIX.md). It explicitly defers PER, phoneme substitutions/deletions/insertions, Tashkeel F1, location metrics, RTF, memory metrics, and mispronunciation-detection metrics until their required labels, model outputs, or runtime instrumentation exist.

> A strong STT score does **not** prove correct madd duration or tajweed execution. Those require a later pronunciation system using encoder features, expected phonetic structure, alignment, and labelled evaluation data.

## Output artifacts

| Artifact | Purpose |
|---|---|
| `artifacts/manifests/experiment_manifest.json` | Reproducible evidence of the held-out-reciter split |
| `artifacts/nemo/manifests/*.jsonl` | Local NeMo audio/text manifests |
| `results/baseline/` | Untouched-model predictions and metrics |
| `results/finetuned/` | Fine-tuned-model predictions and metrics |
| `models/fastconformer-quran.nemo` | Exported final NeMo model |
| `results/comparison/` | Before/After tables, chart, per-reciter/duration analysis, examples |

## References

[1] [Hugging Face — Tarteel AI EveryAyah dataset card](https://huggingface.co/datasets/tarteel-ai/everyayah)

[2] [NVIDIA NeMo — ASR models and FastConformer documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/asr/models.html)

[3] [NVIDIA FastConformer CTC-BPE training entry point](https://github.com/NVIDIA-NeMo/NeMo/blob/main/examples/asr/asr_ctc/speech_to_text_ctc_bpe.py)
