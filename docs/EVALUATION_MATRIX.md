# Evaluation Matrix and Reporting Rules

This document is the **reporting contract** for the current project. A metric may be shown as an experiment result only if its stated inputs and ground truth are actually available. The rule prevents accidental presentation of simulated, inferred, or unsupported measurements as real evidence.

## 1. Metrics currently reported

The existing project is a FastConformer text-ASR baseline evaluated on a locked EveryAyah cohort whose test reciters are absent from training. It reports strict and diagnostic WER/CER, then breaks those text-recognition results down by held-out reciter and duration.

| Metric | Current status | What it measures | Output location |
|---|---|---|---|
| Strict WER | **Active** | Word error with diacritics retained after only safe Unicode/spacing cleanup | `metrics.json`, `metrics_comparison.csv` |
| Strict CER | **Active** | Character error under the same strict protocol | `metrics.json`, `metrics_comparison.csv` |
| Diagnostic WER | **Active** | Word error after diacritic-insensitive diagnostic normalization | `metrics.json`, `metrics_comparison.csv` |
| Diagnostic CER | **Active** | Character error after diagnostic normalization | `metrics.json`, `metrics_comparison.csv` |
| Reciter breakdown | **Active** | Whether text-ASR results vary between reciters | `metrics_by_reciter.csv` |
| Duration breakdown | **Active** | Whether text-ASR results vary by recording-length band | `metrics_by_duration.csv` |

> **Primary result:** strict WER and strict CER are the official before/after scores. Diagnostic values explain error type; they never replace the strict result.

## 2. Metrics reserved for a future phoneme pipeline

The project may later include a phoneme recognizer or a pronunciation-alignment model. The following metrics are appropriate then, but only after there are aligned phoneme references and model phoneme predictions.

| Metric | Required inputs | Interpretation |
|---|---|---|
| PER | Reference phoneme sequence and predicted phoneme sequence | Overall phoneme recognition error |
| Phoneme Substitution Rate | Phoneme alignment | One expected sound is replaced by another |
| Phoneme Deletion Rate | Phoneme alignment | An expected sound is missed |
| Phoneme Insertion Rate | Phoneme alignment | An extra sound is predicted |
| Tashkeel F1 | Reference diacritic labels and predicted diacritic labels | Ability to recover pronunciation-relevant vowel/diacritic decisions |

These metrics are **not emitted by the present Whisper text-ASR code**. Their status is explicitly recorded as `reserved_for_phoneme_pipeline` in `configs/evaluation_matrix.yaml`.

## 3. Metrics requiring word/ayah position references

Ayah and word tracking metrics require a trustworthy reference for the position in time. Ayah-level audio/text alone is insufficient to produce defensible word-level timing metrics.

| Metric | Additional requirement before it can be reported |
|---|---|
| Ayah Accuracy | Reference ayah ID paired with model position output |
| Word Accuracy | Reference word position paired with model position output |
| Location Latency | Word/ayah timestamps and timestamped position predictions |
| Acquisition Time | Session-start marker plus position ground truth |
| Reacquisition Time | Labelled loss/jump event plus subsequent position ground truth |

## 4. Metrics intentionally excluded for now

The dataset used in the present text-ASR experiment contains correct recitations and their transcriptions. It does not provide labelled pronunciation mistakes, their onset times, or expected warning outcomes. Therefore the project must not report the following metrics from this data alone.

| Excluded metric | Why it cannot be measured correctly yet |
|---|---|
| Mistake Precision | Requires knowing whether every issued warning matches a labelled real mistake |
| Mistake Recall | Requires a complete list of actual mistakes to determine which ones were missed |
| Mistake F1 | Depends on valid mistake precision and recall |
| False Alarms/minute | Requires labelled correct/incorrect events and generated warnings over time |
| Detection Latency | Requires the timestamp where each real mistake begins |

A later error-detection study needs intentionally incorrect recitations with error type, location, and onset annotations. Synthetic phoneme errors may help during development, but must be identified as synthetic rather than presented as real-reciter performance.

## 5. Runtime metrics

`RTF`, `Peak VRAM`, and `RAM` are valid metrics, but they require a dedicated runtime instrumentation step. They are not emitted by the current training/evaluation notebooks, so they remain deferred rather than blank fields in a report.

## 6. Source of truth

`configs/evaluation_matrix.yaml` is machine-readable and is the source of truth for metric status. If a new evaluation module is added, its output must be moved from **reserved** or **deferred** to **active** only after the declared prerequisites are met.
