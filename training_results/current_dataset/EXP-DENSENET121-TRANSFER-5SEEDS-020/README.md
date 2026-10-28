# EXP-DENSENET121-TRANSFER-5SEEDS-020 — V09; Holdout03 · 126 images, 4 supported classes

> **Backfill of an experiment archive provided by the project owner.** This is a curated GitHub text/CSV extract; **not** an upload of the entire ANALYSIS ZIP, per-image predictions, PNG figures, original notebook, or binary checkpoints.

## Protocol

- **Source experiment/archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-020_ANALYSIS.zip`
- **Original archive SHA256:** `ecdebc3eaa6e40fbf22a930b0735146e1229c77943ae03b2814e124b69c98262`
- **Data:** V09; Holdout03 · 126 images, 4 supported classes
- **Input/augmentation:** 224×224; source EXP-018
- **Training lineage:** Evaluation-only; REUSED_EXP018 checkpoints (seed selection from EXP-018 Validation)
- **Execution status:** Holdout03 COMPLETE; Training NOT RUN; Final Test NOT RUN.
- **Fixed seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked seed:** **123**. Never choose or change seed using Test/Holdout.
- **Result summary:** Holdout Accuracy 68.25 ± 2.69%; four-class Macro F1 64.60 ± 3.20%.

## 5-seed results

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| holdout | 42 | 69.84% | 67.90% |
| holdout | 123 | 66.67% | 63.14% |
| holdout | 2026 | 64.29% | 59.77% |
| holdout | 3407 | 69.84% | 65.76% |
| holdout | 7777 | 70.63% | 66.43% |

Numbers in this table are rounded to four decimal places in fraction form in [`metrics_5seeds.csv`](./metrics_5seeds.csv); consult the original archive for unrounded source metrics, per-class details and image-level predictions. On Holdout, Macro F1 is calculated for **four classes with true support**; `not_cashew_leaf` has no true Holdout instances.

## Interpretation and guardrails

Compared with Holdout02 (127 images), one Healthy image was removed and 8 Healthy images were modified. This changes evaluation membership/content; not a training improvement. Historical Holdout exposed.

- The original Test is locked; historical Test/Holdouts have been inspected across development runs, so subsequent analysis is **retrospective**, not a new blind field test.
- Different dataset/split versions should not be interpreted as a one-factor architecture or augmentation ablation.
- Dataset edits and any formerly held-out images incorporated into Train must be documented; avoid same-leaf/tree/session leakage.
- **No deployment or model checkpoint is changed by this archival commit.**

### Related experiments

[Experiment archive overview](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019 historical Test/Holdout02](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
