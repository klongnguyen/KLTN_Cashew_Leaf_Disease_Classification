# EXP-DENSENET121-TRANSFER-5SEEDS-022 — V10 · 4,920 Train / 1,445 Val / 686 Test; Holdout04 120

> **Backfill of an experiment archive provided by the project owner.** This is a curated GitHub text/CSV extract; **not** an upload of the entire ANALYSIS ZIP, per-image predictions, PNG figures, original notebook, or binary checkpoints.

## Protocol

- **Source experiment/archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-022_ANALYSIS.zip`
- **Original archive SHA256:** `47954a29b7ba13e0e619df50dc7f1a1fb7aca99867f794c4496693d70e31ae50`
- **Data:** V10 · 4,920 Train / 1,445 Val / 686 Test; Holdout04 120
- **Input/augmentation:** 224×224; no RandomBrightness
- **Training lineage:** Fresh ImageNet pretrained initialization, 5 seeds, EXP-014 two-stage recipe
- **Execution status:** Training+Validation+Test+Holdout COMPLETE.
- **Fixed seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked seed:** **123**. Never choose or change seed using Test/Holdout.
- **Result summary:** Val Acc 96.72 ± 0.20%; Macro F1 96.58 ± 0.21%. Test Acc 94.84 ± 0.35%; Macro F1 94.47 ± 0.37%. Holdout04 Acc 72.17 ± 2.92%; four-class Macro F1 ~71.94%.

## 5-seed results

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| validation | 42 | 96.40% | 96.24% |
| validation | 123 | 96.68% | 96.52% |
| validation | 2026 | 96.75% | 96.62% |
| validation | 3407 | 96.89% | 96.74% |
| validation | 7777 | 96.89% | 96.76% |
| test | 42 | 95.34% | 95.01% |
| test | 123 | 95.04% | 94.67% |
| test | 2026 | 94.75% | 94.36% |
| test | 3407 | 94.46% | 94.10% |
| test | 7777 | 94.61% | 94.22% |
| holdout | 42 | 70.83% | 71.15% |
| holdout | 123 | 70.83% | 70.65% |
| holdout | 2026 | 76.67% | 74.87% |
| holdout | 3407 | 69.17% | 68.73% |
| holdout | 7777 | 73.33% | 74.27% |

Numbers in this table are rounded to four decimal places in fraction form in [`metrics_5seeds.csv`](./metrics_5seeds.csv); consult the original archive for unrounded source metrics, per-class details and image-level predictions. On Holdout, Macro F1 is calculated for **four classes with true support**; `not_cashew_leaf` has no true Holdout instances.

## Interpretation and guardrails

Healthy Recall rose to ~63.08% over 13 Holdout Healthy images vs EXP-021 ~36.92%, but disease→Healthy errors increased. Historical holdout already inspected and not a fresh blind final test.

- The original Test is locked; historical Test/Holdouts have been inspected across development runs, so subsequent analysis is **retrospective**, not a new blind field test.
- Different dataset/split versions should not be interpreted as a one-factor architecture or augmentation ablation.
- Dataset edits and any formerly held-out images incorporated into Train must be documented; avoid same-leaf/tree/session leakage.
- **No deployment or model checkpoint is changed by this archival commit.**

### Related experiments

[Experiment archive overview](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019 historical Test/Holdout02](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
