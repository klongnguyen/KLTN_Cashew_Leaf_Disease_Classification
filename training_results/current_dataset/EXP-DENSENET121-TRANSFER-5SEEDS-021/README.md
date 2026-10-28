# EXP-DENSENET121-TRANSFER-5SEEDS-021 — V09 checkpoints / Holdout04 · 120 images, 4 supported classes
> **Cập nhật điều hướng 28/10/2026.** EVALUATION-ONLY — không retrain, số Holdout thay đổi không đại diện cải thiện learning. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


> **Backfill of an experiment archive provided by the project owner.** This is a curated GitHub text/CSV extract; **not** an upload of the entire ANALYSIS ZIP, per-image predictions, PNG figures, original notebook, or binary checkpoints.

## Protocol

- **Source experiment/archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-021_ANALYSIS.zip`
- **Original archive SHA256:** `4562d6a70f8b0f1ee7cbcf092e9400fe0ae4612c2e9cf20005b6f514f42f634d`
- **Data:** V09 checkpoints / Holdout04 · 120 images, 4 supported classes
- **Input/augmentation:** 224×224; source EXP-018
- **Training lineage:** Evaluation-only; REUSED_EXP018 (NOT trained on Split10)
- **Execution status:** Holdout04 COMPLETE; Training NOT RUN; Final Test NOT RUN.
- **Fixed seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked seed:** **123**. Never choose or change seed using Test/Holdout.
- **Result summary:** Holdout Accuracy 71.33 ± 2.67%; four-class Macro F1 68.31 ± 2.97%.

## 5-seed results

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| holdout | 42 | 71.67% | 68.99% |
| holdout | 123 | 70.00% | 67.70% |
| holdout | 2026 | 67.50% | 63.49% |
| holdout | 3407 | 73.33% | 70.39% |
| holdout | 7777 | 74.17% | 70.95% |

Numbers in this table are rounded to four decimal places in fraction form in [`metrics_5seeds.csv`](./metrics_5seeds.csv); consult the original archive for unrounded source metrics, per-class details and image-level predictions. On Holdout, Macro F1 is calculated for **four classes with true support**; `not_cashew_leaf` has no true Holdout instances.

## Interpretation and guardrails

Six old Holdout Healthy images were removed from evaluation and moved into prospective Split10 Train. The EXP-018 checkpoints were NOT re-trained. Report the rise from EXP-020 as evaluation-set change, not model improvement. Historical Holdout exposed.

- The original Test is locked; historical Test/Holdouts have been inspected across development runs, so subsequent analysis is **retrospective**, not a new blind field test.
- Different dataset/split versions should not be interpreted as a one-factor architecture or augmentation ablation.
- Dataset edits and any formerly held-out images incorporated into Train must be documented; avoid same-leaf/tree/session leakage.
- **No deployment or model checkpoint is changed by this archival commit.**

### Related experiments

[Experiment archive overview](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019 historical Test/Holdout02](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
