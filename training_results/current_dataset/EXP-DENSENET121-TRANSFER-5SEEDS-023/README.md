# EXP-DENSENET121-TRANSFER-5SEEDS-023 — V10 · 4,920 Train / 1,445 Val / 686 locked Test

> **Backfill of an experiment archive provided by the project owner.** This is a curated GitHub text/CSV extract; **not** an upload of the entire ANALYSIS ZIP, per-image predictions, PNG figures, original notebook, or binary checkpoints.

## Protocol

- **Source experiment/archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-023_ANALYSIS.zip`
- **Original archive SHA256:** `6a2b6e8fadca1722565aa3f59bbe75d81806fb6136e0fe7787d2ec372426ce60`
- **Data:** V10 · 4,920 Train / 1,445 Val / 686 locked Test
- **Input/augmentation:** 224×224, EXP-022 augmentation + RandomBrightness(0.08)
- **Training lineage:** Fresh ImageNet initialization; controlled Validation ablation vs EXP-022
- **Execution status:** Training+Validation COMPLETE; Test and Holdout PENDING; RESEARCH_ONLY.
- **Fixed seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked seed:** **123**. Never choose or change seed using Test/Holdout.
- **Result summary:** Val Acc 96.58 ± 0.19%; Macro F1 96.44 ± 0.19%; no Val Healthy Recall improvement vs EXP-022.

## 5-seed results

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| validation | 42 | 96.40% | 96.25% |
| validation | 123 | 96.47% | 96.33% |
| validation | 2026 | 96.54% | 96.40% |
| validation | 3407 | 96.89% | 96.74% |
| validation | 7777 | 96.61% | 96.47% |

Numbers in this table are rounded to four decimal places in fraction form in [`metrics_5seeds.csv`](./metrics_5seeds.csv); consult the original archive for unrounded source metrics, per-class details and image-level predictions. On Holdout, Macro F1 is calculated for **four classes with true support**; `not_cashew_leaf` has no true Holdout instances.

## Interpretation and guardrails

One-factor RandomBrightness addition did not satisfy the preset Macro F1 promotion gate; no automatic deployment promotion and no Final Test/Holdout result in this archive.

- The original Test is locked; historical Test/Holdouts have been inspected across development runs, so subsequent analysis is **retrospective**, not a new blind field test.
- Different dataset/split versions should not be interpreted as a one-factor architecture or augmentation ablation.
- Dataset edits and any formerly held-out images incorporated into Train must be documented; avoid same-leaf/tree/session leakage.
- **No deployment or model checkpoint is changed by this archival commit.**

### Related experiments

[Experiment archive overview](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019 historical Test/Holdout02](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
