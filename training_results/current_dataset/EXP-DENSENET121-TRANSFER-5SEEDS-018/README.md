# EXP-DENSENET121-TRANSFER-5SEEDS-018 — V09 · 4,914 Train / 1,445 Validation / 686 Test
> **Cập nhật điều hướng 28/10/2026.** TRAIN V09 — các EXP-019/020/021 là evaluation lại checkpoint EXP-018. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


> **Backfill of an experiment archive provided by the project owner.** This is a curated GitHub text/CSV extract; **not** an upload of the entire ANALYSIS ZIP, per-image predictions, PNG figures, original notebook, or binary checkpoints.

## Protocol

- **Source experiment/archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-018_ANALYSIS.zip`
- **Original archive SHA256:** `1c23b2219fd2d1cdc9b61c9cec31d3b30bbc7fdc72ed10ce9ad00b614192738c`
- **Data:** V09 · 4,914 Train / 1,445 Validation / 686 Test
- **Input/augmentation:** 224×224, no RandomBrightness
- **Training lineage:** Fresh ImageNet initialization; recipe EXP-014
- **Execution status:** Training+Validation COMPLETE; Final Test and Holdout PENDING (subsequently evaluated by EXP-019).
- **Fixed seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked seed:** **123**. Never choose or change seed using Test/Holdout.
- **Result summary:** Validation Acc 96.76 ± 0.39%; Macro F1 96.62 ± 0.43%.

## 5-seed results

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| validation | 42 | 96.47% | 96.30% |
| validation | 123 | 97.16% | 97.07% |
| validation | 2026 | 96.68% | 96.50% |
| validation | 3407 | 96.33% | 96.16% |
| validation | 7777 | 97.16% | 97.07% |

Numbers in this table are rounded to four decimal places in fraction form in [`metrics_5seeds.csv`](./metrics_5seeds.csv); consult the original archive for unrounded source metrics, per-class details and image-level predictions. On Holdout, Macro F1 is calculated for **four classes with true support**; `not_cashew_leaf` has no true Holdout instances.

## Interpretation and guardrails

EXP-019 is the corresponding final Test / Holdout02 evaluation; do not treat EXP-018 as a separate Test run.

- The original Test is locked; historical Test/Holdouts have been inspected across development runs, so subsequent analysis is **retrospective**, not a new blind field test.
- Different dataset/split versions should not be interpreted as a one-factor architecture or augmentation ablation.
- Dataset edits and any formerly held-out images incorporated into Train must be documented; avoid same-leaf/tree/session leakage.
- **No deployment or model checkpoint is changed by this archival commit.**

### Related experiments

[Experiment archive overview](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019 historical Test/Holdout02](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
