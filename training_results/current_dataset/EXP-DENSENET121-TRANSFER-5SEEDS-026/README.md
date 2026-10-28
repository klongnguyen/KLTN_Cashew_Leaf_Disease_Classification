# EXP-DENSENET121-TRANSFER-5SEEDS-026 — V11 · 4,916 Train / 1,447 Val / 685 Test; Holdout05 120

> **Curated backfill** from the user-provided experiment ANALYSIS archive. This is **not the complete ZIP**: figures, per-image predictions, checkpoint binaries, and original notebook have not been uploaded by this commit.

## Provenance and protocol

- **Original EXP identifier:** `EXP-DENSENET121-TRANSFER-5SEEDS-026-FRESH-337b61af`.
- **Supplied archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-026-FRESH-337b61af_ANALYSIS.zip`.
- **Archive SHA256:** `204ab1cf64be05209f82eb1ff7abdd0106af86bf6c65ec0b23daad0dd3782044`.
- **Dataset:** V11 · 4,916 Train / 1,447 Val / 685 Test; Holdout05 120.
- **Input and image augmentation:** 224×224; NO RandomBrightness.
- **Weights/training:** Fresh ImageNet initialization; original EXP-014 training recipe on the same V11 data.
- **Execution:** Training+Validation COMPLETE; historical Test and Holdout05 evaluation COMPLETE / EXPOSED..
- **Seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked candidate:** **seed 2026**. Do not choose another seed using Test or Holdout.
- **Aggregate source metrics:** Val Acc 97.00 ± 0.14%; Macro F1 96.88 ± 0.14%. Test Acc 96.23 ± 0.56%; Macro F1 95.88 ± 0.61%. Holdout05 Acc 74.83 ± 2.24%; 4-class Macro F1 73.21 ± 1.67%.

## Metrics by seed

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| validation | 42 | 96.75% | 96.63% |
| validation | 123 | 97.03% | 96.89% |
| validation | 2026 | 97.10% | 96.98% |
| validation | 3407 | 97.10% | 96.98% |
| validation | 7777 | 97.03% | 96.89% |
| test | 42 | 95.62% | 95.22% |
| test | 123 | 95.77% | 95.39% |
| test | 2026 | 96.93% | 96.66% |
| test | 3407 | 96.64% | 96.32% |
| test | 7777 | 96.20% | 95.82% |
| holdout | 42 | 73.33% | 72.68% |
| holdout | 123 | 75.83% | 72.49% |
| holdout | 2026 | 76.67% | 73.29% |
| holdout | 3407 | 76.67% | 76.00% |
| holdout | 7777 | 71.67% | 71.58% |

For machine-readable rounded data: [`metrics_5seeds.csv`](./metrics_5seeds.csv). Rounded values are not a substitute for the original archive's full-precision CSV. **Holdout Macro F1 is computed over 4 classes with true support; Test and Validation use 5 classes.**

## Research interpretation and cautions

EXP-026 vs EXP-024/025 changes two factors simultaneously (224/256px AND brightness off/on). Same V11/holdout allows retrospective comparison, but does not isolate resize. Test and Holdout05 were exposed previously; no automatic promotion or WebApp change.

- Preserve archived Test, Holdout, dataset revisions and selection locks. No holdout-informed re-training or seed switching.
- High Test/Validation scores on historical or inspected splits do not certify generalization on a new acquisition session.
- The full original ANALYSIS archive and checkpoint binaries remain outside this Git commit; do not imply all files were uploaded.
- **No deployment changes** made here.

[Overview / experiment timeline](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
