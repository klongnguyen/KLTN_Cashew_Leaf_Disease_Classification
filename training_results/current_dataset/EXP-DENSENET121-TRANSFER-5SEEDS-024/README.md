# EXP-DENSENET121-TRANSFER-5SEEDS-024 — V11 · 4,916 Train / 1,447 Val / 685 Test; Holdout05 120
> **Cập nhật điều hướng 28/10/2026.** TRAIN V11 256px brightness — Test/Holdout đánh giá ở EXP-025, không chọn lại seed bằng Test. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


> **Curated backfill** from the user-provided experiment ANALYSIS archive. This is **not the complete ZIP**: figures, per-image predictions, checkpoint binaries, and original notebook have not been uploaded by this commit.

## Provenance and protocol

- **Original EXP identifier:** `EXP-DENSENET121-TRANSFER-5SEEDS-024-256px-f9fd5d5f`.
- **Supplied archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-024-256px-f9fd5d5f_ANALYSIS.zip`.
- **Archive SHA256:** `6692e5ed4ab836aafea90f7112b36057e80277d154d3cbdbd35064faf934bcda`.
- **Dataset:** V11 · 4,916 Train / 1,447 Val / 685 Test; Holdout05 120.
- **Input and image augmentation:** 256×256; includes RandomBrightness(0.08).
- **Weights/training:** Fresh ImageNet initialization; five seeds; two-stage DenseNet121 Transfer.
- **Execution:** Training+Validation COMPLETE; Final Test and Holdout PENDING in EXP-024 (completed by EXP-025 with frozen checkpoints)..
- **Seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked candidate:** **seed 42**. Do not choose another seed using Test or Holdout.
- **Aggregate source metrics:** Val Acc 97.26 ± 0.21%; Val Macro F1 97.14 ± 0.23%.

## Metrics by seed

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| validation | 42 | 97.17% | 97.05% |
| validation | 123 | 97.58% | 97.49% |
| validation | 2026 | 97.03% | 96.87% |
| validation | 3407 | 97.24% | 97.11% |
| validation | 7777 | 97.30% | 97.17% |

For machine-readable rounded data: [`metrics_5seeds.csv`](./metrics_5seeds.csv). Rounded values are not a substitute for the original archive's full-precision CSV. **Holdout Macro F1 is computed over 4 classes with true support; Test and Validation use 5 classes.**

## Research interpretation and cautions

V11 includes label/content edits and revised split; EXP-023 (V10/224px) is not a one-factor resize control. Final Test/Holdout metrics belong to EXP-025, not EXP-024. Do not select seed on EXP-025 Test.

- Preserve archived Test, Holdout, dataset revisions and selection locks. No holdout-informed re-training or seed switching.
- High Test/Validation scores on historical or inspected splits do not certify generalization on a new acquisition session.
- The full original ANALYSIS archive and checkpoint binaries remain outside this Git commit; do not imply all files were uploaded.
- **No deployment changes** made here.

[Overview / experiment timeline](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
