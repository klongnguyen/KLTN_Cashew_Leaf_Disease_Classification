# EXP-DENSENET121-TRANSFER-5SEEDS-025 — V11 · Final Test 685 images; Real Holdout05 120 images (4 true classes)

> **Curated backfill** from the user-provided experiment ANALYSIS archive. This is **not the complete ZIP**: figures, per-image predictions, checkpoint binaries, and original notebook have not been uploaded by this commit.

## Provenance and protocol

- **Original EXP identifier:** `EXP-DENSENET121-TRANSFER-5SEEDS-025-256px`.
- **Supplied archive:** `EXP-DENSENET121-TRANSFER-5SEEDS-025-256px_ANALYSIS.zip`.
- **Archive SHA256:** `eb2b0e6adcb4594419e09bcf5ea869f89c54d6623b97297e4bac9720e0919fdf`.
- **Dataset:** V11 · Final Test 685 images; Real Holdout05 120 images (4 true classes).
- **Input and image augmentation:** 256×256; frozen EXP-024 checkpoint recipe.
- **Weights/training:** Evaluation-only; SOURCE EXP-024; **NO TRAINING** in EXP-025.
- **Execution:** Final Test COMPLETE; Holdout05 COMPLETE; selection from EXP-024 Validation only..
- **Seeds:** `[42,123,2026,3407,7777]`.
- **Validation-locked candidate:** **seed 42**. Do not choose another seed using Test or Holdout.
- **Aggregate source metrics:** Final Test Acc 96.09 ± 0.42%; Macro F1 95.71 ± 0.46%. Holdout05 Acc 80.00 ± 1.56%; 4-class Macro F1 76.56 ± 0.57%; Healthy Recall ~50.77% over 13 true Healthy photos.

## Metrics by seed

| Split | Seed | Accuracy | Macro F1 |
|---|---:|---:|---:|
| test | 42 | 95.77% | 95.35% |
| test | 123 | 95.77% | 95.35% |
| test | 2026 | 96.06% | 95.69% |
| test | 3407 | 96.79% | 96.48% |
| test | 7777 | 96.06% | 95.67% |
| holdout | 42 | 77.50% | 75.79% |
| holdout | 123 | 80.83% | 76.42% |
| holdout | 2026 | 81.67% | 77.36% |
| holdout | 3407 | 80.00% | 76.72% |
| holdout | 7777 | 80.00% | 76.50% |

For machine-readable rounded data: [`metrics_5seeds.csv`](./metrics_5seeds.csv). Rounded values are not a substitute for the original archive's full-precision CSV. **Holdout Macro F1 is computed over 4 classes with true support; Test and Validation use 5 classes.**

## Research interpretation and cautions

Six Holdout05 image contents differ from older Holdout04; Test V11 differs from earlier V10/V08. Holdout05 lacks not_cashew_leaf, was historically exposed, and cannot be promoted as a fresh blind field test.

- Preserve archived Test, Holdout, dataset revisions and selection locks. No holdout-informed re-training or seed switching.
- High Test/Validation scores on historical or inspected splits do not certify generalization on a new acquisition session.
- The full original ANALYSIS archive and checkpoint binaries remain outside this Git commit; do not imply all files were uploaded.
- **No deployment changes** made here.

[Overview / experiment timeline](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md) · [EXP-019](../EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md)
