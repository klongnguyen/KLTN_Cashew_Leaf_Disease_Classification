# EXP-VIT-TRANSFER-XGBOOST-009 — V08 retrospective Test and historical Holdout

> **Inference-only and retrospective**, backfilled from user-provided `EXP-VIT-TRANSFER-XGBOOST-009_RESULTS(1).zip`. Original archive SHA256: `b55b5ac1d4c64c314e6b09612cc1b99b9865fc66af50f2841bbaed896adfb194`.

## Provenance and protocol

- This is the evaluation continuation of **EXP-VIT-TRANSFER-XGBOOST-008**, not new ViT training.
- The EXP-008 Validation lock **fixed deployment to seed 123 / classifier A** before evaluation. B and C results are exploratory evaluations and **not promoted based on Test/Holdout**.
- Five seeds `[42, 123, 2026, 3407, 7777]`, three classifiers per seed (**A/B/C**), Test V08 = **686 images** (5 supported classes), historical Holdout = **48 images** (4 supported classes). One exact Test/Holdout duplicate had been excluded.
- Test/Holdout were **previously exposed**, so this is **not a fresh blind field test**.
- Classifier replay on frozen cache passed according to source archive; some runtime feature drift is documented therein. Do not confuse `macro_f1_present_classes` (4 on Holdout) with macro across all 5 labels.

## Summary, mean ± seed sample SD

| Classifier | V08 Test Accuracy | Test Macro F1 | 48-image Holdout Accuracy | Holdout Macro F1 (4 classes) |
|---|---:|---:|---:|---:|
| A — selected in EXP-008 | 95.36 ± 0.49% | 95.08 ± 0.56% | 74.58 ± 6.14% | 72.87 ± 6.13% |
| B | 96.68 ± 0.28% | 96.44 ± 0.29% | 83.33 ± 1.47% | 80.32 ± 1.52% |
| C | 96.03 ± 0.24% | 95.73 ± 0.28% | 85.00 ± 1.74% | 81.78 ± 2.67% |

**Important:** The locked, official seed-123/A evaluation is **Test 95.34% Acc / 95.05% Macro F1** and **historical Holdout 70.83% Acc / 69.59% four-class Macro F1**. Although B/C look stronger on historical datasets, switching to them because of these outcomes would be Test/Holdout selection.

## Source artifacts and data

- [30 original full-precision rows — 5 seeds × A/B/C × Test/Holdout](./summary_all_5seeds_ABC.csv).
- This repository backfill includes a **curated README and CSV only**; the original Results ZIP, notebook, image-level files and any binaries are not part of this commit.
- Earlier source/notebooks remain in `Notebook/ViT/vit_transfer_xgboost_009.ipynb`.
- No deployment update and no retraining.

[Experiment timeline](../DENSENET121_TRANSFER_V09_V11_BACKFILL_2026_10.md)
