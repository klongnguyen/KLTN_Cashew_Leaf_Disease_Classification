# EXP-DENSENET121-TRANSFER-5SEEDS-016 — Final Test V08

Inference only on the five EXP-014 Validation-locked checkpoints. No retraining, checkpoint changes, or seed selection using Test results.

Dataset: 686 Test images from `CashewData_Split (8).zip` (SHA256 `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`). The 686 individual image SHA256 values match the frozen EXP-014 manifest.

Deployment seed: **123**, frozen by EXP-014 Validation loss before Test.

| Seed | Correct | Accuracy | Macro F1 | Recall anthracnose | Recall healthy | Recall leaf_miner | Recall red_rust | Recall not_cashew_leaf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 656/686 | 95.63% | 95.32% | 89.34% | 100.00% | 88.64% | 98.73% | 100.00% |
| 123 | 653/686 | 95.19% | 94.84% | 87.70% | 100.00% | 87.88% | 98.73% | 100.00% |
| 2026 | 652/686 | 95.04% | 94.69% | 89.34% | 96.61% | 88.64% | 99.37% | 99.36% |
| 3407 | 655/686 | 95.48% | 95.14% | 87.70% | 99.15% | 88.64% | 100.00% | 100.00% |
| 7777 | 647/686 | 94.31% | 93.82% | 81.97% | 96.61% | 90.15% | 100.00% | 100.00% |

Five-seed Accuracy: **95.13% ± 0.51%** (sample SD across seeds).
Five-seed Macro F1: **94.76% ± 0.58%**.
Deployment seed 123: **653/686**, Accuracy **95.19%**, Macro F1 **94.84%**.

Confusion matrices use true class as rows and predicted class as columns. Each seed folder contains `predictions.csv`, `metrics.json`, `classification_report.csv`, and `confusion_matrix.csv`. The aggregate folder contains five-seed metrics, per-class recall, sample mean/SD, and a sum of five confusion matrices (3,430 predictions, not 3,430 distinct images).

Portable results archive: `EXP-DENSENET121-TRANSFER-5SEEDS-016_ANALYSIS.zip` beside this folder.

The Test set is used once for this final report; the results are descriptive for the frozen V08 split. Seed SD measures training-run variation, not uncertainty for future field photos.

---
**Note to GitHub readers:** This file reproduces the **text of the original ZIP README** as historical source documentation; its references to `predictions.csv` and ZIP "beside this folder" refer to the **original full ANALYSIS archive**, not to the curated GitHub snapshot. The new [GitHub README](./README.md) and [artifact inventory](./ARTIFACT_INVENTORY.md) explain exactly what was imported.
