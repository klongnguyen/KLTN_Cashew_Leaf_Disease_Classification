# EXP-RESNET50-TRANSFER-5SEEDS-025 — 5-seed experimental results (V11)
> Source-backed curated report from supplied ANALYSIS ZIP. Backfilled 2026-10-30. No model retraining or new inference performed during GitHub archival.

## Experiment and provenance

- **Protocol:** ResNet50 transfer: 256px, brightness .08, LR 3e-4/1e-5, epochs 12/30.
- **Cashew_dataV11 / CashewData_Split (11):** original Train **4,916**, Validation **1,447**, Test **685**; if augmented offline, extra Train images noted in protocol above.
- **Dataset ZIP SHA256:** 5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49
- **Base dataset manifest SHA256:** 91d8279239eaad2007b652f1859667359d0cf01f94af917dd27b0a934a7b7e75
- **Classes:** anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf.
- **Seeds:** [42, 123, 2026, 3407, 7777]; source selects **seed 3407 by Validation Loss**, before Test/Holdout.
- **Source archive:** EXP-RESNET50-TRANSFER-5SEEDS-025-256px-2416e1f8_ANALYSIS.zip; **SHA256:** 5cf755a612333475cff96431ad980a2e8c1c0ca961e1240cf246e59d5fc26c0d; **ZIP members:** 366.
- **Original checkpoints / full images and source ZIP in GitHub:** NOT UPLOADED in this curated report.

## Five-seed results

| Seed | Validation Accuracy | Final Test Accuracy | Historical Holdout 05 Accuracy | real_healthy Validation correct |
|---:|---:|---:|---:|---:|
| 42 | 96.20% | 95.62% | 74.17% | 3/7 |
| 123 | 96.06% | 95.33% | 75.83% | 4/7 |
| 2026 | 96.34% | 94.74% | 74.17% | 5/7 |
| 3407 | 96.06% | 94.89% | 74.17% | 3/7 |
| 7777 | 96.13% | 94.74% | 73.33% | 4/7 |

| Metric — mean across 5 seeds | Value |
|---|---:|
| Validation Accuracy | 96.16% |
| Validation Macro F1 | 96.01% |
| Final Test Accuracy | 95.07% |
| Final Test Macro F1 | 94.68% |
| Holdout 05 Accuracy | 74.33% |
| Holdout 05 Macro F1 (four supported true classes) | 67.54% |
| Holdout Healthy Recall (13 actual Healthy) | 26.15% |

**Files:** [per-seed metrics CSV](./metrics_5seeds.csv). The CSV is re-tabulated from the original aggregate and subgroup CSVs. Percentages are displayed to 2 decimal places. Original ANALYSIS ZIP SHA256 above is the provenance handle; neither it nor the model checkpoints are claimed to be stored by this commit.

## Interpretation and constraints

- Val has **7 filenames beginning real_healthy**; the same images are assessed at each seed; 5 seeds do not create 35 independent photos.
- Holdout 05 contains **120 images / 4 ground-truth classes**; it has no ground-truth not_cashew_leaf examples. Reported Holdout Macro F1 averages only the 4 classes with support.
- Test V11 and Holdout 05 have been retrospectively exposed in prior experiments. Use them for descriptive audit, not for further hyperparameter, seed, augmentation-count or checkpoint selection.
- 51 cross-split pHash candidates were flagged in the historical dataset audits; review source grouping rather than declaring exact leakage.
- Read source ZIP for full configs, manifests, calibration, per-image predictions, figures and validation selection lock; curated GitHub material does not replace them.

