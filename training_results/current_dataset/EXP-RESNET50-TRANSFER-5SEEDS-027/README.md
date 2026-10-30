# EXP-RESNET50-TRANSFER-5SEEDS-027 — 5-seed experimental results (V11)
> Source-backed curated report from supplied ANALYSIS ZIP. Backfilled 2026-10-30. No model retraining or new inference performed during GitHub archival.

## Experiment and provenance

- **Protocol:** ResNet50 ImageNet: EXP-026 recipe, only resize 224→256px; epochs 18/8.
- **Cashew_dataV11 / CashewData_Split (11):** original Train **4,916**, Validation **1,447**, Test **685**; if augmented offline, extra Train images noted in protocol above.
- **Dataset ZIP SHA256:** 5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49
- **Base dataset manifest SHA256:** 91d8279239eaad2007b652f1859667359d0cf01f94af917dd27b0a934a7b7e75
- **Classes:** anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf.
- **Seeds:** [42, 123, 2026, 3407, 7777]; source selects **seed 2026 by Validation Loss**, before Test/Holdout.
- **Source archive:** EXP-RESNET50-TRANSFER-5SEEDS-027_ANALYSIS.zip; **SHA256:** 98aba9e2e1e1f565303d6cb0ae6855f2a01d0b260903b97086eccfad1edb93ff; **ZIP members:** 370.
- **Original checkpoints / full images and source ZIP in GitHub:** NOT UPLOADED in this curated report.

## Five-seed results

| Seed | Validation Accuracy | Final Test Accuracy | Historical Holdout 05 Accuracy | real_healthy Validation correct |
|---:|---:|---:|---:|---:|
| 42 | 95.65% | 93.72% | 74.17% | 3/7 |
| 123 | 95.92% | 94.01% | 70.83% | 3/7 |
| 2026 | 95.78% | 93.28% | 71.67% | 3/7 |
| 3407 | 95.78% | 94.01% | 72.50% | 3/7 |
| 7777 | 95.37% | 94.31% | 71.67% | 3/7 |

| Metric — mean across 5 seeds | Value |
|---|---:|
| Validation Accuracy | 95.70% |
| Validation Macro F1 | 95.54% |
| Final Test Accuracy | 93.87% |
| Final Test Macro F1 | 93.38% |
| Holdout 05 Accuracy | 72.17% |
| Holdout 05 Macro F1 (four supported true classes) | 66.89% |
| Holdout Healthy Recall (13 actual Healthy) | 29.23% |

**Files:** [per-seed metrics CSV](./metrics_5seeds.csv). The CSV is re-tabulated from the original aggregate and subgroup CSVs. Percentages are displayed to 2 decimal places. Original ANALYSIS ZIP SHA256 above is the provenance handle; neither it nor the model checkpoints are claimed to be stored by this commit.

## Interpretation and constraints

- Val has **7 filenames beginning real_healthy**; the same images are assessed at each seed; 5 seeds do not create 35 independent photos.
- Holdout 05 contains **120 images / 4 ground-truth classes**; it has no ground-truth not_cashew_leaf examples. Reported Holdout Macro F1 averages only the 4 classes with support.
- Test V11 and Holdout 05 have been retrospectively exposed in prior experiments. Use them for descriptive audit, not for further hyperparameter, seed, augmentation-count or checkpoint selection.
- 51 cross-split pHash candidates were flagged in the historical dataset audits; review source grouping rather than declaring exact leakage.
- Read source ZIP for full configs, manifests, calibration, per-image predictions, figures and validation selection lock; curated GitHub material does not replace them.

