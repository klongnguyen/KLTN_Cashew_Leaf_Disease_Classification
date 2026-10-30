# EXP-VIT-DINOV2SMALL-TRANSFER-5SEEDS-013 — 5-seed experimental results (V11)
> Source-backed curated report from supplied ANALYSIS ZIP. Backfilled 2026-10-30. No model retraining or new inference performed during GitHub archival.

## Experiment and provenance

- **Protocol:** ViT backbone DINOv2-Small self-supervised pretrained; transfer 256px, 5 seeds.
- **Cashew_dataV11 / CashewData_Split (11):** original Train **4,916**, Validation **1,447**, Test **685**; if augmented offline, extra Train images noted in protocol above.
- **Dataset ZIP SHA256:** 5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49
- **Base dataset manifest SHA256:** 91d8279239eaad2007b652f1859667359d0cf01f94af917dd27b0a934a7b7e75
- **Classes:** anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf.
- **Seeds:** [42, 123, 2026, 3407, 7777]; source selects **seed 42 by Validation Loss**, before Test/Holdout.
- **Source archive:** EXP-VIT-DINOV2SMALL-TRANSFER-5SEEDS-013-256px-7dccb051_ANALYSIS.zip; **SHA256:** aaf079c40ef1c0bb4c7b5de976b6ffc40e2f1014bfb35f9c7dfef18d87211a3f; **ZIP members:** 363.
- **Original checkpoints / full images and source ZIP in GitHub:** NOT UPLOADED in this curated report.

## Five-seed results

| Seed | Validation Accuracy | Final Test Accuracy | Historical Holdout 05 Accuracy | real_healthy Validation correct |
|---:|---:|---:|---:|---:|
| 42 | 98.76% | 97.52% | 89.17% | 7/7 |
| 123 | 98.27% | 96.35% | 78.33% | 4/7 |
| 2026 | 98.13% | 97.66% | 92.50% | 6/7 |
| 3407 | 98.48% | 97.66% | 86.67% | 5/7 |
| 7777 | 98.41% | 97.81% | 89.17% | 7/7 |

| Metric — mean across 5 seeds | Value |
|---|---:|
| Validation Accuracy | 98.41% |
| Validation Macro F1 | 98.33% |
| Final Test Accuracy | 97.40% |
| Final Test Macro F1 | 97.12% |
| Holdout 05 Accuracy | 87.17% |
| Holdout 05 Macro F1 (four supported true classes) | 85.70% |
| Holdout Healthy Recall (13 actual Healthy) | 76.92% |

**Files:** [per-seed metrics CSV](./metrics_5seeds.csv). The CSV is re-tabulated from the original aggregate and subgroup CSVs. Percentages are displayed to 2 decimal places. Original ANALYSIS ZIP SHA256 above is the provenance handle; neither it nor the model checkpoints are claimed to be stored by this commit.

## Interpretation and constraints

- Val has **7 filenames beginning real_healthy**; the same images are assessed at each seed; 5 seeds do not create 35 independent photos.
- Holdout 05 contains **120 images / 4 ground-truth classes**; it has no ground-truth not_cashew_leaf examples. Reported Holdout Macro F1 averages only the 4 classes with support.
- Test V11 and Holdout 05 have been retrospectively exposed in prior experiments. Use them for descriptive audit, not for further hyperparameter, seed, augmentation-count or checkpoint selection.
- 51 cross-split pHash candidates were flagged in the historical dataset audits; review source grouping rather than declaring exact leakage.
- Read source ZIP for full configs, manifests, calibration, per-image predictions, figures and validation selection lock; curated GitHub material does not replace them.

