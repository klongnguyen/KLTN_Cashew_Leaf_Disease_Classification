# P0 Error Analysis — DenseNet121 EXP-024/025

## Executive summary
Loaded 600 prediction rows for 120 distinct images. Seed 42: 5 Healthy errors and 3 disease-to-Healthy errors on Holdout. All-five-seed errors on Holdout: 14. Grad-CAM: BLOCKED_MISSING_CHECKPOINT.

## Source and checkpoint integrity
{}

## Prediction error inventory
See `tables/error_inventory_all_seeds.csv` and `tables/p0_priority_cases.csv`. Validation predictions are unavailable when EXP-024 ANALYSIS/FULL is absent.

## Image and label audit
Human review pending; no labels changed.

## Grad-CAM and limitations
No attribution is claimed without verified source images, checkpoint and parity checks.

## Next controlled experiment
Await source assets and blinded human label review. Holdout 05 has only 13 Healthy images and is historically exposed; use a new independent five-class field test for final assessment.
