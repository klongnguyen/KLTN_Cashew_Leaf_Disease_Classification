# EXP-016 — Quality and protocol review (22/10/2026)

**Purpose:** Review results from source file `EXP-DENSENET121-TRANSFER-5SEEDS-016_ANALYSIS.zip`. **Not** model retraining, **not** new independent field test. Source ZIP SHA256: `02c1c0dcd6c8a7c06d8dffab5fe6d8514871590c7e071def6c4f2fdec5a32ea5`.

## Integrity checks

- `evaluation_config.json`: `source_experiment_id=EXP-DENSENET121-TRANSFER-5SEEDS-014`, `evaluation_type=inference_only_five_locked_checkpoints`, `run_full_training=false`, `deployment_seed=123`.
- All five checkpoint SHA256 values in EXP-016 exactly match EXP-014's `validation_selection_lock.json` preserved on main. This validates reported **identities**, not independent possession/re-execution of five binary weights.
- Dataset ZIP and manifest hashes match EXP-014: `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7` and `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b`.
- In original five per-image CSVs: 686 rows per seed, identical ordered `relative_path` / `image_sha256` / `true_label`. Confusions, per-class results and accuracy totals are consistent, all seeds predict the same 2 disease images Healthy.
- Original predictions exist in user-supplied ZIP **but are not in this GitHub curated snapshot**. Without them GitHub reports alone cannot reconstruct image-level confidence and pairwise error auditing fully.

## Metrics & notable failures

- Mean±sample SD (five seeds): Test Accuracy **95.13±0.51%**, Test Macro F1 **94.76±0.58%**.
- Locked seed 123: **653/686** correct, Accuracy 95.19%, Macro F1 94.84%, 2/412 true diseases incorrectly predicted Healthy. A very high Healthy Recall **118/118** is *not* the same as no disease→Healthy false negative.
- Anthracnose 107/122, Leaf miner 116/132; confusion predominantly mutual (11 + 14 instances in seed 123).
- Identical 2 disease→Healthy errors for all five seeds: `test/anthracnose/anthracnose_319.jpg`, `test/leaf_miner/leaf_miner_927.jpg`; inspect image, labels and capture grouping.
- Strongly confident wrong seed 123: `anthracnose_340.jpg` true anthracnose→red_rust confidence .9997878671; `leaf_miner_836.jpg` true leaf_miner→anthracnose .9995212555; `leaf_miner_880.jpg` true leaf_miner→red_rust .9984000325. Verify against images before any relabeling or data changes.
- Filename-based `real_` Test subgroup performance (seed 123) **25/28** correct; remainder **628/658**, indicating possible generalization issue but uncertain provenance.

## Scientific safeguards

1. **Source EXP-014 Val selection is fixed:** seed123. Test-best seed42 must not be chosen post-hoc.
2. All 5 seeds evaluated on same 686 Test images. `confusion_matrix_5seeds_sum.csv` counts 3,430 **model-image inferences**, not 3,430 independent image observations.
3. Historical EXP-015 Holdout 48 after removing exact duplicate with Test; previously exposed and missing true `not_cashew_leaf`. Holdout averages **80.83±0.93%**, Val **97.10±0.20%**: high distribution sensitivity remains.
4. V08 Test has been examined across prior pipelines, so it is **not a pristine blind final model-selection benchmark** for entire study. This run may be inference-only but claim should be qualified.
5. Five-seed SD quantifies training variance, **not** sampling uncertainty of future orchard photographs.
6. 51 near-similarity pHash candidates have been noted by earlier V08 audit; cannot rule out same-leaf/same-session leakage with SHA alone.
7. Web deployment currently points to DenseNet121 **Scratch EXP-009 seed3407**, so do not change `Application/config.py` or README model labels without providing correct signed/hashed EXP-014 deployment package.
8. A fresh 5-class, group-independent field test and on-device CPU/GPU latency/parity checks are necessary before production switch.

## Comparative disposition

EXP-016 closes missing Test evidence for DenseNet121 Transfer 014. This **strengthens DenseNet's staging candidacy** versus ResNet50 Transfer-010 and ViT Transfer-007. ViT Transfer-007 has slightly higher 5-seed Test Accuracy (**95.36%** versus **95.13%**); DenseNet reports higher Validation and historically exposed Holdout, but differences are not statistical proof of superior field performance. Hybrid ViT+XGBoost-008 is Validation-only, so don't compare absent Test values.

[README](./README.md) · [source metrics](./aggregate/test_results_5seeds.csv) · [seed 123 confusion matrix](./seed_123/confusion_matrix.csv).
