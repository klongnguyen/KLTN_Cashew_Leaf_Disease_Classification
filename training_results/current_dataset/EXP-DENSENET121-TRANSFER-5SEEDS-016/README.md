# DenseNet121 Transfer EXP-016 — Final Test V08 of EXP-014 Validation-locked checkpoints

> **Protocol:** EXP-016 is an **inference-only evaluation**, NOT a new model-training experiment. It reuses the exact five checkpoint SHA256 values from **[EXP-014](../EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md)**. The **deployment seed is 123**, selected by minimum Validation Loss in EXP-014 **before** Test. EXP-016 never selects seed by Test scores. [Historical Real Holdout EXP-015](../EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md) likewise evaluates EXP-014 checkpoints, not new training.

**Source:** `EXP-DENSENET121-TRANSFER-5SEEDS-016_ANALYSIS.zip` shared 22/10/2026. ZIP SHA256 `02c1c0dcd6c8a7c06d8dffab5fe6d8514871590c7e071def6c4f2fdec5a32ea5`, **27 members**, archive contains detailed per-image prediction CSVs. Repo snapshot imports **metadata, five-seed summary, five confusion matrices, five per-class reports and key error audits**, but does **not** include five large `predictions.csv` or any binary checkpoint. See [artifact inventory](./ARTIFACT_INVENTORY.md).

## 1. Data, model and checkpoint integrity

- Dataset **V08**, archive `CashewData_Split (8).zip`, SHA256 `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`. Manifest SHA256 `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b`.
- Split: Train 4,790; Validation 1,430; **Test 686**. Test supports by class: Anthracnose **122**, Healthy **118**, Leaf miner **132**, Red rust **158**, Not cashew leaf **156**.
- Output order (critical for loading model): `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`.
- Seeds **42, 123, 2026, 3407, 7777**; **all five original EXP-014 checkpoint hashes agree** with EXP-014 `validation_selection_lock.json`. [Evaluation config](./evaluation_config.json).
- Evaluation runtime reported **TensorFlow 2.21.0 CPU** (`runtime_gpu: []`), `run_full_training: false`. This archive does **not** include source notebook or executable model weights. Results are checked for internal consistency, not independently rerun from weights.
- This V08 Test split has been examined with other research pipelines in previous experiments. Despite the **locked selection for EXP-014**, do **not** call the entire iterative research process blind or infer field performance from V08 alone.

## 2. Final Test — 5 seeds, same 686 images

| Seed | Correct / 686 | Test Accuracy | Test Macro F1 | Anthracnose Recall | Leaf miner Recall |
|---:|---:|---:|---:|---:|---:|
| **42** | **656** | **95.63%** | **95.32%** | 89.34% | 88.64% |
| **123 (locked deployment)** | **653** | **95.19%** | **94.84%** | 87.70% | 87.88% |
| 2026 | 652 | 95.04% | 94.69% | 89.34% | 88.64% |
| 3407 | 655 | 95.48% | 95.14% | 87.70% | 88.64% |
| 7777 | 647 | 94.31% | 93.82% | 81.97% | 90.15% |
| **Mean ± sample SD** | — | **95.13 ± 0.51%** | **94.76 ± 0.58%** | **87.21 ± 3.04%** | **88.79 ± 0.83%** |

- Healthy Recall across 5 seeds **98.47 ± 1.74%**, Red rust **99.37 ± 0.63%**, Not-cashew **99.87 ± 0.29%**.
- The aggregate sum of confusion matrices contains **3,430 predictions across five models on the same 686 images**, **not 3,430 independent images**.
- All five seeds have exactly **2/412 disease Test images incorrectly predicted as Healthy**. Both recurring filenames: `anthracnose_319.jpg` and `leaf_miner_927.jpg`.
- **Seed 123:** Healthy class itself **118/118 correct**; 2 diseased leaves predicted Healthy => Healthy precision **118/120 = 98.33%**, despite Healthy recall **100%**. Do not conflate these measures.

Original numerical tables: [test_results_5seeds.csv](./aggregate/test_results_5seeds.csv) · [mean_std](./aggregate/test_mean_std_summary.csv) · [per_class_test_by_seed](./aggregate/per_class_test_by_seed.csv) · [sum confusion](./aggregate/confusion_matrix_5seeds_sum.csv).

## 3. Deployment seed 123: error analysis

[Seed 123 confusion matrix](./seed_123/confusion_matrix.csv) (true rows → predicted columns):

| Ground truth | Anthracnose | Healthy | Leaf miner | Red rust | Not cashew |
|---|---:|---:|---:|---:|---:|
| Anthracnose (122) | **107** | **1** | 11 | 3 | 0 |
| Healthy (118) | 0 | **118** | 0 | 0 | 0 |
| Leaf miner (132) | **14** | **1** | **116** | 1 | 0 |
| Red rust (158) | 2 | 0 | 0 | **156** | 0 |
| Not cashew (156) | 0 | 0 | 0 | 0 | **156** |

- Dominant confusion: **Leaf miner → Anthracnose (14)** and **Anthracnose → Leaf miner (11)**. Risk to disease subtype selection despite high aggregate accuracy.
- **Disease→Healthy = 2/412 = 0.49%** on this Test split, including one Anthracnose and one Leaf miner; do not extrapolate a guaranteed future miss rate.
- Highly confident wrong predictions worth human checking (seed 123): `anthracnose_340.jpg` → Red rust confidence **0.9997878671**; `leaf_miner_836.jpg` → Anthracnose **0.9995212555**; `leaf_miner_880.jpg` → Red rust **0.9984000325**. High confidence is not evidence ground truth is wrong. [Top confidence errors](./aggregate/top_confident_errors_seed123.csv).
- Real-prefix proxy in Test: **25/28** correct for seed 123; other-prefix **628/658** correct. Filename prefix **does not prove field image independence**. [Real-subgroup summary](./aggregate/real_prefix_test_by_seed.csv).
- 24 Test paths are wrong in all 5 seeds (12 Anthracnose, 12 Leaf miner); examine session/leaf source and annotations before changing data. [Recurrent errors](./aggregate/recurrent_wrong_all_five_seeds.csv).

## 4. Three evaluations of the exact same five DenseNet checkpoints

| Metric | EXP-014 Validation | EXP-016 Final Test | EXP-015 Historical Real Holdout |
|---|---:|---:|---:|
| Images per seed | 1,430 | 686 | 48 |
| 5-seed Accuracy mean ± SD | **97.10 ± 0.20%** | **95.13 ± 0.51%** | **80.83 ± 0.93%** |
| 5-seed Macro F1 mean ± SD | **96.96 ± 0.21%** | **94.76 ± 0.58%** | **80.80%** |
| Deployment seed | 123 (selected on Val) | **123 (locked)** | 123 (locked) |
| Test/field independence | In-study selection | Frozen V08 Test already viewed across model research | Historical exposed, 4 true classes, 1 byte duplicate excluded |

**Interpretation:** Val → Test gap **−1.97 pp**; Test → Historical Holdout gap **−14.30 pp**. Distribution and class composition differ; cannot attribute these gaps entirely to overfitting. The historical Holdout is reused across experiments and does **not** have any true `not_cashew_leaf` samples.

On EXP-015 Holdout **deployment seed 123 = 38/48 = 79.17%** and **0/43 true disease images predicted Healthy**, but only **43 diseased samples**; no guarantee of zero misses on new photographs.

## 5. Context for application model choice

| Pipeline on same V08 | Validation Accuracy mean | Final Test Accuracy mean | Historical Holdout mean |
|---|---:|---:|---:|
| **DenseNet121 transfer 014/015/016** | **97.10%** | **95.13%** | **80.83%** |
| ViT DeiT-Tiny transfer 007 | 94.41% | **95.36%** | 74.58% |
| ResNet50 transfer 010 | 94.60% | 92.48% | 75.42% |
| ViT + Logistic Regression 008 | 96.56% | **NOT_RUN** | **NOT_RUN** |
| ViT + XGBoost 008 | 95.85% | **NOT_RUN** | **NOT_RUN** |

**Web choice:** DenseNet121 Transfer EXP-014 **seed 123** is a strong candidate for a **staging/experimental** replacement of the current web model, but a **fresh independent five-class field test** and exact model-package inference/parity/latency tests are still needed. Do not switch the application's current DenseNet121 Scratch EXP-009 / seed 3407 silently.

**Important:** ViT transfer 009 source not yet verified, do not claim DenseNet wins against an unseen result. Test V08 and historical Holdout are not suitable for repeated post-hoc architecture tuning.

## 6. Next actions

1. Keep the EXP-014 seed **123** model checkpoint immutable; **do not promote seed 42** because it scored higher on Test.
2. Audit recurring anthracnose ↔ leaf miner errors and two disease→Healthy cases with real photographs; validate ground truth with agronomic knowledge without relabeling based on model output alone.
3. Check original-/near-duplicate and same-leaf/capture-session grouping across V08 splits (earlier V08 audits flagged pHash candidate pairs).
4. Create a **new blind field test with five actual classes**; evaluate locked model packages without parameter/threshold selection on new test.
5. For web staging, package source EXP-014 `.keras` + model-internal preprocessing and exact label_map; benchmark API latency CPU and GPU on actual server and compare against current Scratch EXP-009.

[Quality and limitations review](./QUALITY_AND_PROTOCOL_REVIEW.md) · [Artifact inventory](./ARTIFACT_INVENTORY.md) · [Experiment Progress](../EXPERIMENT_PROGRESS_2026_10.md).
