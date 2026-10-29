# EXP-DENSENET121-TRANSFER-5SEEDS-019 — Final Test + Real Holdout (V09, evaluation-only)
> **Cập nhật điều hướng 28/10/2026.** EVALUATION-ONLY — không retrain, số Holdout thay đổi không đại diện cải thiện learning. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


> **Status: COMPLETE — evaluation-only, research report, 27/10/2026.** Source: user-supplied `EXP-DENSENET121-TRANSFER-5SEEDS-019_ANALYSIS.zip` (SHA256 `596c4fa817cb3054de98abe4798584b041a909aac04f91f6b31b538b1ac512d0`). This GitHub directory is a **curated text/CSV extract**, not the complete original ZIP or binary checkpoint bundle.

## 1. Protocol and provenance

- **Source of all weights:** EXP-DENSENET121-TRANSFER-5SEEDS-018 (ImageNet DenseNet121 Transfer, V09). EXP-019 only **reuses** the 5 EXP-018 checkpoints for inference; **no retraining**.
- **Deployment candidate:** seed **123**, selected and locked in **EXP-018 Validation only** before Test/Holdout. Do not switch seed based on metrics below.
- Seeds: `[42,123,2026,3407,7777]`; label order `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`.
- **V09 split:** Train 4,914, Validation 1,445, Test **686**, overall 7,045; ZIP SHA256 `1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4`.
- **Real Holdout 02:** **127** images (46 Anthracnose, 20 Healthy, 36 Leaf Miner, 25 Red Rust, **0 not_cashew_leaf**), ZIP SHA256 `47a5722660e2c8dc7cc04c9ff720e1f889c2481ec99112b41bc0606958a7fdd7`.
- Source Validation lock SHA256 `dc9f56b70d20d692947d4d0927840b6cba52167908ca1937e8f97e0846f405d3`. Evaluation lock SHA256 `3b43f6ce32d1e5dda6bc851fb22cada96a25d27c3ad812b49e92b2132c300a64`.
- Test contains five supported true classes. Holdout contains only four. **Macro Precision/Recall/F1 in aggregate metrics are calculated on the four supported Holdout classes, not all five labels**; see the source reporting discrepancy below.
- Audited: source asserts **zero exact SHA256 overlap** of Holdout with Train/Val/Test. Near-duplicate and same leaf/tree/session provenance still require independent review. The original ZIP checksum manifest verified 167/167 listed files; source archive passes ZIP integrity check.

## 2. Final Test V09 — same 686 held-out image slots, five classes

| Seed | Test Acc | Test Macro F1 | Disease → Healthy / 412 |
|---:|---:|---:|---:|
| 42 | 95.48% | 95.14% | 2 |
| **123 (Val-selected)** | **95.34% (654/686)** | **95.02%** | **2** |
| 2026 | 95.04% | 94.69% | 2 |
| 3407 | 94.46% | 94.07% | 2 |
| 7777 | 95.34% | 95.02% | 2 |
| **Mean ± sample SD** | **95.13 ± 0.41%** | **94.79 ± 0.43%** | **2.0 ± 0.0** |

By comparison, historical EXP-014 weights evaluated in EXP-016 on V08 Test had mean **95.13 ± 0.51% Acc** and **94.76 ± 0.58% Macro F1** (seed123 **653/686**, 95.19%). These nearly identical aggregate Test scores are **not evidence of generalization improvement**, because the training dataset differs and Test V08 has been reviewed repeatedly during research. The supplied EXP-019 package establishes Test class counts and the new dataset hash, but does not by itself independently prove byte-for-byte match of every Test image to V08.

## 3. New Holdout 02 — primary finding

| Seed | Correct/127 | Accuracy | Macro F1 (4 supported classes) | Diseased → Healthy / 107 |
|---:|---:|---:|---:|---:|
| 42 | 88 | 69.29% | 66.99% | 0 |
| **123 (Val-selected)** | **84** | **66.14%** | **62.20%** | **0** |
| 2026 | 81 | 63.78% | 58.99% | 1 |
| 3407 | 88 | 69.29% | 64.85% | 0 |
| 7777 | 89 | 70.08% | 65.59% | 0 |
| **Mean ± sample SD** | — | **67.72 ± 2.67%** | **63.72 ± 3.17%** | **0.2 ± 0.45** |

**Important:** this new 127-image Holdout has a different membership/class distribution from the old 48-image historical Holdout (EXP-015, 80.83 ± 0.93%). Comparing 67.72% and 80.83% directly **cannot isolate a model improvement/regression**. For a paired comparison of EXP-014 vs EXP-018, evaluate their *Validation-locked* checkpoints on the same Holdout 02 under identical preprocess and class mapping; report as retrospective descriptive evaluation, without re-selecting by the Holdout.

### Seed 123 Holdout confusion and per-class performance

| True class | Correct / support | Recall | Main error |
|---|---:|---:|---|
| Anthracnose | 33/46 | **71.74%** | 12 → Red Rust; 1 → Leaf Miner |
| **Healthy** | **5/20** | **25.00%** | **9 → Anthracnose; 6 → Red Rust** |
| Leaf Miner | 26/36 | 72.22% | 8 → Anthracnose; 2 → Red Rust |
| Red Rust | 20/25 | 80.00% | 4 → Anthracnose; 1 → Leaf Miner |
| not_cashew_leaf | 0/0 | **undefined** | No true Holdout examples |

Across the 5 seeds on the **same 20 Healthy images**, only **26/100 Healthy predictions** are correct (mean Healthy Recall **26.00 ± 5.48%**); **74/100** incorrectly label healthy leaves as Anthracnose (52) or Red Rust (22). This is a **false-disease alarm** problem despite the low disease→healthy count. **Do not interpret 100 predictions as 100 independent images.**

Mean Holdout class recall: Anthracnose **75.65%**, Healthy **26.00%**, Leaf Miner **76.11%**, Red Rust **74.40%**. **30/127 Holdout image IDs are wrong in all five seeds; 36/127 are wrong in at least four.** Manually audit groupings, labels and capture conditions; no automatic relabeling based on predictions.

### High-confidence errors of Val-selected seed 123 on Holdout

| True relative path | Ground truth → prediction | Confidence |
|---|---|---:|
| `leaf_miner/real_leaf_miner_100.jpg` | Leaf Miner → Anthracnose | **96.74%** |
| `healthy/real_healthy_031.jpg` | **Healthy → Anthracnose** | **95.12%** |
| `leaf_miner/real_leaf_miner_106.jpg` | Leaf Miner → Red Rust | **95.06%** |
| `anthracnose/real_anthracnose_092.jpg` | Anthracnose → Red Rust | 94.70% |
| `anthracnose/real_anthracnose_065.jpg` | Anthracnose → Red Rust | 92.39% |

See [curated high-confidence error list](./seed_123/real_holdout/top_confident_errors.csv) and [seed-123 confusion matrix](./seed_123/real_holdout/confusion_matrix_raw.csv). The full 5-seed per-image predictions reside only in the original ANALYSIS ZIP.

## 4. Domain shift and calibration

The 5-seed mean Test Accuracy (95.13%) is much higher than Holdout Accuracy (67.72%); however these are **different class distributions** and Macro F1 uses **5 vs 4 supported classes**, so the gap is descriptive, not a controlled causal estimate.

**ECE (10 confidence bins)** averaged **2.02% (Test)** vs **15.62% (Holdout)**, and mean multiclass Brier score **0.0780 vs 0.4632**. Confidence calibration degrades on the new Holdout. These are small-sample estimates, particularly for just 20 Healthy images. Calibration must be treated as an evaluation, not a justification to tune on Holdout.

Optional exploratory *Student-t 95% confidence intervals of the 5-seed means* (df=4; shared images/splits; **not** population field-photo CIs): Test Accuracy **[94.63%, 95.64%]**, Holdout Accuracy **[64.40%, 71.03%]**, Test Macro F1 **[94.25%, 95.33%]**, Holdout Macro F1 **[59.79%, 67.66%]**. Seed SD should be emphasized alongside these CIs.

### Reporting discrepancy in source files

The original `seed_*/real_holdout/classification_report.csv` includes a fifth class (`not_cashew_leaf`, support 0) as a zero-F1 row in sklearn's `macro avg` output. In seed123, that file's 5-label Macro F1 is **49.76%**, while source `metrics.json` explicitly uses **four supported true classes** and reports **62.20%**. The values are not interchangeable; use **four-class Macro F1** for the Holdout aggregate and label the evaluation scope. Do not compare the 4-class and 5-class macro scores as if they used the same denominator.

## 5. Recommended next actions and non-actions

1. **P0: Review the 15/20 Healthy errors** in seed123, especially `real_healthy_027–043` vs `real_healthy_022–026`: this is a conspicuous contiguous filename subgroup, *not proven* capture-session provenance. Check actual leaves, labels, imaging conditions and whether Healthy definition is consistent between Train/Val and Holdout.
2. **P0: Field-data provenance**: independent class-label verification, source leaf/tree/session grouping, similarity audit beyond exact-byte SHA, and enough genuinely new `not_cashew_leaf` Holdout images.
3. **P1: Paired EXP-014 vs EXP-018 comparison on identical Holdout 02** with checkpoints already fixed by their respective Validations; report differences without Holdout-informed model selection.
4. **P1: Future Train/Validation improvement**: collect challenging **Healthy vs Anthracnose/Red Rust**, Anthracnose ↔ Red Rust, and Leaf Miner ↔ Anthracnose cases from *new* field sessions, keeping group-level separation.
5. **No deployment switch** from EXP-014 seed123 to EXP-018 seed123 based solely on EXP-019; no new Test/Holdout-driven training changes. Document exploratory status and obtain a fresh five-class field test.

## 6. Provenance and artifact inventory

- [Per-seed Test metrics](./aggregate/test_results_5seeds.csv) · [Per-seed Holdout metrics](./aggregate/real_holdout_results_5seeds.csv).
- [Holdout recall by class/seed](./aggregate/per_class_real_holdout_by_seed.csv) · [Seed123 Holdout confusion matrix](./seed_123/real_holdout/confusion_matrix_raw.csv) · [Seed123 Test confusion matrix](./seed_123/test/confusion_matrix_raw.csv).
- [Protocol/audit excerpt](./PROVENANCE_AND_LIMITATIONS.md). Original archive contains 169 entries, including notebook, nine aggregate/figure groups, 5-seed per-image predictions, calibration, ROC/PR, confusion PNGs and checksum manifest. 167/167 listed checksum entries verified; **binary checkpoints and the original 3.8 MB ZIP are not stored here**.
- Historical comparison: [EXP-014 train](../EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md), [EXP-015 old Holdout](../EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md), [EXP-016 old Test](../EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md). The EXP-018 source experiment is referenced by validated metadata in the attached archive and **has not been imported as a standalone GitHub result folder**.
