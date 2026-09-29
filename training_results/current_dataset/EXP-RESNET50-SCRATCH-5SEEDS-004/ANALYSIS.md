# Analysis — EXP-RESNET50-SCRATCH-5SEEDS-004

## 1. Scope

This document analyzes the uploaded `EXP-RESNET50-SCRATCH-5SEEDS-004_ANALYSIS` package. The run contains complete 5-seed training/validation artifacts but no final Test evaluation.

The dataset is `Cashew_dataV04` with the fixed split:

- Train: 4,822
- Validation: 1,402
- Test: 687
- Total: 6,911

The Test Set is marked locked, and `run_final_test=false`.

## 2. Main result

The strongest result of EXP-004 is not a single seed score but the combination of:

- Validation Accuracy: **90.23 ± 1.45%**
- Validation Macro-F1: **90.03 ± 1.44%**
- Balanced Accuracy: **89.84 ± 1.43%**

Compared with EXP-003 validation:

- Accuracy mean increased by **2.01 percentage points**.
- Macro-F1 mean increased by **1.97 percentage points**.
- Accuracy seed SD decreased from **2.49% to 1.45%**.
- Macro-F1 seed SD decreased from **2.52% to 1.44%**.

This is a favorable rerun because both central performance and cross-seed stability improved.

However, the environment changed from the EXP-003 `MirroredStrategy`/2-GPU run to a single NVIDIA L4 `OneDeviceStrategy` run. Even with the same seeds and core hyperparameters, GPU execution and distributed vs single-device numerical behavior can change the stochastic trajectory. The result therefore demonstrates a better observed validation run, but does not by itself prove that a modeling change caused the improvement.

## 3. Seed-level behavior

| Seed | Val Acc | Macro F1 | Best epoch | Interpretation |
|---:|---:|---:|---:|---|
| 42 | 88.52% | 88.52% | 19 | weaker than its 003 counterpart |
| 123 | 89.02% | 88.60% | 9 | clear recovery vs 003 |
| 2026 | 91.44% | 91.24% | 20 | strong run and lowest checkpoint val loss |
| 3407 | 90.37% | 90.22% | 16 | still strong, slightly lower than 003 |
| 7777 | **91.80%** | **91.59%** | 28 | highest validation Macro-F1 |

The seed spread remains meaningful, but no single seed catastrophically underperforms as happened in EXP-003 around the mid-85% validation range.

### 003 → 004 changes by seed

| Seed | Accuracy Δ | Macro-F1 Δ |
|---:|---:|---:|
| 42 | −1.28 pp | −1.16 pp |
| 123 | +3.50 pp | +3.02 pp |
| 2026 | +5.78 pp | +6.05 pp |
| 3407 | −0.64 pp | −0.59 pp |
| 7777 | +2.71 pp | +2.53 pp |

The mean gain is therefore driven primarily by seeds `123`, `2026`, and `7777`, not by a uniform uplift across all runs.

## 4. Per-class behavior

| Class | Precision | Recall | F1 ± SD |
|---|---:|---:|---:|
| Anthracnose | 81.72% | 85.44% | **83.46 ± 2.87%** |
| Healthy | 91.07% | 88.00% | **89.38 ± 2.83%** |
| Leaf Miner | 93.28% | 86.18% | **89.55 ± 0.95%** |
| Not Cashew Leaf | 93.25% | 91.46% | **92.28 ± 2.35%** |
| Red Rust | 93.05% | **98.13%** | **95.50 ± 0.95%** |

### Anthracnose

`anthracnose` remains the main weakness. Its mean F1 (**83.46%**) is approximately 6 percentage points below the next disease/healthy groups and more than 12 points below `red_rust`.

The class has lower precision (**81.72%**) than recall (**85.44%**), meaning other classes are also being predicted as anthracnose relatively often.

### Leaf Miner

`leaf_miner` has strong precision (**93.28%**) but lower recall (**86.18%**). The model is comparatively conservative when assigning this class: predictions called leaf miner tend to be correct, but some true leaf-miner images are diverted to other labels.

### Red Rust

`red_rust` is the strongest class with **98.13% recall** and **95.50 ± 0.95% F1**. The remaining limitation is precision rather than recall, consistent with some anthracnose images being predicted as red rust.

## 5. Confusion analysis

The confusion matrix was summed across the five validation runs. Since the same 1,402 validation images are predicted five times, counts below are recurring prediction events across seeds.

Top recurring errors:

- `healthy → anthracnose`: **97**
- `leaf_miner → anthracnose`: **92**
- `not_cashew_leaf → anthracnose`: **80**
- `anthracnose → red_rust`: **80**
- `anthracnose → healthy`: **68**
- `leaf_miner → not_cashew_leaf`: **61**
- `anthracnose → leaf_miner`: **51**

Two patterns stand out.

First, `anthracnose` attracts errors from several classes, which explains its relatively low precision.

Second, true anthracnose itself is split mainly toward `red_rust`, `healthy`, and `leaf_miner`, suggesting that the class likely contains visually heterogeneous or mild/ambiguous cases. This should be verified by image-level error review rather than inferred from the matrix alone.

## 6. Learning-curve behavior

At the best checkpoint:

- mean Train–Validation Accuracy gap: **4.21 pp**;
- mean actual training length: **26.4 epochs**;
- mean best epoch: **18.4**.

The best epochs are substantially earlier than the stopping epochs because EarlyStopping patience is 8. This is working as intended.

Several seeds continue improving training accuracy after validation loss has already reached its minimum:

- seed `123`: final validation loss is about **0.106** above its minimum;
- seed `7777`: final validation loss is about **0.080** above its minimum;
- seed `3407`: about **0.070** above minimum.

This is evidence of post-checkpoint overfitting behavior. It does not invalidate the run because the model checkpoint is selected at minimum validation loss and `restore_best_weights=true`.

Seed `7777` has the highest Validation Macro-F1 but also a relatively large train/validation gap at the best checkpoint (~6.56 pp) and the largest checkpoint loss gap. Seed `2026` has a slightly lower Macro-F1 but the lowest validation loss (**0.2775**). This distinction should be preserved: “best Macro-F1 seed” and “lowest validation-loss seed” are not the same criterion.

## 7. Reproducibility and environment

EXP-004 records:

- Python 3.13.15
- TensorFlow 2.20.0
- Keras 3.13.2
- CUDA 12.5.1
- cuDNN 9
- NVIDIA L4
- OneDeviceStrategy

The split SHA-256 values are stored in `experiment_config.json`, which is an improvement for dataset identity verification.

The five reported checkpoint sizes are ~282.23 MB each; large checkpoint binaries should remain in Drive/external storage rather than GitHub.

## 8. Scientific interpretation

EXP-004 is a stronger **validation rerun** than EXP-003. It provides two useful findings:

1. ResNet50 Scratch on V04 can reach around 90% validation Macro-F1 when the run is stable.
2. Seed variability remains non-negligible, so reporting five-seed mean ± SD is justified.

It does **not** yet establish a new official Test benchmark, because the locked Test Set was intentionally not executed. The existing EXP-003 Test result remains the repository benchmark until EXP-004 is finalized.

## 9. Recommended decision

### Keep EXP-004

Do not mark this as a failed experiment. The validation result is useful and materially stronger/more stable than the previous ResNet50 validation aggregate.

### Do not tune it further before Test

The five checkpoints and configuration should now be frozen. Further hyperparameter changes based on these validation results would create a new experiment ID.

### Run final Test once

Evaluate all five frozen checkpoints on the same locked Test Set, then report mean ± sample SD. Do not select a seed based on Test.

### Candidate single checkpoint

Under the existing “Validation Macro-F1 first” selection rule, seed `7777` is the candidate single checkpoint with **91.59% Validation Macro-F1**. If another rule is desired, define it before looking at Test.

## 10. Repository status

Until final Test is available:

- EXP-004 is stored as **Validation complete — Test pending**.
- EXP-003 remains the official ResNet50 Test benchmark in the V04 comparison table.
- The repository README should surface EXP-004 as the latest ResNet50 rerun without replacing the official benchmark row.