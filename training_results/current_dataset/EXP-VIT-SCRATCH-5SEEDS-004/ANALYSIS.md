# Analysis — EXP-VIT-SCRATCH-5SEEDS-004

## 1. Scope

This document analyzes the uploaded Compact Vision Transformer rerun on `Cashew_dataV04`.

The source archive identifies itself as `EXP-VIT-SCRATCH-5SEEDS-003`. That experiment ID is already occupied by the repository's official V04 ViT benchmark with locked-Test results, so the new rerun is versioned as **EXP-VIT-SCRATCH-5SEEDS-004** instead of overwriting prior evidence.

The uploaded run contains training and Validation outputs for all five seeds but intentionally does **not** run the final Test Set.

## 2. Main validation result

Across seeds `42, 123, 2026, 3407, 7777`:

- Validation Accuracy: **87.13 ± 0.62%**
- Macro Precision: **87.18 ± 0.82%**
- Macro Recall: **86.43 ± 0.71%**
- Macro F1: **86.68 ± 0.70%**
- Balanced Accuracy: **86.43 ± 0.71%**

The relatively small standard deviation is the main strength of this rerun: the Compact ViT is much less seed-sensitive here than in its previous official EXP-003 validation run.

## 3. Seed behavior

| Seed | Val Accuracy | Val Macro F1 | Val Loss | Best epoch |
|---:|---:|---:|---:|---:|
| 42 | **88.09%** | **87.73%** | **0.3596** | 18 |
| 123 | 87.09% | 86.74% | 0.3803 | 30 |
| 2026 | 86.52% | 85.95% | 0.4176 | 23 |
| 3407 | 86.66% | 86.11% | 0.3852 | 25 |
| 7777 | 87.30% | 86.85% | 0.4017 | 20 |

Seed `42` is the best checkpoint under the predefined Validation Macro-F1 selection rule and also has the lowest Validation loss. Therefore no tie-break ambiguity exists for deployment selection in this rerun.

The spread from the lowest to highest Validation Accuracy is only about **1.57 percentage points**, indicating considerably improved seed stability compared with official EXP-003.

## 4. Comparison with official EXP-003

Official EXP-003 validation:

- Accuracy: **85.28 ± 1.28%**
- Macro F1: **84.68 ± 1.38%**

Rerun EXP-004 validation:

- Accuracy: **87.13 ± 0.62%**
- Macro F1: **86.68 ± 0.70%**

Changes:

- Accuracy mean: **+1.85 pp**
- Macro-F1 mean: **+2.00 pp**
- Accuracy SD: **−0.66 pp**
- Macro-F1 SD: **−0.68 pp**

Per-seed Validation Accuracy changes are approximately:

- seed 42: **+4.50 pp**
- seed 123: **+1.07 pp**
- seed 2026: **+1.14 pp**
- seed 3407: **+2.14 pp**
- seed 7777: **+0.42 pp**

Per-seed Macro-F1 changes are also positive for all five seeds.

This consistency is stronger evidence of a reproducible rerun improvement than a mean increase driven by one favorable seed. However, because the old run used 2-GPU `MirroredStrategy` and this rerun uses a single-GPU `OneDeviceStrategy`, the difference must not be attributed solely to an architecture or hyperparameter improvement.

## 5. Per-class behavior

### Anthracnose

Validation F1: **80.28 ± 1.68%**

- Precision: **78.68%**
- Recall: **82.18%**

This remains the hardest class. Low precision indicates substantial false-positive attraction toward `anthracnose`.

### Healthy

Validation F1: **83.15 ± 3.31%**

- Precision: **87.06%**
- Recall: **79.64%**

Healthy has the largest class-level F1 standard deviation and relatively low recall. The summed confusion matrix shows `healthy → anthracnose` as the second most frequent off-diagonal error.

### Leaf Miner

Validation F1: **85.37 ± 2.13%**

- Precision: **88.28%**
- Recall: **82.81%**

The most frequent confusion event overall is `leaf_miner → anthracnose`, showing that the model still struggles to separate some lesion appearances between these two classes.

### Not Cashew Leaf

Validation F1: **91.84 ± 1.82%**

This class remains strong, which supports the usefulness of the out-of-scope class in full-image classification. However, `healthy → not_cashew_leaf` and `not_cashew_leaf → healthy` remain visible secondary confusions.

### Red Rust

Validation F1: **92.74 ± 0.96%**

Red Rust is the strongest and most stable class in this rerun, with recall **95.63%**. The main red-rust-related error comes from `anthracnose → red_rust`, not the reverse direction.

## 6. Confusion structure

Summing the five Validation confusion matrices gives recurring prediction events:

1. `leaf_miner → anthracnose`: 144
2. `healthy → anthracnose`: 129
3. `anthracnose → red_rust`: 116
4. `anthracnose → leaf_miner`: 66
5. `healthy → not_cashew_leaf`: 65
6. `anthracnose → healthy`: 65

This shows two main research targets:

1. better representation of `anthracnose`, particularly its overlap with Leaf Miner and Red Rust;
2. improving Healthy recall and separating Healthy from early/subtle Anthracnose.

These patterns are useful targets for qualitative image review before changing architecture or augmentation.

## 7. Learning-curve interpretation

Mean training length is **33.2 epochs** and the mean selected checkpoint is around epoch **23.2**. EarlyStopping patience is 10, so the stopping behavior is consistent with the configured protocol.

An important interpretation detail: the training pipeline includes image augmentation. Therefore training accuracy is measured on augmented training samples while Validation accuracy is measured on unaugmented Validation samples. For seeds `42` and `7777`, Validation accuracy is above training accuracy at the minimum-`val_loss` checkpoint. This is plausible under augmentation and should not be described as evidence of data leakage.

Several runs show Validation loss increasing after the selected checkpoint. Checkpointing by minimum `val_loss` is therefore appropriate.

## 8. Efficiency observations

The model has only **347,717 parameters**, far fewer than the scratch CNN baselines used elsewhere in the project.

In this rerun:

- mean training time: **5.81 min/seed**
- total five-seed time: **29.03 min**
- checkpoint weights: approximately **4.41 MB** each

These values make the Compact ViT operationally lightweight for experimentation. They are **training-environment measurements**, not a substitute for standardized deployment inference benchmarking.

## 9. Scientific status

The archive specifies:

- `run_final_test = false`
- `final_test_completed = false`
- locked Test Set preserved
- deployment seed selection from Validation only

This is correct experimental discipline. The rerun should not replace the official ViT EXP-003 Test row until the frozen five checkpoints are evaluated on the locked Test Set.

## 10. Recommended next action

No further tuning should be done on EXP-004 before final Test if this configuration is intended to become the rerun benchmark.

Freeze:

- model architecture;
- preprocessing;
- augmentation;
- optimizer/LR schedule;
- five seeds;
- checkpoints.

Then run the same locked Test Set once for all five checkpoints and aggregate the predefined metrics.

If further attempts are desired to specifically improve Anthracnose, create a **new experiment ID** rather than altering EXP-004 after seeing these Validation results.
