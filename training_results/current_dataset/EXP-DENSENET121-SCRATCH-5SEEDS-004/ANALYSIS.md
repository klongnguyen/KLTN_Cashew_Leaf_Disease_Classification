# Analysis — DenseNet121 Scratch 5-Seed Rerun (repository EXP-004)

## 1. Scope and provenance

The uploaded archive was generated on 2026-09-29 and self-identifies as `EXP-DENSENET121-SCRATCH-5SEEDS-003`. The repository already contains an earlier experiment with exactly that ID and completed locked-Test evaluation. To avoid overwriting scientific history, this uploaded rerun is registered as **`EXP-DENSENET121-SCRATCH-5SEEDS-004`**, while the exact source config is preserved separately.

The archive contains complete 5-seed training histories, validation predictions/reports/confusion matrices and aggregate validation metrics, but **no final Test evaluation** (`run_final_test=false`). All conclusions below therefore concern **training/validation only**.

## 2. Aggregate result

The rerun reaches **92.13 ± 1.31% Validation Accuracy** and **91.90 ± 1.37% Validation Macro-F1** across five seeds. This is a strong and relatively stable validation result: the standard deviation is around 1.3–1.5 percentage points for the main aggregate metrics.

Compared with the repository's official DenseNet121 EXP-003 validation result, the rerun increases mean Validation Accuracy by **2.26 pp** and Macro-F1 by **2.37 pp**, while reducing their seed variability by about **0.54 pp** and **0.59 pp**, respectively.

Importantly, the gain is not driven by a single lucky seed: all five seeds improve relative to the previous EXP-003 validation values. However, because the execution strategy changed from 2-GPU `MirroredStrategy` to 1-GPU `OneDeviceStrategy`, this should not be described as a proven architectural/model improvement. It is best interpreted as a better rerun under a different execution environment.

## 3. Seed-level behavior

Seed `42` is strongest under the predefined selection metric, reaching **93.11% Validation Macro-F1** with Validation Accuracy **93.22%** and `val_loss=0.2426`. Seed `7777` is very close at **92.95% Macro-F1** and **93.15% Accuracy**. Seed `2026` is the weakest of this rerun at **90.00% Macro-F1** and **90.37% Accuracy**, but it still exceeds its previous EXP-003 validation result.

The best checkpoint occurs between epochs **16 and 29**, with a mean best epoch of **22.8**. Actual training lasts **24–37 epochs**, averaging **30.8**, which is consistent with EarlyStopping patience 8 after the best validation-loss checkpoint.

## 4. Learning-curve interpretation

The training panel shows very noisy validation accuracy/loss during the first roughly 10–15 epochs for all seeds, especially seeds `42`, `123`, `3407`, and `7777`. After that stage, validation curves become substantially more stable while training accuracy continues approaching 0.97–0.98.

At the minimum-`val_loss` checkpoints, the mean Train–Validation accuracy gap is **4.59 pp**. The gaps by seed are:

- seed `42`: **3.29 pp**
- seed `123`: **4.69 pp**
- seed `2026`: **5.90 pp**
- seed `3407`: **4.92 pp**
- seed `7777`: **4.13 pp**

This indicates a moderate generalization gap rather than catastrophic overfitting. Seed `2026` has the largest checkpoint gap, about **5.90 pp**, matching its weaker validation result. Because validation loss usually rises after its minimum while training accuracy continues improving, checkpointing by minimum `val_loss` remains appropriate.

## 5. Per-class performance

| Class | Precision | Recall | F1 ± SD |
|---|---:|---:|---:|
| `anthracnose` | 83.77% | 89.46% | **86.47 ± 2.66%** |
| `healthy` | 91.46% | 89.69% | **90.54 ± 2.45%** |
| `leaf_miner` | 95.57% | 86.75% | **90.94 ± 1.36%** |
| `not_cashew_leaf` | 95.03% | 93.38% | **94.18 ± 0.71%** |
| `red_rust` | 95.61% | 99.25% | **97.39 ± 0.35%** |

`anthracnose` remains the central classification difficulty. Its mean precision (**83.77%**) is substantially lower than its recall (**89.46%**), meaning the model is comparatively willing to predict anthracnose for samples belonging to other classes. This is consistent with the summed confusion matrix.

`leaf_miner` shows the opposite pattern: very high precision (**95.57%**) but lower recall (**86.75%**). When the model predicts leaf miner it is usually correct, but a meaningful fraction of true leaf-miner examples are missed, most often as anthracnose or not-cashew-leaf.

`red_rust` is the strongest and most stable class: **97.39 ± 0.35% F1** with about **99.25% recall**. `not_cashew_leaf` is also strong at **94.18 ± 0.71% F1**.

## 6. Confusion-pattern analysis

Across five validation passes, the most frequent off-diagonal prediction events are:

1. `leaf_miner → anthracnose`: **104**
2. `healthy → anthracnose`: **82**
3. `not_cashew_leaf → anthracnose`: **63**
4. `anthracnose → healthy`: **56**
5. `anthracnose → red_rust`: **50**
6. `leaf_miner → not_cashew_leaf`: **43**
7. `anthracnose → leaf_miner`: **34**
8. `not_cashew_leaf → healthy`: **29**
9. `healthy → not_cashew_leaf`: **16**
10. `anthracnose → not_cashew_leaf`: **15**

The pattern is coherent across seeds. `anthracnose` acts as a major confusion sink: `leaf_miner`, `healthy`, and `not_cashew_leaf` are repeatedly classified as anthracnose. At the same time, true anthracnose is most often confused with `healthy` or `red_rust`.

This suggests that future data review should prioritize borderline anthracnose examples: mild symptoms, visually similar healthy discoloration, mixed/ambiguous lesions, and cases where lesion morphology overlaps with red rust. Any relabeling should be performed under the existing dataset/annotation rules rather than by inspecting model errors and opportunistically changing labels.

## 7. Comparison against DenseNet121 official EXP-003

| Seed | Old Val Acc | New Val Acc | Δ Acc | Old Macro-F1 | New Macro-F1 | Δ F1 |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 88.45% | 93.22% | +4.77 pp | 87.97% | 93.11% | +5.14 pp |
| 123 | 88.59% | 91.08% | +2.49 pp | 88.10% | 90.91% | +2.81 pp |
| 2026 | 89.87% | 90.37% | +0.50 pp | 89.61% | 90.00% | +0.39 pp |
| 3407 | 89.44% | 92.80% | +3.36 pp | 89.18% | 92.54% | +3.36 pp |
| 7777 | 93.01% | 93.15% | +0.14 pp | 92.81% | 92.95% | +0.14 pp |

All five seeds improve, which makes this rerun more convincing as a stable validation result than a mean increase caused by one outlier seed. Still, the environment difference prevents attributing the improvement to a specific modeling change.

## 8. What can and cannot be concluded

Supported by this archive:

- DenseNet121 scratch reaches about **92.1% validation accuracy** and **91.9% validation Macro-F1** on V04 across five seeds.
- Seed sensitivity is lower than in the previous DenseNet121 EXP-003 validation run.
- `anthracnose` is the weakest class, with false-positive pressure from several other classes.
- `red_rust` is very strong and stable.
- Validation-only selection would currently choose seed `42` under the predefined Macro-F1 rule.

Not supported yet:

- Whether the rerun improves **locked Test** performance.
- Whether it should replace the official DenseNet121 benchmark.
- Whether the apparent improvement is caused by the single-GPU environment, nondeterminism, framework/runtime differences, or another execution-level detail.
- Deployment inference time/FPS for this rerun.

## 9. Recommended next step

Do **not** tune this experiment further before Test. Freeze the five checkpoints and configuration, run the same locked Test Set once for all five seeds, save full per-seed predictions/reports, aggregate mean ± sample SD, and measure inference speed in one documented runtime. Only after that should the repository's official DenseNet121 benchmark row be reconsidered.
