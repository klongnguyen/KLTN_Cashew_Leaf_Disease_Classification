# EXP-RESNET50-SCRATCH-5SEEDS-003

**Model:** ResNet50 from scratch  
**Dataset:** Cashew_dataV04  
**Seeds:** `[42, 123, 2026, 3407, 7777]`

## Dataset

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| anthracnose | 945 | 294 | 122 | 1,361 |
| healthy | 806 | 225 | 118 | 1,149 |
| leaf_miner | 893 | 249 | 132 | 1,274 |
| not_cashew_leaf | 1,101 | 314 | 157 | 1,572 |
| red_rust | 1,077 | 320 | 158 | 1,555 |
| **TOTAL** | **4,822** | **1,402** | **687** | **6,911** |

## Configuration

- Input: `224×224`
- Batch size: `32`
- Initial LR: `0.001`
- Max epochs: `50`
- Optimizer: Adam
- Loss: SparseCategoricalCrossentropy
- Head: GlobalAveragePooling → Dense(512) → BatchNorm → Dropout(0.40) → Softmax
- Checkpoint: minimum `val_loss`
- Backbone call: `x = backbone(x)`; không force `training=True`
- Total parameters: **24,641,413**

## Validation Mean ± Std

- Accuracy: **88.22 ± 2.49%**
- Macro Precision: **88.48 ± 2.31%**
- Macro Recall: **87.97 ± 2.65%**
- Macro F1: **88.06 ± 2.52%**
- Balanced Accuracy: **87.97 ± 2.65%**

## Final Test Mean ± Std

- Accuracy: **90.63 ± 2.11%**
- Macro Precision: **90.36 ± 1.79%**
- Macro Recall: **90.24 ± 2.16%**
- Macro F1: **90.17 ± 2.04%**
- Balanced Accuracy: **90.24 ± 2.16%**
- Weighted F1: **90.68 ± 2.01%**
- Inference: **8.07 ± 0.10 ms/image**
- Throughput: **123.97 ± 1.50 FPS** on the experiment environment

## Per-seed summary

| Seed | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 |
|---:|---:|---:|---:|---:|
| 42 | 89.80% | 89.67% | 92.58% | 92.05% |
| 123 | 85.52% | 85.58% | 87.92% | 87.82% |
| 2026 | 85.66% | 85.19% | 88.79% | 88.09% |
| 3407 | **91.01%** | **90.81%** | 91.85% | 91.28% |
| 7777 | 89.09% | 89.05% | 91.99% | 91.60% |

Deployment seed should be **3407**, selected by Validation Macro-F1 only. Test performance is not used for seed selection.

## Per-class Test results — 5-seed mean ± SD

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| `anthracnose` | 78.57% | 80.66% | **79.35 ± 3.84%** |
| `healthy` | 89.33% | 92.37% | **90.69 ± 2.05%** |
| `leaf_miner` | 91.58% | 90.00% | **90.75 ± 1.06%** |
| `not_cashew_leaf` | 97.20% | 91.21% | **94.06 ± 2.48%** |
| `red_rust` | 95.09% | 96.96% | **95.99 ± 1.56%** |

`anthracnose` remains the most difficult class. `red_rust` is the strongest class.

Most frequent average cross-class errors:

- `not_cashew_leaf → anthracnose`: **7.90%**
- `anthracnose → healthy`: **7.70%**
- `anthracnose → leaf_miner`: **7.05%**
- `leaf_miner → anthracnose`: **5.45%**
- `healthy → anthracnose`: **5.08%**

## Training behavior

- Best epoch: **21.80 ± 9.58**
- Training time: **30.97 ± 9.74 min/seed**
- Best checkpoint size: approximately **282.23 MB**
- Seeds `123` and `2026` are notably weaker than seeds `42`, `3407`, and `7777`, which increases between-seed variance.

## V04 model comparison

| Model | Test Accuracy | Macro F1 | Parameters |
|---|---:|---:|---:|
| DenseNet121 Scratch | **91.82 ± 0.86%** | **91.37 ± 0.79%** | 7.57M |
| ResNet50 Scratch | **90.63 ± 2.11%** | **90.17 ± 2.04%** | 24.64M |
| ViT Scratch | **85.30 ± 1.68%** | **84.41 ± 1.71%** | **0.35M** |

On `Cashew_dataV04`, DenseNet121 currently has the highest mean Accuracy/Macro-F1 and the lowest variation among the CNN baselines. ResNet50 remains strong but is more seed-sensitive and uses substantially more parameters. ViT is much smaller but lower in classification performance.

## Scientific protocol

- Same Train/Val/Test split for all 5 seeds.
- Test Set is locked and not used for tuning.
- Checkpoint selection uses Validation only.
- Deployment seed selection uses Validation only.
- Results are reported as **Mean ± sample standard deviation (`ddof=1`)**.
- V03 and V04 metrics are not treated as a controlled comparison because the dataset content/split changed after cleaning and leakage removal.

## GitHub package

GitHub stores lightweight analysis artifacts only. Checkpoints `.weights.h5`, deployment `.keras` models and FULL ZIP should remain in external archive storage for model reuse / YOLO integration.
