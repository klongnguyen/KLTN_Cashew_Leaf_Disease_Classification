# EXP-RESNET50-SCRATCH-5SEEDS-003

**Model:** ResNet50 Scratch  
**Training mode:** From scratch  
**Dataset:** `Cashew_dataV04`  
**Seeds:** `42, 123, 2026, 3407, 7777`

## Dataset

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 945 | 294 | 122 | 1,361 |
| `healthy` | 806 | 225 | 118 | 1,149 |
| `leaf_miner` | 893 | 249 | 132 | 1,274 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,077 | 320 | 158 | 1,555 |
| **TOTAL** | **4,822** | **1,402** | **687** | **6,911** |

V04 được tạo sau khi tiếp tục loại ảnh mờ/chất lượng thấp và xử lý các trường hợp có nguy cơ data leakage. Split được khóa cho toàn bộ baseline V04.

## Configuration

- **Input:** `224×224`
- **Batch size:** `32`
- **Optimizer:** `Adam`
- **Initial LR:** `1e-3`
- **Max epochs:** `50`
- **Head:** `GAP → Dense(512) → BatchNorm → Dropout(0.40) → Dense(5)`
- **Parameters:** `24,641,413`
- **Checkpoint:** minimum `val_loss`
- **Backbone:** `x = backbone(x)`, không force `training=True`
- **Test:** locked; không dùng để tuning hoặc chọn seed

## 5-seed results

| Seed | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 42 | 89.80% | 89.67% | 92.58% | 92.05% | 24 |
| 123 | 85.52% | 85.58% | 87.92% | 87.82% | 9 |
| 2026 | 85.66% | 85.19% | 88.79% | 88.09% | 15 |
| 3407 | 91.01% | 90.81% | 91.85% | 91.28% | 30 |
| 7777 | 89.09% | 89.05% | 91.99% | 91.60% | 31 |

### Aggregate

| Metric | Mean ± sample SD |
|---|---:|
| Validation Accuracy | **88.22 ± 2.49%** |
| Validation Macro F1 | **88.06 ± 2.52%** |
| Test Accuracy | **90.63 ± 2.11%** |
| Test Macro Precision | **90.36 ± 1.79%** |
| Test Macro Recall | **90.24 ± 2.16%** |
| Test Macro F1 | **90.17 ± 2.04%** |
| Balanced Accuracy | **90.24 ± 2.16%** |
| Weighted F1 | **90.68 ± 2.01%** |

## Per-class Test classification report — 5-seed mean

| Class | Precision | Recall | F1 ± SD |
|---|---:|---:|---:|
| `anthracnose` | 78.57% | 80.66% | **79.35 ± 3.84%** |
| `healthy` | 89.33% | 92.37% | **90.69 ± 2.05%** |
| `leaf_miner` | 91.58% | 90.00% | **90.75 ± 1.06%** |
| `not_cashew_leaf` | 97.20% | 91.21% | **94.06 ± 2.48%** |
| `red_rust` | 95.09% | 96.96% | **95.99 ± 1.56%** |

`anthracnose` là class khó nhất; `red_rust` ổn định và mạnh nhất trong ResNet50 V04.

## Available figures

- [`figures/per_seed_accuracy.svg`](./figures/per_seed_accuracy.svg)
- [`figures/per_class_f1.svg`](./figures/per_class_f1.svg)

Các panel confusion matrix / learning curve đầy đủ được giữ trong ANALYSIS/FULL archive nếu chưa được commit vào GitHub.

## Deployment selection

Deployment seed: **3407**, được chọn bằng **Validation Macro-F1** với tie-break là Validation Loss thấp hơn. Test không tham gia lựa chọn.

Các model/checkpoint nặng không lưu trực tiếp trong GitHub; FULL archive được giữ riêng để tái sử dụng và tích hợp YOLO.

## Reproducibility

- Cùng Train/Validation/Test split cho toàn bộ seed.
- Cùng hyperparameters giữa 5 seed.
- Checkpoint chọn theo Validation.
- Test chỉ dùng sau khi cấu hình đã khóa.
- Báo cáo `Mean ± sample Standard Deviation (ddof=1)`.

Machine-readable metrics nằm trong [`aggregate/`](./aggregate/).
