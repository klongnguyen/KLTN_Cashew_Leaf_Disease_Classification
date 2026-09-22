# EXP-DENSENET121-SCRATCH-5SEEDS-003

**Model:** DenseNet121 Scratch  
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
- **Parameters:** `7,566,917`
- **Checkpoint:** minimum `val_loss`
- **Backbone:** `x = backbone(x)`, không force `training=True`
- **Test:** locked; không dùng để tuning hoặc chọn seed

## 5-seed results

| Seed | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 42 | 88.45% | 87.97% | 92.14% | 91.59% | 26 |
| 123 | 88.59% | 88.10% | 92.29% | 91.97% | 23 |
| 2026 | 89.87% | 89.61% | 91.41% | 91.01% | 30 |
| 3407 | 89.44% | 89.18% | 90.54% | 90.18% | 19 |
| 7777 | 93.01% | 92.81% | 92.72% | 92.13% | 25 |

### Aggregate

| Metric | Mean ± sample SD |
|---|---:|
| Validation Accuracy | **89.87 ± 1.85%** |
| Validation Macro F1 | **89.53 ± 1.96%** |
| Test Accuracy | **91.82 ± 0.86%** |
| Test Macro Precision | **91.61 ± 0.77%** |
| Test Macro Recall | **91.40 ± 0.80%** |
| Test Macro F1 | **91.37 ± 0.79%** |
| Balanced Accuracy | **91.40 ± 0.80%** |
| Weighted F1 | **91.88 ± 0.76%** |

## Per-class Test classification report — 5-seed mean

| Class | Precision | Recall | F1 ± SD |
|---|---:|---:|---:|
| `anthracnose` | 80.12% | 83.11% | **81.30 ± 2.00%** |
| `healthy` | 90.09% | 93.39% | **91.68 ± 1.51%** |
| `leaf_miner` | 94.33% | 88.64% | **91.29 ± 1.37%** |
| `not_cashew_leaf` | 98.39% | 93.76% | **96.02 ± 1.35%** |
| `red_rust` | 95.12% | 98.10% | **96.58 ± 0.42%** |

## Training curves — 5 seeds

![Training curves](./figures/training_curves_5seeds_panel.svg)

Các confusion-matrix panel đầy đủ được giữ trong ANALYSIS/FULL archive nếu chưa được commit vào GitHub.

## Deployment selection

Deployment seed: **7777**, được chọn bằng **Validation Macro-F1** với tie-break là Validation Loss thấp hơn. Test không tham gia lựa chọn.

Các model/checkpoint nặng không lưu trực tiếp trong GitHub; FULL archive được giữ riêng để tái sử dụng và tích hợp YOLO.

## Reproducibility

- Cùng Train/Validation/Test split cho toàn bộ seed.
- Cùng hyperparameters giữa 5 seed.
- Checkpoint chọn theo Validation.
- Test chỉ dùng sau khi cấu hình đã khóa.
- Báo cáo `Mean ± sample Standard Deviation (ddof=1)`.

Machine-readable metrics nằm trong [`aggregate/`](./aggregate/).
