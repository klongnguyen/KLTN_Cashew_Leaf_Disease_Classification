# EXP-VIT-SCRATCH-5SEEDS-003

**Model:** Vision Transformer Scratch  
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
- **Patch size:** `16×16`
- **Projection dim:** `64`
- **Attention heads:** `4`
- **Transformer blocks:** `8`
- **Transformer MLP:** `128`
- **Batch size:** `32`
- **Optimizer:** `AdamW`
- **Initial LR:** `3e-4`
- **Weight decay:** `1e-4`
- **Max epochs:** `60`
- **Parameters:** `347,717`
- **Checkpoint:** minimum `val_loss`
- **Test:** locked; không dùng để tuning hoặc chọn seed

## 5-seed results

| Seed | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Best epoch |
|---:|---:|---:|---:|---:|---:|
| 42 | 83.59% | 82.97% | 83.55% | 82.87% | 20 |
| 123 | 86.02% | 85.40% | 86.17% | 85.39% | 32 |
| 2026 | 85.38% | 84.63% | 86.75% | 85.79% | 25 |
| 3407 | 84.52% | 83.86% | 86.61% | 85.76% | 35 |
| 7777 | 86.88% | 86.54% | 83.41% | 82.25% | 28 |

### Aggregate

| Metric | Mean ± sample SD |
|---|---:|
| Validation Accuracy | **85.28 ± 1.28%** |
| Validation Macro F1 | **84.68 ± 1.38%** |
| Test Accuracy | **85.30 ± 1.68%** |
| Test Macro Precision | **85.04 ± 1.29%** |
| Test Macro Recall | **84.65 ± 1.69%** |
| Test Macro F1 | **84.41 ± 1.71%** |
| Balanced Accuracy | **84.65 ± 1.69%** |
| Weighted F1 | **85.38 ± 1.61%** |

## Per-class Test classification report — 5-seed mean

| Class | Precision | Recall | F1 ± SD |
|---|---:|---:|---:|
| `anthracnose` | 74.41% | 67.87% | **70.43 ± 4.44%** |
| `healthy` | 73.13% | 88.14% | **79.72 ± 2.22%** |
| `leaf_miner` | 85.18% | 85.00% | **84.81 ± 2.45%** |
| `not_cashew_leaf` | 98.69% | 95.92% | **97.28 ± 1.37%** |
| `red_rust` | 93.78% | 86.33% | **89.81 ± 1.90%** |

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
