# EXP-VIT-SCRATCH-5SEEDS-002

## Vision Transformer Scratch — 5-Seed Baseline

Experiment này huấn luyện **Vision Transformer (ViT) from scratch** trên `Cashew_dataV03` với cùng protocol 5-seed đang dùng cho ResNet50 và DenseNet121.

## Dataset

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 1,096 | 313 | 156 | 1,565 |
| `healthy` | 818 | 225 | 128 | 1,171 |
| `leaf_miner` | 919 | 262 | 131 | 1,312 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,115 | 319 | 159 | 1,593 |
| **TOTAL** | **5,049** | **1,433** | **731** | **7,213** |

## Configuration

| Item | Value |
|---|---|
| Input | 224×224×3 |
| Patch size | 16×16 |
| Patch tokens | 196 |
| Sequence length | 197 |
| Projection dim | 64 |
| Attention heads | 4 |
| Transformer blocks | 8 |
| Transformer MLP units | 128 |
| Head | Dense(256, GELU) + Dropout(0.40) |
| Optimizer | AdamW |
| Initial LR | 3e-4 |
| Weight decay | 1e-4 |
| Batch size | 32 |
| Max epochs | 60 |
| Seeds | 42, 123, 2026, 3407, 7777 |

## Validation — 5 seeds

| Seed | Val Accuracy | Val Macro F1 | Val Loss | Best Epoch |
|---:|---:|---:|---:|---:|
| 42 | 80.18% | 79.16% | 0.5817 | 22 |
| 123 | 83.81% | 83.35% | 0.5115 | 24 |
| 2026 | 73.06% | 71.31% | 0.7314 | 16 |
| 3407 | 84.72% | 84.16% | 0.5022 | 21 |
| 7777 | **85.97%** | **85.54%** | **0.4605** | 28 |
| **Mean ± Std** | **81.55 ± 5.21%** | **80.70 ± 5.77%** | — | — |

## Final Test — 5 seeds

| Seed | Accuracy | Macro F1 | Test Loss |
|---:|---:|---:|---:|
| 42 | 85.09% | 84.77% | 0.4230 |
| 123 | 85.09% | 84.93% | 0.4126 |
| 2026 | 86.18% | 85.99% | 0.4059 |
| 3407 | 85.36% | 85.08% | 0.4045 |
| 7777 | **87.96%** | **87.61%** | **0.3427** |
| **Mean ± Std** | **85.94 ± 1.22%** | **85.68 ± 1.18%** | — |

### Thesis-ready summary

| Metric | Mean ± Std |
|---|---:|
| Accuracy | **85.94 ± 1.22%** |
| Macro Precision | **85.69 ± 1.14%** |
| Macro Recall | **85.93 ± 1.18%** |
| Macro F1 | **85.68 ± 1.18%** |
| Balanced Accuracy | **85.93 ± 1.18%** |
| Weighted F1 | **85.87 ± 1.20%** |

## Per-class Test performance

| Class | Precision Mean | Recall Mean | F1 Mean ± Std |
|---|---:|---:|---:|
| `anthracnose` | 80.69% | 71.15% | **75.61 ± 1.67%** |
| `healthy` | 77.14% | 86.09% | **81.34 ± 1.14%** |
| `leaf_miner` | 85.28% | 86.41% | **85.83 ± 1.93%** |
| `not_cashew_leaf` | **96.89%** | **95.29%** | **96.07 ± 1.53%** |
| `red_rust` | 88.47% | 90.69% | **89.53 ± 1.91%** |

## Main observations

- ViT có **Test Accuracy 85.94 ± 1.22%**, thấp hơn ResNet50 và DenseNet121 scratch.
- Test variance khá thấp (`±1.22%`), nhưng Validation variance lớn (`±5.21%` Accuracy; `±5.77%` Macro-F1), cho thấy quá trình học ViT scratch nhạy với initialization/optimization trên dataset hiện tại.
- `anthracnose` là lớp khó nhất: F1 **75.61 ± 1.67%**, Recall khoảng **71.15%**.
- Nhầm lẫn trung bình nổi bật: `anthracnose → healthy` (~12.18%), `healthy → anthracnose` (~8.75%), `anthracnose → leaf_miner` (~8.08%), `leaf_miner → anthracnose` (~7.63%).
- `not_cashew_leaf` là lớp mạnh nhất: F1 **96.07 ± 1.53%**.
- Model rất nhẹ: **347,717 parameters**, best weights khoảng **4.41 MB**.
- Inference trung bình khoảng **5.36 ms/image (~186 FPS)** trên Kaggle T4×2 trong protocol đo hiện tại.

## Deployment package

Deployment seed được chọn bằng **Validation Macro-F1**, không dùng Test:

```text
deployment seed = 7777
Val Macro-F1    = 85.54%
Val Accuracy    = 85.97%
Val Loss        = 0.4605
```

FULL archive có `deployment_model.keras`, `vit_token_feature_extractor.keras`, `vit_embedding_model.keras`, best weights cả 5 seeds, code mẫu YOLO → ViT classifier, metrics, predictions, manifests và checksums.

Checkpoint/model nặng **không commit vào GitHub**. GitHub chỉ lưu config và kết quả nhẹ; FULL archive dùng cho tái sử dụng model/YOLO.

## Protocol

- Fixed Train / Validation / Test split cho cả 5 seeds.
- Same hyperparameters giữa các seeds.
- Validation dùng cho checkpoint và chọn deployment seed.
- Test không dùng để tuning hoặc chọn seed.
- Kết quả báo cáo bằng **Mean ± sample Standard Deviation (`ddof=1`)**.
