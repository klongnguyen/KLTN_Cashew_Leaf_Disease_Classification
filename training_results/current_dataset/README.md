# Current Dataset Experiments

Thư mục này quản lý các baseline classification được huấn luyện trên **Master Dataset hiện tại — Cashew_dataV04**.

## Dataset hiện tại — Cashew_dataV04

Sau lần làm sạch gần nhất, các ảnh mờ/chất lượng thấp và các trường hợp có nguy cơ **data leakage** đã được loại khỏi bộ dữ liệu. Dataset hiện tại có **6,911 ảnh / 5 lớp**, với split cố định dùng chung cho các baseline mới.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 945 | 294 | 122 | 1,361 |
| `healthy` | 806 | 225 | 118 | 1,149 |
| `leaf_miner` | 893 | 249 | 132 | 1,274 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,077 | 320 | 158 | 1,555 |
| **TOTAL** | **4,822** | **1,402** | **687** | **6,911** |

- Test Set được khóa và không dùng để tuning hoặc chọn seed.
- Mỗi cấu hình chạy trên 5 seed: `42, 123, 2026, 3407, 7777`.
- Kết quả được báo cáo theo **Mean ± sample Standard Deviation (ddof=1)**.
- Các kết quả V03 được giữ lại như lịch sử thực nghiệm nhưng **không so sánh trực tiếp** với V04.

## Baseline V04 hiện có

| Experiment | Model | Mode | Seeds | Validation Accuracy | Test Accuracy | Macro F1 | Params | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| [`EXP-DENSENET121-SCRATCH-5SEEDS-003`](./EXP-DENSENET121-SCRATCH-5SEEDS-003/) | DenseNet121 | Scratch | 5 | **89.87 ± 1.85%** | **91.82 ± 0.86%** | **91.37 ± 0.79%** | 7.57M | ✅ V04 baseline |
| [`EXP-RESNET50-SCRATCH-5SEEDS-003`](./EXP-RESNET50-SCRATCH-5SEEDS-003/) | ResNet50 | Scratch | 5 | **88.22 ± 2.49%** | **90.63 ± 2.11%** | **90.17 ± 2.04%** | 24.64M | ✅ V04 baseline |
| [`EXP-VIT-SCRATCH-5SEEDS-003`](./EXP-VIT-SCRATCH-5SEEDS-003/) | Vision Transformer | Scratch | 5 | **85.28 ± 1.28%** | **85.30 ± 1.68%** | **84.41 ± 1.71%** | **0.35M** | ✅ V04 baseline |

## Nhận xét nhanh

- **DenseNet121** hiện có kết quả tốt nhất trên V04: `91.82 ± 0.86%` Test Accuracy và `91.37 ± 0.79%` Macro F1.
- **ResNet50** đứng thứ hai với `90.63 ± 2.11%` Accuracy và `90.17 ± 2.04%` Macro F1. Hai seed `123` và `2026` yếu hơn rõ so với ba seed còn lại nên độ lệch chuẩn cao hơn DenseNet121.
- **ViT compact** thấp hơn hai CNN về Accuracy/F1 nhưng chỉ có khoảng `0.35M` tham số.
- DenseNet121 cao hơn ResNet50 khoảng **1.19 điểm % Test Accuracy** và **1.20 điểm % Macro F1**, đồng thời chỉ dùng khoảng 31% số tham số của ResNet50.
- `anthracnose` tiếp tục là lớp khó nhất ở cả ba baseline. Với ResNet50 V04, F1 của Anthracnose là **79.35 ± 3.84%**.
- `red_rust` là lớp mạnh nhất của ResNet50 V04 với **F1 = 95.99 ± 1.56%**.

### ResNet50 — per-seed Validation/Test Accuracy

<p align="center">
  <img src="./EXP-RESNET50-SCRATCH-5SEEDS-003/figures/per_seed_accuracy.svg" width="95%" alt="ResNet50 V04 per-seed accuracy">
</p>

### ResNet50 — per-class Test F1

<p align="center">
  <img src="./EXP-RESNET50-SCRATCH-5SEEDS-003/figures/per_class_f1.svg" width="90%" alt="ResNet50 V04 per-class F1">
</p>

### DenseNet121 — 5-seed training curves

<p align="center">
  <img src="./EXP-DENSENET121-SCRATCH-5SEEDS-003/figures/training_curves_5seeds_panel.svg" width="100%" alt="DenseNet121 V04 5-seed training curves">
</p>

### Vision Transformer — 5-seed training curves

<p align="center">
  <img src="./EXP-VIT-SCRATCH-5SEEDS-003/figures/training_curves_5seeds_panel.svg" width="100%" alt="ViT V04 5-seed training curves">
</p>

## Previous dataset snapshot — V03

Các experiment `*-002` trong thư mục này được huấn luyện trên **Cashew_dataV03 (7,213 ảnh)** trước lần làm sạch mới. Chúng được giữ để theo dõi lịch sử thay đổi dataset nhưng không được đưa vào bảng benchmark trực tiếp với V04.

## YOLO26 detection experiments

Các lần train YOLO26 đang ở giai đoạn annotation/data iteration được lưu tại [`../../failure/YOLO26/README.md`](../../failure/YOLO26/README.md).

## Lưu ý dung lượng

GitHub chỉ lưu config, metrics, CSV và figures nhẹ. Checkpoint `.weights.h5`, `.keras` và FULL ZIP được quản lý riêng để phục vụ tái sử dụng model và pipeline YOLO.
