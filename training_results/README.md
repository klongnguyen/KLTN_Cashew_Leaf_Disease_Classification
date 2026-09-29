# Training Results

Thư mục này quản lý kết quả huấn luyện theo **phiên bản dataset** để tránh so sánh sai giữa các experiment sử dụng dữ liệu khác nhau.

## Current benchmark — Cashew_dataV04

Dataset hiện tại có **6,911 ảnh / 5 lớp**:

```text
Train      4,822
Validation 1,402
Test         687
```

| Experiment | Model | Seeds | Test Accuracy | Macro F1 | Params | Status |
|---|---|---:|---:|---:|---:|---|
| [`EXP-DENSENET121-SCRATCH-5SEEDS-003`](./current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/) | DenseNet121 Scratch | 5 | **91.82 ± 0.86%** | **91.37 ± 0.79%** | 7.57M | ✅ V04 baseline |
| [`EXP-RESNET50-SCRATCH-5SEEDS-003`](./current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-003/) | ResNet50 Scratch | 5 | **90.63 ± 2.11%** | **90.17 ± 2.04%** | 24.64M | ✅ V04 baseline |
| [`EXP-VIT-SCRATCH-5SEEDS-003`](./current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/) | Compact ViT Scratch | 5 | **85.30 ± 1.68%** | **84.41 ± 1.71%** | **0.35M** | ✅ V04 baseline |

➡️ Chi tiết dataset, class-level metrics và figures: [`current_dataset/README.md`](./current_dataset/README.md)

## Latest validation-only rerun

[`EXP-RESNET50-SCRATCH-5SEEDS-004`](./current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-004/) đã hoàn tất 5-seed training/validation trên Colab NVIDIA L4:

- Validation Accuracy: **90.23 ± 1.45%**
- Validation Macro-F1: **90.03 ± 1.44%**
- Per-class F1 thấp nhất: `anthracnose` — **83.46 ± 2.87%**
- Highest validation Macro-F1 seed: `7777` — **91.59%**
- Final locked Test: **chưa chạy**

Do đó EXP-004 hiện có status **Validation complete — Test pending** và chưa thay thế ResNet50 EXP-003 trong bảng official Test benchmark.

## Quy tắc benchmark

- Mỗi dataset version phải có fixed Train/Validation/Test split.
- Baseline hiện tại dùng 5 seeds: `42, 123, 2026, 3407, 7777`.
- Validation dùng cho model selection/tuning.
- Test không dùng để tuning hoặc chọn seed.
- Báo cáo `Mean ± sample Standard Deviation (ddof=1)`.
- Không ghi đè experiment cũ; thay dataset/config quan trọng phải tạo ID mới.
- Chỉ so sánh model trực tiếp khi chúng dùng **cùng dataset version và cùng evaluation protocol**.
- Validation-only experiment không được ghi vào official Test benchmark cho tới khi Test khóa được chạy trên cấu hình đã freeze.

## Dataset history

| Version | Total | Train | Val | Test | Status |
|---|---:|---:|---:|---:|---|
| `Cashew_dataV04` | **6,911** | 4,822 | 1,402 | 687 | **Current** |
| `Cashew_dataV03` | 7,213 | 5,049 | 1,433 | 731 | Historical |

V04 được tạo sau khi tiếp tục loại ảnh mờ/chất lượng thấp và xử lý các trường hợp có nguy cơ data leakage.

Xem [`../DATASET_CHANGELOG.md`](../DATASET_CHANGELOG.md).

## Thư mục

```text
training_results/
├── current_dataset/            # V04 benchmark + validation reruns + historical V03 paths kept for compatibility
└── archive_legacy_dataset/     # experiments cũ hơn V03
```

## YOLO detection

YOLO26 chưa được đưa vào `current_dataset/` vì detector cuối chưa khóa. Toàn bộ Take 01–06 được lưu tại [`../failure/YOLO26/README.md`](../failure/YOLO26/README.md).

## Dung lượng

GitHub ưu tiên lưu:
- config;
- aggregate CSV/JSON;
- README;
- panel figures nhẹ.

Checkpoint `.weights.h5`, model `.keras`/`.pt` và FULL ZIP không commit trực tiếp; lưu ở external archive/Drive để phục vụ tái sử dụng.
