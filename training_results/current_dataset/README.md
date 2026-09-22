# Current Dataset Experiments — Cashew_dataV04

Thư mục này quản lý các **classification baseline chính thức** trên Master Dataset hiện tại `Cashew_dataV04`.

## Dataset snapshot

V04 được tạo sau lần làm sạch bổ sung: loại ảnh mờ/chất lượng thấp và xử lý các trường hợp có nguy cơ **data leakage** giữa các split.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 945 | 294 | 122 | 1,361 |
| `healthy` | 806 | 225 | 118 | 1,149 |
| `leaf_miner` | 893 | 249 | 132 | 1,274 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,077 | 320 | 158 | 1,555 |
| **TOTAL** | **4,822** | **1,402** | **687** | **6,911** |

![Dataset distribution](./dataset_distribution_v04.svg)

## Benchmark V04

| Experiment | Model | Val Accuracy | Test Accuracy | Macro F1 | Params | Inference* |
|---|---|---:|---:|---:|---:|---:|
| [`EXP-DENSENET121-SCRATCH-5SEEDS-003`](./EXP-DENSENET121-SCRATCH-5SEEDS-003/) | DenseNet121 | **89.87 ± 1.85%** | **91.82 ± 0.86%** | **91.37 ± 0.79%** | 7.57M | 13.20 ms/img |
| [`EXP-RESNET50-SCRATCH-5SEEDS-003`](./EXP-RESNET50-SCRATCH-5SEEDS-003/) | ResNet50 | **88.22 ± 2.49%** | **90.63 ± 2.11%** | **90.17 ± 2.04%** | 24.64M | 8.07 ms/img |
| [`EXP-VIT-SCRATCH-5SEEDS-003`](./EXP-VIT-SCRATCH-5SEEDS-003/) | Compact ViT | **85.28 ± 1.28%** | **85.30 ± 1.68%** | **84.41 ± 1.71%** | **0.35M** | **6.08 ms/img** |

\* Inference benchmark được đo trong môi trường Kaggle T4×2/MirroredStrategy của từng experiment; không xem là tốc độ deployment trên web/mobile.

![V04 benchmark](./benchmark_v04.svg)

### Kết luận benchmark hiện tại

- **DenseNet121** có hiệu năng tổng thể cao nhất và độ ổn định tốt nhất: Test Accuracy Std chỉ **0.86%**.
- **ResNet50** đứng thứ hai về Accuracy/Macro-F1 nhưng có 24.64M tham số và biến thiên seed lớn hơn DenseNet121.
- **Compact ViT** nhẹ nhất rõ rệt và nhanh nhất trong benchmark GPU hiện tại, nhưng hiệu năng thấp hơn hai CNN.
- `anthracnose` tiếp tục là lớp khó nhất ở cả ba kiến trúc.

## Per-class Macro comparison

| Class | DenseNet121 F1 | ResNet50 F1 | ViT F1 |
|---|---:|---:|---:|
| `anthracnose` | 81.30 ± 2.00% | 79.35 ± 3.84% | 70.43 ± 4.44% |
| `healthy` | 91.68 ± 1.51% | 90.69 ± 2.05% | 79.72 ± 2.22% |
| `leaf_miner` | 91.29 ± 1.37% | 90.75 ± 1.06% | 84.81 ± 2.45% |
| `not_cashew_leaf` | 96.02 ± 1.35% | 94.06 ± 2.48% | 97.28 ± 1.37% |
| `red_rust` | 96.58 ± 0.42% | 95.99 ± 1.56% | 89.81 ± 1.90% |

## Protocol

- Fixed split trong toàn bộ `Cashew_dataV04`.
- Seeds: `42, 123, 2026, 3407, 7777`.
- Validation dùng cho checkpoint, EarlyStopping/LR scheduling và deployment-seed selection.
- Test Set không dùng để tuning hoặc chọn seed.
- Kết quả báo cáo theo **Mean ± sample Standard Deviation (`ddof=1`)**.
- Các figure so sánh seed phải dùng panel chung để tránh chọn hình đẹp nhất.

## Machine-readable summaries

- [`benchmark_v04.csv`](./benchmark_v04.csv)
- [`class_f1_comparison_v04.csv`](./class_f1_comparison_v04.csv)
- [`dataset_cashew_v04.csv`](../../dataset_cashew_v04.csv)

## Dataset versioning

Các experiment `*-002` dùng `Cashew_dataV03` vẫn còn trong thư mục này để giữ tương thích với các đường dẫn cũ, nhưng **không còn là current benchmark**. Không so sánh trực tiếp V03 với V04 như một controlled model comparison vì Train/Validation/Test đã thay đổi.

Các experiment rất cũ hơn nữa được lưu tại [`../archive_legacy_dataset/`](../archive_legacy_dataset/).

## Object detection

Các YOLO26 take đang ở giai đoạn annotation/data iteration được quản lý riêng tại [`../../failure/YOLO26/README.md`](../../failure/YOLO26/README.md). Take 006 là iteration tốt nhất hiện tại nhưng chưa được khóa làm final detector.
