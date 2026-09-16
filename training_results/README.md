# Training Results

Thư mục này quản lý các kết quả huấn luyện **được xem là candidate/benchmark chính thức** theo phiên bản dataset để tránh so sánh sai giữa các experiment dùng dữ liệu khác nhau.

## Cấu trúc

```text
training_results/
├── current_dataset/          # Baseline/candidate dùng Master Dataset hiện tại
└── archive_legacy_dataset/   # Experiment dùng dataset cũ, chỉ tham khảo
```

> Các lần train YOLO26 đang ở giai đoạn phát triển/iteration được lưu riêng tại [`../failure/YOLO26/`](../failure/YOLO26/), không nằm trong `current_dataset/` cho đến khi một detector cuối được chốt.

## Current Dataset — Quick Snapshot

Master Dataset hiện tại có **7,213 ảnh / 5 lớp**, được chia cố định thành **5,049 Train / 1,433 Validation / 731 Test**. Test Set được khóa và chỉ dùng cho đánh giá cuối cùng.

| Experiment | Model | Mode | Seeds | Test Accuracy | Macro F1 | Params | Status |
|---|---|---|---:|---:|---:|---:|---|
| [`EXP-RESNET50-SCRATCH-5SEEDS-002`](./current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-002/) | ResNet50 | Scratch | 5 | **90.10 ± 1.82%** | **90.12 ± 1.81%** | 24.64M | ✅ Baseline hợp lệ |
| [`EXP-DENSENET121-SCRATCH-5SEEDS-002`](./current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-002/) | DenseNet121 | Scratch | 5 | **89.00 ± 3.27%** | **89.22 ± 2.92%** | 7.57M | ✅ Baseline hợp lệ |
| [`EXP-VIT-SCRATCH-5SEEDS-002`](./current_dataset/EXP-VIT-SCRATCH-5SEEDS-002/) | Vision Transformer | Scratch | 5 | **85.94 ± 1.22%** | **85.68 ± 1.18%** | **0.35M** | ✅ Baseline hợp lệ |

### Current baseline comparison

- ResNet50 hiện có hiệu năng trung bình cao nhất.
- DenseNet121 giảm mạnh số tham số nhưng hiệu năng chỉ thấp hơn khoảng 1 điểm %.
- ViT Scratch nhẹ nhất, Test variance thấp, nhưng Accuracy/Macro-F1 thấp hơn hai CNN baseline và Validation variance lớn hơn đáng kể.
- `anthracnose` tiếp tục là class khó nhất trong cả ba baseline.

➡️ Xem dataset, bảng benchmark và phân tích chi tiết tại [`current_dataset/README.md`](./current_dataset/README.md).

## YOLO26 Development Archive

Toàn bộ các take YOLO26 hiện tại được quản lý tại [`../failure/YOLO26/README.md`](../failure/YOLO26/README.md). Các take này vẫn có giá trị cho Failure Analysis / Discussion và nghiên cứu annotation strategy, nhưng **chưa được xem là final detector**.

## Quy ước quản lý experiment

- `current_dataset/`: chỉ lưu các experiment dùng dataset hiện tại và được phép đưa vào bảng so sánh chính thức của khóa luận.
- `archive_legacy_dataset/`: lưu các experiment đã chạy bằng dataset cũ. Các kết quả này **không dùng để so sánh trực tiếp** với experiment mới.
- `failure/`: lưu các thử nghiệm chưa đạt final, data iteration, annotation-policy iteration hoặc các cấu hình chưa đủ điều kiện chốt.
- Mỗi experiment mới dùng ID duy nhất theo format `EXP-{MODEL}-{MODE}-{NO}` và nên lưu config, metrics, figures, predictions hoặc file tổng hợp tương đương.
- Không ghi đè experiment cũ; mọi thay đổi cấu hình đáng kể phải tạo experiment mới.
- Kết quả nhiều seed được báo cáo theo **Mean ± sample Standard Deviation (ddof=1)**.
- Test Set không dùng để tuning hyperparameter hoặc chọn seed.

## Dataset transition

Các experiment dùng dataset cũ được chuyển vào archive và không so sánh trực tiếp với các baseline hiện tại.
