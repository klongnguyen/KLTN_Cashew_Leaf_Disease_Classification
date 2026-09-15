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
| [`EXP-DENSENET121-SCRATCH-5SEEDS-002`](./current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-002/) | DenseNet121 | Scratch | 5 | **89.00 ± 3.27%** | **89.22 ± 2.92%** | **7.57M** | ✅ Baseline hợp lệ |

### DenseNet121 — 5-seed training curves

<p align="center">
  <img src="./current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-002/figures/training_curves_5seeds_panel.svg" width="100%" alt="DenseNet121 5-seed training curves">
</p>

ResNet50 hiện đạt hiệu năng trung bình và độ ổn định tốt hơn, trong khi DenseNet121 có lợi thế rõ về số tham số. Các kết luận chính thức luôn dựa trên **Mean ± Std của cả 5 seeds**.

➡️ Xem dataset, bảng benchmark và hình trực quan chi tiết tại [`current_dataset/README.md`](./current_dataset/README.md).

## YOLO26 Development Archive

Toàn bộ các take YOLO26 hiện tại (`EXP-Y26S-SMALL-001` → `EXP-Y26S-SMALL-006`) được quản lý tại:

[`../failure/YOLO26/README.md`](../failure/YOLO26/README.md)

Các take này vẫn có giá trị cho Failure Analysis / Discussion và nghiên cứu annotation strategy, nhưng **chưa được xem là final detector**.

## Quy ước quản lý experiment

- `current_dataset/`: chỉ lưu các experiment dùng dataset hiện tại và được phép đưa vào bảng so sánh chính thức của khóa luận.
- `archive_legacy_dataset/`: lưu các experiment đã chạy bằng dataset cũ. Các kết quả này **không dùng để so sánh trực tiếp** với experiment mới.
- `failure/`: lưu các thử nghiệm chưa đạt final, data iteration, annotation-policy iteration hoặc các cấu hình chưa đủ điều kiện chốt.
- Mỗi experiment mới dùng ID duy nhất theo format `EXP-{MODEL}-{MODE}-{NO}` và nên lưu config, metrics, figures, predictions hoặc file tổng hợp tương đương.
- Không ghi đè experiment cũ; mọi thay đổi cấu hình đáng kể phải tạo experiment mới.
- Kết quả nhiều seed được báo cáo theo **Mean ± sample Standard Deviation (ddof=1)**.
- Test Set không dùng để tuning hyperparameter hoặc chọn seed.

## Dataset transition

Dataset hiện tại đã thay đổi so với dataset dùng cho các experiment cũ. Vì vậy hai experiment sau đã được chuyển vào archive:

- `EXP-DENSENET121-SCRATCH-001`
- `EXP-VIT-SCRATCH-001`

Các lần YOLO26 được quản lý riêng trong Failure Experiment Archive cho đến khi có detector cuối được xác nhận.
