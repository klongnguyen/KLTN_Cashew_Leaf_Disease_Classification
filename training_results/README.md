# Training Results

Thư mục này quản lý toàn bộ kết quả huấn luyện của dự án theo **phiên bản dataset** để tránh so sánh sai giữa các experiment dùng dữ liệu khác nhau.

## Cấu trúc

```text
training_results/
├── current_dataset/          # Experiment dùng Master Dataset hiện tại
└── archive_legacy_dataset/   # Experiment dùng dataset cũ, chỉ tham khảo
```

## Current Dataset — Quick Snapshot

Master Dataset hiện tại có **7,213 ảnh / 5 lớp**, được chia cố định thành **5,049 Train / 1,433 Validation / 731 Test**. Test Set được khóa và chỉ dùng cho đánh giá cuối cùng.

| Experiment | Model | Mode | Seeds | Test Accuracy | Macro F1 | Status |
|---|---|---|---:|---:|---:|---|
| [`EXP-RESNET50-SCRATCH-5SEEDS-002`](./current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-002/) | ResNet50 | Scratch | 5 | **90.10 ± 1.82%** | **90.12 ± 1.81%** | ✅ Baseline hợp lệ |

### Representative training curve — Seed 42

> Seed 42 được dùng làm hình minh họa vì đây là seed cố định đầu tiên trong protocol; kết luận chính thức luôn dựa trên **Mean ± Std của cả 5 seeds**, không chọn seed tốt nhất theo Test.

<p align="center">
  <img src="./current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-002/5_seed_resnet50_v02/5_seed_resnet50_v02/42/accuracy_curve.png" width="760" alt="ResNet50 Seed 42 Accuracy Curve">
</p>

➡️ Xem dataset, bảng benchmark và hình trực quan chi tiết tại [`current_dataset/README.md`](./current_dataset/README.md).

## Quy ước quản lý experiment

- `current_dataset/`: chỉ lưu các experiment được huấn luyện bằng dataset hiện tại và được phép đưa vào bảng so sánh chính thức của khóa luận.
- `archive_legacy_dataset/`: lưu các experiment đã chạy bằng dataset cũ. Các kết quả này **không dùng để so sánh trực tiếp** với experiment mới.
- Mỗi experiment mới dùng ID duy nhất theo format `EXP-{MODEL}-{MODE}-{NO}` và nên lưu config, metrics, figures, predictions hoặc file tổng hợp tương đương.
- Không ghi đè experiment cũ; mọi thay đổi cấu hình đáng kể phải tạo experiment mới.
- Kết quả nhiều seed được báo cáo theo **Mean ± sample Standard Deviation (ddof=1)**.
- Test Set không dùng để tuning hyperparameter hoặc chọn seed.

## Dataset transition

Dataset hiện tại đã thay đổi so với dataset dùng cho các experiment cũ. Vì vậy hai experiment sau đã được chuyển vào archive:

- `EXP-DENSENET121-SCRATCH-001`
- `EXP-VIT-SCRATCH-001`

> Thư mục `failure/` ở root repo vẫn được giữ riêng làm Failure Experiment Archive. Các kết quả trong đó không được xem là benchmark chính thức của dataset hiện tại trừ khi README của experiment ghi rõ điều ngược lại.
