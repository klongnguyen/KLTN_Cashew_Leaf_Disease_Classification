# Current Dataset Experiments

Thư mục này dành riêng cho các kết quả huấn luyện sử dụng **Master Dataset hiện tại** sau khi dataset được cập nhật.

## Trạng thái

Đã có baseline chính thức đầu tiên trên dataset hiện tại.

## Experiments

| Experiment | Model | Mode | Seeds | Test Accuracy | Macro F1 | Status |
|---|---|---|---:|---:|---:|---|
| `EXP-RESNET50-SCRATCH-5SEEDS-002` | ResNet50 | Scratch | 5 | **90.10 ± 1.82%** | **90.12 ± 1.81%** | Baseline hợp lệ |

## Quy tắc lưu experiment

Mỗi experiment nên có cấu trúc tối thiểu:

```text
EXP-{MODEL}-{MODE}-{NO}/
├── README.md
├── experiment_config.json
├── aggregate/
└── seed_*/ hoặc file tổng hợp per-seed
```

Chỉ các experiment trong thư mục này mới được xem là ứng viên cho bảng so sánh chính thức của dataset hiện tại, trừ khi có ghi chú khác.

Không sử dụng kết quả trong `../archive_legacy_dataset/` để so sánh trực tiếp với các experiment mới vì chúng được huấn luyện trên dataset cũ.

## Lưu ý dung lượng

Các checkpoint `.weights.h5` dung lượng lớn không commit trực tiếp vào GitHub. Repo ưu tiên lưu config, metrics và các bảng tổng hợp để bảo đảm khả năng kiểm tra experiment mà không làm repository quá nặng.
