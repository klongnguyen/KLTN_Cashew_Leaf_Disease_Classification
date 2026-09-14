# Current Dataset Experiments

Thư mục này dành riêng cho các kết quả huấn luyện sử dụng **Master Dataset hiện tại** sau khi dataset được cập nhật.

## Trạng thái

Chưa có experiment mới được thêm.

## Quy tắc lưu experiment

Mỗi experiment nên có cấu trúc tối thiểu:

```text
EXP-{MODEL}-{MODE}-{NO}/
├── README.md
├── experiment_config.json
├── checkpoints/
├── logs/
├── metrics/
├── figures/
└── predictions/
```

Chỉ các experiment trong thư mục này mới được xem là ứng viên cho bảng so sánh chính thức của dataset hiện tại, trừ khi có ghi chú khác.

Không sử dụng kết quả trong `../archive_legacy_dataset/` để so sánh trực tiếp với các experiment mới vì chúng được huấn luyện trên dataset cũ.
