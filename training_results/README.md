# Training Results

Thư mục này quản lý toàn bộ kết quả huấn luyện của dự án theo **phiên bản dataset** để tránh nhầm lẫn giữa các experiment cũ và các experiment được thực hiện trên Master Dataset mới.

## Cấu trúc

```text
training_results/
├── current_dataset/          # Kết quả huấn luyện từ dataset hiện tại
└── archive_legacy_dataset/   # Kết quả từ dataset cũ, chỉ dùng tham khảo
```

## Quy ước

- `current_dataset/`: chỉ lưu các experiment được huấn luyện bằng dataset hiện tại và được phép dùng trong bảng so sánh chính thức của khóa luận.
- `archive_legacy_dataset/`: lưu các experiment đã chạy bằng dataset cũ. Các kết quả này **không dùng để so sánh trực tiếp** với experiment mới và chỉ giữ lại để tham khảo lịch sử.
- Mỗi experiment mới nên dùng ID duy nhất theo format `EXP-{MODEL}-{MODE}-{NO}` và lưu đầy đủ config, metrics, figures, predictions và README.
- Không ghi đè experiment cũ.

## Dataset transition

Dataset hiện tại đã được thay đổi so với dataset dùng cho các experiment cũ. Vì vậy hai experiment cũ sau đây đã được chuyển vào archive:

- `EXP-DENSENET121-SCRATCH-001`
- `EXP-VIT-SCRATCH-001`

Các kết quả huấn luyện tiếp theo phải được lưu trong `current_dataset/`.

> Thư mục `failure/` ở root repo vẫn được giữ riêng làm Failure Experiment Archive cho các lần thử không đạt yêu cầu. Các kết quả trong đó không được xem là benchmark chính thức của dataset hiện tại trừ khi README của experiment ghi rõ điều ngược lại.
