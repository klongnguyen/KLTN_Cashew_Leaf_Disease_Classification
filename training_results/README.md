# Training Results

Kết quả thí nghiệm dùng trong khóa luận. Tệp trọng số, ZIP và ảnh gốc không nằm trong Git; thư mục chỉ chứa README, CSV, JSON và bảng kết quả.

| Thư mục | Nội dung |
|---|---|
| `current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4/` | ResNet50 học chuyển giao, 5 seed: Validation / Test / ảnh thực địa, dự đoán từng ảnh |
| `current_dataset/EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-256px-1192fd08/` | DeiT-S (LLRD), 5 seed: cùng cấu trúc trên |
| `current_dataset/EXP-COMBINED-3CLASSIFIERS-YOLO26S-V04-037/` | So sánh ba mô hình phân loại khi kết hợp YOLO26s (F0, F3a, F3b, F3), `thesis_tables/` |
| `yolo_leaf_miner_class_weight/` | Trọng số lớp cho YOLOv8s và YOLO26s (Validation) |
| `yolo_negative_v02/`, `yolo_negative_v03/` | Bổ sung ảnh âm và đánh giá báo bệnh nhầm trên ảnh âm (Validation) |
| `yolo_negative_v04/EXP-Y26S-NEG-V04-001/` | YOLO26s phiên bản cuối, dùng trong mô hình kết hợp và ứng dụng web |

Mô hình kết hợp triển khai (EXP-035) nằm ở thư mục gốc: `../EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/`.
