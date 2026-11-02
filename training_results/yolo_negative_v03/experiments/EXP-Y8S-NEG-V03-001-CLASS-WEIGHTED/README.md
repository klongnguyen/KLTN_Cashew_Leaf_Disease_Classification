# EXP-Y8S-NEG-V03-001-CLASS-WEIGHTED

| Thuộc tính | Giá trị |
|---|---|
| Model | `yolov8s.pt` |
| Dataset | `CASHEW_YOLO_MASTER_FINAL_NEG_V03` |
| Train / Validation / Test (locked) | **3497 / 951 / 412** |
| Negative images Train / Val | **621 / 58** |
| Seed | **42** |
| Class Weighting | **cls_pw=0.25**, Leaf Miner weight **1.57524955** |
| Img size / batch / optimizer | **640 / 8 / AdamW** |
| Epochs thực chạy | **50** |
| Training time | **9.10 phút** |
| Precision | **66.52%** |
| Recall | **65.80%** |
| mAP50 | **70.15%** |
| mAP50-95 | **38.00%** |
| Leaf Miner AP50-95 | **47.56%** |

## Các nguồn số liệu

- [Chỉ số per-class](per_class_metrics.csv), [metadata và metrics tổng hợp](metrics_summary.json)
- [Đánh giá đồng nhất sáu checkpoint và False Positive](../../negative_image_evaluation/README.md)
- [So sánh V03 YOLOv8s vs YOLO26s](../../README.md)

Archive gốc đã đối chiếu: `EXP-Y8S-NEG-V03-001_ANALYSIS.zip` — SHA256 `c9e1375b1456dd70c16d8a6b9cb908724744cc0f871c9062b7036ea6a5b1b29a`; inner ANALYSIS ZIP SHA256 `c9e1375b1456dd70c16d8a6b9cb908724744cc0f871c9062b7036ea6a5b1b29a`.

**Trạng thái artifact:** File ZIP gốc chứa đầy đủ biểu đồ, `results.csv`, phân tích lỗi, hình dự đoán và file preflight. File nguồn nhị phân chưa nằm trong GitHub; bản Git hiện chứa báo cáo, CSV/JSON tổng hợp và importer xác minh SHA256. [Hướng dẫn đồng bộ](../../import_v03_archives.py).

**Giới hạn:** Validation-only và chỉ 1 seed. Test 412 ảnh đã khóa, không dùng Test để chọn mô hình.
