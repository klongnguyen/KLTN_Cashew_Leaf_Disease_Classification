# EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../../../README.md) · [Danh mục 95 README](../../../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


| Thuộc tính | Giá trị |
|---|---|
| Model | `yolo26s.pt` |
| Dataset | `CASHEW_YOLO_MASTER_FINAL_NEG_V03` |
| Train / Validation / Test (locked) | **3497 / 951 / 412** |
| Negative images Train / Val | **621 / 58** |
| Seed | **42** |
| Class Weighting | **cls_pw=0.25**, Leaf Miner weight **1.57524955** |
| Img size / batch / optimizer | **640 / 8 / AdamW** |
| Epochs thực chạy | **56** |
| Training time | **16.08 phút** |
| Precision | **68.84%** |
| Recall | **66.69%** |
| mAP50 | **71.63%** |
| mAP50-95 | **39.81%** |
| Leaf Miner AP50-95 | **51.49%** |

## Các nguồn số liệu

- [Chỉ số per-class](per_class_metrics.csv), [metadata và metrics tổng hợp](metrics_summary.json)
- [Đánh giá đồng nhất sáu checkpoint và False Positive](../../negative_image_evaluation/README.md)
- [So sánh V03 YOLOv8s vs YOLO26s](../../README.md)

Archive gốc đã đối chiếu: `EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED-20261102T043914Z-1-001.zip` — SHA256 `d65d59c4b3e410b19fbcb41b09d5620c6d3d4ac6b2c029b9dd03ea796f340c46`; inner ANALYSIS ZIP SHA256 `3e7663a9575fc88f67915cddf3fe32c85c59761e794bca30c0647af7b38abb57`.

**Trạng thái artifact:** File ZIP gốc chứa đầy đủ biểu đồ, `results.csv`, phân tích lỗi, hình dự đoán và file preflight. File nguồn nhị phân chưa nằm trong GitHub; bản Git hiện chứa báo cáo, CSV/JSON tổng hợp và importer xác minh SHA256. [Hướng dẫn đồng bộ](../../import_v03_archives.py).

**Giới hạn:** Validation-only và chỉ 1 seed. Test 412 ảnh đã khóa, không dùng Test để chọn mô hình.
