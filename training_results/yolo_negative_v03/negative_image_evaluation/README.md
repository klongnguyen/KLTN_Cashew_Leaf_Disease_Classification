# Negative Image Evaluation — 6 checkpoint, V003 / NEG V02 / NEG V03

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../../README.md) · [Danh mục 95 README](../../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


Nguồn: `NEG_V03_NEGATIVE_IMAGE_EVALUATION_ANALYSIS.zip`; SHA256: `91435db7b9fbe9ac408417bc0290d9656c9c08829601401a8899754153c5dcbb`. **Validation-only**, không sử dụng Test. Sáu checkpoint được đánh giá lại trên cùng một tập **951 ảnh**, trong đó **58 ảnh negative**: 38 Healthy và 20 Not Cashew Leaf.

**Evaluation contract:** Ultralytics 8.4.197, `imgsz=640`, `batch=4`, `iou=0.7`, `half=False`, `rect=False`; báo bệnh nhầm nếu ≥1 box trên ảnh âm với `conf >= 0.25`. Tính P/R/mAP từ `model.val(conf=0.001)`. Validation SHA256: `9a502be84e2d619a6d75bb76b979c7d40eabc8d6688124d05d013965555ee78f`. Các metric re-check không gộp với metric gốc trong notebook (train Ultralytics 8.4.198).

| Kiến trúc | Checkpoint | Healthy sai | Not Cashew sai | Tổng ảnh sai | Box sai | P | R | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| yolov8s | v003 | 8/38 | 5/20 | **13/58 (22.41%)** | 19 | 66.69% | 64.09% | 68.98% | 37.38% |
| yolov8s | neg_v02 | 8/38 | 6/20 | **14/58 (24.14%)** | 19 | 70.34% | 64.31% | 69.77% | 38.47% |
| yolov8s | neg_v03 | 10/38 | 3/20 | **13/58 (22.41%)** | 14 | 66.63% | 65.01% | 69.65% | 38.07% |
| yolo26s | v003 | 6/38 | 9/20 | **15/58 (25.86%)** | 24 | 69.57% | 65.33% | 70.54% | 39.84% |
| yolo26s | neg_v02 | 2/38 | 1/20 | **3/58 (5.17%)** | 3 | 67.76% | 66.54% | 70.73% | 39.71% |
| yolo26s | neg_v03 | 2/38 | 1/20 | **3/58 (5.17%)** | 3 | 67.79% | 67.03% | 71.36% | 39.82% |

- YOLO26s: `V003 → NEG V02` báo sai **15 → 3/58**, giảm tương đối 80%; `NEG V02 → NEG V03` vẫn **3/58**.
- YOLOv8s: `V003 → NEG V02 → NEG V03` là **13 → 14 → 13/58**; thêm ảnh âm chưa chứng minh hiệu quả ổn định.
- Với YOLO26s NEG V03: **hai ảnh Healthy sai Anthracnose**, một ảnh Not Cashew sai Red Rust; **không có false box Leaf Miner** trong 58 ảnh âm tại ngưỡng trên.
- Paired McNemar exact `p=0.004180908203125` cho YOLO26s `V003 → NEG V03` và `p=1.0` cho `NEG V02 → NEG V03`. Chỉ có 58 ảnh và một seed, kiểm định mang tính thăm dò.
- Phương pháp NMS có thể khác giữa YOLOv8 và kiến trúc YOLO26 end-to-end. Inference contract giữ thống nhất về thông số, nhưng ảnh hưởng cụ thể của ngưỡng IoU không nhất thiết đồng nhất.

## CSV lưu trên GitHub

- [Đánh giá sáu checkpoint trên cùng Validation](rechecked_validation_metrics_6.csv)
- [False Alarm và số box sai](false_alarm_6.csv)
- [Per-class hai checkpoint V03 (đánh giá lại)](per_class_v03_recheck.csv)
- [Chuyển trạng thái từng cặp theo ảnh](paired_transitions.csv)
- [False box theo class](negative_false_boxes_by_class.csv)
- [Ba ảnh sai YOLO26s NEG V03](remaining_false_alarms_yolo26s_v03.csv)
- [Protocol](evaluation_protocol.json)

**Giới hạn artifact:** Báo cáo và thống kê trong GitHub là backfill trung thực từ ZIP. Tệp gốc `comparison_paired_images.csv`, `negative_per_image.csv`, `negative_boxes.csv`, từng `report.json` nằm trong ZIP nguồn; chạy [import_v03_archives.py](../import_v03_archives.py) để đồng bộ đầy đủ. Không giả vờ những file đó đã được push.
