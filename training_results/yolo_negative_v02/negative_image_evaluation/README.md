# Negative Image Evaluation — cùng tập Validation NEG V02

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../../README.md) · [Danh mục 95 README](../../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


Phân tích bốn checkpoint `best.pt` trên cùng **951 ảnh Validation**, gồm **893 ảnh có GT bệnh**, **58 ảnh âm (Healthy 38, Not Cashew Leaf 20)**. Không sử dụng Test để tuning.

- Environment: Ultralytics **8.4.197**, input 640px, batch 4, `iou=0.7`, FP16 disabled.
- False alarm: có ít nhất **một box dự đoán trên ảnh âm** tại `confidence >= 0.25`.
- Official P/R/AP: `conf=0.001`, `iou=0.7` và cùng Validation để đảm bảo so sánh đồng nhất.
- Fingerprint Validation: `9a502be84e2d619a6d75bb76b979c7d40eabc8d6688124d05d013965555ee78f`.

| Model | Checkpoint | Healthy false alarm | Not Cashew false alarm | Total false alarm | False boxes |
|---|---|---:|---:|---:|---:|
| YOLOv8s | V003 | 8/38 | 5/20 | **13/58 (22.41%)** | 19 |
| YOLOv8s | NEG V02 | 8/38 | 6/20 | **14/58 (24.14%)** | 19 |
| YOLO26s | V003 | 6/38 | 9/20 | **15/58 (25.86%)** | 24 |
| YOLO26s | NEG V02 | 2/38 | 1/20 | **3/58 (5.17%)** | 3 |

**Kết luận quan sát:** YOLO26s giảm 12/58 ảnh âm báo nhầm (từ 15 xuống 3), giảm tương đối **80%**, tổng box sai 24 → 3. Với YOLOv8s, tỷ lệ báo nhầm tăng từ 13/58 lên 14/58.

Paired exact McNemar `p=0.00418` cho YOLO26s, một kiểm định **thăm dò** trên chỉ 58 ảnh, một seed. Không diễn giải như bằng chứng tổng quát hóa cho nhiều vườn/dataset chưa thấy.

Các file trong folder này là bảng số liệu tóm tắt (không phải toàn bộ CSV gốc). Archive `NEGATIVE_IMAGE_EVALUATION_ANALYSIS.zip` còn chứa danh sách từng ảnh/từng box, JSON và report theo từng mô hình; đồng bộ đầy đủ bằng `sync_all_yolo_analysis.py`.

**Nguồn:** `NEGATIVE_IMAGE_EVALUATION_ANALYSIS.zip`, SHA256 `6489e0c41bc19b804302f7c12edcad38e6d069670cae95b759c73119ec5cbd13`.
