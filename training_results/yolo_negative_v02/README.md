# YOLO Negative Image V02 — YOLOv8s và YOLO26s

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../README.md) · Danh mục 95 README · Tiến độ · [Protocol](../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


Hai thí nghiệm `cls_pw=0.25`, seed 42, ảnh 640px, **cùng snapshot NEG V02**.

| Mô hình | Precision | Recall | mAP50 | mAP50-95 | Leaf Miner AP50-95 |
|---|---:|---:|---:|---:|---:|
| [YOLOv8s](experiments/EXP-Y8S-NEG-V02-001-CLASS-WEIGHTED/README.md) | 69.00% | 64.74% | 69.90% | 38.67% | 49.40% |
| [YOLO26s](experiments/EXP-Y26S-NEG-V02-001-CLASS-WEIGHTED/README.md) | 68.95% | 66.16% | 71.31% | 40.08% | 52.85% |

Dataset: Train **3152** (276 negative), Validation **951** (58 negative), Test **412** (0 negative, khóa). Chỉ ba classes bệnh; Healthy & Not Cashew là background 0 box.

## Đánh giá false positive trên cùng Validation

[Negative Image Evaluation (58 ảnh âm)](negative_image_evaluation/README.md) xác nhận YOLO26s giảm báo bệnh nhầm **15/58 → 3/58 ảnh**, trong khi YOLOv8s tăng **13/58 → 14/58**. Cần phân biệt:
- **mAP** từ đánh giá Validation toàn bộ (`conf=0.001`).
- **False alarm rate** trên ảnh âm (`conf>=0.25`).

**Giới hạn nghiên cứu:** một seed, tập ảnh âm nhỏ; NEG V02 có bớt 2 annotation Train so với V003, chưa phải ablation độc lập tuyệt đối; Test giữ khóa. Không mô tả bằng chứng Validation là Test.

## Files

- [Tổng hợp chỉ số hai mô hình](validation_overview.csv)
- [Dataset / weights config](dataset_statistics_summary.json)
- [Báo cáo FP](negative_image_evaluation/README.md)
- [Checksum của 7 ZIP đã gửi](all_sources_sha256.csv)
- [Bốn thí nghiệm class weighting trước đó](../yolo_leaf_miner_class_weight/README.md)

**Artifact completeness:** Hiện thư mục GitHub chứa README, bảng CSV, JSON tổng hợp và nguồn gốc 7 archive. Chưa chứa tất cả đồ thị PNG/JPG hoặc từng file CSV/JSON gốc. Dùng `sync_all_yolo_analysis.py` để nhập 7 archive nguyên bản (không checkpoint).
