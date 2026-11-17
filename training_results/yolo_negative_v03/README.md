# YOLO Negative Images V03 — so sánh YOLOv8s / YOLO26s

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../README.md) · Danh mục 95 README · Tiến độ · [Protocol](../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


Thí nghiệm **Validation-only** sử dụng **NEG V03**, Train **3497 ảnh (621 ảnh âm)**, Validation **951 ảnh (58 ảnh âm)** và Test **412 ảnh được giữ khóa**. Train 3 lớp Anthracnose, Leaf Miner, Red Rust; Healthy/Not Cashew là background 0 box.

- Dataset fingerprint: `1acc0edcf7f736f9754e4977b61133a7ec5582b44d8cf3eef27208f49a202add`.
- Cả hai notebook huấn luyện: `cls_pw=0.25`, seed **42**, `imgsz=640`, batch **8**, AdamW, Mosaic **0.4**, 80 epochs tối đa, patience 20.
- Phiên bản Ultralytics train **8.4.198**. Kiểm tra FP đánh giá đồng nhất **8.4.197** với cấu hình riêng.

## Chỉ số gốc từ notebook (Validation)

| Model | Precision | Recall | mAP50 | mAP50-95 | Anthracnose AP50-95 | Leaf Miner AP50-95 | Red Rust AP50-95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| [YOLOv8s](experiments/EXP-Y8S-NEG-V03-001-CLASS-WEIGHTED/README.md) | 66.52% | 65.80% | 70.15% | **38.00%** | 30.97% | 47.56% | 35.47% |
| [YOLO26s](experiments/EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED/README.md) | 68.84% | 66.69% | 71.63% | **39.81%** | 31.92% | 51.49% | 36.03% |

YOLO26s dẫn đầu 4 chỉ số macro và AP50-95 của cả 3 bệnh. YOLOv8s cao hơn về Recall Leaf Miner (77.33% vs 73.42%) và Recall Red Rust (70.57% vs 64.30%). Một seed, chênh lệch chưa đủ để khẳng định ưu thế ổn định.

## Đánh giá False Positive — đánh giá lại trên cùng 951 ảnh

Dùng [báo cáo Negative Image Evaluation V03](negative_image_evaluation/README.md) làm **nguồn so sánh chính thức** giữa các checkpoint, không trộn mAP gốc trong notebook.

- YOLOv8s NEG V03: **13/58 (22.41%)** ảnh âm báo bệnh nhầm; **14 false boxes**.
- YOLO26s NEG V03: **3/58 (5.17%)** ảnh âm báo bệnh nhầm; **3 false boxes**.
- So YOLO26s NEG V02, ảnh báo sai **3/58 → 3/58** khi tăng negative Train **276 → 621**, chưa có cải thiện False Alarm Rate.
- mAP50-95 khi re-check trên cùng Validation: YOLOv8s NEG V03 **38.07%** và YOLO26s NEG V03 **39.82%**.

## Kết luận và ranh giới

YOLO26s NEG V03 là ứng viên ưu tiên về AP và False Positive trên Validation. Cần kiểm chứng nhiều seed, lỗi annotation, ảnh cùng lá/cây/phiên và tập dữ liệu độc lập. **Không chọn hyperparameter theo Test**.

## Nội dung và provenance

- [Validation overview nguyên giá trị](validation_overview.csv) · [Dataset / config](dataset_statistics_summary.json)
- [Kết quả false positive trên 6 checkpoint](negative_image_evaluation/README.md)
- [Thông tin checksum 3 archive nguồn](source_archive_provenance.json)
- [Importer để đồng bộ 128 file analysis gốc](import_v03_archives.py)
- [Nhánh NEG V02](../yolo_negative_v02/README.md) · [Class-weight baseline](../yolo_leaf_miner_class_weight/README.md)

**Trạng thái đồng bộ:** GitHub hiện chứa **metadata, báo cáo, CSV tổng hợp và mã importer**. Toàn bộ 128 file PNG/JPG/CSV/JSON nguồn (37 + 39 + 52) được chuẩn bị trong bộ ZIP để nhập; chưa được push nhị phân trong lần cập nhật tóm tắt. Không commit `FULL.zip`, `best.pt` hoặc dataset dung lượng lớn.
