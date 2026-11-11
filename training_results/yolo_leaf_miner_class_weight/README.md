# YOLOv8s / YOLO26s — tác động Class-Weighted Loss trên Cashew YOLO Master Final

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../README.md) · Danh mục 95 README · Tiến độ · [Protocol](../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


## Phạm vi

Bốn thí nghiệm trên **Validation-only** của cùng dataset fingerprint `9cd75fd917a0edbce88e43d535c823ae464e348468d9a5e5ef2c98cf328349d7`: Train **2.876**, Validation **893**, Test **412** (chưa sử dụng để chọn mô hình). So sánh `cls_pw=0` và `cls_pw=0.25` cho từng kiến trúc, seed **42**, 640px, batch 8, AdamW, max 80 epochs.

| Kiến trúc | cls_pw | P | R | mAP50 | mAP50-95 | Leaf Miner AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| YOLOv8s | 0.25 | 67.01% | 65.10% | 69.43% | **37.68%** | **48.53%** |
| YOLO26s | 0.25 | 69.44% | 65.97% | 71.20% | **40.30%** | **53.07%** |
| YOLOv8s | 0 | 70.06% | 66.22% | 70.88% | **38.27%** | **50.89%** |
| YOLO26s | 0 | 67.90% | 67.74% | 71.57% | **39.98%** | **51.99%** |

## Kết quả và giới hạn

- **YOLOv8s:** weighting làm mAP50-95 giảm **0,59 điểm phần trăm**; Leaf Miner AP50-95 giảm **2,36 điểm**. Baseline không weighting tốt hơn trên các chỉ số này.
- **YOLO26s:** weighting làm mAP50-95 tăng **0,32 điểm** và Leaf Miner AP50-95 tăng **1,08 điểm**, nhưng Leaf Miner Recall giảm **2,14 điểm**.
- Leaf Miner có ít box Train (**2.076**) so với Anthracnose (**34.887**) và Red Rust (**73.072**), nhưng AP50-95 Leaf Miner vẫn cao hơn hai lớp kia. Không thể suy luận độ khó chỉ từ số box.
- Chưa có lặp lại đa seed để khẳng định khác biệt ổn định; chỉ dùng Validation để so sánh. Không mô tả đây là test accuracy.
- **Không nhầm** class-weighted loss với weighted image sampling: EXP-004 ở đây là **NO-CLASS-WEIGHT**, không phải Weighted Sampling V004.

## Dữ liệu trong GitHub

Có cấu trúc so sánh đầy đủ bốn case dưới `experiments/`, mỗi case gồm **README**, **metrics_summary.json**, **per_class_metrics.csv** và **class_weights_summary.json**. Tệp so sánh:
- [Validation overview](validation_overview.csv)
- [Per-class metrics](validation_by_class.csv)
- [Ảnh hưởng weighting](class_weight_effect.csv)
- [Nguồn gốc và SHA-256 các ZIP](source_provenance.csv)

**Trạng thái artifact:** đây là bản **backfill các chỉ số và provenance** từ bốn ANALYSIS archive đã cung cấp, **chưa push toàn bộ archive nhị phân**, learning curves PNG, confusion matrix PNG, bộ 10–20 ảnh dự đoán, `results.csv` đầy đủ từng epoch hay `leaf_miner_failure_analysis.csv` nguyên bản. Các source zip không bị chỉnh sửa. Có thể đồng bộ các thành phần thiếu từ ZIP nguồn trên máy bằng script `import_analysis_archives.py` (nếu xuất hiện trong folder này).

## Các experiment

- [EXP-Y8S-LM-003-CLASS-WEIGHTED](experiments/EXP-Y8S-LM-003-CLASS-WEIGHTED/README.md) — mAP50-95 37.68%, Leaf Miner AP50-95 48.53%.
- [EXP-Y26S-LM-003-CLASS-WEIGHTED](experiments/EXP-Y26S-LM-003-CLASS-WEIGHTED/README.md) — mAP50-95 40.30%, Leaf Miner AP50-95 53.07%.
- [EXP-Y8S-LM-004-NO-CLASS-WEIGHT](experiments/EXP-Y8S-LM-004-NO-CLASS-WEIGHT/README.md) — mAP50-95 38.27%, Leaf Miner AP50-95 50.89%.
- [EXP-Y26S-LM-004-NO-CLASS-WEIGHT](experiments/EXP-Y26S-LM-004-NO-CLASS-WEIGHT/README.md) — mAP50-95 39.98%, Leaf Miner AP50-95 51.99%.

## Quy tắc đánh giá

Tập Test của Cashew dataset đã khóa. Không dùng số Test để lựa chọn `cls_pw`, model hoặc confidence threshold. Khi cần kết luận học thuật, báo cáo giá trị mỗi seed và đánh giá lặp trên cùng split.
