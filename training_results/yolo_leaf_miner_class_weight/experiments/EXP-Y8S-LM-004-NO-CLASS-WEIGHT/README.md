# EXP-Y8S-LM-004-NO-CLASS-WEIGHT

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../../../README.md) · [Danh mục 95 README](../../../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


- Mô hình: `yolov8s.pt`; class weighting: `cls_pw=0`; seed: **42**.
- Dataset YOLO Master Final leakage fixed (Train **2.876**, Validation **893**, Test khóa **412**); fingerprint: `9cd75fd917a0edbce88e43d535c823ae464e348468d9a5e5ef2c98cf328349d7`.
- **Validation:** Precision 70.06%, Recall 66.22%, mAP50 70.88%, mAP50-95 38.27%.
- **Leaf Miner:** P 74.34%, R 71.65%, AP50 76.39%, AP50-95 50.89%.
- Epoch chạy 61; epoch tốt nhất (theo lịch sử) 41; thời gian huấn luyện ~9.44 phút.
- Cấu hình chung: 640px, batch 8, AdamW LR=0.001, tối đa 80 epochs, patience=20, mosaic=0.4, seed 42.
- Thống kê Train box: Anthracnose 34.887, Leaf Miner 2.076, Red Rust 73.072.
- Nguồn tải lên hội thoại: `EXP-Y8S-LM-004-NO-CLASS-WEIGHT-20261101T124902Z-1-001.zip` (SHA-256: `506586dcef2f1826fcc86dea4952f8ff1f82827bcf9b74651d002bd7517962ab`).

**Phạm vi bản ghi GitHub hiện tại:** thư mục này chứa README, thống kê tái lập, bảng kết quả cấu trúc; chưa chứa toàn bộ PNG, hình dự đoán Validation, `results.csv` theo epoch và các file phân tích lỗi chi tiết từ archive. Tên archive và SHA-256 được ghi lại trong `source_provenance.csv`. Không nhầm tóm tắt này với bản sao toàn bộ ANALYSIS.zip.

**Giới hạn:** chỉ một seed, đánh giá trên Validation, chưa dùng Test để chọn mô hình. Các thống kê FP/FN từ phân tích lỗi riêng không thay thế official detection metrics.

[Xem bảng so sánh 4 thí nghiệm](../../README.md).
