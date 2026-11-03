# EXP-Y26S-LM-003-CLASS-WEIGHTED

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** YOLO Validation-only (không đánh giá trên Test khóa). Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../../../README.md) · [Danh mục 95 README](../../../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


- Mô hình: `yolo26s.pt`; class weighting: `cls_pw=0.25`; seed: **42**.
- Dataset YOLO Master Final leakage fixed (Train **2.876**, Validation **893**, Test khóa **412**); fingerprint: `9cd75fd917a0edbce88e43d535c823ae464e348468d9a5e5ef2c98cf328349d7`.
- **Validation:** Precision 69.44%, Recall 65.97%, mAP50 71.20%, mAP50-95 40.30%.
- **Leaf Miner:** P 75.19%, R 70.77%, AP50 78.05%, AP50-95 53.07%.
- Epoch chạy 74; epoch tốt nhất (theo lịch sử) 54; thời gian huấn luyện ~18.10 phút.
- Cấu hình chung: 640px, batch 8, AdamW LR=0.001, tối đa 80 epochs, patience=20, mosaic=0.4, seed 42.
- Thống kê Train box: Anthracnose 34.887, Leaf Miner 2.076, Red Rust 73.072.
- Nguồn tải lên hội thoại: `EXP-Y26S-LM-003_ANALYSIS(1).zip` (SHA-256: `a987c0c65f059c9f4a5a432256f903dec1cbf63fa1b88cd79e700aa65c3dc131`).

**Phạm vi bản ghi GitHub hiện tại:** thư mục này chứa README, thống kê tái lập, bảng kết quả cấu trúc; chưa chứa toàn bộ PNG, hình dự đoán Validation, `results.csv` theo epoch và các file phân tích lỗi chi tiết từ archive. Tên archive và SHA-256 được ghi lại trong `source_provenance.csv`. Không nhầm tóm tắt này với bản sao toàn bộ ANALYSIS.zip.

**Giới hạn:** chỉ một seed, đánh giá trên Validation, chưa dùng Test để chọn mô hình. Các thống kê FP/FN từ phân tích lỗi riêng không thay thế official detection metrics.

[Xem bảng so sánh 4 thí nghiệm](../../README.md).
