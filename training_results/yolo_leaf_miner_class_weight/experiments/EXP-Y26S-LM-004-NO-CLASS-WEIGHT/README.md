# EXP-Y26S-LM-004-NO-CLASS-WEIGHT

- Mô hình: `yolo26s.pt`; class weighting: `cls_pw=0`; seed: **42**.
- Dataset YOLO Master Final leakage fixed (Train **2.876**, Validation **893**, Test khóa **412**); fingerprint: `9cd75fd917a0edbce88e43d535c823ae464e348468d9a5e5ef2c98cf328349d7`.
- **Validation:** Precision 67.90%, Recall 67.74%, mAP50 71.57%, mAP50-95 39.98%.
- **Leaf Miner:** P 71.63%, R 72.91%, AP50 78.03%, AP50-95 51.99%.
- Epoch chạy 74; epoch tốt nhất (theo lịch sử) 54; thời gian huấn luyện ~18.16 phút.
- Cấu hình chung: 640px, batch 8, AdamW LR=0.001, tối đa 80 epochs, patience=20, mosaic=0.4, seed 42.
- Thống kê Train box: Anthracnose 34.887, Leaf Miner 2.076, Red Rust 73.072.
- Nguồn tải lên hội thoại: `EXP-Y26S-LM-004-NO-CLASS-WEIGHT-20261101T124909Z-1-001.zip` (SHA-256: `3e317ad67a95c9d3c593000c437490ba6aa841df56da13e2a0d0e028079a0cc8`).

**Phạm vi bản ghi GitHub hiện tại:** thư mục này chứa README, thống kê tái lập, bảng kết quả cấu trúc; chưa chứa toàn bộ PNG, hình dự đoán Validation, `results.csv` theo epoch và các file phân tích lỗi chi tiết từ archive. Tên archive và SHA-256 được ghi lại trong `source_provenance.csv`. Không nhầm tóm tắt này với bản sao toàn bộ ANALYSIS.zip.

**Giới hạn:** chỉ một seed, đánh giá trên Validation, chưa dùng Test để chọn mô hình. Các thống kê FP/FN từ phân tích lỗi riêng không thay thế official detection metrics.

[Xem bảng so sánh 4 thí nghiệm](../../README.md).
