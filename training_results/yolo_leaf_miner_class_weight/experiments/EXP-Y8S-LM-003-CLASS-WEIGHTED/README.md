# EXP-Y8S-LM-003-CLASS-WEIGHTED

- Mô hình: `yolov8s.pt`; class weighting: `cls_pw=0.25`; seed: **42**.
- Dataset YOLO Master Final leakage fixed (Train **2.876**, Validation **893**, Test khóa **412**); fingerprint: `9cd75fd917a0edbce88e43d535c823ae464e348468d9a5e5ef2c98cf328349d7`.
- **Validation:** Precision 67.01%, Recall 65.10%, mAP50 69.43%, mAP50-95 37.68%.
- **Leaf Miner:** P 65.48%, R 71.46%, AP50 73.44%, AP50-95 48.53%.
- Epoch chạy 48; epoch tốt nhất (theo lịch sử) 28; thời gian huấn luyện ~7.48 phút.
- Cấu hình chung: 640px, batch 8, AdamW LR=0.001, tối đa 80 epochs, patience=20, mosaic=0.4, seed 42.
- Thống kê Train box: Anthracnose 34.887, Leaf Miner 2.076, Red Rust 73.072.
- Nguồn tải lên hội thoại: `EXP-Y8S-LM-003_ANALYSIS(1).zip` (SHA-256: `5edc9cff6a76a7c50db23ee752749cf14b355785cdffdfc4a10b8209d0059497`).

**Phạm vi bản ghi GitHub hiện tại:** thư mục này chứa README, thống kê tái lập, bảng kết quả cấu trúc; chưa chứa toàn bộ PNG, hình dự đoán Validation, `results.csv` theo epoch và các file phân tích lỗi chi tiết từ archive. Tên archive và SHA-256 được ghi lại trong `source_provenance.csv`. Không nhầm tóm tắt này với bản sao toàn bộ ANALYSIS.zip.

**Giới hạn:** chỉ một seed, đánh giá trên Validation, chưa dùng Test để chọn mô hình. Các thống kê FP/FN từ phân tích lỗi riêng không thay thế official detection metrics.

[Xem bảng so sánh 4 thí nghiệm](../../README.md).
