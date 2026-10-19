# Tiến độ huấn luyện và đối chiếu phiên bản (19/10/2026)

Nguồn: cây thư mục GitHub `main` trước cập nhật, các gói ANALYSIS mà người dùng đã gửi, và cấu hình thí nghiệm được ghi trong ZIP. **Validation và Test là hai giao thức khác nhau**; không so sánh các số từ hai tập như cùng một benchmark.

## Tổng hợp trạng thái quan trọng

| Kiến trúc | Experiment | Dataset | Tình trạng | Chỉ số chính |
|---|---|---|---|---|
| DenseNet121 | [EXP-005](./EXP-DENSENET121-SCRATCH-5SEEDS-005/README.md) | V05 | **Final locked Test xong** | Test Acc **91.97 ± 1.11%**, F1 **91.41 ± 1.13%** |
| ResNet50 | [EXP-005](./EXP-RESNET50-SCRATCH-5SEEDS-005/README.md) | V05 | **Final locked Test xong** | Test Acc **89.78 ± 2.50%**, F1 **89.36 ± 2.47%** |
| Compact ViT | [EXP-005](./EXP-VIT-SCRATCH-5SEEDS-005/README.md) | V05 | **Final locked Test xong** | Test Acc **85.04 ± 0.44%**, F1 **83.93 ± 0.37%** |
| DenseNet121 | [EXP-006](./EXP-DENSENET121-SCRATCH-5SEEDS-006/README.md) | V05 | Test xong | Test Acc **88.44 ± 3.14%**, F1 **87.95 ± 3.14%** |
| DenseNet121 | [EXP-009](./EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL/EXP-DENSENET121-SCRATCH-5SEEDS-009/README.md) | V07 | Validation xong; Test pending | Val Acc **94.55 ± 1.11%**, F1 **94.38 ± 1.13%** |
| DenseNet121 | [EXP-011](./EXP-DENSENET121-SCRATCH-5SEEDS-011/README.md) | V07 | Validation xong; **số liệu trùng 009** | Val Acc **94.55 ± 1.11%**, F1 **94.38 ± 1.13%** |
| **DenseNet121** | **[EXP-012](./EXP-DENSENET121-SCRATCH-5SEEDS-012/README.md)** | **V08** | **Validation xong; Test pending** | Val Acc **93.99 ± 1.05%**, F1 **93.81 ± 1.13%** |

Các phiên bản `002–005` của DenseNet121, ResNet50 và ViT đều có README trong thư mục tương ứng. Bản ghi V04 `003` là Test benchmark lịch sử; `004` là validation rerun, **chưa có Test cho chính rerun 004**.

**Tình trạng chưa đủ bằng chứng:** thư mục kết quả chuẩn `EXP-007`, `008`, `010` **không xuất hiện** trong bản kiểm kê GitHub hiện tại; có notebook backup 008 nhưng không thể tự coi đó là hoàn tất training. Không tự điền metrics khi không có kết quả nguồn.

## Những thay đổi quan trọng phải nêu trong từng README

| Chuyển phiên bản | Dataset | Config | Kết luận có căn cứ |
|---|---|---|---|
| V03 → V04 (003) | 7,213 → 6,911 ảnh sau cleaning | Tuỳ kiến trúc; xem README riêng | Không đánh giá cải tiến mô hình bằng so sánh chéo test split |
| V04 EXP-003 → 004 | V04 cố định | Rerun / thay đổi execution strategy từ 2 GPU sang single GPU ở nhiều thí nghiệm | Val thay đổi; không mặc nhiên do kiến trúc |
| V04 → V05 (005) | Cùng 6,911 ảnh, Train/Val fingerprint khác; Test giữ fingerprint | 5 seeds và locked Test benchmark chính thức | DenseNet121 mạnh nhất V05 |
| DenseNet V05 005 → 006 | V05 | EXP-006: batch 32, LR 3e-4, Dropout 0.4 | Kết quả Test 006 thấp hơn 005, các khác biệt config cần đọc đầy đủ |
| DenseNet EXP-006 → 009 | V05 → V07 6,907 ảnh | Batch 16, LR 1e-4, Dropout 0.3 | Val 009 cao nhưng không có Real Val |
| DenseNet 009 → 011 | Cùng V07, cùng manifest SHA | Cấu hình chính trùng | Cả 5 hàng Val metrics **trùng chính xác**; chưa xác minh 2 lần train độc lập |
| DenseNet 011 → 012 | V07 → V08 6,906 ảnh; 35 `real_` chuyển Train→Val; loại 1 Test | Cấu hình chính giữ nguyên | Val 012 giảm nhẹ do thay split; seed 42 real_ đúng 22/35 |

## Tiến độ / Checklist

- [x] V05 official 3-model Final Test, 5 seeds, report Mean ± sample SD.
- [x] Hồ sơ mới EXP-011, EXP-012: README, CSV aggregate Validation, metadata snapshot.
- [x] Chuẩn hóa ID nguồn ZIP `...-01` thành EXP-012; giữ mapping ID gốc.
- [x] Gắn cảnh báo kết quả 009 và 011 trùng nhau.
- [ ] Kiểm tra ảnh nghi tương tự cross-split (pHash) và checkpoint independence 009/011.
- [ ] Chạy **locked Test V08 686 ảnh** khi được khóa/đảm bảo không tuning bằng Test.
- [ ] Đánh giá **Real-world Holdout 49 ảnh** nếu holdout chưa dùng để lựa chọn model; nếu đã tuning cần holdout mới.
- [ ] Thử pipeline **YOLO crop → DenseNet121** với label map/inference preprocessing được kiểm tra.
- [ ] Xác minh và liên kết FULL ZIP/checkpoints/figures còn lưu ngoài GitHub.
- [ ] Kiểm tra phiên bản thực tế đang chạy trên web: repo hiện mô tả web dùng **EXP-009**, không tự nhận đang dùng EXP-012.

## Chuẩn README cho mọi thí nghiệm

1. Experiment ID chuẩn và ID ghi trên archive gốc (nếu khác).
2. Dataset version, Train/Val/Test, fingerprint/manifest nếu có, có `real_` hay không.
3. Config/augmentation/optimizer/head và sự thay đổi so với bản trước (nếu biết).
4. Training/Validation/Test metrics ghi rõ split; seed table, Mean ± sample SD.
5. Cách chọn checkpoint/seed **Validation only**.
6. Hạn chế, failure modes, domain shift, leakage suspects và trạng thái xác minh.
7. Artifact đã commit (và artifact vẫn chỉ có trong FULL/ANALYSIS ZIP), tiến độ tiếp theo.

> **Không tự suy diễn thông số chưa có trong ZIP/README, không đổi Test set sau khi tuning mà vẫn gọi là cùng benchmark.**
