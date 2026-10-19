# Tiến độ huấn luyện và đối chiếu phiên bản (19/10/2026)

> **EXP-RESNET50-TRANSFER-5SEEDS-008 đã hoàn thành (20/10/2026):** 5-seed Validation + **Final Test V08 686 ảnh** + **Real Holdout lịch sử 48 ảnh**. Val Accuracy **95.09 ± 0.22%**; Test Accuracy **93.24 ± 0.24%**; historical Holdout Accuracy **77.08 ± 2.55%**. **ImageNet pretrained nhưng frozen backbone ở cả Stage 1/2** (chỉ refinement head). **Seed 3407 chọn theo min Val Loss**; [README](./EXP-RESNET50-TRANSFER-5SEEDS-008/README.md) · [Audit](./EXP-RESNET50-TRANSFER-5SEEDS-008/ANALYSIS_AND_PROTOCOL_AUDIT.md).


> **Bổ sung ResNet50 Scratch EXP-006/007/008 (19/10/2026):** 006 trên V05: Val Acc **86.46 ± 2.27%**, Macro F1 **86.12 ± 2.56%**; 007/008 trên V07: **89.20 ± 2.04%**, Macro F1 **88.61 ± 2.19%**; 007 và 008 có predictions/history/aggregate Validation **trùng byte**, chưa có bằng chứng 008 retrain độc lập. **Final Test cả ba pending**. [Audit/Changelog](./RESNET50_SCRATCH_006_008_AUDIT_2026_10.md).


> **Cập nhật 19/10/2026 (EXP-014/015):** V08 đã có **huấn luyện 5 seed DenseNet121 ImageNet Transfer Learning EXP-014** và **đánh giá 48 ảnh Real Holdout lịch sử bằng checkpoint EXP-014 trong EXP-015**. Seed triển khai **123** (khóa theo Val Loss). **Final Test V08 686 ảnh vẫn chưa chạy**. Nhóm Holdout này đã được xem trước EXP-014, nên không phải blind holdout mới.


Nguồn: cây thư mục GitHub `main` trước cập nhật, các gói ANALYSIS mà người dùng đã gửi, và cấu hình thí nghiệm được ghi trong ZIP. **Validation và Test là hai giao thức khác nhau**; không so sánh các số từ hai tập như cùng một benchmark.

## Tổng hợp trạng thái quan trọng

| Kiến trúc | Experiment | Dataset | Tình trạng | Chỉ số chính |
|---|---|---|---|---|
| **ResNet50 ImageNet frozen** | **[TRANSFER EXP-008](./EXP-RESNET50-TRANSFER-5SEEDS-008/README.md)** | **V08** | **5 seeds Val + Final Test + historical Holdout xong**; **không fine-tune backbone** | Val Acc **95.09 ± 0.22%**, Test Acc **93.24 ± 0.24%**, historical Holdout Acc **77.08 ± 2.55%** |
| ResNet50 Scratch | [EXP-006](./EXP-RESNET50-SCRATCH-5SEEDS-006/README.md) | V05 | 5-seed Validation xong; **Test 006 pending** | Val Acc **86.46 ± 2.27%**, F1 **86.12 ± 2.56%**, candidate seed 42 (min loss) |
| ResNet50 Scratch | [EXP-007](./EXP-RESNET50-SCRATCH-5SEEDS-007/README.md) | V07 | 5-seed Validation xong; **Test pending** | Val Acc **89.20 ± 2.04%**, F1 **88.61 ± 2.19%**, candidate seed 123 (min loss) |
| ResNet50 Scratch | [EXP-008](./EXP-RESNET50-SCRATCH-5SEEDS-008/README.md) | V07 | **Same exact 5-seed Val/predictions as 007**; Test pending | Val Acc **89.20 ± 2.04%**, F1 **88.61 ± 2.19%**; independent run not established |
| DenseNet121 | [EXP-005](./EXP-DENSENET121-SCRATCH-5SEEDS-005/README.md) | V05 | **Final locked Test xong** | Test Acc **91.97 ± 1.11%**, F1 **91.41 ± 1.13%** |
| ResNet50 | [EXP-005](./EXP-RESNET50-SCRATCH-5SEEDS-005/README.md) | V05 | **Final locked Test xong** | Test Acc **89.78 ± 2.50%**, F1 **89.36 ± 2.47%** |
| Compact ViT | [EXP-005](./EXP-VIT-SCRATCH-5SEEDS-005/README.md) | V05 | **Final locked Test xong** | Test Acc **85.04 ± 0.44%**, F1 **83.93 ± 0.37%** |
| DenseNet121 | [EXP-006](./EXP-DENSENET121-SCRATCH-5SEEDS-006/README.md) | V05 | Test xong | Test Acc **88.44 ± 3.14%**, F1 **87.95 ± 3.14%** |
| DenseNet121 | [EXP-009](./EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL/EXP-DENSENET121-SCRATCH-5SEEDS-009/README.md) | V07 | Validation xong; Test pending | Val Acc **94.55 ± 1.11%**, F1 **94.38 ± 1.13%** |
| DenseNet121 | [EXP-011](./EXP-DENSENET121-SCRATCH-5SEEDS-011/README.md) | V07 | Validation xong; **số liệu trùng 009** | Val Acc **94.55 ± 1.11%**, F1 **94.38 ± 1.13%** |
| DenseNet121 | EXP-013 (chưa có directory chuẩn trên repo) | V08 | 5-seed Scratch; Val/Test/Real Holdout lịch sử | Val Acc **94.17 ± 0.90%**, Val F1 **94.02 ± 0.98%**; Holdout Acc **51.25 ± 8.15%** |
| **DenseNet121 Transfer** | **[EXP-014](./EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md)** | **V08** | **5-seed training + Validation xong; Final Test pending** | **Val Acc 97.10 ± 0.20%; Macro F1 96.96 ± 0.21%; Real Val 79.43 ± 1.28%** |
| **DenseNet121 Transfer evaluation** | **[EXP-015](./EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md)** | **V08 checkpoints + historical holdout** | **Không retrain; đã chấm 48 ảnh holdout; Test pending** | **Holdout Acc 80.83 ± 0.93%; Macro F1 80.80 ± 0.19%; seed 123 đúng 38/48** |
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
| ResNet50 Scratch-008 → ImageNet Transfer-008 | **V07 → V08** (manifest khác) | Scratch weights=None → **ImageNet pretrained frozen features**; 2 stages head-only | Val **89.20 ± 2.04% V07 → 95.09 ± 0.22% V08**, **không phải controlled pretrained ablation**; Transfer-008 đã chạy Test V08 **93.24 ± 0.24%** |
| ResNet50 Transfer-008 → proposed partial FT | V08 dự kiến giữ nguyên | 008: **0 trainable backbone params cả 2 stages**; cần experiment ID mới nếu mở conv5 | Chưa chạy, không tự tạo kết quả/không chọn theo Holdout |
| ResNet50 Scratch 005 → 006 | V05 giữ nguyên số ảnh/split | 006: Adam LR 3e-4, batch32, head512/Dropout0.4, rescale /255; 005 config lịch sử xem README riêng | 006 Val **86.46 ± 2.27%**, Test 006 pending; **không dùng Test 005 cho 006** |
| ResNet50 Scratch 006 → 007 | **V05 → V07**, 6.911 → 6.907 ảnh | 007: batch16, LR1e-4, head Dropout0.3, Caffe preprocessing, bỏ augmentation contrast | Val **86.46 → 89.20%**, **không phải ablation cùng dataset** |
| ResNet50 Scratch 007 → 008 | V07 manifest trùng | Cấu hình chính trùng; 008 bổ sung figure/notebook, predictions + history của 5 seeds trùng byte 007 | Val **89.20% → 89.20%**; **chưa chứng minh retrain độc lập** |
| DenseNet 012 → 013 | Cùng V08 | EXP-013 Scratch AdamW, Dropout 0.4, augmentation rotation/zoom/contrast tăng; 5 seeds | Val Acc 94.17 ± 0.90%; Holdout lịch sử 51.25 ± 8.15% |
| DenseNet 013 → 014 | **V08 không đổi**, manifest SHA trùng | **ImageNet pretrained + frozen-head stage 1 → conv5 partial fine-tune stage 2**; giữ augmentation, AdamW/batch/dropout | Val Acc **94.17 → 97.10%**, Val F1 **94.02 → 96.96%**; real_ Val **67.43 → 79.43%** |
| DenseNet 014 → 015 | **Không thay dataset, không train** | **5 checkpoint 014 locked**; đánh giá Holdout đã phân tích trước, loại trùng byte 1 ảnh Test | Accuracy lịch sử **51.25% EXP-013 → 80.83% EXP-015** trên cùng 48 ảnh; seed 123 Val-selected 38/48 |
| DenseNet 011 → 012 | V07 → V08 6,906 ảnh; 35 `real_` chuyển Train→Val; loại 1 Test | Cấu hình chính giữ nguyên | Val 012 giảm nhẹ do thay split; seed 42 real_ đúng 22/35 |

## Tiến độ / Checklist

- [x] V05 official 3-model Final Test, 5 seeds, report Mean ± sample SD.
- [x] Hồ sơ EXP-011, EXP-012: README, CSV aggregate Validation, metadata snapshot.
- [x] Nhập báo cáo ResNet50 **TRANSFER-008**: Validation, Final Test, historical Holdout đầy đủ, confusion matrix 5 seeds, metadata frozen/preprocessing/selection, audit và bảng tiến độ.
- [ ] Audit 51 V08 pHash candidates + nguồn cùng lá/cùng phiên chụp và thu thập **fresh blind field test** đủ 5 true classes.
- [ ] Thiết kế **experiment ID mới ResNet50 partial fine-tuning thực** (mở conv5 non-BN), không ghi đè Transfer-008, chọn bằng Validation.
- [x] Nhập hồ sơ ResNet50 Scratch **006/007/008**: 3 README versioned, 5-seed Validation CSV, Mean ± SD, 15 confusion matrix CSV, config snapshot, audit V07, biểu đồ SVG; đánh dấu 007/008 không độc lập.
- [ ] Kiểm tra binary checkpoint ResNet50 007/008 trong FULL ZIP để xác định thực chất retrain/re-export; kiểm tra 52 cặp pHash V07.
- [ ] Chạy locked Test riêng cho ResNet50 Scratch 006 (V05) và 007 (V07) khi protocol cho phép; không chấm 008 như lần test độc lập nếu là bản tái đóng gói.
- [x] Thêm EXP-014/015: README báo cáo, aggregate CSV, manifest/audit metadata, 5 confusion-matrix CSV Holdout, báo cáo chênh lệch EXP-013 vs 015.
- [ ] Chưa commit original ANALYSIS ZIP/FULL ZIP, binary model .keras/.h5, per-image predictions đầy đủ, full notebook/PNG của 014/015; xem source archive.
- [x] Chuẩn hóa ID nguồn ZIP `...-01` thành EXP-012; giữ mapping ID gốc.
- [x] Gắn cảnh báo kết quả 009 và 011 trùng nhau.
- [ ] Kiểm tra ảnh nghi tương tự cross-split (pHash) và checkpoint independence 009/011.
- [ ] Chạy **locked Test V08 686 ảnh** cho checkpoint EXP-014 đã khóa bằng Validation; chưa được chấm trong EXP-015.
- [x] **EXP-015 đã đánh giá 48 ảnh historical Real Holdout** (49 ảnh gốc trừ 1 exact Test duplicate) trên đủ 5 checkpoint của EXP-014; **không** dùng kết quả này để chọn lại seed/model.
- [ ] Thu thập **blind independent field holdout mới** đầy đủ cả 5 lớp, từ cây/phiên/vườn độc lập; đánh giá sau khi khóa pipeline.
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
