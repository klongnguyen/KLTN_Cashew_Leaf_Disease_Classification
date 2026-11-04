# Kế hoạch cải tiến ViT (DeiT-Small) sau EXP-015

**Cơ sở:** `EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015-256px-d4379a48` (đã huấn luyện xong trên Colab GPU, 5 seed).
**Đối chứng:** `EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05` (cùng công thức Transfer).
**Dataset:** V13 (Train 4.972 / Val 1.452 / Test 685), Holdout 09 (126 ảnh, 4 lớp, chỉ dùng mô tả).
**Phạm vi:** giữ nguyên backbone DeiT-Small distilled (không đổi checkpoint pretrained, không dùng DINO); chỉ cải tiến công thức huấn luyện.

---

## 1. Mục tiêu và tiêu chí "ngang DenseNet121"

Trên Test, ViT-015 đã ngang DenseNet121-029. Khoảng cách nằm ở ảnh thật (Holdout) và độ ổn định giữa các seed. Vì vậy mục tiêu cụ thể là:

| Tiêu chí | ViT-015 hiện tại | DenseNet121-029 | Mục tiêu cho ViT |
|---|---|---|---|
| Test macro-F1 (%) | 95.92 ± 0.63 | 95.89 ± 0.52 | giữ ≥ 95.4, SD ≤ 0.5 |
| Holdout macro-F1 (%) | 77.05 ± 2.13 | 81.67 ± 1.24 | ≥ 80.2 (kém DenseNet ≤ 1.5 điểm) |
| Holdout accuracy (%) | 78.9 ± 2.1 | 82.7 ± 1.3 | ≥ 81 |
| Validation val_loss (trung bình 5 seed) | 0.098 | 0.075 | ≤ 0.085 |
| Healthy thật, Validation (%) | 66.7 ± 11.8 | 78.3 ± 4.6 | ≥ 75 (chỉ 12 ảnh, rất nhiễu) |

Các ngưỡng Holdout chỉ dùng để đánh giá cuối cùng, không dùng để chọn cấu hình.

## 2. Chẩn đoán từ log EXP-015

1. **Overfit mạnh hơn DenseNet ở Stage 2.** Seed 42, epoch 15: ViT train accuracy 98.0% so với DenseNet 95.7%, nhưng val accuracy xấp xỉ nhau (97.3% và 96.9%). ViT mở 7.1M tham số trên 4.972 ảnh (DenseNet mở 2.1M).
2. **Không có regularization nội tại.** Backbone được dựng với `dropout_rate = 0` và `attention_dropout = 0`; chỉ có Dropout 0.4 ở đầu phân loại. KerasHub 0.32 `DeiTBackbone` không có tham số drop-path (đã kiểm tra chữ ký hàm).
3. **Stage 1 chưa hội tụ.** Cả 5 seed chạm trần 12 epoch, val_loss vẫn giảm (0.1995 ở epoch cuối), val accuracy 93.3% so với DenseNet 94.7% cùng giai đoạn.
4. **Stage 2 dùng LR phẳng 1e-5 cho 4 khối cuối.** Số khối "4/12" do người viết notebook chọn theo tỉ lệ tham số (32,7% so với 30,3% của DenseNet), chưa được thực nghiệm kiểm chứng.
5. **Chọn checkpoint nhiễu.** Epoch tốt nhất dao động từ 6 đến 29; val_loss giữa các seed trải 0.088–0.111 (DenseNet 0.070–0.080). Đây là nguồn chính của SD Test lớn hơn (0.66 so với 0.49).
6. **Lỗi Holdout tập trung ở ảnh thật.** Recall: red_rust 60.0 ± 12.3% (DenseNet 74.4%), healthy 61.1 ± 9.6% (DenseNet 65.3%). Lỗi chính là red_rust→anthracnose (9.8/25 ảnh) và healthy→anthracnose (7.2/19 ảnh).
7. **Cặp nhầm chính trên Test (anthracnose ↔ leaf_miner) giống nhau ở cả hai mô hình**, nên không phải điểm yếu riêng của ViT và không nên kỳ vọng thay đổi kiến trúc sẽ sửa được.

## 3. Nguyên tắc công bằng và quy tắc quyết định

- Mỗi thí nghiệm là **một EXP ID mới**, đánh số tiếp theo 015. EXP-015 giữ nguyên làm baseline cùng công thức.
- **Chọn cấu hình chỉ bằng Validation**: val_loss (cross-entropy thường, không làm mịn nhãn), macro-F1, false_healthy_rate, accuracy nhóm `real_`. Test và Holdout chỉ mở sau `validation_selection_lock.json`, và chỉ cho các cấu hình cuối cùng (tối đa 2–3).
- Mỗi thay đổi so với cấu hình đang giữ (incumbent): **chấp nhận** nếu val_loss trung bình giảm ≥ 0.005, val macro-F1 không giảm quá 0.3 điểm và false_healthy_rate không tăng. Nếu val_loss ngang nhau (± 0.005), chọn cấu hình đơn giản hơn.
- Nhóm `real_` trên Validation chỉ có 59 ảnh (12 ảnh healthy), chỉ dùng làm tín hiệu phụ.
- Sàng lọc bằng 3 seed (42, 123, 2026), xác nhận bản thắng bằng đủ 5 seed. Ghi rõ số cấu hình đã thử.
- Các thay đổi **không phụ thuộc kiến trúc** (label smoothing, EMA, Stage 1 dài hơn) phải chạy song song cho DenseNet121 (Track B, mục 6) để có thể viết "mỗi kiến trúc được tinh chỉnh với ngân sách ngang nhau".

## 4. Các thí nghiệm

Thời gian tham chiếu: EXP-015 mất khoảng 190–320 giây mỗi seed, tức khoảng 20–25 phút cho 5 seed trên Colab GPU.

### EXP-016 — Layer-wise LR decay và độ sâu mở (ưu tiên cao nhất)

- **Thay đổi:** LR theo khối: `lr_khối_i = base_lr × 0.75^(11 − i)`; embedding (patch, position) đóng băng hoặc dùng LR thấp nhất; đầu phân loại dùng `base_lr`.
- **Lưới sàng lọc:** số khối mở N ∈ {4, 8, 12} × `base_lr` ∈ {3e-5, 5e-5} (6 cấu hình × 3 seed ≈ 1 giờ).
- **Cách cài trong Keras 3:** Adam chuẩn hóa gradient nên **không thể** thay LR bằng cách nhân gradient. Cần lớp con của `AdamW` ghi đè `update_step` để nhân `learning_rate` theo từng biến (map theo đường dẫn biến → hệ số).
- **Giữ nguyên:** Stage 1, scheduler, early stopping, augmentation, batch 16, seed.
- **Chấp nhận:** theo mục 3.

### EXP-017 — Regularization (trên cấu hình thắng của 016)

Làm từng yếu tố một (OFAT), mỗi yếu tố 3 seed:

| Bước | Thay đổi | Lưu ý |
|---|---|---|
| 017a | Label smoothing 0.1 | Dùng `CategoricalCrossentropy(label_smoothing=0.1)` với nhãn one-hot; **val_loss để chọn checkpoint vẫn phải là cross-entropy thường** (thêm làm metric riêng) để so được với 029 |
| 017b | `dropout_rate` backbone 0.1 (DeiT không có drop-path) | Phải cập nhật `fairness contract` vì đây là tham số kiến trúc khai báo |
| 017c | Mixup nhẹ (α = 0.2), chỉ trên Train | Chỉ thử nếu 017a/b chưa đủ; không dùng CutMix vì có thể xóa vết bệnh nhỏ |

Kết hợp các yếu tố đã được chấp nhận thành 017-final và chạy 5 seed.

### EXP-018 — Ổn định chọn checkpoint

- **Phương án A (ưu tiên):** trung bình trọng số của K = 3 checkpoint có val_loss thấp nhất (chọn bằng Validation).
- **Phương án B:** EMA trọng số (momentum ≈ 0.999, tức khoảng 3 epoch với 311 bước/epoch). Với Keras 3 cần callback tự viết để đánh giá và lưu checkpoint bằng **trọng số EMA**; `use_ema=True` mặc định chỉ hoán đổi ở cuối quá trình.
- **Mục tiêu:** giảm SD giữa các seed và độ dao động của epoch tốt nhất.

### EXP-019 — Stage 1 dài hơn

- **Thay đổi:** `STAGE1_MAX_EPOCHS` 12 → 30, giữ early stopping patience 4.
- **Lý do:** Stage 1 chạm trần ở cả 5 seed; đầu phân loại tốt hơn giúp Stage 2 khởi động ổn định hơn.
- **Tách riêng** vì nó đổi cấu trúc lịch huấn luyện chung.

### EXP-020 — Oversample ảnh `real_` (ưu tiên thấp)

- Với DenseNet (EXP-030) cách này chỉ cải thiện nhẹ (Holdout F1 82.1 so với 81.7, Test giảm từ 96.2 xuống 95.9), nên chỉ thử nếu khoảng cách trên ảnh thật vẫn còn sau 016–019.

## 5. Lộ trình

| Giai đoạn | Nội dung | Thời gian Colab ước tính |
|---|---|---|
| 1 | EXP-016 sàng lọc (6 cấu hình × 3 seed) và xác nhận 5 seed | ~1,5 giờ |
| 2 | EXP-017a/b/c và EXP-018, EXP-019 (OFAT, 3 seed rồi 5 seed cho bản kết hợp) | ~2 giờ |
| 3 | EXP-020 (oversample `real_`) chỉ khi khoảng cách trên ảnh thật vẫn còn | ~0,5 giờ |
| 4 | Chốt 1–2 cấu hình cuối, chạy 5 seed, mở Test và Holdout một lần | ~1 giờ |
| 5 (Track B) | Chạy các thay đổi không phụ thuộc kiến trúc trên DenseNet121 để đối chứng ngang ngân sách | ~1,5 giờ |

## 6. Track B — đối chứng công bằng cho DenseNet121

Áp dụng cho DenseNet121 những yếu tố không phụ thuộc kiến trúc: label smoothing, trung bình checkpoint hoặc EMA, Stage 1 dài hơn. Nếu DenseNet cũng cải thiện tương tự thì khoảng cách không đổi, và đó là kết quả trung thực cần báo cáo.

## 7. Cách báo cáo trong khóa luận

- Bảng ablation trên **Validation** cho mọi cấu hình đã thử, kèm số cấu hình và số seed.
- Test và Holdout chỉ báo cáo cho baseline (015) và 1–2 cấu hình cuối, kèm SD.
- Nêu rõ: một split cố định; Holdout 09 nhỏ (126 ảnh), đã từng được xem, chỉ mang tính mô tả; việc chọn trên Validation nhiều lần có thể làm lạc quan nhẹ.
- Phân biệt "ViT cùng công thức với DenseNet" (015) và "mỗi kiến trúc được tinh chỉnh ngang ngân sách" (Track A + B).

## 8. Điều chưa kiểm chứng

- LR theo khối qua lớp con `AdamW` trong Keras 3 (backend TensorFlow, có thể ảnh hưởng đến XLA/`jit_compile`) cần smoke test trước khi chạy thật.
- Callback EMA và việc lưu trọng số EMA cần kiểm tra tải lại được.
- Không có gì đảm bảo ViT sẽ bắt kịp DenseNet trên Holdout; các giả thuyết ở mục 2 chưa được thí nghiệm kiểm chứng.
- Vì giữ nguyên checkpoint pretrained (DeiT-Small chỉ được huấn luyện trước trên ImageNet-1k), trần cải thiện có thể thấp hơn so với việc đổi backbone. Nếu sau EXP-016–019 khoảng cách Holdout vẫn còn, kết quả đó nên được báo cáo như một phát hiện trung thực chứ không cần ép cho bằng.

## 9. Các quyết định cần xác nhận

1. Có chạy Track B cho DenseNet121 để đối chứng ngang ngân sách không.
2. Số khối mở và LR tối đa trong lưới EXP-016 có cần thu hẹp để tiết kiệm thời gian Colab không.
