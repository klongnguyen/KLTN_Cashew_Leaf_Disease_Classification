# EXP-DENSENET121-TRANSFER-5SEEDS-014 — ImageNet Transfer Learning (V08)

**Experiment:** `EXP-DENSENET121-TRANSFER-5SEEDS-014` · **đào tạo mới 5 seeds**. Source: user-supplied `EXP-DENSENET121-TRANSFER-5SEEDS-014_ANALYSIS.zip`. **Cập nhật hồ sơ: 2026-10-19.**

> **Phân biệt:** EXP-014 là lần **huấn luyện 5 seed bằng transfer learning**; [EXP-015](../EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md) dùng **nguyên 5 checkpoint 014** để đánh giá Real Holdout, không retrain. EXP-014 chưa chạy Final Test trên 686 ảnh.

## 1. Dữ liệu cố định, quy tắc chống leakage

- **Dataset:** `Cashew_dataV08_real_val` — `CashewData_Split (8).zip`, **Train 4.790 / Validation 1.430 / Test 686 = 6.906** ảnh, 5 lớp.
- Dataset ZIP SHA256: `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`.
- Manifest SHA256: `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b`; giống EXP-012 và EXP-013 V08.
- Phân bố ảnh tên `real_`: Train 135, Val 35, Test 28; **tiền tố file không bảo đảm provenance thực địa**.
- Exact duplicate khác split: 0; **51 cặp pHash nghi gần trùng/cùng ảnh cần xác minh thủ công**. Kiểm toán chưa chứng minh không có leakage theo lá hoặc buổi chụp.
- Class order **bất biến khi infer**: `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`.

## 2. Cấu hình và thay đổi so với EXP-013

| Thành phần | EXP-013 Scratch | EXP-014 Transfer |
|---|---|---|
| Backbone | DenseNet121 `weights=None` | **DenseNet121 `weights=imagenet`**, `include_top=False`, GAP |
| Cách huấn luyện | Scratch một giai đoạn | **Stage 1 đóng băng backbone; Stage 2 fine-tune conv5** |
| Head | Dropout 0.4 + Dense(5, softmax) | Giữ |
| Optimizer | AdamW, weight_decay 1e-4 | Giữ, optimizer mới mỗi stage |
| Batch / image | 16 / 224×224 RGB | Giữ |
| Train augmentation | Flip ngang; rotation 0.08; zoom 0.10; contrast 0.10 | Giữ |
| Input & preprocessing | RGB pixel 0–255; chia 255 rồi chuẩn hóa ImageNet **bên trong model** | Giữ |
| Dataset, split, seeds | V08; [42,123,2026,3407,7777] | Giữ |
| Checkpoint chọn theo | Validation loss thấp nhất | Giữ (thêm stage selection) |

Stage 1: chỉ train head, AdamW LR **3e-4**, tối đa 12 epoch, EarlyStopping 4, ReduceLROnPlateau patience 2, min LR 1e-6.

Stage 2: khôi phục best stage-1, mở các lớp `conv5_*` **không phải BatchNorm** (2.129.920 backbone parameters trainable) và head; AdamW mới LR **1e-5**, tối đa 30 epoch, EarlyStopping 8, ReduceLROnPlateau patience 4, min LR 1e-7. `backbone(..., training=False)` giữ BatchNorm ở chế độ inference. Tổng tham số mô hình 7.042.629. Loss: SparseCategoricalCrossentropy; Weight Decay 1e-4.

Đây là **so sánh phương pháp ImageNet + 2-stage versus Scratch**, **không** cô lập chỉ một yếu tố pretrained weights.

## 3. Kết quả 5 seeds — Validation V08

| Seed | Stage / best epoch | Val Loss | Accuracy | Macro F1 | Val `real_` đúng |
|---:|---|---:|---:|---:|---:|
| 42 | stage2 / 24 | 0.100702 | 96.92% | 96.78% | 27/35 |
| **123** | **stage2 / 29** | **0.092535** | **97.27%** | **97.14%** | **28/35** |
| 2026 | stage2 / 20 | 0.101971 | 96.85% | 96.70% | 28/35 |
| 3407 | stage2 / 24 | 0.097457 | 97.20% | 97.05% | 28/35 |
| 7777 | stage2 / 22 | 0.097242 | 97.27% | 97.13% | 28/35 |

- **Val Accuracy 97.10 ± 0.20%**, **Macro F1 96.96 ± 0.21%** (mean ± sample SD qua 5 seed).
- Macro Precision **96.94 ± 0.19%**, Macro Recall **96.98 ± 0.23%**.
- Anthracnose Recall **93.83 ± 0.32%**; Leaf miner Recall **95.15 ± 0.49%**.
- **False Healthy** trên 900 ảnh bệnh/seed: **0.89 ± 0.08%** (mean ± sample SD).
- Nhóm **35 ảnh `real_` trong Validation: 79.43 ± 1.28%**; nhóm này nhỏ và **đã dùng trong Validation**, không phải unseen external Holdout.
- Thời gian huấn luyện tổng 2 stage: **249.64 ± 6.57 giây/seed** theo archive; inference Validation tính decode/resize ~**0.62 ± 0.01 ms/ảnh** trong runtime ghi nhận (không đại diện web production).
- **Deployment seed = 123**, chọn bằng **Validation Loss thấp nhất** (trước khi xem Holdout), không chọn seed theo Accuracy Test/Holdout.

Tham khảo [bảng Validation từng seed](./aggregate/validation_results_5seeds.csv), [Mean ± SD](./aggregate/validation_mean_std_summary.csv), [35 ảnh real_ theo seed](./aggregate/real_validation_by_seed.csv), [config](./experiment_config.json), [validation lock](./validation_selection_lock.json), [audit](./dataset_audit_summary.json).

## 4. So sánh có kiểm soát với EXP-013

| Metric | EXP-013 Scratch | EXP-014 Transfer | Chênh lệch |
|---|---:|---:|---:|
| Val Accuracy | 94.17 ± 0.90% | **97.10 ± 0.20%** | **+2.94 điểm %** |
| Val Macro F1 | 94.02 ± 0.98% | **96.96 ± 0.21%** | **+2.94 điểm %** |
| Val `real_` Accuracy | 67.43% | **79.43%** | **+12.00 điểm %** |

Cùng V08, nhưng chỉ số nhóm `real_` có **35 ảnh**, dễ dao động theo từng ảnh. Việc cải thiện Validation không thay thế cho đánh giá độc lập.

## 5. Hoàn thành / giới hạn / artifact

- ✅ 5-seed training, 2-stage selection, validation, class-level error tracking.
- ✅ Validation lock `2026-10-19T08:38:55.528406+00:00`, config SHA256 `fcf1ebc509cd5c4dd54f9ad073e5af02ba5b412f72210dfee47a998c921292b2`.
- ✅ EXP-015 đã đánh giá Real Holdout lịch sử **48 ảnh**, ghi ở folder **khác**; **không sửa execution status gốc EXP-014**.
- ⏳ **Final locked Test V08 686 ảnh chưa chạy** với EXP-014.
- ⏳ Cần audit 51 cặp pHash và thu thập real-world blind holdout mới.
- ⚠️ Bản ghi trên GitHub bao gồm **báo cáo, config/lock, aggregate CSV, figures tái lập**; **ANALYSIS/FULL ZIP gốc, toàn bộ per-image predictions, training notebooks và binary checkpoint .h5/.keras chưa được commit**. Không thể dùng repository này một mình để tái tạo checkpoint hoặc xác minh binary hash; xem FULL ZIP ngoài GitHub.
- ⚠️ Ứng dụng web `Application/` trên repo vẫn mô tả chạy **EXP-009 seed 3407**, **chưa thay deployment** sang EXP-014.

**Đọc tiếp:** [EXP-015 — Holdout checkpoint evaluation](../EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md) · [Tiến độ](../EXPERIMENT_PROGRESS_2026_10.md).

