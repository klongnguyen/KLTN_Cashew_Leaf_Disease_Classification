# EXP-DENSENET121-CBAM-TRANSFER-5SEEDS-017

> **Research-only • V08 Validation • 5 paired seeds • Final Test/Holdout: NOT_RUN**

## Giao thức

- Dataset V08: Train 4.790 / Val 1.430 / Test 686; 5 classes và seed cố định `[42,123,2026,3407,7777]`.
- So sánh **baseline_no_cbam vs cbam trong cùng thí nghiệm**, không sử dụng baseline lịch sử khác recipe.

## Kết quả 5 seed

| Seed | Baseline Accuracy (%) | CBAM Accuracy (%) | Δ (điểm %) | Baseline Macro-F1 (%) | CBAM Macro-F1 (%) |
|---:|---:|---:|---:|---:|---:|
| 42 | 97.203 | 96.224 | -0.979 | 97.061 | 96.013 |
| 123 | 96.993 | 95.245 | -1.748 | 96.833 | 94.938 |
| 2026 | 97.203 | 96.014 | -1.189 | 97.085 | 95.752 |
| 3407 | 96.993 | 95.315 | -1.678 | 96.833 | 95.026 |
| 7777 | 96.993 | 96.224 | -0.769 | 96.840 | 96.004 |

## Tổng hợp

| Metric | Baseline | CBAM | Δ |
|---|---:|---:|---:|
| Val Accuracy mean ± SD | 97.077 ± 0.115% | 95.804 ± 0.487% | -1.273 pp |
| Val Macro-F1 mean ± SD | 96.930 ± 0.130% | 95.547 ± 0.527% | -1.384 pp |
| Paired accuracy CI95 (pp) | — | — | [-1.806; -0.740] |
| Paired t-test accuracy (exploratory) | — | — | p=0.0027 |

## Per-class F1 trung bình

| Class | Baseline (%) | CBAM (%) | Δ pp |
|---|---:|---:|---:|
| anthracnose | 94.33 | 91.85 | -2.47 |
| healthy | 96.37 | 93.41 | -2.96 |
| leaf_miner | 94.94 | 93.89 | -1.04 |
| red_rust | 99.24 | 99.00 | -0.24 |
| not_cashew_leaf | 99.78 | 99.58 | -0.19 |

- Disease→Healthy Validation trung bình trên 5 seed: 9.6 → 13.8 ảnh.
- Parameters (theo file metrics): 7,042,629 → 7,174,888 (Δ +132,259).

## Nhận xét và giới hạn
- Thêm CBAM không chứng minh lợi ích tổng thể trên Validation trong cấu hình này.
- Các heatmap ở ZIP là **attention diagnostics**, chưa được overlay chính xác lên ảnh gốc; cần code lớp custom CBAM và input để phân tích localization.
- Không chuyển ảnh giữa split, không tune bằng Test, không khẳng định hiệu quả real-world khi chưa được kiểm chứng.

Nguồn: `EXP-DENSENET121-CBAM-TRANSFER-5SEEDS-017-Analy.zip`. [Bảng so sánh chung](../CBAM_ABLATION_3MODELS_V08/README.md).
