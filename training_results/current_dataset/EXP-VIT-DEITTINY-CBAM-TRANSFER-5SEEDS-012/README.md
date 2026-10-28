# EXP-VIT-DEITTINY-CBAM-TRANSFER-5SEEDS-012

> **Research-only • V08 Validation • 5 paired seeds • Final Test/Holdout: NOT_RUN**

## Giao thức

- Dataset V08: Train 4.790 / Val 1.430 / Test 686; 5 classes và seed cố định `[42,123,2026,3407,7777]`.
- So sánh **baseline_no_cbam vs cbam trong cùng thí nghiệm**, không sử dụng baseline lịch sử khác recipe.

## Kết quả 5 seed

| Seed | Baseline Accuracy (%) | CBAM Accuracy (%) | Δ (điểm %) | Baseline Macro-F1 (%) | CBAM Macro-F1 (%) |
|---:|---:|---:|---:|---:|---:|
| 42 | 96.084 | 95.804 | -0.280 | 95.930 | 95.619 |
| 123 | 95.804 | 95.944 | 0.140 | 95.635 | 95.761 |
| 2026 | 96.084 | 96.014 | -0.070 | 95.917 | 95.801 |
| 3407 | 95.385 | 95.035 | -0.350 | 95.214 | 94.907 |
| 7777 | 95.524 | 95.455 | -0.070 | 95.389 | 95.293 |

## Tổng hợp

| Metric | Baseline | CBAM | Δ |
|---|---:|---:|---:|
| Val Accuracy mean ± SD | 95.776 ± 0.319% | 95.650 ± 0.406% | -0.126 pp |
| Val Macro-F1 mean ± SD | 95.617 ± 0.317% | 95.476 ± 0.376% | -0.141 pp |
| Paired accuracy CI95 (pp) | — | — | [-0.367; 0.115] |
| Paired t-test accuracy (exploratory) | — | — | p=0.2205 |

## Per-class F1 trung bình

| Class | Baseline (%) | CBAM (%) | Δ pp |
|---|---:|---:|---:|
| anthracnose | 91.83 | 91.68 | -0.15 |
| healthy | 94.40 | 93.78 | -0.62 |
| leaf_miner | 94.64 | 94.93 | 0.29 |
| red_rust | 99.06 | 98.97 | -0.09 |
| not_cashew_leaf | 98.15 | 98.02 | -0.13 |

- Disease→Healthy Validation trung bình trên 5 seed: 19.6 → 20.6 ảnh.
- Parameters (theo file metrics): 5,526,725 → 5,531,636 (Δ +4,911).

## Nhận xét và giới hạn
- Thêm CBAM không chứng minh lợi ích tổng thể trên Validation trong cấu hình này.
- Các heatmap ở ZIP là **attention diagnostics**, chưa được overlay chính xác lên ảnh gốc; cần code lớp custom CBAM và input để phân tích localization.
- Không chuyển ảnh giữa split, không tune bằng Test, không khẳng định hiệu quả real-world khi chưa được kiểm chứng.

Nguồn: `EXP-VIT-DEITTINY-CBAM-TRANSFER-5SEEDS-012_analy.zip`. [Bảng so sánh chung](../CBAM_ABLATION_3MODELS_V08/README.md).
