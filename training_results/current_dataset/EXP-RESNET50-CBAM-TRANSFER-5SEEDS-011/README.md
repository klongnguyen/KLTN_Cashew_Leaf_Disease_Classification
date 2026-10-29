# EXP-RESNET50-CBAM-TRANSFER-5SEEDS-011
> **Cập nhật điều hướng 28/10/2026.** VALIDATION-ONLY / CBAM — Test/Holdout chưa chạy. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


> **Research-only • V08 Validation • 5 paired seeds • Final Test/Holdout: NOT_RUN**

## Giao thức

- Dataset V08: Train 4.790 / Val 1.430 / Test 686; 5 classes và seed cố định `[42,123,2026,3407,7777]`.
- So sánh **baseline_no_cbam vs cbam trong cùng thí nghiệm**, không sử dụng baseline lịch sử khác recipe.

## Kết quả 5 seed

| Seed | Baseline Accuracy (%) | CBAM Accuracy (%) | Δ (điểm %) | Baseline Macro-F1 (%) | CBAM Macro-F1 (%) |
|---:|---:|---:|---:|---:|---:|
| 42 | 95.315 | 96.224 | 0.909 | 95.118 | 96.078 |
| 123 | 95.664 | 96.364 | 0.699 | 95.487 | 96.225 |
| 2026 | 95.035 | 96.573 | 1.538 | 94.812 | 96.446 |
| 3407 | 95.175 | 95.944 | 0.769 | 94.986 | 95.753 |
| 7777 | 95.245 | 95.524 | 0.280 | 95.023 | 95.358 |

## Tổng hợp

| Metric | Baseline | CBAM | Δ |
|---|---:|---:|---:|
| Val Accuracy mean ± SD | 95.287 ± 0.235% | 96.126 ± 0.407% | 0.839 pp |
| Val Macro-F1 mean ± SD | 95.085 ± 0.250% | 95.972 ± 0.426% | 0.887 pp |
| Paired accuracy CI95 (pp) | — | — | [0.273; 1.405] |
| Paired t-test accuracy (exploratory) | — | — | p=0.0147 |

## Per-class F1 trung bình

| Class | Baseline (%) | CBAM (%) | Δ pp |
|---|---:|---:|---:|
| anthracnose | 90.96 | 92.39 | 1.43 |
| healthy | 94.21 | 95.24 | 1.03 |
| leaf_miner | 92.33 | 94.09 | 1.76 |
| red_rust | 98.50 | 98.78 | 0.27 |
| not_cashew_leaf | 99.42 | 99.36 | -0.06 |

## Nhận xét và giới hạn
- CBAM cải thiện Accuracy/Macro-F1 ở cả 5 seed. Kết quả khả quan trên Validation, nhưng còn kém DenseNet121 baseline xét Accuracy tuyệt đối.
- Các heatmap ở ZIP là **attention diagnostics**, chưa được overlay chính xác lên ảnh gốc; cần code lớp custom CBAM và input để phân tích localization.
- Không chuyển ảnh giữa split, không tune bằng Test, không khẳng định hiệu quả real-world khi chưa được kiểm chứng.

Nguồn: `EXP-RESNET50-CBAM-TRANSFER-5SEEDS-011-analy.zip`. [Bảng so sánh chung](../CBAM_ABLATION_3MODELS_V08/README.md).
