# EXP-RESNET50-SCRATCH-5SEEDS-008 — V07 reporting revision / repackage
> **Cập nhật điều hướng 28/10/2026.** SCRATCH LỊCH SỬ — không phải model WebApp Transfer EXP-014. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


> **Đính chính mức độ độc lập:** Source `EXP-RESNET50-SCRATCH-5SEEDS-008_ANALYSIS.zip` ghi ID 008 nhưng **cùng tập V07, cùng cấu hình trọng yếu và dữ liệu huấn luyện/Validation trùng byte với EXP-007 ở cả 5 seeds**. Do đó **chưa có bằng chứng một lần train cải thiện độc lập**; 008 nên được mô tả là **bản tái xuất/hoàn thiện notebook và visualization** cho đến khi đối chiếu được binary checkpoint từ FULL ZIP.

## Kế thừa dataset và recipe

- Dataset **Cashew_dataV07_clean_val** — Train **4.825** / Val **1.395** / Test **687** = **6.907**; class order `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`.
- Dataset manifest SHA256 `f9e7d9a39f33bd631b3133a77d0260055bcdfbcd79cbebbb4d87877b893c6d74` (cùng 007).
- ResNet50 `weights=None` (**Scratch**), GAP → Dropout 0.3 → Dense 5, Adam LR 1e-4, batch 16, max 50 epochs, EarlyStop 10, ReduceLR 4/factor 0.5, preprocessing ResNet Caffe **bên trong model**, flip/rotation .04/zoom .05, no contrast.
- 5 seeds cố định: `[42, 123, 2026, 3407, 7777]`. Best seed **123 theo min Validation Loss**, không phải seed max Accuracy.
- 52 pHash candidates và 1 thumbnail candidate chưa audit thủ công; Val V07 không có ảnh `real_`.

## So sánh byte với EXP-007

| Đối tượng so sánh | Kết quả kiểm tra ZIP |
|---|---|
| `aggregate/validation_results_5seeds.csv` | **SHA-256 trùng** |
| `aggregate/validation_mean_std_summary.csv` | **SHA-256 trùng** |
| 5 `seed_*/validation/predictions.csv` | **trùng SHA-256 từng seed** |
| 5 `seed_*/logs/history.csv` | **trùng SHA-256 từng seed** |
| 5 confusion matrix Val | **trùng byte** |
| Source notebook/figures | 008 bổ sung/đổi cách trình bày |
| `training_summary_5seeds.csv` | Có khác **timing/đường dẫn artifact**, không chứng minh run độc lập |
| FULL ZIP checkpoint binary | **Chưa có để xác minh** |

Hashes of identical source CSV:
- Validation five-seed CSV: `c2c9d7b12163fdc17e1cb337135c80a25029dc30477f6f03ad870a3bdc615119`.
- Validation aggregate CSV: `15dab01ed3b6f6e32677d4dc3ed2d83f4ad28f819a6f241c68b604f64354a7bb`.

## Kết quả Validation (trùng 007)

| Seed | Val Loss | Val Accuracy | Val Macro F1 |
|---:|---:|---:|---:|
| 42 | 0.368771 | 90.97% | 90.59% |
| **123** | **0.332464** | 87.81% | 86.91% |
| 2026 | 0.338971 | **91.83%** | **91.38%** |
| 3407 | 0.366160 | 88.10% | 87.29% |
| 7777 | 0.353775 | 87.31% | 86.87% |

Accuracy **89.20 ± 2.04%**, Macro F1 **88.61 ± 2.19%**, Precision **89.73 ± 2.23%**, Recall **88.27 ± 2.10%**, đều là **Validation 5 seeds**. Không có Test/Holdout mới.

## Những khác biệt thật sự có bằng chứng từ archive

EXP-008 thêm các hình/biểu đồ về Val Macro F1, learning rate, per-class F1 và notebook `resnet50_5seed_008.ipynb`. **Không có bằng chứng thay đổi data, architecture/weights, lựa chọn seed, hiệu năng tổng thể**.

## Artifact đã lưu

[Validation 5-seed CSV](./aggregate/validation_results_5seeds.csv) · [Mean ± SD](./aggregate/validation_mean_std_summary.csv) · [Config snapshot](./config_snapshot.json) · [Dataset audit](./dataset_audit_summary.json) · [Seed 123 confusion matrix](./seed_123/validation/confusion_matrix.csv). Original figure PNG/notebook/checkpoint không được đưa vào commit tài liệu này.

Source ZIP SHA256: `5bdcd4382a13afb7f805030e478a8f7bd63e0ea95355a3b35f97e1a43f43a6a9`.

**Không sử dụng EXP-008 như một dòng độc lập thứ hai trong bảng benchmark**: nếu muốn nhắc đến cả hai, chú thích `EXP-008 ≈ EXP-007; reporting revision, not demonstrated independent retrain`.
