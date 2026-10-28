# EXP-RESNET50-TRANSFER-5SEEDS-008 — ImageNet Frozen Feature Extraction / Head Refinement
> **Cập nhật điều hướng 28/10/2026.** TRANSFER V08 FROZEN — benchmark/Test/Holdout cũ; phân biệt EXP-009 partial fine-tuning. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


**Trạng thái:** Đã chạy 5 seeds, Validation V08, **Final Test V08 686 ảnh**, **historical Real Holdout 48 ảnh**. Nguồn: `EXP-RESNET50-TRANSFER-5SEEDS-008_ANALYSIS.zip` do chủ dự án cung cấp (kiểm tra 20/10/2026).

> **Đính chính tên phương pháp:** EXP-008 sử dụng **ResNet50 pretrained ImageNet**, nhưng **đóng băng backbone trong CẢ Stage 1 và Stage 2**. Do đó đây là **frozen feature extraction + classification-head refinement**, **chưa phải partial-backbone fine-tuning**. Smoke test xác nhận `stage1_backbone_trainable_params=0`, `stage2_backbone_trainable_params=0`. EXP này **khác** `EXP-RESNET50-SCRATCH-5SEEDS-008` (weights=None, V07).

## 1. Dataset, splits và reproducibility

- **Dataset:** `Cashew_dataV08_real_val`, ZIP `CashewData_Split (8).zip` (6.906 ảnh).
- **Train / Validation / Test:** **4.790 / 1.430 / 686**; class names theo **đúng thứ tự model outputs**: `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`.
- Dataset ZIP SHA256: `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`.
- Dataset manifest SHA256: `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b` (**cùng V08 của DenseNet121 Transfer EXP-014**).
- 5 seeds `[42,123,2026,3407,7777]`; mọi Mean ± SD là **sample SD (ddof=1)**, không phải confidence interval.
- 35 ảnh có tên `real_` trong Validation (9 anthracnose, 3 healthy, 18 leaf_miner, 5 red_rust); **prefix là proxy**, chưa chứng minh provenance nguồn chụp.
- Kiểm toán nguồn: **0 exact cross-split duplicates**, **51 pHash candidates** cần review thủ công; 10 đường dẫn ảnh Val nghi tương tự Train đều được 5 seeds dự đoán đúng, nhưng bỏ 10 ảnh này thì Val seed 3407 vẫn **95.35% trên 1.420 ảnh**. Không thể loại trừ leakage cấp độ lá/cây/buổi chụp chỉ bằng pHash.
- 48 ảnh **Real Holdout lịch sử** sau loại 1 ảnh trùng byte với Test từ archive 49 ảnh: Anthracnose 13, Healthy 5, Leaf miner 24, Red rust 6, **0 ảnh true not_cashew_leaf**. Tập này đã được phân tích ở EXP trước → **không phải blind field test mới**.

## 2. Kiến trúc và cấu hình thực tế từ notebook

| Thành phần | Cấu hình nguồn |
|---|---|
| Backbone | `keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")` |
| Backbone training | **Stage 1 frozen, Stage 2 frozen** (0 backbone params trainable) |
| Head | Dropout 0.40 + Dense(5, softmax) |
| Total params | **23.597.957** |
| Input | 224×224 RGB float32, pixel 0–255, bilinear resize |
| Preprocessing | **ResNet50 Caffe mode BGR mean subtraction trong model**, **không /255 ở ngoài** |
| Augmentation train | RandomFlip horizontal, rotation 0.08, zoom 0.10, contrast 0.10 |
| Optimizer | AdamW, weight_decay 1e-4, batch 16 |
| Stage 1 | Head training, LR **3e-4**, tối đa **12 epochs**, EarlyStopping patience 4 |
| Stage 2 | **Vẫn chỉ head training**, LR **3e-5**, tối đa **6 epochs**, EarlyStopping patience 2 |
| Checkpoint selection | `val_loss` nhỏ nhất qua 2 stages, tolerance `1e-6`, tie-break Validation Macro F1 rồi Stage 1 |
| Deployment seed selection | `val_loss` nhỏ nhất qua 5 seed, tie-break F1 rồi seed thấp hơn |

**Smoketest:** `pipeline_smoke.json` báo PASS cho ImageNet init, preprocessing (`max_abs_error=0`), reload model/weights, clean process parity. Đây là thông tin từ artifacts, không phải lần chạy kiểm thử độc lập bằng checkpoint binary trong repo.

**Vì sao Accuracy cao ngay epoch đầu:** Pre-fit Validation Accuracy chỉ **9.23–27.06%** tùy seed, first batch Accuracy thấp; tại cuối epoch 1, head đã qua **300 minibatches** trên 4.790 ảnh và được đánh giá không augmentation/dropout. ImageNet features đã học sẵn + epoch-average Train vs end-of-epoch Val giải thích việc Validation tăng nhanh; **không tự chứng minh leakage/overfitting**.

## 3. Validation (5 seeds, V08 1.430 ảnh)

| Seed | Selected stage / epoch | Val Loss | Val Accuracy | Macro F1 | `real_` Val correct |
|---:|---|---:|---:|---:|---:|
| 42 | Stage 2 / 4 | 0.144610 | 94.90% | 94.74% | 24/35 |
| 123 | Stage 2 / 1 | 0.149440 | 94.90% | 94.66% | 24/35 |
| 2026 | Stage 2 / 2 | 0.144690 | 95.24% | 95.08% | 25/35 |
| **3407** | **Stage 1 / 10** | **0.142581** | **95.38%** | **95.23%** | **26/35** |
| 7777 | Stage 2 / 2 | 0.149725 | 95.03% | 94.84% | 26/35 |

**Mean ± SD:** Val Accuracy **95.09 ± 0.22%**, Macro F1 **94.91 ± 0.23%**, Anthracnose Recall **90.58 ± 1.17%**, Leaf miner Recall **91.82 ± 1.22%**.

- Nhóm `real_` Val: **71.43%** trung bình 5 seeds (125 đúng / 175 lượt đánh giá); không có ground truth `not_cashew_leaf` trong nhóm 35 ảnh.
- 500-image **clean Train Probe**: **97.08 ± 0.46%** so với Val 95.09%; gap **1.99 ± 0.32 điểm %**, không phải dấu hiệu overfitting Train/Val nghiêm trọng nếu chỉ xét metric này.
- Bệnh `→ Healthy` trên Val: **1.33 ± 0.14%** trên 900 ảnh bệnh/seed. Nên báo cáo bên cạnh Accuracy.
- Stage 2 chỉ cải thiện Validation Loss rất ít, và với seed 3407 **Stage 1 tốt hơn Stage 2** nên được chọn hợp lệ.
- 10 pHash Val candidates có đúng 10/10 ở cả 5 seed; Val còn lại vẫn ~94.9–95.35%. Cần mở ảnh xác minh, **không tự cho là data leakage**.

Chi tiết: [per-seed Validation](./aggregate/validation_results_5seeds.csv), [mean ± SD](./aggregate/validation_mean_std_summary.csv), [per-class](./aggregate/per_class_validation_mean_std.csv), [`real_` subgroup](./aggregate/real_validation_by_seed.csv), [diagnostics](./aggregate/overfit_diagnostics_by_seed.csv).

## 4. Final Test V08 — 686 ảnh, có dữ liệu thật trong EXP-008

| Seed | Correct / 686 | Test Accuracy | Macro F1 |
|---:|---:|---:|---:|
| 42 | 641 / 686 | 93.44% | 92.96% |
| 123 | 639 / 686 | 93.15% | 92.59% |
| 2026 | 640 / 686 | 93.29% | 92.77% |
| **3407 (Val-selected)** | **637 / 686** | **92.86%** | **92.35%** |
| 7777 | 641 / 686 | 93.44% | 92.92% |

**Mean ± SD:** **Test Accuracy 93.24 ± 0.24%**, Test Macro F1 **92.72 ± 0.25%**. 28 ảnh `real_` trong V08 Test là subgroup nhỏ, **không được thay thế** test thực địa ngoài nguồn.

Chi tiết: [Test per seed](./aggregate/test_results_5seeds.csv), [Test Mean ± SD](./aggregate/test_mean_std_summary.csv), các confusion matrix theo seed trong `seed_*/test/`.

## 5. Historical Real Holdout — 48 ảnh sau khi loại exact duplicate

| Seed | Correct / 48 | Holdout Accuracy | Holdout Macro F1 (4 classes) |
|---:|---:|---:|---:|
| 42 | 37/48 | 77.08% | 74.98% |
| 123 | 38/48 | 79.17% | 75.43% |
| 2026 | 35/48 | 72.92% | 66.30% |
| **3407 (Val-selected)** | **37/48** | **77.08%** | **71.92%** |
| 7777 | 38/48 | 79.17% | 76.20% |

**Mean ± SD:** Accuracy **77.08 ± 2.55%**; Macro F1 **72.97 ± 4.07%** (chỉ 4 lớp có ground truth).

**Per-class Recall tại deployment seed 3407:** Anthracnose **7/13 = 53.85%**, Healthy **4/5 = 80%**, Leaf miner **22/24 = 91.67%**, Red rust **4/6 = 66.67%**. Anthracnose và Red rust là vấn đề lớn nhất; ảnh bệnh vẫn có thể bị dự đoán Healthy.

- Đánh giá từ cùng **historical Holdout** với DenseNet121-015 nhưng khác mô hình; đây **không phải** một ước lượng accuracy khách quan trên mọi điều kiện thực địa.
- Macro metrics **4 lớp true có support**, không thể suy ra hiệu năng trên `not_cashew_leaf`.
- Không đổi deployment seed sang 123/7777 vì Holdout của các seed đó cao hơn: **seed 3407 được khóa theo Validation loss trước khi test**.

Chi tiết: [Holdout 5 seeds](./real_holdout/aggregate/real_holdout_results_5seeds.csv) · [Holdout Mean ± SD](./real_holdout/aggregate/real_holdout_mean_std.csv) · [Deployment seed confusion](./real_holdout/seed_3407/confusion_matrix.csv) · [Data audit](./real_holdout/dataset_audit.json).

## 6. So sánh với DenseNet121 Transfer 014/015 (cùng V08)

| Metric | ResNet50 Transfer 008 | DenseNet121 Transfer 014 / evaluation 015 |
|---|---:|---:|
| Val Accuracy | **95.09 ± 0.22%** | **97.10 ± 0.20%** |
| Val Macro F1 | **94.91 ± 0.23%** | **96.96 ± 0.21%** |
| Val `real_` Accuracy | **71.43%** | **79.43%** |
| Historical Holdout Acc | **77.08 ± 2.55%** | **80.83 ± 0.93%** |
| Final Test V08 | **93.24 ± 0.24%** | **Chưa chạy cho DenseNet121-014** |

**Không kết luận DenseNet121 vượt ResNet50 đã fine-tune đúng nghĩa:** ResNet50-008 chưa mở backbone, còn DenseNet121-014 đã fine-tune conv5. Phép so sánh hiện tại là **hai pipeline/methods khác nhau trên cùng dataset**, không phải controlled architecture-only comparison.

## 7. Kết luận và next steps

1. **Không xóa/ghi đè EXP-008**: baseline frozen-feature tốt, có đủ Validation/Test/Historical Holdout.
2. Tạo **experiment ID mới** cho ResNet50 **partial fine-tuning thực sự** (ví dụ chỉ mở `conv5_block3_*` không phải BatchNorm, LR ~3e-6), trên cùng V08 và 5 seeds. Không làm Test/Holdout-guided tuning; chọn bằng Validation.
3. Kiểm toán thủ công 51 pHash candidate pairs, nhóm cùng lá/buổi chụp, và lỗi Anthracnose ↔ Red Rust / disease → Healthy.
4. Thu thập **fresh independent field test** từ nhiều cây/phiên/vườn và có lớp `not_cashew_leaf` để xác nhận generalization; 48 ảnh Holdout cũ chỉ dùng làm descriptive historical comparison.
5. **Không thay web deployment**: ứng dụng `Application/` đang ghi dùng DenseNet121 Scratch EXP-009 seed 3407, không tự chuyển sang ResNet50-008.

## Artifact đã/ chưa đưa lên GitHub

- **Đã lưu trong báo cáo:** các bảng 5-seed Validation/Test/Holdout, confusion matrices theo seed, preprocessing/smoke metadata, nguồn và protocol, chẩn đoán Overfit, hình so sánh dạng SVG (tái lập từ CSV).
- **Chưa đưa lên GitHub:** original FULL/ANALYSIS ZIP, file `.keras` / `.h5` >90MB, **original Jupyter notebook**, full per-image prediction logs và original PNG panels; những file này cần import riêng từ bản gốc nếu muốn hoàn chỉnh reproducibility. Không xem metadata checksum là bằng chứng kiểm chứng binary từ repository.

**Source ANALYSIS ZIP SHA256:** `a408af9b540208045f2214c71cb6072b75e8b1442efacd12d88ae510107bc24f`. **Source config SHA256:** `167a748a76308806849a5c7dfe9f64e903c766751ca03cd22b9cea3a084debcc`. **Selection lock SHA256:** `aabd917cb021e6591b45350083ff763b6cf26234451d87a7034b28322a1daece`.

[Audit chi tiết](./ANALYSIS_AND_PROTOCOL_AUDIT.md) · [Tiến độ EXP](../EXPERIMENT_PROGRESS_2026_10.md) · [DenseNet121-014](../EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md).

