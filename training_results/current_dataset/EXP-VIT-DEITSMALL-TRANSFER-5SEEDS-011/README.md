# EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011 — Results (23/10/2026)

> **Trạng thái: COMPLETED — Validation-only; NOT PROMOTED.** Cả 5 seed đã hoàn tất. Quyết định này nghĩa là không có checkpoint đạt toàn bộ tiêu chí lựa chọn đã khóa; không đồng nghĩa lỗi huấn luyện.

## 1. Mục tiêu và nguồn

Khảo sát liệu kéo dài recipe **DeiT-Small distilled pretrained ImageNet** hai giai đoạn của EXP-010 có giúp cải thiện hiệu quả phân loại bệnh lá điều. **Đây là một lần huấn luyện 5 seed mới từ pretrained weights**, không tiếp tục fine-tune checkpoint EXP-010.

- **Notebook/mã nguồn:** [EXP-011 source branch](https://github.com/klongnguyen/KLTN_Cashew_Leaf_Disease_Classification/tree/exp/vit-deitsmall-extended-epochs-011) · [Colab notebook](https://github.com/klongnguyen/KLTN_Cashew_Leaf_Disease_Classification/tree/exp/vit-deitsmall-extended-epochs-011/Notebook/ViT/vit_deit_small_transfer_5seeds_011.ipynb).
- **Dữ liệu:** `CashewData_Split (8).zip` / V08 (`train 4790, val 1430, test 686`); giữ nguyên split và SHA256 ZIP `75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7`.
- **5 seed:** `42, 123, 2026, 3407, 7777`; batch size 16; 224×224; 5 lớp (Anthracnose, Healthy, Leaf Miner, Red Rust, Not Cashew Leaf).
- **Stage 1:** backbone frozen; train classification head; tối đa **30 epochs**, Early Stopping patience **7**.
- **Stage 2:** tiếp tục từ Stage 1 best checkpoint **cùng seed**; fine-tune hai encoder blocks cuối + head; tối đa **30 epochs**, patience **7**, 3-epoch warmup + cosine LR trong horizon 30 epochs.
- **Đánh giá:** chỉ Train cập nhật weights, chỉ Validation chọn checkpoint/seed. **Không infer Test V08 hoặc Historical Holdout.** Test member chỉ được đọc cho audit tính toàn vẹn ZIP.

## 2. Kết quả Validation V08

Trung bình ± sample SD trên 5 seed; Disease→Healthy là số ảnh mắc bệnh bị nhận nhầm thành Healthy trên **900 ảnh bệnh Validation**.

| Mô hình | Accuracy | Macro F1 | Disease→Healthy / 900 |
|---|---:|---:|---:|
| DeiT-Tiny A EXP-008 (đối chứng) | **94,41 ± 0,73%** | **94,29 ± 0,70%** | 20,8 ± 4,44 |
| DeiT-Small EXP-010 Stage 1 | 86,27 ± 1,34% | 85,95 ± 1,43% | 17,8 ± 5,45 |
| DeiT-Small EXP-010 Stage 2 | 92,81 ± 1,00% | 92,74 ± 0,96% | 21,2 ± 6,72 |
| DeiT-Small EXP-011 Stage 1 | 88,90 ± 0,48% | 88,76 ± 0,48% | 20,2 ± 5,36 |
| **DeiT-Small EXP-011 Stage 2** | **93,45 ± 1,28%** | **93,40 ± 1,30%** | **19,2 ± 5,26** |

Chênh lệch EXP-011 trừ EXP-010 trên **cùng seed**: Stage 1 Macro F1 **+2,81 pp**; Stage 2 Macro F1 **+0,66 pp**, Acc **+0,64 pp**, D2H trung bình **−2 ảnh/900**. EXP-011 vẫn kém DeiT-Tiny A khoảng **0,89 pp Macro F1**. Việc tăng giới hạn epoch, patience, và độ dài cosine schedule diễn ra đồng thời nên không quy kết nhân quả cho riêng số epochs.

### Stage 2 theo seed

| Seed | Validation Accuracy | Macro F1 | Disease→Healthy /900 | Best epoch | Guardrails |
|---|---:|---:|---:|---:|---|
| 42 | 91,96% | 91,84% | 22 | 6 | Không đạt |
| 123 | 94,06% | 94,07% | 14 | 27 | Không đạt |
| 2026 | 92,59% | 92,56% | 27 | 19 | Không đạt |
| 3407 | 93,43% | 93,37% | 16 | 15 | Không đạt |
| 7777 | **95,24%** | **95,17%** | 17 | 27 | Không đạt |

**Selection lock:** `selection/validation_selection_lock.json` → `status: NOT PROMOTED`, `selected: null`; không đổi model WebApp hoặc seed được khóa của EXP-008.

## 3. Tài liệu và dữ liệu kết quả

- [Scientific report](analysis/SCIENTIFIC_REPORT.md) · [Failure analysis](analysis/FAILURE_ANALYSIS.md) · [Giới hạn nghiên cứu](analysis/LIMITATIONS.md).
- [Metrics 5 seeds](aggregate/validation_5seeds.csv) · [Mean ± SD](aggregate/validation_mean_std.csv) · [Đối chiếu EXP-010 theo cặp](aggregate/paired_exp011_minus_exp010.csv) · [Metrics per class](aggregate/per_class_validation.csv).
- [Learning curves](figures/accuracy_loss_curves_5seeds.png) · [Macro F1 by seed](figures/macro_f1_by_seed.png) · [Confusion matrices 2×5](figures/confusion_matrices_5seeds.png) · [Normalized CM](figures/confusion_matrices_normalized_5seeds.png) · [Recall/Precision/F1 theo lớp](figures/per_class_precision_recall_f1.png).
- Mỗi `seed_<seed>/` có `metrics_stage1/2.json`, `history_stage1/2.csv`, `classification_report_val_stage1/2.csv`, `confusion_matrix_val_stage1/2.csv`, `errors_val_stage1/2.csv` và `seed_complete.json`.
- [Analysis ZIP gốc trên Drive](https://drive.google.com/file/d/1AXddpdWxuuRDYEJU20GWtf3LzsvQSR4s/view), [toàn bộ thư mục Colab/Drive](https://drive.google.com/drive/folders/1nFBgCJwNnZZ2i6-nZdtFOHkyLTPw4KhU).
- [Ảnh lỗi seed 123 (3,7 MB) trên Drive](https://drive.google.com/file/d/1mt8tJJWkDwZPqP6f8-07B9eQ7yOnG431/view). PNG lớn này được dẫn link thay vì commit vào GitHub.

## 4. Hạn chế và quy tắc diễn giải

- Cùng một tập Validation cho cả 5 seed, nên sample SD **không** phải độ bất định từ 5 tập dữ liệu độc lập.
- Audit có **226 cặp pHash candidates cross-split** cần truy vết lá/cây/buổi chụp để xác minh; chưa đồng nghĩa có 226 trường hợp leakage.
- 35 ảnh có prefix `real_` trong Validation chưa thay thế một tập kiểm định thực địa mới, độc lập đủ 5 lớp.
- `not_cashew_leaf` Recall Stage 2 còn hạn chế (khoảng 85,48% trung bình); không được tự đổi nhãn từ kết quả dự đoán.
- **Không dùng Test/Holdout để tối ưu, chọn seed, threshold hay tuyên bố cải thiện.** EXP-011 chưa có final Test/field test mới; WebApp giữ nguyên.

_Artifacts được sao lưu từ kết quả đã xuất trên Google Drive ngày 23/10/2026. Checkpoint `.keras`, dữ liệu gốc, cache và log runtime lớn được giữ trên Drive, không commit vào repository._
