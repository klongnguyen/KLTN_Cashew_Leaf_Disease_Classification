# EXP-DENSENET121-SCRATCH-5SEEDS-011 — V07
> **Cập nhật điều hướng 28/10/2026.** LỊCH SỬ / SCRATCH — Không phải model WebApp Transfer EXP-014. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


**Dataset:** `Cashew_dataV07_clean_val`; **Source:** `EXP-DENSENET121-SCRATCH-5SEEDS-011_ANALYSIS.zip` (người dùng cung cấp); **Status:** Validation complete, Final Test pending.

> **Lưu ý khoa học:** Năm hàng `validation_results_5seeds.csv` của EXP-011 **trùng từng giá trị** với bản [EXP-009](../EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL/EXP-DENSENET121-SCRATCH-5SEEDS-009/README.md) đã có trên GitHub; manifest SHA-256 và recipe chính cũng trùng. **Chưa xác minh checkpoint độc lập, vì vậy không coi EXP-011 là bằng chứng cải thiện hay lần train độc lập.**

## Dataset và thay đổi so với EXP-009

| Class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| anthracnose | 948 | 299 | 122 | 1369 |
| healthy | 806 | 213 | 118 | 1137 |
| leaf_miner | 893 | 246 | 132 | 1271 |
| red_rust | 1077 | 323 | 158 | 1558 |
| not_cashew_leaf | 1101 | 314 | 157 | 1572 |
| **TOTAL** | **4825** | **1395** | **687** | **6907** |

- **Manifest SHA-256:** `f9e7d9a39f33bd631b3133a77d0260055bcdfbcd79cbebbb4d87877b893c6d74` (**giống EXP-009**).
- Đếm ảnh có tên bắt đầu `real_`: Train **170**, Val **0**, Test **28** (proxy tên file, không phải phân loại nguồn chụp đã xác minh).
- Audit báo 0 duplicate tuyệt đối cross-split, nhưng 52 ứng viên pHash và 1 ứng viên trùng thumbnail cần rà soát thủ công.
- **Thay đổi từ EXP-009:** chưa tìm thấy thay đổi có bằng chứng về dataset, mô hình, optimizer, augmentation hoặc checkpoint rule. Không được quy kết 011 là một cải tiến độc lập.

## Cấu hình huấn luyện

- **Model:** DenseNet121 Scratch (`weights=None`, `include_top=False`); tổng **7,042,629** tham số theo training summary; head Dropout **0.30**.
- **Input/Preprocess:** ảnh RGB `224×224×3`; DenseNet `preprocess_input` tương đương nằm trong mô hình. Không chuẩn hóa hai lần ở inference.
- **Optimizer:** Adam; learning rate ban đầu `1e-4`; batch size **16**, tối đa 50 epochs; loss SparseCategoricalCrossentropy; không class weighting.
- **Augmentation Train:** horizontal flip, rotation factor `0.04`, zoom factor `0.05`, không brightness/contrast.
- **Callbacks:** chọn best checkpoint theo **min val_loss**; EarlyStopping patience **10**; ReduceLROnPlateau factor **0.5**, patience **4**, min LR `1e-7`.
- **Seed:** `42, 123, 2026, 3407, 7777`; checkpoint và deployment-seed chọn trên **Validation**, không dùng Test.
- **Số liệu:** Mean ± sample SD, `ddof=1`.

## Kết quả Validation — 5 seeds

| Seed | Best epoch | Val Loss | Accuracy | Macro F1 |
|---:|---:|---:|---:|---:|
| 42 | 28 | 0.2070 | 93.55% | 93.27% |
| 123 | 28 | 0.1574 | 95.05% | 94.85% |
| 2026 | 34 | 0.1417 | 95.34% | 95.21% |
| 3407 | 39 | 0.1410 | 95.63% | 95.51% |
| 7777 | 28 | 0.2106 | 93.19% | 93.07% |

| Metric | Mean ± sample SD |
|---|---:|
| Val Accuracy | **94.55 ± 1.11%** |
| Val Macro Precision | **94.69 ± 1.01%** |
| Val Macro Recall | **94.20 ± 1.18%** |
| Val Macro F1 | **94.38 ± 1.13%** |

Checkpoint chọn triển khai theo min val_loss: **seed 3407**, loss **0.140977**.

## Hạn chế, đánh giá và bước tiếp theo

Validation không có file `real_` nên **chưa kiểm tra được domain shift** trong nhóm này. Nhóm 52 ứng viên near-duplicate chưa được xác nhận. Kiểm tra fingerprint/checkpoint và làm rõ vì sao kết quả 009–011 trùng trước khi mô tả là một thí nghiệm mới. Đánh giá tiếp [EXP-012](../EXP-DENSENET121-SCRATCH-5SEEDS-012/README.md) với 35 ảnh `real_` trong Validation; kết quả V07 và V08 **không so sánh trực tiếp như đối chứng cố định**.

## Tình trạng hoàn thành / Khả năng tái lập

- Đã có **5-seed Training + Validation**. `run_final_test=false` trong config; **không có Final Test hay Real-world Holdout hoàn chỉnh** trong ANALYSIS ZIP.
- FULL ZIP, file `.keras`/`.weights.h5`, notebook và ảnh panel không được đẩy trong bản commit tài liệu này. Cần liên kết archive bên ngoài đã xác minh nếu muốn triển khai YOLO.
- Thứ tự class index **theo config thực nghiệm**: `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`. Khi tích hợp bắt buộc tải đúng label map từ model package và thử inference smoke test.
- Không lấy Validation Accuracy hoặc confidence của phân loại ảnh nguyên gốc làm độ chính xác cho pipeline YOLO crop → DenseNet121.

## File kèm trên GitHub

- [Aggregate Validation theo seed](./aggregate/validation_results_5seeds.csv)
- [Mean ± SD](./aggregate/validation_mean_std_summary.csv)
- [Metadata snapshot (chỉ gồm các trường xác minh)](./config_snapshot.json)
