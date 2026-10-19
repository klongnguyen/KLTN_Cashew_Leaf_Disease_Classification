# EXP-DENSENET121-SCRATCH-5SEEDS-012 — V08 Real Validation

**Dataset:** `Cashew_dataV08_real_val`; **Status:** Validation complete, Final Test pending.

> **Đặt sai tên ZIP:** Source là `EXP-DENSENET121-SCRATCH-5SEEDS-01_ANALYSIS.zip`; `experiment_config.json` ghi experiment `...-01`. Người dùng xác nhận notebook/run này là **phiên bản 012**; vì vậy repository chuẩn hóa thành **EXP-012**, đồng thời giữ dấu vết ID gốc trong `config_snapshot.json`. Không ghi đè EXP-001.

## Thay đổi dataset EXP-011 (V07) → EXP-012 (V08)

| Tập | V07 | V08 | Chênh lệch |
|---|---:|---:|---:|
| Train | 4825 | **4790** | −35 |
| Validation | 1395 | **1430** | +35 |
| Test | 687 | **686** | −1 |
| **Tổng** | **6907** | **6906** | **−1** |

- **35** ảnh tên bắt đầu `real_` được chuyển từ Train sang Validation: phân bố `real_` V08 là Train **135**, Val **35**, Test **28**.
- Config liệt kê ảnh Test loại đi: `test/not_cashew_leaf/not_cashew_leaf_1514.JPG`; lý do nghi near-duplicate cần đối chiếu audit gốc, không tự coi đã loại hết leakage.
- SHA-256 manifest V08: `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b` (khác V07).
- Dataset audit V08: **0** cặp duplicate tuyệt đối, **51** ứng viên pHash vẫn cần rà soát. Tập Val lúc này gồm 35 ảnh `real_` và 1395 ảnh còn lại.
- **Thay đổi cấu hình so với EXP-011:** theo metadata hai ZIP, **recipe chính không đổi** (batch, LR, optimizer, augment, dropout, checkpoint). Khác biệt có cơ sở là dataset/split.

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
| 42 | 40 | 0.1630 | 94.62% | 94.47% |
| 123 | 28 | 0.1996 | 93.57% | 93.47% |
| 2026 | 21 | 0.1752 | 94.13% | 93.99% |
| 3407 | 27 | 0.1709 | 95.17% | 95.03% |
| 7777 | 29 | 0.2145 | 92.45% | 92.07% |

| Metric | Mean ± sample SD |
|---|---:|
| Val Accuracy | **93.99 ± 1.05%** |
| Val Macro Precision | **94.16 ± 0.99%** |
| Val Macro Recall | **93.58 ± 1.20%** |
| Val Macro F1 | **93.81 ± 1.13%** |

**Seed triển khai = 42**, được chọn theo **min val_loss = 0.162995**. Seed 3407 có Val Accuracy cao hơn nhưng không được đổi quy tắc chọn seed theo một chỉ số khác sau khi xem kết quả.

## Phân tích domain shift: nhóm ảnh `real_` trong Validation

Dựa vào `seed_*/validation/predictions.csv` của ANALYSIS ZIP: 

| Seed | Đúng/Tổng real_ | Accuracy real_ | Đúng/Tổng khác | Accuracy khác |
|---:|---:|---:|---:|---:|
| 42 | 22/35 | 62.86% | 1331/1395 | 95.41% |
| 123 | 23/35 | 65.71% | 1315/1395 | 94.27% |
| 2026 | 23/35 | 65.71% | 1323/1395 | 94.84% |
| 3407 | 25/35 | 71.43% | 1336/1395 | 95.77% |
| 7777 | 24/35 | 68.57% | 1298/1395 | 93.05% |

- **Seed 42:** ảnh `real_` chỉ **22/35 = 62.86%**, còn ảnh khác **1331/1395 = 95.41%**; chênh **32.55 điểm phần trăm**.
- Với nhóm `real_` của seed 42, Anthracnose đúng **3/9**; Leaf miner đúng **12/18**; Red rust **4/5**; Healthy **3/3**.
- Đây là dấu hiệu mạnh của **domain shift hoặc khác biệt dữ liệu/nhãn**, chưa chứng minh nguyên nhân gốc. **35** ảnh là mẫu nhỏ và thiếu lớp `not_cashew_leaf`, không phải đánh giá triển khai ngoài hiện trường toàn diện.

## Hạn chế và kế hoạch bước tiếp

1. Kiểm tra 51 cặp pHash nghi trùng, xác minh các ảnh `real_` và label; làm rõ ảnh Test V08 bị loại.
2. Xác minh pipeline input (resize, RGB, preprocessing trong model) và thứ tự nhãn khi tích hợp YOLO; kiểm thử ảnh crop so với ảnh nguyên.
3. Đóng băng cấu hình rồi mới chạy **Final Test V08 686 ảnh** trên đủ 5 seeds; không dùng Test để tuning.
4. Đánh giá tập holdout thực tế **49 ảnh/4 lớp** nếu chưa dùng cho quyết định mô hình; nếu từng dùng để tuning cần holdout mới.
5. Xem EXP-012 là **baseline kỹ thuật có điều kiện** cho YOLO→DenseNet, **chưa chứng minh tổng quát hóa trong triển khai thực tế**.

## Tình trạng hoàn thành / Khả năng tái lập

- Đã có **5-seed Training + Validation**. `run_final_test=false` trong config; **không có Final Test hay Real-world Holdout hoàn chỉnh** trong ANALYSIS ZIP.
- FULL ZIP, file `.keras`/`.weights.h5`, notebook và ảnh panel không được đẩy trong bản commit tài liệu này. Cần liên kết archive bên ngoài đã xác minh nếu muốn triển khai YOLO.
- Thứ tự class index **theo config thực nghiệm**: `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`. Khi tích hợp bắt buộc tải đúng label map từ model package và thử inference smoke test.
- Không lấy Validation Accuracy hoặc confidence của phân loại ảnh nguyên gốc làm độ chính xác cho pipeline YOLO crop → DenseNet121.

## File kèm trên GitHub

- [Aggregate Validation theo seed](./aggregate/validation_results_5seeds.csv)
- [Mean ± SD](./aggregate/validation_mean_std_summary.csv)
- [Accuracy theo nhóm real_/khác](./aggregate/real_validation_by_seed.csv)
- [Metadata snapshot (ghi ID nguồn gốc)](./config_snapshot.json)
