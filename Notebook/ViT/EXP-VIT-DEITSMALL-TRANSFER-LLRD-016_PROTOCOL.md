# EXP-VIT-DEITSMALL-TRANSFER-LLRD-016 — protocol và trạng thái

**Trạng thái:** notebook đã dựng và kiểm tra cục bộ trên CPU với dữ liệu giả; **chưa huấn luyện trên dữ liệu thật**. Chưa có Accuracy, F1, checkpoint hay kết quả Test/Holdout nào cho EXP-016.

- Notebook: `Notebook/ViT/vit_deit_small_transfer_llrd_016_256px.ipynb`
- Script dựng: `Notebook/ViT/build_vit_transfer_016.py` (đọc notebook EXP-015, giữ nguyên mọi cell dùng chung, chỉ thay các cell mang công thức Stage 2).
- Kế hoạch gốc: `KE_HOACH_CAI_TIEN_VIT_SAU_EXP-015.md`, mục "EXP-016".

## Yếu tố duy nhất thay đổi so với EXP-015

Công thức **Stage 2**: LR phẳng 1e-5 cho 4 khối cuối được thay bằng LR giảm dần theo khối, và số khối mở trở thành biến khảo sát.

| | EXP-015 (baseline) | EXP-016 |
|---|---|---|
| Khối encoder mở | 4 cuối + LayerNorm cuối | N ∈ {4, 8, 12} cuối + LayerNorm cuối |
| LR khối i (i = 0…11) | 1e-5 phẳng | `base_lr × 0.75^(11 − i)` |
| LR LayerNorm cuối, hai đầu phân loại | 1e-5 | `base_lr` |
| `base_lr` | 1e-5 | {3e-5, 5e-5} |
| Embedding (patch, position) | đóng băng | đóng băng (mọi cấu hình) |
| Optimizer Stage 2 | `AdamW` | `LayerwiseLRAdamW` (lớp con `AdamW`, ghi đè `update_step`) |

Giữ nguyên: dataset V13, augmentation, Stage 1 (LR 3e-4, tối đa 12 epoch), batch 16, Dropout 0,40, weight decay 1e-4, EarlyStopping patience 8 và ReduceLROnPlateau patience 4 ở Stage 2, seed, luật chọn checkpoint (val_loss thấp nhất, tie-break macro-F1). *Fairness contract* (cell 03a) kiểm các giá trị này; chỉ `stage2.lr` được khai báo là sai khác có chủ đích.

Weight decay vẫn dùng `base_lr` chưa nhân hệ số (Keras áp dụng nó ngoài `update_step`). Độ co mỗi bước khoảng 1e-4 × 5e-5 = 5e-9, không đáng kể.

## Hai pha (đổi `PHASE` ở cell 03)

### `SCREEN` — sàng lọc, chỉ Train + Validation

- 7 cấu hình × 3 seed (42, 123, 2026): lưới 3 × 2 = 6 cấu hình **cộng cấu hình `B0_flat_n04_lr1e-05`** tái lập công thức EXP-015 ngay trong notebook này. B0 là phép đối chứng cùng mã nguồn, cùng runtime; nó cũng kiểm tra rằng đường mã LLRD với hệ số 1,0 cho kết quả cùng cỡ với EXP-015. Nếu muốn dùng bảng Validation đã lưu của EXP-015 thay vì chạy lại B0, đặt `BASELINE_015_VALIDATION_CSV`.
- Stage 1 huấn luyện **một lần mỗi seed**; mọi cấu hình bắt đầu Stage 2 từ cùng checkpoint Stage 1. So sánh theo cặp, giảm phương sai và tiết kiệm thời gian. (Hệ quả: Stage 1 của sàng lọc không phải 7 lần huấn luyện độc lập.)
- Không đọc Test, không đọc Holdout. `RUN_FINAL_TEST` và `RUN_REAL_HOLDOUT` tắt trong pha này, và khóa sàng lọc ghi `test_and_holdout_read: false`.
- Đầu ra: `screening/screening_unit_results.csv`, `screening/screening_config_summary.csv`, `screening/screening_decision.json`, `screening/screening_val_loss_by_config.png`, `screening_selection_lock.json`, và `<EXP>_ANALYSIS.zip` (không chứa checkpoint, tự upload lên Drive nếu chạy Colab).

**Quy tắc quyết định (kế hoạch mục 3, chỉ Validation):**

1. Một cấu hình qua cổng nếu so với baseline: val_loss trung bình giảm ≥ 0,005, val macro-F1 không giảm quá 0,003 (0,3 điểm) và `false_healthy_rate` trung bình không tăng.
2. Nếu có nhiều cấu hình qua cổng, lấy các cấu hình có val_loss trong 0,005 so với tốt nhất, rồi chọn cấu hình đơn giản nhất (ít khối mở hơn, rồi `base_lr` thấp hơn).
3. Nếu không cấu hình nào qua cổng: `decision = KEEP_INCUMBENT`. Đó là một kết quả hợp lệ ("LLRD không cải thiện công thức EXP-015") và pha `CONFIRM` từ chối chạy, trừ khi đặt `CONFIRM_CONFIG_OVERRIDE` có chủ đích.

Nhóm `real_` của Validation chỉ có 59 ảnh (12 ảnh healthy) nên `real_subgroup_accuracy` được ghi nhưng **không** tham gia quy tắc.

### `CONFIRM` — xác nhận 5 seed

Đọc cấu hình thắng từ `screening_selection_lock.json` (cùng runtime), hoặc từ `CONFIRM_CONFIG_OVERRIDE` / `SCREENING_LOCK_PATH_OVERRIDE`. Sau đó chạy đúng quy trình EXP-015: 5 seed, khóa Validation, Final Test, Holdout 09 (mô tả), xuất model và gói kết quả V2. Mã thực nghiệm của hai pha giống nhau, chỉ khác `PHASE`, nên `rev8` trong hai EXP ID trùng nhau:

- `EXP-VIT-DEITSMALL-TRANSFER-LLRD-SCREEN-016-256px-<rev8>`
- `EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-256px-<rev8>`

## Mục tiêu so với baseline (kế hoạch mục 1; ngưỡng Holdout chỉ để đánh giá cuối)

| Tiêu chí | EXP-015 | DenseNet121-029 | Mục tiêu |
|---|---|---|---|
| Test macro-F1 (%) | 95,92 ± 0,63 | 95,89 ± 0,52 | giữ ≥ 95,4, SD ≤ 0,5 |
| Holdout macro-F1 (%) | 77,05 ± 2,13 | 81,67 ± 1,24 | ≥ 80,2 |
| Validation val_loss (TB 5 seed) | 0,098 | 0,075 | ≤ 0,085 |

Không có gì đảm bảo đạt các mục tiêu này; nếu LLRD không giúp, kết quả đó được báo cáo đúng như vậy.

## Kiểm tra đã làm (cục bộ, TensorFlow 2.21.0, Keras 3.15.1, KerasHub 0.32.0, CPU)

Các cổng trong cell smoke (dừng huấn luyện nếu lỗi):

- Số tham số mở theo độ sâu: 4 khối = 7.098.624 (khớp EXP-015), 8 và 12 khối theo công thức `N × 1.774.464 + 768`; embedding luôn đóng băng.
- **Cổng LLRD:** cùng gradient, cùng trọng số khởi đầu, so một bước `AdamW` thường với `LayerwiseLRAdamW`; tỉ lệ độ dài bước từng biến phải bằng hệ số `0.75^(11 − i)` trong 2%. Bước Adam tuyến tính theo LR nên đây là phép đo trực tiếp, không phụ thuộc cách Keras cài đặt nội bộ.
- Lưu/nạp lại `.weights.h5` và `.keras` khi optimizer là `LayerwiseLRAdamW`.
- Toàn bộ cổng cũ của EXP-015 (chép tensor, nội suy position embedding, preprocessing, augmentation, photometric).

Chạy thử đầu-cuối trên CPU (dữ liệu giả 30 ảnh Train / 20 ảnh Validation, trọng số "pretrained" ngẫu nhiên, 1 + 2 epoch, 2 seed, 3 cấu hình): pha `SCREEN` chạy hết (Stage 1 dùng chung, các đơn vị Stage 2, bảng tổng hợp, quyết định, khóa sàng lọc, ZIP) và pha `CONFIRM` chạy hết (huấn luyện, khóa Validation, Final Test). Cổng LLRD đo được hệ số 0,0422 (khối 1) … 0,75 (khối 11) … 1,0 (khối 12, LayerNorm, đầu phân loại), khớp `0.75^(11 − i)`. Các con số Accuracy của lần chạy đó vô nghĩa. Các cell xuất model, Holdout, thống kê V2 và nén gói kết quả không có thay đổi logic so với EXP-015 (chỉ đổi tên) và **chưa được chạy lại** trong lần kiểm tra này.

## Chưa kiểm tra được (cần Colab GPU)

- Thời gian thực: 7 cấu hình × 3 seed. Ước lượng thô theo log EXP-015 (190–320 giây mỗi seed cho cả hai stage ở N = 4): Stage 2 của N = 12 chậm hơn vì lan truyền ngược qua cả 12 khối, nên pha SCREEN có thể mất khoảng 1–1,5 giờ trên Colab GPU. Đây là ước lượng, chưa đo.
- `jit_compile=False` được giữ nguyên như EXP-015 nên lớp con `AdamW` không đụng tới XLA; chưa kiểm trên GPU.
- Tính tất định trên GPU chỉ ở mức best effort; kết quả hai lần chạy cùng cấu hình có thể lệch nhỏ.

## Việc chưa nằm trong notebook này

EXP-017 (label smoothing, dropout backbone, mixup), EXP-018 (trung bình checkpoint / EMA), EXP-019 (Stage 1 dài hơn), EXP-020 (oversample `real_`) và Track B (đối chứng DenseNet121) là các thí nghiệm riêng với EXP ID mới, chưa có notebook. Quyết định cần xác nhận ở kế hoạch mục 9 (có chạy Track B không; có thu hẹp lưới không) vẫn mở: lưới EXP-016 trong notebook giữ nguyên đầy đủ 3 × 2.
