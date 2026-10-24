# Dataset Changelog

> **DenseNet121 EXP-016 (22/10/2026):** **KHÔNG tạo dataset V09, không thay đổi V08.** Dùng 686 ảnh Test V08 (Train 4.790/Val 1.430/Test 686), cùng dataset SHA256 `75c5928b...` và manifest SHA256 `1810d7e6...` như EXP-014. EXP-016 **chỉ đánh giá 5 checkpoint 014 đã khóa bằng Validation**, Final Test mean **95,13 ± 0,51%**. [Báo cáo](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md). Cần lưu ý historical Holdout EXP-015 **không phải fresh blind test** và 51 pHash candidates V08 chưa review hết.


> **EXP-VIT-TRANSFER-XGBOOST-008 (21/10/2026) không tạo/chỉnh sửa dataset mới:** kế thừa V08 ZIP SHA256 `75C5928B...` (**Train 4790 / Val 1430 / Test 686**). EXP-008 chỉ fit Logistic Regression/XGBoost trên **DeiT embeddings từ checkpoint EXP-007**, không retrain backbone hoặc đổi split. Audit nguồn ghi 0 ảnh trùng byte cross-split, 1 dHash candidate **cần review**, và **group leakage chưa xác minh**. [Báo cáo](./training_results/current_dataset/EXP-VIT-TRANSFER-XGBOOST-008/README.md). Test/Holdout **chưa chạy trong EXP-008**.


> **ResNet50 TRANSFER-009 (20/10/2026) tiếp tục sử dụng V08 không chỉnh sửa:** Train **4.790**, Validation **1.430**, Test **686**, cùng dataset ZIP SHA256 `75c5928b...` và manifest SHA256 `1810d7e6...` với DenseNet121-014 và ResNet50-008. **Không tạo V09**. 51 pHash candidate pairs vẫn cần review; real Holdout 49 ảnh đã loại 1 ảnh trùng Test, còn 48 ảnh, 4 lớp có ground truth. [EXP-009](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-009/README.md). So sánh với 008 là **method comparison**, không phải dataset update hay one-factor fine-tuning ablation.


> **ResNet50 TRANSFER-008 (20/10/2026) dùng nguyên V08, không chỉnh dataset:** `Cashew_dataV08_real_val` Train **4.790**, Validation **1.430**, Test **686**; cùng manifest SHA256 `1810d7e6...` với DenseNet121-014. 51 pHash candidate pairs V08 vẫn chưa được xác minh; 35 ảnh `real_` Val chỉ là proxy. Historical Holdout (49 ảnh gốc) loại **1 ảnh trùng byte với Test** còn 48 ảnh, **thiếu true not_cashew_leaf**. [Báo cáo Transfer-008](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-008/README.md). **Không nhầm với ResNet50 SCRATCH-008 (V07)**.


> **ResNet50 Scratch EXP-006/007/008 dùng dataset hiện có, không tạo V09:** EXP-006 trên **V05 (4.822/1.402/687)**; EXP-007 và 008 trên **V07 (4.825/1.395/687)**, cùng manifest `f9e7d9a3...`. Không đối chiếu Val V05 và V07 như ablation một biến; 007/008 có source predictions/histories trùng nhau. **52 pHash candidates V07** cần xác minh. [Audit](./training_results/current_dataset/RESNET50_SCRATCH_006_008_AUDIT_2026_10.md).


> **EXP-014/015 dùng nguyên V08, không tạo phiên bản dataset mới:** EXP-014 ImageNet Transfer 5 seeds (Train 4.790 / Val 1.430 / Test 686, manifest `1810d7e6...`); EXP-015 chỉ đánh giá 5 checkpoint EXP-014 trên Real Holdout 49 ảnh gốc, **loại 1 ảnh trùng byte với Test**, còn **48 ảnh thuộc 4 lớp có ground truth**. Holdout này đã được kiểm tra trong EXP-013 nên là **historical exposed**, không phải blind test. Chi tiết: [014](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md) · [015](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md). 51 cặp pHash nghi trùng trong V08 vẫn cần audit thủ công.


> **Cập nhật V07/V08 ngày 19/10/2026:** xem [experiment progress](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md). V05 là official Test đa mô hình; V07/V08 là các dataset DenseNet mới hơn.

Tài liệu này theo dõi các phiên bản classification dataset để tránh trộn lẫn kết quả giữa những split khác nhau.

## Current development — Cashew_dataV05

**First recorded experiments:** 2026-10-03 → 2026-10-04

V05 giữ nguyên tổng số ảnh và class counts của V04:

```text
Train      4,822
Validation 1,402
Test         687
Total      6,911
```

Tuy nhiên, ResNet50 experiment metadata ghi split fingerprints khác V04 ở Train và Validation:

- Train V04: `45a124724d96084add52d3781ff28c0d93adabc6adf4d67067327aab16ea92c7`
- Train V05: `7443257702dbba7a392a28b497a49933f5cfac7e187e991043007f8e309a795f`
- Val V04: `7529e4e0c07af37e1a9c4a366f05846cf72dea530d5f2121da5acb83c64782a0`
- Val V05: `ea4bba016c9f3abb69c0bf6f7deb3d502929707edfbeafa89f36898b7747e279`

The Test fingerprint is unchanged across V04 and V05:

`ee49ef59c791a7519c172316afd48554f92fbb1b4f47d11b57c82d6ae1624dc4`

All three latest V05 archives have identical normalized split/class/filename manifests. V05 đã hoàn tất locked Test chính thức cho cả ba mô hình.


## V07 — Cashew_dataV07_clean_val (DenseNet EXP-009/011)

- 6,907 ảnh: Train **4,825**, Val **1,395**, Test **687**.
- Tên file `real_` theo split: Train 170, Validation 0, Test 28.
- SHA-256 manifest: `f9e7d9a39f33bd631b3133a77d0260055bcdfbcd79cbebbb4d87877b893c6d74`.
- 009/011 có kết quả 5 seeds Validation trùng nhau; không khẳng định 2 run độc lập khi chưa xác minh checkpoint.
- Không có Real Validation tách riêng để đo domain shift.

## V08 — Cashew_dataV08_real_val (DenseNet EXP-012)

- 6,906 ảnh: Train **4,790**, Val **1,430**, Test **686**.
- So với V07, 35 ảnh tên `real_` chuyển từ Train sang Validation, và loại 1 ảnh Test `not_cashew_leaf_1514.JPG` được liệt kê trong source config.
- Tên file `real_`: Train 135, Val 35, Test 28.
- SHA-256 manifest: `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b`.
- Audit có 51 cặp pHash nghi tương tự chưa xác nhận thủ công. Final Test EXP-012 `run_final_test=false`.
- Validation Accuracy seed 42 trên nhóm real_: **22/35 = 62.86%**, nhóm còn lại **1331/1395 = 95.41%**. Đây là dấu hiệu domain shift, không chứng minh nguyên nhân chỉ do ánh sáng/nền.
- Test V08 đã thay đổi một ảnh so với V07, nên không được quy so sánh là cố định 100% cùng Test.

## Previous completed benchmark — Cashew_dataV04

**Ngày khóa phiên bản:** 2026-09-22

V04 được tạo sau khi tiếp tục:
- loại ảnh mờ/chất lượng thấp;
- loại hoặc điều chỉnh các trường hợp có nguy cơ data leakage giữa Train/Validation/Test;
- giữ nguyên nguyên tắc nhóm các ảnh cùng nguồn/cùng lá/các biến thể liên quan trong cùng split khi cần để hạn chế leakage.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 945 | 294 | 122 | 1,361 |
| `healthy` | 806 | 225 | 118 | 1,149 |
| `leaf_miner` | 893 | 249 | 132 | 1,274 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,077 | 320 | 158 | 1,555 |
| **TOTAL** | **4,822** | **1,402** | **687** | **6,911** |

## Previous — Cashew_dataV03

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 1,096 | 313 | 156 | 1,565 |
| `healthy` | 818 | 225 | 128 | 1,171 |
| `leaf_miner` | 919 | 262 | 131 | 1,312 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,115 | 319 | 159 | 1,593 |
| **TOTAL** | **5,049** | **1,433** | **731** | **7,213** |

## V03 → V04

| Class | V03 | V04 | Change |
|---|---:|---:|---:|
| `anthracnose` | 1,565 | 1,361 | -204 |
| `healthy` | 1,171 | 1,149 | -22 |
| `leaf_miner` | 1,312 | 1,274 | -38 |
| `not_cashew_leaf` | 1,572 | 1,572 | +0 |
| `red_rust` | 1,593 | 1,555 | -38 |
| **TOTAL** | **7,213** | **6,911** | **-302** |

V04 có ít hơn **302 ảnh**. Mức giảm không được diễn giải là mất dữ liệu đơn thuần; mục tiêu là tăng chất lượng benchmark bằng việc loại dữ liệu chất lượng thấp và giảm rủi ro leakage.

## Quy tắc so sánh

- Không so sánh trực tiếp metric V03 với V04 như một controlled architecture comparison.
- So sánh model chính thức phải dùng cùng dataset version.
- V05 is an official completed multi-model locked-Test benchmark; DenseNet V07/V08 are later development snapshots.
- Test Set của từng version được khóa sau khi version đó được chốt.
