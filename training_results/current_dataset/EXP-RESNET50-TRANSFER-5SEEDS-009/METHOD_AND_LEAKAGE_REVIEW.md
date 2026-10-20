# Kiểm tra kỹ thuật và hạn chế — ResNet50 TRANSFER-009

Nguồn duy nhất cho các số liệu/thao tác trong tệp này: `EXP-RESNET50-TRANSFER-5SEEDS-009_ANALYSIS.zip` (do tác giả cung cấp); giải thích dưới đây là nhận định phương pháp, không phải kiểm định mới của checkpoint binary.

## 1. Kiểm tra phương pháp

- `experiment_config.json`: Stage 1 frozen backbone, LR 1e-4, max 18 epochs. Stage 2 `conv5_block3_*` **non-BatchNorm + head** được train, LR 3e-6, max 8 epochs; optimizer reset. Dropout .4; AdamW WD1e-4; batch16; image224 RGB.
- `pipeline_smoke.json`: PASS, `stage1_backbone_trainable_params=0`, `stage2_backbone_trainable_params=4459520`, 53 BN layers; ImageNet initialization PASS; Caffe preprocessing max abs error=0; model weights reload PASS; inference clean process parity PASS.
- Gọi backbone `training=False` giữ BN frozen trong fine-tune, **không làm tắt gradients qua Conv trainable**. EXP-009 đã cải thiện lỗi frozen toàn bộ Stage2 trong EXP-008.
- Stage2 Val Loss cải thiện so với best Stage1 từ **0.0010 (seed 42)** đến **0.0125 (seed 7777)** nhưng Test/Real Holdout trung bình không cải thiện chắc chắn. **Tác động mở backbone chưa được cô lập**, do Stage1 recipe EXP-008 khác EXP-009.

## 2. Giải thích Val Accuracy cao từ epoch đầu

- Pre-fit Accuracy random head/ResNet ImageNet: 9.23–27.06% theo seed, không phải 95%.
- Stage1 epoch1 Training Accuracy **45.78–53.34%**, Validation Accuracy **78.32–86.08%**: 300 weight updates/head trong một epoch, cộng pretrained visual features; online Train metric trung bình cả epoch và có augmentation/dropout; Val tính bằng cuối epoch, inference/no augmentation.
- Matched inference probes theo 50 minibatches trong epoch1 cho thấy sự tăng dần, không có chỉ báo rằng mô hình biết nhãn ngay từ step0. 100-image probe chỉ là chẩn đoán, **không phải full Validation**.
- Val Recall Red rust **98.35%**, Not-cashew **98.54%** trên 642/1430 ảnh làm overall Accuracy cao; không thay cho class-wise scores.

## 3. Overfit & domain shift

- Clean 500-image Train Probe Accuracy **97.16 ± 0.26%**; Val Accuracy **95.23 ± 0.35%**; gap **1.93 ± 0.20 percentage points**; chưa có chứng cứ severe Train–Val overfit nội bộ.
- Val 35 `real_` tên file **73.71%**; non-real prefix Val ~**95.77%**. Test V08 Accuracy **92.89%**, historical Holdout Accuracy **75.83%**. Khác biệt nguồn/cỡ mẫu/class-mix → **hướng điều tra domain shift**, không chỉ ra cụ thể thiết bị/nền/độ sáng khi chưa có verified source metadata.
- Stage2 Loss có rebound nhẹ ở một số seed, nhất là 3407 và 7777; xem `aggregate/overfit_diagnostics_by_seed.csv`. Không chọn epochs bằng Test hoặc Holdout.

## 4. Leakage & provenance

- Dataset ZIP SHA256: `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`; V08 manifest SHA256 `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b`.
- 0 cross-split exact byte duplicate; 51 pHash candidates giữa splits/labels, **chưa xác minh thủ công**. 10 suspect Val paths đúng 10/10 mỗi seed; Val sau loại subset vẫn từ **94.65% đến 95.56%** tùy seed. **Không xem 10 candidates là xác nhận leakage**, nhưng cần grouped split theo original leaf/orchard/capture.
- Holdout từ archive 49: loại `red_rust/IMG_3244.jpg` trùng byte `test/red_rust/real_red_rust_029.jpg`, còn 48 ảnh, **4 true classes**, đã phân tích nhiều lần trước EXP-009; không mới, không blind.
- Test V08 cũng từng chạy và đánh giá trước khi hình thành EXP-009, mặc dù checkpoint EXP-009 được lựa chọn bằng Validation. Không xem kết quả là final untouched benchmark cho toàn bộ quá trình chọn chiến lược nghiên cứu.

## 5. Các lỗi đáng phân tích

- `val/anthracnose/anthracnose_1486.jpg` → Healthy, confidence sai tối đa **0.9955382**.
- `val/anthracnose/anthracnose_1482.jpg` → Healthy, sai confidence tối đa **0.9943441**.
- `val/red_rust/red_rust_1346.jpg` → Anthracnose, sai confidence tối đa **0.99700016**.
- `val/leaf_miner/leaf_miner_1216.jpg` → Anthracnose, sai confidence tối đa **0.9953909**.
- Các ảnh `anthracnose_1480..1486` thường lặp lại sai nhiều seeds; kiểm tra grouping, nhãn và ảnh tương tự; **không đổi nhãn tự động theo prediction**.
- Val False Healthy: **1.29 ± 0.19%** / 900 ảnh bệnh/seed. Model softmax confidence không phải calibration; nên đo ECE/Brier/reliability diagram trên Validation và test source độc lập sau khi chốt.

## 6. Action items

1. Giữ nguyên EXP-009 và checkpoint seed **2026**, không thay vì Test/Holdout có seed khác tốt hơn.
2. Audit ảnh sai lặp, pHash 51 candidates, capture grouping; nếu sửa dữ liệu tạo **dataset snapshot/EXP ID mới**, không âm thầm sửa V08.
3. Thiết kế ablation xuất phát **same frozen Stage1 checkpoint**, giữ LR/epochs/augmentation/callback giống nhau, so sánh Stage2 frozen vs `conv5_block3_*`. Chọn config bằng Val, **không tune theo historical Holdout**.
4. Thu thập holdout thực địa **mới**, khác vườn/cây/buổi, có đủ `not_cashew_leaf`. Đánh giá deployment seed khóa và YOLO crop→classifier end-to-end.
5. Application web hiện vẫn ghi sử dụng DenseNet121 Scratch EXP-009, **chưa đổi deployment ResNet50-009**.

### Giới hạn lưu trữ
Github snapshot chỉ chứa artifact văn bản/CSV/SVG được import, chưa có original notebook, binary weights, full predictions và tất cả PNG. Original ANALYSIS archive: SHA256 `65b77db413d3b9e0c559686116f023c4d299a666e6533fe905d8d2246ec03762`. Hash gốc không áp dụng cho CSV tái-serialize.

