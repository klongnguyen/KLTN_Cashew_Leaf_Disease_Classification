# EXP-RESNET50-TRANSFER-5SEEDS-009 — ImageNet ResNet50 Partial Fine-tuning, V08
> **Cập nhật điều hướng 28/10/2026.** TRANSFER V08 PARTIAL FINE-TUNING — seed chọn bằng Validation, Test/Holdout đã tiếp xúc. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


> **Trạng thái:** 5-seed Training/Validation **COMPLETED**; Final Test V08 **COMPLETED**; Real Holdout lịch sử 48 ảnh **COMPLETED**; **fresh independent real-world test NOT RUN**. Nguồn: gói `EXP-RESNET50-TRANSFER-5SEEDS-009_ANALYSIS.zip` do tác giả cung cấp và kiểm tra ngày 20/10/2026. **Không nhầm với ResNet50 Scratch-009 (nếu có) hoặc ResNet50-TRANSFER-008 frozen-only.**

**Tóm tắt:** EXP-009 mở thật `conv5_block3_*` không phải BatchNorm ở Stage 2: **4.459.520 tham số backbone trainable**, khác [EXP-008 frozen ImageNet](../EXP-RESNET50-TRANSFER-5SEEDS-008/README.md) (**0 tham số backbone** ở cả hai giai đoạn). Mặc dù Validation tăng nhẹ so với EXP-008, Test và Holdout lịch sử trung bình **không cải thiện**. Accuracy cao sớm trên Validation **không đủ để khẳng định leakage**; ImageNet pretrained + 300 training steps mỗi epoch giải thích việc học nhanh.

## 1. Dataset cố định và giao thức

- Dataset **Cashew_dataV08_real_val**, archive `CashewData_Split (8).zip`; **4.790 Train / 1.430 Validation / 686 Test = 6.906 ảnh**.
- Dataset ZIP SHA256: `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`.
- Dataset manifest SHA256: `1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b`.
- Đúng thứ tự 5 labels trong output: **`anthracnose`, `healthy`, `leaf_miner`, `red_rust`, `not_cashew_leaf`**.
- Seeds cố định: `[42,123,2026,3407,7777]`; báo cáo **mean ± sample standard deviation (ddof=1)**.
- Nhóm tên `real_`: **135 Train / 35 Validation / 28 Test**; tiền tố chỉ là proxy, **không chứng minh nguồn ảnh chụp thực địa độc lập**.
- Dataset audit xác nhận **0 exact cross-split duplicates**, nhưng **51 pHash near-similarity candidate pairs** chưa review tất cả theo lá/cây/session. Không kết luận không có leakage.
- 10 ảnh Val thuộc nhóm pHash nghi tương tự Train **đúng 10/10 với cả 5 seed**. Khi loại 10 ảnh ấy ra khỏi tính toán mô tả, seed3407 còn **1.357/1.420 = 95,56%**; chính nhóm 10 ảnh chưa giải thích mức Accuracy cao.
- Real Holdout archive 49 ảnh, **1 ảnh trùng byte với Test được loại**, còn **48 ảnh** gồm Anthracnose **13**, Healthy **5**, Leaf miner **24**, Red rust **6**, **không có ground-truth not_cashew_leaf**. Đã xem và phân tích qua các EXP trước → **historical exposed**, không phải blind field test mới.
- Test V08 đã được phân tích từ EXP-008 trước khi thiết kế EXP-009: có locked checkpoint selection nội bộ nhưng **không thể gọi là Test mù hoàn toàn của toàn bộ lịch sử nghiên cứu**.

## 2. Cấu hình và kiểm tra kỹ thuật

| Thành phần | EXP-009 |
|---|---|
| Backbone | `tf.keras.applications.ResNet50(weights="imagenet", include_top=False, pooling="avg")` |
| Stage 1 | Frozen backbone; chỉ train classification head; **LR 1e-4**, tối đa **18 epochs** |
| Stage 2 | **Mở `conv5_block3_*` non-BatchNorm + head**, các lớp khác frozen; **LR 3e-6**, tối đa **8 epochs** |
| Backbone trainable params | Stage 1 **0** → Stage 2 **4.459.520** |
| Total parameters | **23.597.957** |
| Image | RGB 224×224, pixel float32 **0–255**, bilinear resize |
| Preprocessing | ResNet50 **Caffe BGR mean subtraction trong model**, **không /255 ngoài model** |
| Optimizer | AdamW, weight decay **1e-4**, batch **16**, fresh optimizer per stage |
| Head | Global Average Pooling → Dropout **0.40** → Dense(5, softmax) |
| Augmentation (Train only) | HorizontalFlip, Rotation **0.08**, Zoom **0.10**, Contrast **0.10** |
| BatchNorm | `training=False` khi gọi backbone, BN frozen, **Conv trainable vẫn có gradients** |
| Checkpoint rule | **Minimum Val Loss** qua 2 stages, tie-break Val Macro F1, rồi Stage 1 |
| Seed deployment | **Minimum Val Loss** trong 5 seeds, tie-break Val Macro F1, rồi seed nhỏ hơn |

`pipeline_smoke.json`: **PASS**, ImageNet initialization PASS, preprocessing parity `max_abs_error=0`, Clean Process inference parity PASS, reload weights/model PASS, backbone training counts như trên. Đây là **log thí nghiệm**, GitHub chưa chứa binary checkpoints để tái chạy xác minh model độc lập.

## 3. Validation — 5 seeds trên 1.430 ảnh V08

| Seed | Stage / selected epoch | Validation Loss | Val Accuracy | Val Macro F1 | `real_` 35 ảnh |
|---:|---|---:|---:|---:|---:|
| 42 | Stage 2 / 1 | 0,150205 | 95,10% | 94,97% | 25/35 |
| 123 | Stage 2 / 7 | 0,149586 | 94,69% | 94,43% | 25/35 |
| **2026** | **Stage 2 / 7** | **0,142782** | **95,45%** | **95,31%** | 25/35 |
| 3407 | Stage 2 / 6 | 0,143298 | 95,59% | 95,42% | 27/35 |
| 7777 | Stage 2 / 6 | 0,151626 | 95,31% | 95,12% | 27/35 |

**Mean ± SD:** Val Accuracy **95,23 ± 0,35%**; Val Macro F1 **95,05 ± 0,39%**; Val Macro Precision **95,03 ± 0,30%**; Val Macro Recall **95,09 ± 0,47%**.

- Anthracnose Val Recall **90,84 ± 0,53%**; Leaf miner **92,80 ± 0,54%**.
- 900 ảnh bệnh/seed: **1,29 ± 0,19%** bị dự đoán nhầm thành Healthy (rủi ro bỏ sót bệnh cần audit).
- Clean Train Probe 500 ảnh: **97,16 ± 0,26%**; gap probe–Val **1,93 ± 0,20 điểm %**, không đủ bằng chứng severe Train–Val overfit; không loại trừ domain shift.
- `real_` Validation Accuracy: **73,71%** trung bình (129/175 lượt trên 5 seeds), trong khi ảnh không có prefix `real_` đạt gần **95,77%** trung bình. Đây là dấu hiệu bất đồng phân phối cần điều tra.
- **Selected deployment seed 2026** (Val Loss thấp nhất `0.14278186857700348`), được khóa trước Test/Holdout; không chuyển sang seed tốt nhất trên Test/Holdout.

Số liệu chi tiết: [Val per seed](./aggregate/validation_results_5seeds.csv) · [Val mean ± SD](./aggregate/validation_mean_std_summary.csv) · [Per-class](./aggregate/per_class_validation_mean_std.csv) · [Real Val](./aggregate/real_validation_by_seed.csv) · [Overfit diagnostics](./aggregate/overfit_diagnostics_by_seed.csv).

## 4. Final Test V08 — 686 ảnh, đã chạy

| Seed | Đúng/686 | Test Accuracy | Test Macro F1 |
|---:|---:|---:|---:|
| 42 | 633 | 92,27% | 91,69% |
| 123 | 637 | 92,86% | 92,38% |
| **2026 (deploy)** | **639** | **93,15%** | **92,55%** |
| 3407 | 642 | 93,59% | 93,05% |
| 7777 | 635 | 92,57% | 91,94% |

**Mean ± SD:** Test Accuracy **92,89 ± 0,51%**, Test Macro F1 **92,32 ± 0,53%**. Seed 3407 có Test Accuracy cao hơn nhưng **không được chọn lại** dựa trên Test.

Số liệu: [Test per seed](./aggregate/test_results_5seeds.csv) · [Test mean ± SD](./aggregate/test_mean_std_summary.csv). Original confusion matrix per seed trong `seed_*/test/`.

## 5. Historical Real Holdout — 48 ảnh (4 true classes)

| Seed | Correct/48 | Accuracy | Macro F1 (4 true classes) |
|---:|---:|---:|---:|
| 42 | 37 | 77,08% | 72,42% |
| 123 | 35 | 72,92% | 66,79% |
| **2026 (Val-selected)** | **35** | **72,92%** | **67,02%** |
| 3407 | 37 | 77,08% | 71,96% |
| 7777 | 38 | 79,17% | 76,07% |

**Mean ± SD:** Accuracy **75,83 ± 2,80%**, Macro F1 **70,85 ± 3,94%** (mean across four supported true classes). **Deployment seed 2026:** Anthracnose **7/13**, Healthy **3/5**, Leaf miner **20/24**, Red rust **5/6**; đúng **35/48** ảnh. **Không có sample true not_cashew_leaf**, không đánh giá được reject performance.

**Lưu ý:** Đây là historical comparison trên dữ liệu đã xem, **không** được dùng để chọn lại seed hay khẳng định chính xác accuracy trên ảnh thực địa mới.

Số liệu: [Holdout per seed](./real_holdout/aggregate/real_holdout_results_5seeds.csv), [Mean ± SD](./real_holdout/aggregate/real_holdout_mean_std.csv), [selected-seed metrics](./real_holdout/aggregate/deployment_seed_result.json), [excluded byte duplicate](./real_holdout/excluded_overlap_images.csv).

## 6. So sánh EXP-008 frozen và DenseNet121-014

| Metrics | ResNet50 Transfer-008 (frozen) | **ResNet50 Transfer-009 (partial FT)** | DenseNet121 Transfer-014/015 |
|---|---:|---:|---:|
| Val Acc | 95,09 ± 0,22% | **95,23 ± 0,35%** | 97,10 ± 0,20% |
| Val Macro F1 | 94,91 ± 0,23% | **95,05 ± 0,39%** | 96,96 ± 0,21% |
| Final Test Acc | **93,24 ± 0,24%** | **92,89 ± 0,51%** | **Chưa có cho EXP-014** |
| Historical Holdout Acc | **77,08 ± 2,55%** | **75,83 ± 2,80%** | 80,83 ± 0,93% |
| Real `real_` Val Acc | 71,43% | **73,71%** | 79,43% |

- EXP-008 → 009 Val +**0,14 pp**, nhưng Final Test **−0,35 pp**, Historical Holdout **−1,25 pp**; differences small and 48-img sample uncertain. Hai EXP đồng thời khác Stage-1 LR và epochs → **không cô lập tác động của unfreeze**.
- DenseNet121-014 đã fine-tune conv5 nhưng **khác architecture**, nên không dùng bảng này để khẳng định ResNet50 kém một cách phổ quát.
- Những phép so sánh đã xem Test V08 nhiều lần là **retrospective exploratory research**, không được nói blind cross-architecture final evaluation.

## 7. Vì sao Val Accuracy cao từ epoch đầu?

1. Backbone ResNet50 khởi tạo **ImageNet weights**, không học đặc trưng từ con số 0. Validation **pre-fit** trung bình thấp: theo 5 seeds từ **9,23% đến 27,06%**; không có biểu hiện đã biết nhãn trước khi train.
2. **Epoch đầu = 300 mini-batches** (`ceil(4790 / 16)`). Val Accuracy 78,32%–86,08% sau khi đầu phân loại đã cập nhật khoảng 300 lần; không phải accuracy tại step0.
3. Train Accuracy trong epoch là trung bình trên quá trình học liên tục, augmentation + dropout; Val Accuracy đo tại cuối epoch, inference mode không augmentation. Vì vậy Val có thể lớn hơn Train ngay epoch đầu.
4. Hai lớp `red_rust` và `not_cashew_leaf` có recall rất cao, góp phần nâng overall Accuracy; Val Macro F1 và per-class recall phải được báo cáo song song.
5. **51 pHash candidate pairs** chưa review, nên không tuyệt đối loại trừ leakage cấp leaf/session. 10 nghi Val→Train đều đúng nhưng bỏ ra thì Val vẫn ~95%; không đủ bằng chứng leakage là nguyên nhân chính.

Đọc thêm [báo cáo phương pháp/giới hạn](./METHOD_AND_LEAKAGE_REVIEW.md), [các ảnh sai lặp lại](./aggregate/recurrent_validation_errors_top.csv) và [learning progress snapshots](./aggregate/early_training_diagnostics.csv).

## 8. Các điểm yếu cần ưu tiên

- **Domain shift:** khoảng cách 95,23% Val vs 73,71% real_ Val và 75,83% historical Holdout; thiếu field blind holdout 5 lớp.
- **False Healthy tự tin cao:** ví dụ Anthracnose `anthracnose_1486.jpg` → Healthy 99,55%, `anthracnose_1482.jpg` → Healthy 99,43%; cần rà nhãn và model calibration. **Không tự sửa nhãn dựa trên dự đoán model**.
- **Fine-tuning benefit chưa chứng minh:** Test/Holdout giảm so với EXP-008, Stage 2 loss rebound ở một số seeds.
- **Data audit:** 51 pHash candidates và nhóm ảnh cùng lá/buổi/vườn có thể bị bỏ sót bởi hash; cần review chuyên gia, quản lý grouping và holdout độc lập.
- Nên thực hiện ablation **Stage-1 checkpoint chung, LR/callback cố định, chỉ khác backbone unfreeze** và đánh giá bằng Val; tiếp đó mới **kiểm chứng bằng ảnh thực địa mới**.
- **Không thay deployment web**: `Application/` hiện vẫn dùng DenseNet121 Scratch EXP-009 seed 3407, không tự đổi sang ResNet50-009.

## 9. Phạm vi import vào GitHub

Nhánh tài liệu này **lưu báo cáo, bảng thống kê, per-class, confusion matrices, lỗi lặp, config/audit/selection metadata và SVG phân tích**, không mặc định có đủ model binaries. **Không commit original 3,785,100-byte ANALYSIS ZIP, FULL ZIP, checkpoints `.weights.h5`, `.keras`, notebook 204KB, toàn bộ per-image predictions/epoch histories hoặc original PNG**, trừ khi được import riêng. Các SHA256 trong source metadata tham chiếu file gốc, không phải bản snapshot được biên soạn lại.

**ANALYSIS ZIP SHA256:** `65b77db413d3b9e0c559686116f023c4d299a666e6533fe905d8d2246ec03762`.

[Artifact inventory](./ARTIFACT_INVENTORY.md) · [Current experiment progress](../EXPERIMENT_PROGRESS_2026_10.md).

