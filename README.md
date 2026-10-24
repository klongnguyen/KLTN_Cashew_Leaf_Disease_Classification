# KLTN — Cashew Leaf Disease Classification & Detection

> **Cập nhật 24/10/2026** · Nghiên cứu chính đang sử dụng **Cashew_dataV08_real_val (6.906 ảnh, 5 lớp)**. Đã hoàn thành nhiều thí nghiệm phân loại 5 seeds với ResNet50, DenseNet121, DeiT-Tiny, DeiT-Small và các classifier Logistic Regression/XGBoost. **Ứng dụng hiện tại vẫn dùng DenseNet121 Scratch EXP-009/seed 3407**; **DenseNet121 Transfer EXP-014/seed 123 là ứng viên để thử nghiệm trên staging, chưa triển khai thay thế**. Nhánh YOLO26s vẫn đang nghiên cứu, chưa khóa detector cuối.

Khóa luận xây dựng một hệ thống hỗ trợ **nhận diện và phân loại bệnh trên lá cây điều** bằng học sâu, tiến tới **phát hiện vị trí tổn thương (bounding box)** và tích hợp kết quả vào ứng dụng web. Dự án bao gồm quản lý dữ liệu, kiểm định leakage, huấn luyện nhiều seed, đánh giá trên các tập độc lập theo giao thức và phân tích các dự đoán sai.

## 1. Tổng quan tiến độ

| Hạng mục | Tình trạng đến 24/10/2026 | Tài liệu |
|---|---|---|
| Dữ liệu phân loại V08 | **Đã dùng cho thí nghiệm mới**; Train 4.790 / Val 1.430 / Test 686 | [Dataset changelog](./DATASET_CHANGELOG.md) |
| DenseNet121 Transfer | **Đã train 5 seed (EXP-014)**, đánh giá Test bằng EXP-016 và Holdout lịch sử bằng EXP-015 | [014](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md) · [015](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md) · [016](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md) |
| ResNet50 Transfer | Đã đánh giá các phương án frozen / partial fine-tune **EXP-008/009**; **EXP-010** mới hơn có kết quả trên Drive | [008](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-008/README.md) · [009](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-009/README.md) · [010 (Drive)](https://drive.google.com/drive/folders/14gyYhTB-k59xLM8wX0p6dxdEFXO-PlxN) |
| DeiT-Tiny + ML classifiers | EXP-008 so sánh Softmax / Logistic Regression / XGBoost trên Validation; **EXP-009 đánh giá Test/Holdout bằng checkpoint đã khóa** | [008](./training_results/current_dataset/EXP-VIT-TRANSFER-XGBOOST-008/README.md) · [009 (Drive)](https://drive.google.com/drive/folders/1Qp87Jhcs6xGE31o6oyLW_OcJXjmqmP4i) |
| DeiT-Small Transfer | **EXP-010 và EXP-011 đã hoàn thành 5 seed Validation**, chưa đánh giá Test/Holdout; EXP-011 **NOT PROMOTED** | [EXP-011](./training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011/README.md) · [EXP-010 (Drive)](https://drive.google.com/drive/folders/1enEpCG61_KCMmztHC47rsw_gpLoONBHa) |
| Detection — YOLO26s | Đã nghiên cứu **Take 01–07**; Take 006 có chỉ số tổng hợp cao nhất trong các take lưu trữ, nhưng **chưa có detector cuối** | [YOLO26 Failure/Development archive](./failure/YOLO26/README.md) |
| Web application | **Có giao diện và API FastAPI**, vẫn sử dụng DenseNet121 Scratch V07 EXP-009, seed 3407 | [Application](./Application/README.md) |

> **Trạng thái triển khai khác với kết quả nghiên cứu:** không có thí nghiệm mới nào được tự động chuyển vào WebApp chỉ vì đạt Accuracy cao; cần kiểm tra checkpoint, preprocessing, label order, tốc độ inference và chất lượng trên dữ liệu thực địa mới.

## 2. Dataset đang nghiên cứu — V08

**Dataset:** `Cashew_dataV08_real_val` · **ZIP:** `CashewData_Split (8).zip` · **SHA256:** `75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7`.

| Lớp | Train | Validation | Test | Tổng |
|---|---:|---:|---:|---:|
| `anthracnose` — Thán thư | 939 | 308 | 122 | 1.369 |
| `healthy` — Khỏe | 803 | 216 | 118 | 1.137 |
| `leaf_miner` — Sâu vẽ bùa | 875 | 264 | 132 | 1.271 |
| `red_rust` — Rỉ đỏ | 1.072 | 328 | 158 | 1.558 |
| `not_cashew_leaf` — Không phải lá điều | 1.101 | 314 | 156 | 1.571 |
| **Tổng** | **4.790** | **1.430** | **686** | **6.906** |

- **Thứ tự nhãn của các pipeline V08:** `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`; không được tự sắp xếp lại khi load model hoặc tính confusion matrix.
- **Seeds cố định:** `[42, 123, 2026, 3407, 7777]`. Mỗi thí nghiệm báo cáo **mean ± sample SD (ddof=1)**; đây là 5 lần train trên **cùng split**, không phải 5 tập Test độc lập.
- **Quy trình chọn:** Train cập nhật trọng số; Validation dùng cho checkpoint, hyperparameter/seed, guardrails theo từng experiment. **Test không được dùng để lựa chọn hoặc tuning**. Các đánh giá Test V08 đã được quan sát qua nhiều pipeline nên **không còn là blind benchmark toàn cục cho quá trình nghiên cứu lặp lại**.
- **Kiểm định dữ liệu:** không thấy exact-byte cross-split duplicates trong các audit đã đối chiếu; các báo cáo ghi nhận **51 cặp** hoặc **226 cặp nghi gần trùng pHash** theo quy trình/phạm vi audit khác nhau. Đây là **candidates cần xác minh theo lá/cây/buổi chụp**, không phải số lượng leakage đã chứng minh.
- **Real Holdout lịch sử:** 48 ảnh sau khi loại một ảnh trùng với Test; chỉ có ground truth ở **4 lớp** (không có `not_cashew_leaf`) và đã được xem trong các thí nghiệm trước. **Không phải fresh blind field test**.

Lịch sử các dataset V03/V04/V05/V07 được bảo lưu trong [DATASET_CHANGELOG.md](./DATASET_CHANGELOG.md). **V05 là benchmark Scratch lịch sử, không còn là snapshot chính cho các Transfer experiment V08.**

## 3. So sánh mô hình phân loại trên cùng V08

Các giá trị bên dưới là **Accuracy/Macro F1 trung bình ± độ lệch chuẩn mẫu qua 5 seed (%)**, không phải chỉ số seed tốt nhất. Một dòng có kết quả Test/Holdout chỉ khi phép đánh giá đó đã thực hiện. **Không chọn lại mô hình theo Test/Holdout.**

| Mô hình / experiment | Val Accuracy | Val Macro F1 | Test Accuracy | Test Macro F1 | Holdout Accuracy |
|---|---:|---:|---:|---:|---:|
| **[DenseNet121 Transfer 014/015/016](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md)** | **97,10 ± 0,20** | **96,96 ± 0,21** | 95,13 ± 0,51 | 94,76 ± 0,58 | 80,83 ± 0,93 |
| [DeiT-Tiny + Logistic B (008/009)](https://drive.google.com/drive/folders/1Qp87Jhcs6xGE31o6oyLW_OcJXjmqmP4i) | 96,56 ± 0,17 | 96,40 ± 0,18 | **96,68 ± 0,28** | **96,44 ± 0,28** | 83,33 ± 1,47 |
| [DeiT-Tiny + XGBoost C (008/009)](https://drive.google.com/drive/folders/1Qp87Jhcs6xGE31o6oyLW_OcJXjmqmP4i) | 95,85 ± 0,20 | 95,64 ± 0,20 | 96,04 ± 0,24 | 95,73 ± 0,28 | **85,00 ± 1,74** |
| [ResNet50 partial fine-tune 009](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-009/README.md) | 95,23 ± 0,35 | 95,05 ± 0,39 | 92,89 ± 0,51 | 92,32 ± 0,53 | 75,83 ± 2,80 |
| [ResNet50 frozen 008](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-008/README.md) | 95,09 ± 0,22 | 94,91 ± 0,23 | 93,24 ± 0,24 | 92,72 ± 0,25 | 77,08 ± 2,55 |
| [ResNet50 augmented fine-tune 010](https://drive.google.com/drive/folders/14gyYhTB-k59xLM8wX0p6dxdEFXO-PlxN) | 94,60 ± 0,48 | 94,39 ± 0,52 | 92,48 ± 0,80 | 91,93 ± 0,78 | 75,42 ± 3,42 |
| [DeiT-Tiny Softmax A (007/008/009)](./training_results/current_dataset/EXP-VIT-TRANSFER-XGBOOST-008/README.md) | 94,41 ± 0,73 | 94,29 ± 0,70 | 95,36 ± 0,49 | 95,08 ± 0,56 | 74,58 ± 6,14 |
| [DeiT-Small Stage 2 — 011](./training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011/README.md) | 93,45 ± 1,28 | 93,40 ± 1,30 | — | — | — |
| [DeiT-Small Stage 2 — 010](https://drive.google.com/drive/folders/1enEpCG61_KCMmztHC47rsw_gpLoONBHa) | 92,81 ± 1,00 | 92,74 ± 0,96 | — | — | — |

**Cách diễn giải:**

1. **DenseNet121 Transfer EXP-014/seed 123 là ứng viên staging ưu tiên**: theo [EXP-016](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md), seed 123 đã được chọn **bằng Validation Loss trước khi xem Test**, đạt **653/686 = 95,19% Test Accuracy**, chỉ **2/412 ảnh bệnh → Healthy** trên Test. Điểm Holdout của seed 123 là **38/48**; các số trung bình trong bảng trên là qua 5 seed.
2. **Logistic Regression B có Test Accuracy/Macro F1 trung bình cao nhất**, nhưng ở seed 123 đã khóa, số lỗi bệnh → Healthy trên Validation là **21/900**, so với **18/900** của Softmax A; XGBoost C là **23/900**. Theo selection lock của EXP-008, **A/seed 123 vẫn được chọn trong nhánh DeiT-Tiny**, không nâng B/C chỉ nhờ điểm Test.
3. **DeiT-Small EXP-011 cải thiện** so với EXP-010 (Macro F1 Validation Stage 2: **92,74 → 93,40%**) nhưng **chưa đạt guardrails / NOT PROMOTED**; chưa chạy Test/Holdout cho EXP-010/011.
4. **Holdout 48 ảnh** có khác biệt phân phối, đã bị exposed và thiếu một lớp. Mọi phép xếp hạng trên tập này đều chỉ là **mô tả hồi cứu**, không phải kết luận khái quát hóa thực địa.

**[Google Sheet đối chiếu 9 cấu hình và biểu đồ (V08)](https://docs.google.com/spreadsheets/d/1mEY7LxsQyOO_AY54Dwr1SKqtxzHYlGmat8y3OIK6OZo/edit)** · [Bảng tiến độ/đổi version chi tiết](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md).

## 4. Ứng dụng Web và lộ trình triển khai

- **Đang được cấu hình trong repo:** `DenseNet121 Scratch EXP-009 / seed 3407` trên **V07**, input **224×224 RGB**; Validation của seed 3407: **Accuracy 95,63%, Macro F1 95,51%, Loss 0,1410**. Đây là kết quả của **V07 Validation**, không được so thẳng với V08 như một thí nghiệm có cùng split.
- **Công nghệ:** FastAPI, HTML/CSS/JavaScript, tải ảnh đơn lẻ, dự đoán 5 lớp và hiển thị khuyến nghị. [Chi tiết và cách chạy](./Application/README.md).
- **Ứng viên nâng cấp:** `DenseNet121 Transfer EXP-014 / seed 123` — **chưa thay vào WebApp**. Cần đóng gói checkpoint đúng nhãn, kiểm tra xử lý ảnh và preprocessing trong model, parity với notebook, benchmark độ trễ API và chạy thử staging.
- **Yêu cầu trước khi khẳng định sử dụng ngoài thực địa:** bổ sung bộ **blind field test mới đủ 5 lớp** với ảnh độc lập theo lá/cây/buổi chụp; đánh giá lỗi bỏ sót bệnh, nhầm bệnh, hiệu chuẩn xác suất và khả năng từ chối khi không đủ tin cậy.

Chạy ứng dụng theo hướng dẫn của [Application/README.md](./Application/README.md). Ví dụ từ thư mục `Application/` khi đã cài các dependency phù hợp:

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Mở `http://127.0.0.1:8000`. Việc chạy được ứng dụng còn phụ thuộc vào model package/checkpoint đã hiện diện ở đường dẫn mà backend cấu hình.

## 5. Nhánh Object Detection — YOLO26s

YOLO26s dùng 3 bounding-box classes: `anthracnose`, `leaf_miner`, `red_rust`. Ảnh `healthy` và `not_cashew_leaf` có thể dùng làm **negative samples 0 box** theo quy tắc annotation. [Guideline gán nhãn](./CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md).

- Đã thử **Take 01–07**, lưu trong [Failure/Development Archive](./failure/YOLO26/README.md).
- **Take 006:** Precision **0,7473**, Recall **0,6648**, mAP@0.50 **0,7298**, mAP@0.50:0.95 **0,3877** — cao nhất về chỉ số tổng hợp trong các take đã ghi nhận.
- **Take 007** sử dụng dataset/Test dày GT hơn và đổi phạm vi dữ liệu; vì vậy **không so điểm Take 006/007 như một ablation chỉ thay model**.
- **Chưa chọn detector cuối, chưa tích hợp detector vào WebApp như một chức năng production đã được kiểm định.** Cần khóa split, tiêu chuẩn annotation, bổ sung negative images và chốt ngưỡng trên Validation trước khi Test.

## 6. Benchmark lịch sử và cấu trúc repository

Để giữ khả năng truy vết, các kết quả Scratch đời trước vẫn được lưu riêng. **Benchmark V05 (6.911 ảnh, khác V08):**

| Model Scratch (V05) | Test Accuracy (5 seed) | Test Macro F1 |
|---|---:|---:|
| [DenseNet121](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-005/README.md) | **91,97 ± 1,11%** | **91,41 ± 1,13%** |
| [ResNet50](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-005/README.md) | 89,78 ± 2,50% | 89,36 ± 2,47% |
| [Compact ViT](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-005/README.md) | 85,04 ± 0,44% | 83,93 ± 0,37% |

**Không gộp V04/V05/V07/V08 vào cùng một benchmark chỉ để so Accuracy.** Các so sánh có ý nghĩa nhất là giữa pipeline trên **đúng cùng V08** và có ghi rõ split/selection.

```text
.
├── Application/                      # Ứng dụng Web FastAPI; hiện dùng DenseNet121 Scratch EXP-009
├── Notebook/                         # Notebook huấn luyện và kiểm định
├── training_results/
│   ├── current_dataset/              # V08 Transfer/ViT và benchmark các version khác
│   ├── dataset_quality_study/        # Nghiên cứu dữ liệu/ảnh thực địa
│   └── archive_legacy_dataset/       # Lưu trữ thí nghiệm cũ
├── failure/YOLO26/                   # YOLO Take 01–07, chưa có final detector
├── CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md
├── DATASET_CHANGELOG.md
├── DISEASE_REFERENCE.md
└── README.md
```

**Tài liệu/điểm truy cập quan trọng:**

- [Danh mục các thí nghiệm phân loại](./training_results/current_dataset/README.md) · [Tiến độ tháng 10](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md)
- [Kết quả DenseNet Transfer & Test](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md) · [DeiT-Small EXP-011: Scientific report](./training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011/analysis/SCIENTIFIC_REPORT.md)
- [ViT + classifiers EXP-008: Method & leakage review](./training_results/current_dataset/EXP-VIT-TRANSFER-XGBOOST-008/METHOD_AND_LEAKAGE_REVIEW.md)
- [Notebook DeiT-Small EXP-011 (nhánh nguồn)](https://github.com/klongnguyen/KLTN_Cashew_Leaf_Disease_Classification/blob/exp/vit-deitsmall-extended-epochs-011/Notebook/ViT/vit_deit_small_transfer_5seeds_011.ipynb)
- [Dataset changelog](./DATASET_CHANGELOG.md) · [Disease reference](./DISEASE_REFERENCE.md) · [YOLO archive](./failure/YOLO26/README.md)
- **Lưu trữ:** GitHub ưu tiên báo cáo, config, metrics, source và figures. **Checkpoint lớn/FULL ZIP** của một số experiment nằm trên Google Drive; README experiment tương ứng ghi nguồn/trạng thái artifact. Không giả định mọi checkpoint có thể tải trực tiếp từ repo.

## 7. Công việc tiếp theo

1. **Kiểm chứng và thử nghiệm staging DenseNet121 Transfer EXP-014/seed 123** với model package, preprocess, label map và API parity; không thay model đang phục vụ khi chưa hoàn thành.
2. **Thu thập fresh blind field test đủ 5 lớp** và khóa nguồn chụp theo lá/cây/buổi/vườn; đánh giá các candidate đã khóa mà không tuning trên Test.
3. **Audit lỗi lặp lại** Anthracnose ↔ Leaf Miner, Anthracnose → Healthy, `not_cashew_leaf` và các cặp ảnh pHash nghi tương tự; xác nhận nhãn bằng đánh giá chuyên môn.
4. **Nghiên cứu DeiT-Small + Logistic/XGBoost** từ checkpoint EXP-011 trên Train/Validation; giữ Softmax làm đối chứng, chưa dùng Test/Holdout để lựa chọn.
5. **Tiếp tục YOLO26s**: chuẩn hóa annotations và negative examples, khóa split trước khi đánh giá detector và pipeline crop → classifier.
6. **Đồng bộ báo cáo khóa luận**, tài liệu thí nghiệm và bảng so sánh theo kết quả đã được xác minh.

> **Phạm vi:** dự án hỗ trợ nhận diện/đánh giá triệu chứng bằng ảnh và phục vụ nghiên cứu; dự đoán của mô hình không thay cho việc xác nhận hiện trường/chuyên môn nông nghiệp.
