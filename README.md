# KLTN — Cashew Leaf Disease Classification & Detection

> **Trạng thái 24/10/2026:** DenseNet121 Transfer EXP-014 seed 123 là cấu hình mặc định của web; EXP-016 đã hoàn thành Final Test V08 cho 5 seed. [DeiT-Small Transfer EXP-011](./training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011/README.md) mới ở giai đoạn chuẩn bị chạy, chưa có metrics. Checkpoint `.keras` và dataset ZIP không lưu trong Git; xem [hướng dẫn ứng dụng](./Application/README.md) và [tiến độ thí nghiệm](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md).

> **ResNet50 ImageNet Partial Fine-tuning EXP-009 — cập nhật 20/10/2026:** [Báo cáo kỹ thuật](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-009/README.md) · [Kiểm tra Accuracy cao/domain shift](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-009/METHOD_AND_LEAKAGE_REVIEW.md). V08 5 seeds: **Val Acc 95,23 ± 0,35% / Macro F1 95,05 ± 0,39%**; **Final Test 686 ảnh 92,89 ± 0,51% Acc / Macro F1 92,32 ± 0,53%**; **Real Holdout lịch sử 48 ảnh 75,83 ± 2,80% Acc / Macro F1 70,85 ± 3,94%**. Stage 2 đã mở **4.459.520 tham số `conv5_block3_*` không phải BN**, khác [EXP-008 frozen](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-008/README.md). **Seed 2026 chọn bằng Val Loss trước Test/Holdout**; chỉ cải thiện Val nhẹ so với EXP-008 và **chưa chứng minh tăng generalization**. Holdout đã được phân tích từ trước, thiếu lớp `not_cashew_leaf`; mã nguồn Application hiện cấu hình DenseNet121 Transfer EXP-014. [Tiến độ](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md).


> **EXP-RESNET50-TRANSFER-5SEEDS-008 — đánh giá mới 20/10/2026:** [Báo cáo kỹ thuật](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-008/README.md) · [Audit code/phương pháp](./training_results/current_dataset/EXP-RESNET50-TRANSFER-5SEEDS-008/ANALYSIS_AND_PROTOCOL_AUDIT.md). Dùng **ResNet50 pretrained ImageNet, nhưng backbone vẫn frozen ở cả Stage 1 và Stage 2** (head-only training; **không phải partial fine-tuning**), dataset V08. **5 seeds: Val Acc 95.09 ± 0.22% / Macro F1 94.91 ± 0.23%; Final Test 686 ảnh Acc 93.24 ± 0.24% / F1 92.72 ± 0.25%; Real Holdout lịch sử 48 ảnh Acc 77.08 ± 2.55% / Macro F1 72.97 ± 4.07%**. Seed **3407 chọn bằng Validation Loss**, không chọn bằng Holdout. Holdout là **previously exposed, 4 true classes**, chưa có field blind test mới. **Khác với ResNet50 Scratch-008 trên V07** và **chưa cập nhật mô hình web đang chạy**. [Tiến độ](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md).


> **ResNet50 Scratch EXP-006/007/008 (cập nhật 19/10/2026):** [EXP-006 V05](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-006/README.md) Val Acc **86.46 ± 2.27%**, Macro F1 **86.12 ± 2.56%**; [EXP-007 V07](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-007/README.md) **89.20 ± 2.04%**, F1 **88.61 ± 2.19%**; [EXP-008 V07](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-008/README.md) có **cùng metrics, predictions và training histories 5 seeds với 007**, chưa chứng minh retrain độc lập. Đây đều là **Scratch (weights=None)**, không phải Transfer Learning; **chưa chạy Test riêng các EXP 006–008**, không dùng Val V05 vs V07 như controlled ablation. [Audit + config changes](./training_results/current_dataset/RESNET50_SCRATCH_006_008_AUDIT_2026_10.md) · [Tiến độ](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md).


> **DenseNet121 Transfer V08 — Mô hình chính thức triển khai Web App:** [EXP-014 training 5 seeds](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md): **97.10 ± 0.20% Val Accuracy**, **96.96 ± 0.21% Val Macro F1**; [EXP-015 Holdout lịch sử](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md): **80.83 ± 0.93% Accuracy** trên 48 ảnh; [EXP-016 Final Test V08](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md): **95.13 ± 0.51% Accuracy, 94.76 ± 0.58% Macro F1** trên 686 ảnh. **Seed 123 chọn bằng Validation Loss (97.27% Val Accuracy, 653/686 ảnh Test V08)** đã được **triển khai chính thức vào [Ứng dụng Web AI](Application/)**. [Tiến độ](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md).


> **Mốc lịch sử EXP-011/012 (trước EXP-014/015):** [Bảng tiến độ và thay đổi từng EXP](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md). **DenseNet121 EXP-012 (V08)** đạt **93.99 ± 1.05% Val Accuracy**, **93.81 ± 1.13% Val Macro F1**; **Final Test riêng EXP-012 chưa có**. Seed 42 chỉ đúng **22/35** ảnh `real_` trong Validation. [Chi tiết EXP-012](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-012/README.md). EXP-011 **trùng 5 seed metrics** với EXP-009, chưa xác minh là run độc lập.

Khóa luận xây dựng hệ thống thị giác máy tính cho **phân loại bệnh trên lá cây điều** và hướng mở rộng sang **phát hiện vùng tổn thương bằng bounding box**.

---

## 🌟 Cập nhật tiến độ dự án mới nhất

### 1. 🚀 Đã triển khai Ứng Dụng Web Chuẩn Đoán AI (Application/)
- Hệ thống web hoàn chỉnh phục vụ chuẩn đoán bệnh lá cây điều, chạy trên nền **FastAPI** và giao diện **HTML5/CSS3/Vanilla JS** responsive hiện đại.
- **Mô hình tích hợp chính thức**: DenseNet121 (Transfer v014) — checkpoint tối ưu **Seed 123**:
  - **Validation Accuracy**: **97.27%** (so với v009 Scratch: 95.63%, tăng **+1.64%**)
  - **Validation Macro F1**: **97.14%** (so với v009 Scratch: 95.51%, tăng **+1.63%**)
  - **Validation Loss**: **0.0925** (thấp nhất trong cả 5 seeds)
  - **Final Test V08 (EXP-016)**: **95.19% (653/686 ảnh kiểm thử độc lập)**
- **Tính năng nổi bật**:
  - Tải ảnh đơn lẻ trực quan (hỗ trợ kéo thả, kiểm tra định dạng .png, .jpg, .jpeg, .webp).
  - Phân tích và xếp hạng độ tin cậy từ cao xuống thấp của 5 lớp nhãn (`anthracnose`, `healthy`, `leaf_miner`, `not_cashew_leaf`, `red_rust`).
  - Cẩm nang nông nghiệp: Tự động cung cấp triệu chứng nhận diện và phác đồ điều trị, hoạt chất thuốc bảo vệ thực vật phòng trừ tương ứng.
  - Bộ 3 mẫu thử nhanh 1-click (Bệnh Thán Thư, Sâu Vẽ Bùa, Bệnh Rỉ Đỏ).
- **Khởi chạy nhanh**:
  ```powershell
  cd Application
  python server.py
  # Hoặc: uvicorn server:app --host 127.0.0.1 --port 8000
  # Truy cập http://localhost:8000
  ```
- ➡️ [Chi tiết tài liệu ứng dụng web](./Application/README.md)

### 2. 🔬 DenseNet121 Transfer Learning v014 (EXP-DENSENET121-TRANSFER-5SEEDS-014)
- Huấn luyện chuyển giao tri thức từ ImageNet kết hợp 2 giai đoạn (Stage 1: freeze backbone; Stage 2: mở conv5 non-BN fine-tuning) với 5 seeds cố định [42, 123, 2026, 3407, 7777] trên dataset Cashew_dataV08_real_val (6.906 ảnh).
- Kết quả Validation 5 seeds:
  - **Val Accuracy**: **97.10 ± 0.20%** (Tối ưu: Seed 123 đạt **97.27%**)
  - **Val Macro F1**: **96.96 ± 0.21%** (Tối ưu: Seed 123 đạt **97.14%**)
  - **Val Macro Precision**: **96.94 ± 0.19%**
  - **Val Macro Recall**: **96.98 ± 0.23%**
- Đã đánh giá Real Holdout lịch sử 48 ảnh ([EXP-015](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md)) đạt **80.83 ± 0.93%** và Final Test V08 686 ảnh ([EXP-016](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md)) đạt **95.13 ± 0.51%**.
- Checkpoint Seed 123 được đóng gói hoàn chỉnh trong model_package (deployment_model.keras, embedding, feature extractor, label map, metadata) và triển khai làm mô hình mặc định trong [Application/](./Application/README.md).
- ➡️ [Báo cáo chi tiết EXP-014](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md) · [Baseline EXP-009 Scratch cũ](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL/EXP-DENSENET121-SCRATCH-5SEEDS-009/README.md)

### 3. 🔍 Kiểm định chất lượng dữ liệu (`EXP-DENSENET121-DATASET-AUDIT-001`)
- Phân tích rà soát tính toàn vẹn bộ dữ liệu: kiểm tra trùng lặp pHash cross-split, đo độ trôi phân phối đặc trưng (Train/Val feature shift), rà soát ảnh thực địa (real images).
- Cung cấp contact sheets trực quan phục vụ đánh giá chất lượng và phát hiện data leakage.
- ➡️ [Chi tiết thư mục audit](./training_results/current_dataset/EXP-DENSENET121-DATASET-AUDIT-001/)

### 4. 📓 Bộ Notebook huấn luyện và kiểm định chuẩn hóa (`Notebook/`)
- [`Notebook/DenseNet121/densenet121_5seed.ipynb`](./Notebook/DenseNet121/densenet121_5seed.ipynb): Pipeline chuẩn hóa 5 seeds cho DenseNet121 (EXP-009 recipe) với đầy đủ logs và outputs.
- [`Notebook/DenseNet121/densenet121_007_diagnostic.ipynb`](./Notebook/DenseNet121/densenet121_007_diagnostic.ipynb): Diagnostic setup chẩn đoán gradient và learning rate.
- [`Notebook/DenseNet121/densenet121_dataset_audit_scratch.ipynb`](./Notebook/DenseNet121/densenet121_dataset_audit_scratch.ipynb): Notebook kiểm định chất lượng bộ dữ liệu.
- [`Notebook/ResNet50/resnet50_5seed_reference_recipe.ipynb`](./Notebook/ResNet50/resnet50_5seed_reference_recipe.ipynb): Pipeline chuẩn hóa 5 seeds cho ResNet50.

---

### Nghiên cứu chất lượng dữ liệu: Kaggle base → CashewData_Split (7)

Báo cáo thực nghiệm 1 seed DenseNet121 Scratch riêng biệt nhằm đánh giá cải thiện dữ liệu và generalization ảnh thực địa: **IMPROVED001 Test Accuracy 93,01%; Real Test 67,86% (19/28); phát hiện 39 cặp nghi gần trùng pHash xuyên split**. Không nhập kết quả này vào benchmark V05 vì ZIP có 6.907 ảnh và giao thức thí nghiệm khác.

➡️ [Kế hoạch điều tra nguyên nhân sai trên ảnh Real (WP0–WP7)](./training_results/dataset_quality_study/INVESTIGATION_PLAN.md)

➡️ [Báo cáo WP3 — IMPROVED002 có 35 ảnh Real Validation](./training_results/dataset_quality_study/EXP-DENSENET121-SCRATCH-IMPROVED-002/ANALYSIS.md)

➡️ [Báo cáo dataset IMPROVED001](./training_results/dataset_quality_study/EXP-DENSENET121-SCRATCH-IMPROVED-001/ANALYSIS.md)

---

## Project status

### Classification — official V05 benchmark (historical cross-model comparison)

Current dataset: **`Cashew_dataV05` — 6,911 images / 5 classes**. Training, Validation, and the locked Test are complete for all three scratch baselines.

| Model | V05 Validation Accuracy | V05 Validation Macro F1 | Candidate seed | Params |
|---|---:|---:|---:|---:|
| **DenseNet121 Scratch** | **94.58 ± 0.95%** | **94.43 ± 0.93%** | 2026 | 7.57M |
| **ResNet50 Scratch** | **92.43 ± 1.94%** | **92.30 ± 1.98%** | 3407 | 24.64M |
| **Compact ViT Scratch** | **88.33 ± 1.44%** | **87.81 ± 1.52%** | 3407 | **0.35M** |

Detailed V05 analysis: [`training_results/current_dataset/V05_VALIDATION_ANALYSIS.md`](./training_results/current_dataset/V05_VALIDATION_ANALYSIS.md).

### Final V05 locked-Test benchmark

| Model | Test Accuracy | Macro Precision | Macro Recall | Macro F1 | Balanced Accuracy | Params |
|---|---:|---:|---:|---:|---:|---:|
| **DenseNet121 Scratch** | **91.97 ± 1.11%** | **91.48 ± 1.13%** | **91.42 ± 1.15%** | **91.41 ± 1.13%** | **91.42 ± 1.15%** | 7.57M |
| **ResNet50 Scratch** | 89.78 ± 2.50% | 89.72 ± 2.08% | 89.37 ± 2.56% | 89.36 ± 2.47% | 89.37 ± 2.56% | 24.64M |
| **Compact ViT Scratch** | 85.04 ± 0.44% | 84.35 ± 0.30% | 84.12 ± 0.35% | 83.93 ± 0.37% | 84.12 ± 0.35% | **0.35M** |

DenseNet121 is the strongest final V05 classifier by both Test Accuracy and Macro F1.

For historical comparison, the previous V04 Test benchmark is retained below.

| Model | V04 Test Accuracy | V04 Macro F1 | Params |
|---|---:|---:|---:|
| **DenseNet121 Scratch** | **91.82 ± 0.86%** | **91.37 ± 0.79%** | 7.57M |
| **ResNet50 Scratch** | **90.63 ± 2.11%** | **90.17 ± 2.04%** | 24.64M |
| **Compact ViT Scratch** | **85.30 ± 1.68%** | **84.41 ± 1.71%** | **0.35M** |

➡️ Chi tiết: [`training_results/current_dataset/README.md`](./training_results/current_dataset/README.md)

#### Previous V04 DenseNet121 validation rerun — EXP-004

[`EXP-DENSENET121-SCRATCH-5SEEDS-004`](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-004/) đã hoàn tất 5-seed training/validation:

- Validation Accuracy: **92.13 ± 1.31%**
- Validation Macro-F1: **91.90 ± 1.37%**
- cao hơn official EXP-003 ở validation mean và ổn định hơn theo seed;
- cả 5 seed đều cải thiện validation so với lần chạy trước;
- final locked Test **chưa chạy**.

Archive nguồn vô tình tái sử dụng ID `EXP-DENSENET121-SCRATCH-5SEEDS-003`; repository lưu rerun này thành **EXP-004** để không ghi đè official experiment cũ. Bảng benchmark Test phía trên vẫn giữ DenseNet121 EXP-003 cho tới khi EXP-004 hoàn tất locked Test.

#### Previous V04 ResNet50 validation rerun — EXP-004

[`EXP-RESNET50-SCRATCH-5SEEDS-004`](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-004/) đã hoàn tất 5-seed training/validation trên Colab NVIDIA L4:

- Validation Accuracy: **90.23 ± 1.45%**
- Validation Macro-F1: **90.03 ± 1.44%**
- tốt hơn EXP-003 ở validation mean và ổn định seed, nhưng **final locked Test chưa chạy**.

#### Previous V04 Compact ViT validation rerun — EXP-004

[`EXP-VIT-SCRATCH-5SEEDS-004`](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-004/) đã hoàn tất 5-seed training/validation trên single-GPU `OneDeviceStrategy`:

- Validation Accuracy: **87.13 ± 0.62%**
- Validation Macro-F1: **86.68 ± 0.70%**
- tăng **+1.85 pp** Validation Accuracy và **+2.00 pp** Macro-F1 so với official EXP-003;
- cả 5 seed đều cải thiện validation, đồng thời seed variability giảm rõ rệt;
- final locked Test **chưa chạy**.

Archive nguồn tái sử dụng ID `EXP-VIT-SCRATCH-5SEEDS-003`; repository lưu lần chạy mới thành **EXP-004** để không ghi đè official benchmark cũ.

Vì cả ba rerun EXP-004 chưa chạy locked Test, bảng benchmark chính thức phía trên vẫn giữ các EXP-003.

### Object detection — development

YOLO26s đã được thử nghiệm qua **Take 01 → Take 06** để nghiên cứu annotation policy và chất lượng bounding box.

Take 006 hiện là iteration tốt nhất:

```text
Precision     0.7473
Recall        0.6648
F1            0.7036
mAP@0.50      0.7298
mAP@0.50:0.95 0.3877
Mean IoU      0.7603
```

Take 006 vẫn được giữ trong Failure Archive vì detector cuối chưa khóa split/threshold/protocol và detection dataset còn class imbalance + thiếu negative supervision.

➡️ Chi tiết: [`failure/YOLO26/README.md`](./failure/YOLO26/README.md)

---

## Current dataset — Cashew_dataV05

V05 giữ nguyên class counts của V04 nhưng split fingerprints ghi trong ResNet50 metadata đã thay đổi ở Train/Validation; Test fingerprint vẫn giữ nguyên. V04 trước đó được tạo sau lần làm sạch bổ sung:
- loại ảnh mờ/chất lượng thấp;
- xử lý các trường hợp có nguy cơ data leakage;
- giữ split cố định cho benchmark V04.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 945 | 294 | 122 | 1,361 |
| `healthy` | 806 | 225 | 118 | 1,149 |
| `leaf_miner` | 893 | 249 | 132 | 1,274 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,077 | 320 | 158 | 1,555 |
| **TOTAL** | **4,822** | **1,402** | **687** | **6,911** |

Files:
- [`dataset_cashew_v04.csv`](./dataset_cashew_v04.csv) — previous V04 count snapshot;
- [`training_results/current_dataset/dataset_consistency_v05.json`](./training_results/current_dataset/dataset_consistency_v05.json) — V05 manifest/fingerprint traceability;
- [`DATASET_CHANGELOG.md`](./DATASET_CHANGELOG.md) — lịch sử V03 → V04 → V05.

> File Excel V03 cũ đã được loại khỏi root để tránh nhầm với current dataset. Git history vẫn giữ bản cũ.

---

## Classification protocol

Latest V05 scratch baselines:
- dùng cùng fixed V05 manifest;
- train 5 seeds: `42, 123, 2026, 3407, 7777`;
- checkpoint chọn theo Validation;
- Test không dùng để tuning hoặc chọn seed;
- báo cáo `Mean ± sample Standard Deviation (ddof=1)`.

Current experiments:

- [`EXP-DENSENET121-SCRATCH-5SEEDS-005`](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-005/) — V05 validation + locked Test completed
- [`EXP-RESNET50-SCRATCH-5SEEDS-005`](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-005/) — V05 validation + locked Test completed
- [`EXP-VIT-SCRATCH-5SEEDS-005`](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-005/) — V05 validation + locked Test completed; source archive ID was EXP-004

Previous V04 benchmark/reruns:

- [`EXP-DENSENET121-SCRATCH-5SEEDS-003`](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/) — current official Test benchmark
- [`EXP-DENSENET121-SCRATCH-5SEEDS-004`](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-004/) — latest validation rerun; Test pending
- [`EXP-RESNET50-SCRATCH-5SEEDS-003`](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-003/) — current official Test benchmark
- [`EXP-RESNET50-SCRATCH-5SEEDS-004`](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-004/) — latest validation rerun; Test pending
- [`EXP-VIT-SCRATCH-5SEEDS-003`](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/) — current official Test benchmark
- [`EXP-VIT-SCRATCH-5SEEDS-004`](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-004/) — latest validation rerun; Test pending

`anthracnose` hiện là class khó nhất nhất quán qua các baseline classification và tiếp tục là class yếu nhất trong cả ba validation rerun mới.

---

## Bounding-box annotation policy

Detection dùng 3 box classes:

```text
anthracnose
leaf_miner
red_rust
```

`healthy` và `not_cashew_leaf` là **0-box negative images**.

Policy hiện tại là **selective clear-lesion annotation**:
- ưu tiên lesion rõ, đủ lớn và có ý nghĩa thị giác;
- gom cluster hợp lý;
- không cố bounding mọi chấm cực nhỏ;
- giữ cùng annotation policy giữa Train/Validation/Test.

➡️ Guideline hiện tại: [`CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md`](./CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md)

---

## Repository structure

```text
.
├── Application/                   # FastAPI Web Application & UI chẩn đoán AI
│   ├── server.py                  # Backend API service
│   ├── model_service.py           # Inference engine & image preprocessing
│   ├── config.py                  # Cấu hình nhãn bệnh, cẩm nang nông nghiệp
│   ├── static/                    # Giao diện HTML5, CSS3, JavaScript & assets
│   └── README.md                  # Hướng dẫn chi tiết triển khai Web App
├── Notebook/                      # Jupyter Notebooks huấn luyện & chẩn đoán
│   ├── DenseNet121/               # 5-seed training EXP-009, diagnostic, audit
│   └── ResNet50/                  # 5-seed reference recipe EXP-007
├── training_results/              # Lưu trữ kết quả thí nghiệm
│   ├── current_dataset/           # V05/V07 experiments, EXP-009, Dataset Audit
│   ├── dataset_quality_study/     # Nghiên cứu chất lượng dữ liệu & Real holdout
│   └── archive_legacy_dataset/    # Thí nghiệm cũ
├── img_check/                     # Ảnh mẫu tham khảo bệnh thực địa
├── failure/                       # Nghiên cứu YOLO detection (Take 01–06)
├── README.md                      # Báo cáo tổng quan dự án
├── DATASET_CHANGELOG.md           # Lịch sử thay đổi dataset qua các phiên bản
├── CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md
└── DISEASE_REFERENCE.md
```

### Result management

- `Application/`: Ứng dụng web phục vụ người dùng cuối kiểm thử mô hình trực quan.
- `Notebook/`: Toàn bộ mã nguồn thí nghiệm và notebook huấn luyện chuẩn hóa theo seed.
- `training_results/current_dataset/`: Benchmark chính thức, kết quả EXP-009 5 seeds, dataset audit, và validation reruns.
- `training_results/archive_legacy_dataset/`: Thí nghiệm cũ.
- `failure/YOLO26/`: Báo cáo failure analysis nhánh object detection.
- Checkpoint/model/FULL ZIP lớn **không commit trực tiếp**; repo chỉ giữ config, metrics, summaries, panels và code adapters.

---

## Dataset versioning

- Cashew_dataV03 (7,213 ảnh): Phiên bản ban đầu.
- Cashew_dataV04 / V05 (6,911 ảnh): Làm sạch ảnh mờ, loại bỏ nguy cơ data leakage và cố định split benchmark cho 3 baseline scratch ban đầu.
- Cashew_dataV07_clean_val (6,907 ảnh): Chuẩn hóa phục vụ thực nghiệm EXP-009 scratch.
- Cashew_dataV08_real_val (6,906 ảnh): **Dataset chính thức hiện tại** — Train 4.790 / Validation 1.430 / Test 686, bổ sung ảnh thực địa và phân bổ đồng đều, làm nền tảng cho chuỗi mô hình Transfer Learning (EXP-014, EXP-015, EXP-016) và triển khai ứng dụng Web AI.

V03, V04/V05, V07 và V08 có phân chia Train/Validation/Test khác nhau, vì vậy **không dùng so sánh trực tiếp chéo phiên bản dataset như một controlled model comparison**.

Git history vẫn lưu trữ đầy đủ toàn bộ phiên bản cũ.

---

## Tài liệu chính

- [Ứng dụng Web AI Chuẩn Đoán Bệnh](./Application/README.md)
- [Báo cáo DenseNet121 Transfer Learning 5-Seed v014 (Mô hình triển khai)](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md)
- [Đánh giá Real Holdout EXP-015](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md)
- [Đánh giá Final Test V08 EXP-016](./training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md)
- [Báo cáo DenseNet121 Scratch 5-Seed v009](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL/EXP-DENSENET121-SCRATCH-5SEEDS-009/README.md)
- [Báo cáo Kiểm định Dữ liệu Toàn diện](./training_results/current_dataset/EXP-DENSENET121-DATASET-AUDIT-001/)
- [Bảng tổng hợp tiến độ thực nghiệm 2026](./training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md)
- [Classification training results](./training_results/README.md)
- [Current V04/V05 benchmark](./training_results/current_dataset/README.md)
- [Dataset changelog](./DATASET_CHANGELOG.md)
- [Bounding-box annotation guideline](./CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md)
- [YOLO failure/development archive](./failure/YOLO26/README.md)
- [Disease reference](./DISEASE_REFERENCE.md)

---

## Disease reference samples

Ảnh tham khảo nhận diện bệnh thực địa được lưu trong [`img_check/`](./img_check/).

Ba nhóm tổn thương chính:
- **Anthracnose**
- **Leaf Miner**
- **Red Rust**

Phần mô tả bệnh chi tiết không được dùng thay cho ground-truth review chuyên môn; annotation/model evaluation tuân theo dataset guideline và protocol của project.
