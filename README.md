# KLTN — Cashew Leaf Disease Classification & Detection

Khóa luận xây dựng hệ thống thị giác máy tính cho **phân loại bệnh trên lá cây điều** và hướng mở rộng sang **phát hiện vùng tổn thương bằng bounding box**.

## Project status

### Classification — latest results

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

- [`EXP-DENSENET121-SCRATCH-5SEEDS-005`](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-005/) — V05 validation complete; Test pending
- [`EXP-RESNET50-SCRATCH-5SEEDS-005`](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-005/) — V05 validation complete; Test pending
- [`EXP-VIT-SCRATCH-5SEEDS-005`](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-005/) — V05 validation complete; Test pending; source archive ID was EXP-004

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
├── README.md
├── DATASET_CHANGELOG.md
├── dataset_cashew_v04.csv
├── CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md
├── DISEASE_REFERENCE.md
├── img_check/
├── training_results/
│   ├── README.md
│   ├── current_dataset/
│   └── archive_legacy_dataset/
└── failure/
    ├── README.md
    └── YOLO26/
```

### Result management

- `training_results/current_dataset/`: benchmark V04 chính thức và các validation rerun đang chờ final Test.
- `training_results/archive_legacy_dataset/`: experiment rất cũ.
- `failure/YOLO26/`: detection iteration/failure analysis.
- Checkpoint/model/FULL ZIP lớn **không commit trực tiếp**; repo chỉ giữ config, metrics, summaries và figures nhẹ.

---

## Dataset versioning

`Cashew_dataV03` có 7,213 ảnh. Sau cleaning bổ sung, V04 còn 6,911 ảnh.

V03 và V04 có Train/Validation/Test khác nhau, vì vậy **không dùng metric V03 ↔ V04 như một controlled model comparison**.

Git history vẫn giữ toàn bộ phiên bản cũ.

---

## Tài liệu chính

- [Classification training results](./training_results/README.md)
- [Current V04 benchmark](./training_results/current_dataset/README.md)
- [Latest DenseNet121 EXP-004 analysis](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-004/ANALYSIS.md)
- [Latest ResNet50 EXP-004 analysis](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-004/ANALYSIS.md)
- [Latest ViT EXP-004 analysis](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-004/ANALYSIS.md)
- [Dataset changelog](./DATASET_CHANGELOG.md)
- [Bounding-box annotation guideline](./CASHEW_BOUNDING_BOX_ANNOTATION_GUIDELINE.md)
- [YOLO failure/development archive](./failure/YOLO26/README.md)
- [Failure archive index](./failure/README.md)
- [Disease reference](./DISEASE_REFERENCE.md)

---

## Disease reference samples

Ảnh tham khảo nhận diện bệnh thực địa được lưu trong [`img_check/`](./img_check/).

Ba nhóm tổn thương chính:
- **Anthracnose**
- **Leaf Miner**
- **Red Rust**

Phần mô tả bệnh chi tiết không được dùng thay cho ground-truth review chuyên môn; annotation/model evaluation tuân theo dataset guideline và protocol của project.
