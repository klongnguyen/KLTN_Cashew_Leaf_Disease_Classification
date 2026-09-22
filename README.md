# KLTN — Cashew Leaf Disease Classification & Detection

Khóa luận xây dựng hệ thống thị giác máy tính cho **phân loại bệnh trên lá cây điều** và hướng mở rộng sang **phát hiện vùng tổn thương bằng bounding box**.

## Project status

### Classification — current benchmark

Current dataset: **`Cashew_dataV04` — 6,911 images / 5 classes**

| Model | Test Accuracy | Macro F1 | Params |
|---|---:|---:|---:|
| **DenseNet121 Scratch** | **91.82 ± 0.86%** | **91.37 ± 0.79%** | 7.57M |
| **ResNet50 Scratch** | **90.63 ± 2.11%** | **90.17 ± 2.04%** | 24.64M |
| **Compact ViT Scratch** | **85.30 ± 1.68%** | **84.41 ± 1.71%** | **0.35M** |

➡️ Chi tiết: [`training_results/current_dataset/README.md`](./training_results/current_dataset/README.md)

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

## Current dataset — Cashew_dataV04

V04 được tạo sau lần làm sạch bổ sung:
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
- [`dataset_cashew_v04.csv`](./dataset_cashew_v04.csv) — machine-readable snapshot hiện tại;
- [`DATASET_CHANGELOG.md`](./DATASET_CHANGELOG.md) — lịch sử V03 → V04.

> File Excel V03 cũ đã được loại khỏi root để tránh nhầm với current dataset. Git history vẫn giữ bản cũ.

---

## Classification protocol

Mỗi baseline current:
- dùng cùng fixed split V04;
- train 5 seeds: `42, 123, 2026, 3407, 7777`;
- checkpoint chọn theo Validation;
- Test không dùng để tuning hoặc chọn seed;
- báo cáo `Mean ± sample Standard Deviation (ddof=1)`.

Current experiments:

- [`EXP-DENSENET121-SCRATCH-5SEEDS-003`](./training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/)
- [`EXP-RESNET50-SCRATCH-5SEEDS-003`](./training_results/current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-003/)
- [`EXP-VIT-SCRATCH-5SEEDS-003`](./training_results/current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/)

`anthracnose` hiện là class khó nhất nhất quán qua cả ba kiến trúc.

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

- `training_results/current_dataset/`: benchmark V04 chính thức.
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
