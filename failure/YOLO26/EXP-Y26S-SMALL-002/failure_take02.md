# FAILURE TAKE 02 — YOLO26s Resolution 640×640

> **Source result archive:** `EXP-Y26S-SMALL-002.zip`

## 1. Experiment Information

| Field | Value |
|---|---|
| External experiment name | `EXP-Y26S-SMALL-002` |
| Internal metadata recorded by notebook | `EXP-Y26S-SMALL-001` ⚠️ |
| Failure Take | `failure_take02` |
| Model | YOLO26s |
| Task | Direct disease object detection |
| Classes | `anthracnose`, `leaf_miner`, `red_rust` |
| Main change from Take 01 | Roboflow dataset resize changed to `640×640` |
| Training `imgsz` | `640` |
| Seed | `42` |
| Epochs | `50` |
| Batch size | `8` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Failure type | `PERFORMANCE_FAILURE` + `DATA_FAILURE` |
| Final decision | **Không chọn làm final model** |

---

## 2. Mục tiêu của Take 02

Take 02 được thực hiện để kiểm tra giả thuyết:

> Việc tăng kích thước ảnh dataset từ mức resize thấp ở Take 01 lên `640×640` có giúp YOLO26s phát hiện tốt hơn các lesion nhỏ trên lá điều hay không?

Theo kế hoạch ban đầu, đây được xem như một **resolution ablation**: giữ nguyên model và hyperparameter, chỉ thay đổi resolution của dữ liệu.

Tuy nhiên, sau khi kiểm tra artifact của hai experiment, Train set giữa Take 01 và Take 02 **không hoàn toàn giống nhau**. Vì vậy Take 02 chưa phải một ablation resolution sạch và kết quả chỉ được dùng như bằng chứng thử nghiệm sơ bộ.

---

## 3. Cấu hình huấn luyện

```text
Model            : yolo26s.pt
Epochs           : 50
Patience         : 20
Training imgsz   : 640
Batch size       : 8
Optimizer        : AdamW
Learning rate    : 0.001
Weight decay     : 0.0005
Mosaic           : 0.5
MixUp            : 0.0
Copy-Paste       : 0.0
Horizontal Flip  : 0.5
Seed             : 42
Prediction conf  : 0.25
NMS IoU          : 0.70
Matching IoU     : 0.50
```

So với Take 01, mục tiêu thay đổi duy nhất là **resize dataset thành 640×640**.

---

## 4. Dataset Statistics

### 4.1. Take 02

| Split | Images | Boxes | Anthracnose | Leaf Miner | Red Rust |
|---|---:|---:|---:|---:|---:|
| Train | 356 | 4,796 | 1,371 | 253 | 3,172 |
| Validation | 34 | 466 | 119 | 25 | 322 |
| Test | 16 | 202 | 73 | 11 | 118 |

### 4.2. So sánh với Take 01

| Split / Metric | Take 01 | Take 02 | Difference |
|---|---:|---:|---:|
| Train images | 355 | 356 | **+1** |
| Train boxes | 4,828 | 4,796 | **-32** |
| Train Anthracnose boxes | 1,414 | 1,371 | **-43** |
| Train Leaf Miner boxes | 243 | 253 | **+10** |
| Train Red Rust boxes | 3,171 | 3,172 | **+1** |
| Validation images | 34 | 34 | 0 |
| Validation boxes | 466 | 466 | 0 |
| Test images | 16 | 16 | 0 |
| Test boxes | 202 | 202 | 0 |

### 4.3. Vấn đề thực nghiệm

Nếu chỉ thay đổi resolution, lý tưởng phải giữ nguyên:

```text
same images
same labels
same boxes
same split
same seed
same hyperparameters
```

Nhưng Train set đã thay đổi số ảnh và số bounding box. Do đó:

> Không thể khẳng định toàn bộ chênh lệch giữa Take 01 và Take 02 chỉ đến từ resolution.

Đây là lý do Take 02 được gắn thêm `DATA_FAILURE`.

---

## 5. Test Metrics — Take 02

| Metric | Result |
|---|---:|
| Precision | **0.5811** |
| Recall | **0.5099** |
| F1-score | **0.5432** |
| mAP@0.50 | **0.5336** |
| mAP@0.75 | **0.3135** |
| mAP@0.50:0.95 | **0.3073** |
| Mean IoU of matched TP | **0.7622** |
| Median IoU of matched TP | **0.7707** |

---

## 6. So sánh Take 01 và Take 02

| Metric | Take 01 | Take 02 | Change |
|---|---:|---:|---:|
| Precision | 0.6196 | 0.5811 | **-0.0385** |
| Recall | 0.5637 | 0.5099 | **-0.0538** |
| F1 | 0.5903 | 0.5432 | **-0.0471** |
| mAP@0.50 | 0.5589 | 0.5336 | **-0.0253** |
| mAP@0.75 | 0.3215 | 0.3135 | **-0.0080** |
| mAP@0.50:0.95 | 0.2988 | **0.3073** | **+0.0085** |
| Mean IoU | 0.7505 | **0.7622** | **+0.0117** |
| Median IoU | 0.7565 | **0.7707** | **+0.0142** |

### Nhận xét chính

Resolution cao hơn tạo ra tín hiệu tích cực ở **localization quality**:

- mAP@0.50:0.95 tăng nhẹ;
- Mean IoU tăng;
- Median IoU tăng.

Nhưng hiệu năng detection tổng thể không cải thiện:

- Precision giảm;
- Recall giảm;
- F1 giảm;
- mAP@0.50 giảm.

Vì vậy kết luận của Take 02 là:

> **Resize 640×640 tạo cải thiện nhẹ về độ khớp bounding box khi detect đúng, nhưng chưa cải thiện khả năng phát hiện tổng thể.**

---

## 7. Custom IoU Analysis

Cấu hình custom evaluation:

```text
Prediction confidence threshold : 0.25
NMS IoU                         : 0.70
Matching IoU threshold          : 0.50
```

### 7.1. Overall

| Metric | Take 01 | Take 02 | Change |
|---|---:|---:|---:|
| TP | 67 | **73** | +6 |
| FP | 53 | **83** | **+30** |
| FN | 135 | **129** | -6 |
| Custom Precision | 0.5583 | 0.4679 | -0.0904 |
| Custom Recall | 0.3317 | 0.3614 | +0.0297 |
| Custom F1 | 0.4161 | 0.4078 | -0.0083 |
| Mean IoU | 0.7505 | 0.7622 | +0.0117 |

Take 02 phát hiện thêm một số lesion thật:

```text
TP +6
FN -6
```

nhưng đồng thời tạo ra rất nhiều detection dư:

```text
FP +30
```

Đây là lý do Precision giảm mạnh trong custom evaluation.

---

## 8. Per-Class Performance

### 8.1. Take 02

| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Anthracnose | 0.4537 | 0.2055 | 0.2829 | 0.3061 | 0.1227 |
| Leaf Miner | 0.8379 | 0.9091 | 0.8720 | 0.9050 | 0.6063 |
| Red Rust | 0.4517 | 0.4153 | 0.4327 | 0.3899 | 0.1928 |

### 8.2. So sánh với Take 01

| Class | Metric | Take 01 | Take 02 | Observation |
|---|---|---:|---:|---|
| Anthracnose | Precision | 0.3781 | **0.4537** | tăng |
| Anthracnose | Recall | **0.3499** | 0.2055 | giảm mạnh |
| Anthracnose | mAP50 | 0.3022 | 0.3061 | gần như không đổi |
| Anthracnose | mAP50-95 | 0.1220 | 0.1227 | gần như không đổi |
| Leaf Miner | mAP50 | **0.9399** | 0.9050 | giảm nhẹ |
| Leaf Miner | mAP50-95 | 0.5896 | **0.6063** | tăng nhẹ |
| Red Rust | Precision | **0.6120** | 0.4517 | giảm mạnh |
| Red Rust | Recall | 0.4322 | 0.4153 | giảm nhẹ |
| Red Rust | mAP50 | **0.4347** | 0.3899 | giảm |
| Red Rust | mAP50-95 | 0.1847 | **0.1928** | tăng nhẹ |

### Phân tích

`leaf_miner` vẫn là class mạnh nhất, nhưng Test chỉ có **11 bounding boxes**, nên metric có độ biến động cao và chưa đủ để kết luận đây chắc chắn là class dễ nhất.

`anthracnose` vẫn là class yếu nhất. Recall giảm xuống khoảng **0.205**, nghĩa là phần lớn lesion Anthracnose vẫn bị bỏ sót.

`red_rust` có mAP50-95 tăng nhẹ nhưng Precision giảm mạnh. Điều này cho thấy model có thể localization tốt hơn ở một số prediction đúng, nhưng đồng thời tạo thêm nhiều prediction không chính xác.

---

## 9. Failure Analysis

### 9.1. Resolution không phải bottleneck duy nhất

Nếu bottleneck chính chỉ là ảnh quá nhỏ, việc tăng resolution lên `640×640` đáng lẽ phải cải thiện rõ Recall và mAP tổng thể. Kết quả không cho thấy xu hướng đó.

Resolution cao hơn có giúp chất lượng localization tăng nhẹ, nhưng chưa giải quyết được:

- bỏ sót lesion;
- false positive;
- class imbalance về độ khó;
- annotation của các cụm lesion nhỏ.

### 9.2. Annotation của Anthracnose và Red Rust cần review

Hai class này thường có:

- nhiều đốm nhỏ;
- lesion nằm sát nhau;
- nhiều bounding box trên cùng một lá;
- vùng bệnh có texture/màu sắc không đồng nhất;
- nguy cơ annotation quá vụn.

Nếu một cụm lesion được annotate thành quá nhiều box nhỏ, model có thể học ra nhiều detection gần nhau và tăng FP.

### 9.3. Tăng chi tiết có thể làm tăng false positive

Take 02 có `FP = 83`, tăng 30 so với Take 01. Một giả thuyết hợp lý là ảnh 640 giữ lại nhiều texture/đốm nhỏ hơn, khiến model nhạy hơn với các pattern giống lesion nhưng chưa phân biệt đủ tốt.

Đây là **giả thuyết**, không phải kết luận chắc chắn. Cần kiểm tra prediction và annotation trên Validation để xác nhận.

### 9.4. Test set quá nhỏ

Test chỉ có:

```text
16 images
202 boxes
```

Trong đó Leaf Miner chỉ có 11 box. Vì vậy một số metric có thể thay đổi đáng kể chỉ vì vài detection đúng/sai.

### 9.5. Experiment traceability bị sai ID

File ZIP được đặt tên:

```text
EXP-Y26S-SMALL-002.zip
```

nhưng artifact bên trong vẫn ghi:

```text
experiment_id = EXP-Y26S-SMALL-001
```

Checkpoint và run name cũng tiếp tục dùng `EXP-Y26S-SMALL-001`.

Đây không làm thay đổi metric, nhưng là lỗi quản lý thí nghiệm quan trọng vì gây khó truy vết kết quả về sau.

---

## 10. Kết luận Take 02

### Trạng thái

```text
RESOLUTION IMPROVEMENT = MIXED / INCONCLUSIVE
FINAL MODEL             = NO
KEEP FOR FAILURE STUDY  = YES
```

### Kết luận ngắn

Take 02 cho thấy tăng dataset resize lên 640×640:

**Có cải thiện:**

- Mean IoU;
- Median IoU;
- mAP50-95 tăng nhẹ;
- TP tăng 6;
- FN giảm 6.

**Không cải thiện hoặc xấu hơn:**

- Precision;
- Recall YOLO;
- F1;
- mAP50;
- FP tăng mạnh;
- Anthracnose Recall giảm mạnh;
- Red Rust Precision giảm mạnh.

Do Train set giữa hai take không hoàn toàn giống nhau, kết quả này **không được dùng để khẳng định 640×640 tốt hơn hay kém hơn 224×224 một cách khoa học**.

---

## 11. Root Cause Summary

```text
Take 01 cho thấy:
model detect đúng thì box khá ổn
nhưng bỏ sót nhiều lesion
        ↓
Take 02 tăng resolution lên 640
        ↓
localization tốt hơn nhẹ
        ↓
nhưng FP tăng mạnh
và Recall/mAP50 không cải thiện
        ↓
=> resolution không phải nguyên nhân duy nhất
        ↓
ưu tiên kiểm tra annotation + dataset consistency
```

---

## 12. Action Plan trước Take 03

### A. Khóa dataset pilot

Tạo một dataset pilot cố định:

```text
Master Pilot
├── Train
├── Validation
└── Test
```

Từ cùng một master annotation sinh ra các phiên bản resolution khác nhau.

Bắt buộc giữ nguyên:

- số ảnh;
- filename;
- split;
- số bounding box;
- class ID;
- annotation.

### B. Review Anthracnose và Red Rust

Ưu tiên review 30–50 ảnh có:

- nhiều lesion nhỏ;
- nhiều box gần nhau;
- box chồng lấn;
- cụm đốm dày;
- prediction tạo nhiều FP.

Áp dụng đúng Annotation Guideline:

```text
nhiều đốm nhỏ sát nhau
        ↓
gom thành cluster hợp lý
        ↓
1 bounding box có ý nghĩa thị giác
```

### C. Không tune bằng Test

Các thay đổi sau phải quyết định trên Validation:

- confidence threshold;
- NMS IoU;
- augmentation;
- learning rate;
- epochs;
- resolution.

Test chỉ dùng sau khi cấu hình đã khóa.

### D. Sửa Experiment ID

Take tiếp theo phải đổi trong notebook trước khi chạy:

```python
EXPERIMENT_ID = "EXP-Y26S-...-003"
```

Tên ZIP, run directory, metrics JSON và checkpoint phải cùng một Experiment ID.

### E. Resolution ablation sạch

Khuyến nghị không resize nhiều lần bằng Roboflow nếu không cần thiết. Tốt nhất giữ ảnh nguồn ở chất lượng cao và để YOLO thực hiện input resize bằng `imgsz`.

Ví dụ:

```text
same frozen dataset
      ↓
YOLO imgsz=640
      ↓
YOLO imgsz=960
```

Chỉ thay đúng **một biến** giữa hai experiment.

---

## 13. Acceptance Criteria cho experiment tiếp theo

Experiment tiếp theo chỉ được xem là ablation hợp lệ khi:

- [ ] Train/Val/Test hoàn toàn giống nhau.
- [ ] Số box từng class giống nhau giữa các experiment.
- [ ] Experiment ID đúng trong toàn bộ artifact.
- [ ] Seed giống nhau.
- [ ] Hyperparameter giống nhau, trừ đúng biến đang khảo sát.
- [ ] Threshold được lựa chọn từ Validation, không phải Test.
- [ ] Báo cáo đầy đủ Precision, Recall, F1, mAP50, mAP50-95 và per-class AP.
- [ ] Kiểm tra false-positive và false-negative đại diện.

---

## 14. Decision

- [x] Lưu Take 02 trong `failure/` để phục vụ Failure Analysis.
- [x] Không chọn Take 02 làm final YOLO model.
- [x] Ghi nhận resolution 640 có tín hiệu cải thiện localization.
- [x] Không kết luận resolution 640 tốt hơn vì dataset Train không hoàn toàn giống Take 01.
- [ ] Review annotation Anthracnose/Red Rust.
- [ ] Khóa dataset pilot trước Take 03.
- [ ] Thực hiện resolution ablation sạch sau khi dataset được khóa.

---

## 15. Artifact trong thư mục này

```text
EXP-Y26S-SMALL-002/
├── failure_take02.md
├── config/
│   └── experiment_config.json
└── metrics/
    ├── metrics_summary.json
    ├── final_summary.csv
    ├── per_class_metrics.csv
    └── dataset_statistics.csv
```

> Full training ZIP vẫn được giữ ngoài Git repository nếu checkpoint/figure quá lớn. Repository chỉ lưu những artifact cần thiết để tái lập phân tích và đối chiếu kết quả.
