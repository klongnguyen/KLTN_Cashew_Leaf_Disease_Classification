# FAILURE TAKE 04 — YOLO26s Dense Small-Lesion Annotation + Expanded Dataset

> **Source result archive:** `EXP-Y26S-SMALL-004.zip`

## 1. Experiment Information

| Field | Value |
|---|---|
| External experiment name | `EXP-Y26S-SMALL-004` |
| Internal metadata recorded by notebook | `EXP-Y26S-SMALL-003` ⚠️ |
| Failure Take | `failure_take04` |
| Model | YOLO26s |
| Task | Direct disease object detection |
| Classes | `anthracnose`, `leaf_miner`, `red_rust` |
| Input size | `640 × 640` |
| Seed | `42` |
| Max epochs | `50` |
| Actual epochs | `42` (Early Stopping) |
| Batch size | `8` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Main annotation change | Bounding thêm các chi tiết bệnh nhỏ có ý nghĩa, nhưng không bounding toàn bộ mọi dấu hiệu trên lá |
| Dataset change | Tăng số lượng ảnh được annotation |
| YOLO augmentation | `mosaic=0.5`, `fliplr=0.5`, `mixup=0`, `copy_paste=0` |
| Failure type | `PERFORMANCE_FAILURE` + `DATA_POLICY_CHANGE` |
| Final decision | **Chưa chọn làm final detector, nhưng là bước cải thiện rõ so với Take 03** |

> **Traceability warning:** file ZIP bên ngoài là `EXP-Y26S-SMALL-004`, nhưng `experiment_config.json`, `metrics_summary.json`, `args.yaml` và checkpoint bên trong vẫn ghi `EXP-Y26S-SMALL-003`. Cần sửa `EXPERIMENT_ID` trước Take 05.

---

## 2. Mục tiêu của Take 04

Take 04 thay đổi chiến lược annotation theo hướng trung gian giữa Take 02 và Take 03:

- vẫn tránh việc bounding tất cả dấu hiệu li ti trên một lá;
- nhưng không bỏ qua hoàn toàn các chi tiết bệnh nhỏ nếu chúng rõ ràng và có ý nghĩa;
- tăng số lượng ảnh được annotation;
- giữ input size `640×640`;
- giữ nguyên YOLO26s và phần lớn hyperparameter để quan sát tác động của dữ liệu.

Mục tiêu là tăng lượng supervision cho detector nhưng vẫn hạn chế annotation noise từ các box quá mờ hoặc không có giá trị.

---

## 3. Dataset Statistics

| Split | Images | Boxes | Anthracnose | Leaf Miner | Red Rust |
|---|---:|---:|---:|---:|---:|
| Train | 597 | 14,693 | 4,067 | 429 | 10,197 |
| Validation | 59 | 1,468 | 365 | 43 | 1,060 |
| Test | 27 | 672 | 147 | 38 | 487 |

### Mật độ annotation

- Train: **24.61 boxes/image**
- Validation: **24.88 boxes/image**
- Test: **24.89 boxes/image**

So với Take 03:

| Metric | Take 03 | Take 04 | Thay đổi |
|---|---:|---:|---:|
| Train images | 347 | 597 | **+72.0%** |
| Train boxes | 1,174 | 14,693 | **×12.52** |
| Boxes / train image | 3.38 | 24.61 | **×7.27** |
| Test images | 16 | 27 | **+68.8%** |
| Test boxes | 53 | 672 | **×12.68** |

Take 04 vì vậy **không phải ablation trực tiếp** với Take 03. Cả số ảnh, số box và annotation policy đều thay đổi mạnh.

### Class imbalance

Tỷ lệ box trong Train:

- `anthracnose`: 4,067 / 14,693 ≈ **27.7%**
- `leaf_miner`: 429 / 14,693 ≈ **2.9%**
- `red_rust`: 10,197 / 14,693 ≈ **69.4%**

`leaf_miner` hiện là class thiếu dữ liệu rõ rệt, trong khi `red_rust` chiếm gần 70% toàn bộ box Train.

---

## 4. Kết quả Test tổng thể

| Metric | Take 04 |
|---|---:|
| Precision | **0.4830** |
| Recall | **0.5360** |
| F1-score | **0.5081** |
| mAP@0.50 | **0.4929** |
| mAP@0.75 | **0.2272** |
| mAP@0.50:0.95 | **0.2588** |
| Mean IoU (matched TP) | **0.7910** |
| Median IoU (matched TP) | **0.8076** |

Custom matching tại `confidence=0.25`, `IoU≥0.50`:

| Metric | Value |
|---|---:|
| TP | **353** |
| FP | **354** |
| FN | **319** |
| Precision | **0.4993** |
| Recall | **0.5253** |
| F1 | **0.5120** |

---

## 5. Kết quả theo từng class

### YOLO metrics

| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Anthracnose | **0.5778** | **0.5238** | **0.5495** | **0.5638** | **0.2841** |
| Leaf Miner | 0.4469 | 0.4211 | 0.4336 | 0.4063 | 0.2110 |
| Red Rust | 0.4243 | **0.6632** | 0.5175 | 0.5085 | 0.2812 |

### Custom IoU matching

| Class | TP | FP | FN | Precision | Recall | Mean IoU TP |
|---|---:|---:|---:|---:|---:|---:|
| Anthracnose | 63 | 31 | 84 | **0.6702** | 0.4286 | 0.7729 |
| Leaf Miner | 14 | 14 | 24 | 0.5000 | 0.3684 | **0.7981** |
| Red Rust | 276 | 309 | 211 | 0.4718 | **0.5667** | **0.7948** |

---

## 6. So sánh với Take 03

> **Cảnh báo:** Take 03 và Take 04 dùng annotation policy và quy mô dataset khác nhau. Bảng này dùng để theo dõi tiến trình, không được diễn giải như một ablation khoa học hoàn toàn.

| Metric | Take 03 | Take 04 | Delta |
|---|---:|---:|---:|
| Precision | 0.3111 | **0.4830** | **+0.1719** |
| Recall | 0.4295 | **0.5360** | **+0.1065** |
| F1 | 0.3608 | **0.5081** | **+0.1473** |
| mAP50 | 0.3522 | **0.4929** | **+0.1407** |
| mAP50-95 | 0.2148 | **0.2588** | **+0.0440** |
| Mean IoU TP | **0.8163** | 0.7910 | -0.0253 |
| Median IoU TP | **0.8288** | 0.8076 | -0.0212 |

Kết quả cho thấy việc tăng lượng annotation và đưa lại các lesion nhỏ có ý nghĩa đã giúp **khả năng tìm kiếm lesion tổng thể tăng rõ rệt**.

Đổi lại, Mean IoU giảm nhẹ từ `0.8163` xuống `0.7910`. Đây là trade-off hợp lý khi bài toán chuyển từ ít box tương đối lớn/rõ sang nhiều box nhỏ hơn và khó localization hơn.

---

## 7. Phân tích Training / Validation

Training dừng tại **epoch 42/50** do Early Stopping.

- Best validation mAP50 ≈ **0.5408** tại epoch **15**.
- Best validation mAP50-95 ≈ **0.2889** tại epoch **22**.
- Train losses giảm tương đối đều.
- Validation losses giảm mạnh ở giai đoạn đầu rồi dao động.
- Validation mAP50 ổn định quanh khoảng `0.48–0.53` ở giai đoạn sau.

Điều này cho thấy model đã **hội tụ và plateau**, không cần tăng epoch ngay lập tức. Early Stopping đang hoạt động đúng vai trò.

Test mAP50-95 = `0.2588`, khá gần vùng validation tốt nhất `0.2889`, vì vậy Take 04 không cho thấy generalization collapse nghiêm trọng.

---

## 8. Phân tích Confusion Matrix

Confusion Matrix normalized cho thấy:

### Anthracnose

Khoảng **0.88** true Anthracnose được nhận thành Anthracnose ở ma trận normalized. Đây là tín hiệu tốt hơn đáng kể so với Take 03.

### Red Rust

Khoảng **0.87** true Red Rust được nhận thành Red Rust. Việc tăng supervision cho các chi tiết nhỏ giúp Red Rust phục hồi mạnh về Recall.

### Leaf Miner

Leaf Miner là bottleneck mới:

- chỉ khoảng **0.34** true Leaf Miner được nhận đúng;
- khoảng **0.45** bị nhầm thành Anthracnose;
- khoảng **0.16** bị nhầm thành Red Rust.

Điều này phù hợp với tình trạng class imbalance: Leaf Miner chỉ chiếm khoảng **2.9%** số box Train.

### Background False Positive

Cột background cho thấy model vẫn tạo nhiều detection trên vùng không có GT:

- khoảng `0.45` background detection đi vào Anthracnose;
- khoảng `0.46` đi vào Red Rust.

Custom matching cũng xác nhận điều này với **354 FP**, gần bằng **353 TP**.

Đây là bottleneck quan trọng nhất của Take 04.

---

## 9. F1–Confidence Analysis

F1 curve đạt cực đại toàn class khoảng:

- **F1 ≈ 0.50**
- tại confidence khoảng **0.191**

Điều này cho thấy threshold `0.25` trong custom evaluation chưa chắc là operating point tối ưu.

Tuy nhiên **không được dùng Test để chọn threshold cuối**. Take 05 cần tìm confidence threshold tối ưu trên **Validation**, khóa threshold đó, sau đó mới đánh giá Test.

---

## 10. Điểm tích cực của Take 04

### 10.1. Detection tổng thể phục hồi mạnh

So với Take 03:

- Precision tăng;
- Recall tăng;
- F1 tăng;
- mAP50 tăng;
- mAP50-95 tăng.

Đây là bằng chứng tốt rằng annotation quá thưa ở Take 03 đã làm thiếu supervision.

### 10.2. Red Rust được cải thiện rõ

Take 03 Red Rust:

- Recall ≈ `0.205`
- mAP50 ≈ `0.206`

Take 04 Red Rust:

- Recall ≈ **0.663**
- mAP50 ≈ **0.508**

Đây là một cải thiện lớn.

### 10.3. Anthracnose cũng phục hồi

Take 04 Anthracnose:

- Precision ≈ **0.578**
- Recall ≈ **0.524**
- mAP50 ≈ **0.564**

Đây là class có mAP50 tốt nhất trong Take 04.

### 10.4. Localization vẫn tốt

Mean IoU TP = **0.7910** và Median IoU = **0.8076**.

Dù thấp hơn Take 03 một chút, đây vẫn là mức tốt cho một dataset có mật độ box lớn và nhiều lesion nhỏ.

---

## 11. Failure Analysis

### 11.1. False Positive vẫn rất cao

Ở `conf=0.25`, `IoU≥0.5`:

- TP = 353
- FP = 354
- FN = 319

Số FP gần bằng số TP. Model chưa đủ sạch để chọn làm detector cuối.

### 11.2. Leaf Miner thiếu supervision

Train chỉ có **429 Leaf Miner boxes**, so với:

- 4,067 Anthracnose;
- 10,197 Red Rust.

Việc bổ sung dữ liệu lần sau nên ưu tiên **ảnh nguồn Leaf Miner**, thay vì chỉ tiếp tục tăng Red Rust.

### 11.3. Annotation density rất cao

Train có trung bình **24.61 boxes/image**, cao hơn Take 03 hơn 7 lần.

Việc bounding lesion nhỏ là hợp lý nếu lesion:

- nhìn rõ;
- class xác định được;
- có kích thước đủ để học sau resize;
- không phải một dấu hiệu cực mờ/nhiễu.

Nhưng không nên chuyển thành quy tắc:

> mọi chấm bệnh nhìn thấy đều phải có một box.

Cần tiếp tục giữ nguyên nguyên tắc của Take 04: **annotate lesion nhỏ có ý nghĩa, không annotate toàn bộ mọi dấu hiệu trên lá**.

### 11.4. Thiếu negative images có thể góp phần làm FP cao

Trong thống kê hiện tại:

- `images_with_boxes = images` cho Train;
- `images_with_boxes = images` cho Validation;
- `images_with_boxes = images` cho Test.

Điều này cho thấy tập detection đang không có, hoặc có rất ít, ảnh negative không chứa disease box.

Đối với YOLO Direct, nên bổ sung một lượng có kiểm soát:

- Healthy cashew leaves;
- Not Cashew Leaf;
- cashew leaves/background khó nhưng không có lesion cần detect;

với **0 bounding box**.

Negative images có thể giúp model học tốt hơn khái niệm background và giảm False Positive.

### 11.5. Experiment ID vẫn bị ghi sai

Archive là `EXP-Y26S-SMALL-004`, nhưng nội bộ artifact là `EXP-Y26S-SMALL-003`.

Đây là lỗi quản lý experiment và phải sửa trước lần train tiếp theo.

---

## 12. Đánh giá chiến lược bounding mới

Chiến lược Take 04:

> bounding cả các chi tiết nhỏ nếu rõ ràng, nhưng không bounding tất cả mọi dấu hiệu trên lá.

Kết quả hiện tại **ủng hộ hướng này hơn Take 03**.

Take 03 làm annotation quá thưa:

- box sạch;
- IoU TP rất cao;
- nhưng supervision không đủ;
- Recall/mAP thấp.

Take 04 tăng lại các lesion nhỏ có ý nghĩa:

- IoU giảm nhẹ;
- nhưng Precision/Recall/mAP tăng đáng kể.

Do đó, guideline nên được khóa theo hướng **moderate-density annotation**:

```text
Lesion rõ và có ý nghĩa
        ↓
Annotate kể cả nhỏ

Lesion quá mờ / vài pixel / class không chắc
        ↓
Không annotate riêng

Nhiều lesion cực nhỏ gần nhau
        ↓
Gom cluster nếu hợp lý

Không bounding toàn bộ mọi dấu hiệu chỉ để tăng số box
```

---

## 13. Trạng thái Take 04

Take 04 chưa đạt mức final detector vì:

- FP còn cao;
- mAP50-95 mới `0.2588`;
- Leaf Miner yếu;
- class imbalance lớn;
- chưa có negative images rõ ràng;
- experiment ID sai.

Tuy nhiên đây là **bước tiến rõ ràng so với Take 03** và cung cấp hướng annotation hợp lý hơn.

Trạng thái:

`FAILURE FOR FINAL MODEL — PROMISING DATA ITERATION`

---

## 14. Khuyến nghị cho Take 05

### Ưu tiên 1 — Khóa annotation policy của Take 04

Không tiếp tục thay đổi định nghĩa lesion giữa mỗi lần train.

Khóa guideline:

- small-but-clear lesion → annotate;
- tiny/blurred/uncertain lesion → bỏ;
- cluster hợp lý → một box;
- không cố annotate tất cả dấu hiệu trên lá.

### Ưu tiên 2 — Tăng Leaf Miner bằng ảnh nguồn độc lập

Không cần tăng Red Rust trước.

Mục tiêu là cải thiện distribution:

```text
Leaf Miner 2.9%  → tăng đáng kể
```

Ưu tiên thêm:

- lá khác;
- cây khác;
- mức độ bệnh khác;
- background/lighting khác.

### Ưu tiên 3 — Bổ sung negative images

Đưa `healthy` và `not_cashew_leaf` vào YOLO dataset dưới dạng **0-box images**, đặc biệt trong Train và Validation.

Mục tiêu:

- giảm FP;
- giảm detection trên texture/background không phải lesion;
- cải thiện Precision.

### Ưu tiên 4 — Khóa split

Từ Take 05 nên khóa:

- Train;
- Validation;
- Test;
- annotation policy.

Sau đó chỉ thay **một biến** trong mỗi experiment.

### Ưu tiên 5 — Tune confidence trên Validation

F1 curve Test gợi ý operating point quanh `0.19`, nhưng threshold cuối phải được tìm trên Validation.

Quy trình:

```text
Validation
→ tìm confidence threshold
→ khóa threshold
→ Test đúng 1 lần
```

### Ưu tiên 6 — Sửa Experiment ID

Trước khi chạy:

```python
EXPERIMENT_ID = "EXP-Y26S-SMALL-005"
```

và kiểm tra:

- archive name;
- metrics JSON;
- config JSON;
- args.yaml;
- checkpoint name;

đều cùng ID.

---

## 15. Kết luận

Take 04 cho thấy việc **bounding lại các lesion nhỏ có ý nghĩa và tăng số ảnh annotation** là hướng cải thiện khả thi.

So với Take 03:

- F1 tăng từ `0.3608` lên `0.5081`;
- mAP50 tăng từ `0.3522` lên `0.4929`;
- Recall tăng từ `0.4295` lên `0.5360`;
- Mean IoU chỉ giảm nhẹ từ `0.8163` xuống `0.7910`.

Điều này cho thấy Take 03 có thể đã làm annotation quá thưa, còn Take 04 đạt cân bằng tốt hơn giữa:

```text
annotation cleanliness
        +
small lesion supervision
        +
localization quality
```

Bottleneck tiếp theo không còn chỉ là cách vẽ box, mà chuyển sang:

1. **class imbalance — đặc biệt Leaf Miner**;
2. **False Positive cao**;
3. **thiếu negative images**;
4. **cần khóa benchmark/split để bắt đầu ablation sạch**.

Take 04 vì vậy nên được giữ làm **baseline annotation policy mới cho Take 05**, dù chưa đạt yêu cầu final detector.
