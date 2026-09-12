# FAILURE TAKE 03 — YOLO26s Cleaned Bounding Boxes + Offline Augmentation

> **Source result archive:** `EXP-Y26S-SMALL-003.zip`

## 1. Experiment Information

| Field | Value |
|---|---|
| Experiment ID | `EXP-Y26S-SMALL-003` |
| Failure Take | `failure_take03` |
| Model | YOLO26s |
| Task | Direct disease object detection |
| Classes | `anthracnose`, `leaf_miner`, `red_rust` |
| Input size | `640 × 640` |
| Seed | `42` |
| Epochs | `50` |
| Batch size | `8` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Main annotation change | Giảm mạnh box quá nhỏ, tránh box chồng đè, hạn chế vùng bệnh quá mờ |
| Offline augmentation | Horizontal Flip + Rotate 90° |
| YOLO augmentation | `mosaic=0.5`, `fliplr=0.5`, `mixup=0`, `copy_paste=0` |
| Failure type | `PERFORMANCE_FAILURE` + `DATA_POLICY_CHANGE` |
| Final decision | **Chưa chọn làm final detector** |

---

## 2. Mục tiêu của Take 03

Take 03 kiểm tra một hướng cải thiện quan trọng ở mức dữ liệu:

1. giảm số lượng bounding box quá nhỏ;
2. không để nhiều box chồng đè lên cùng một vùng tổn thương;
3. bỏ qua/hạn chế annotation các vùng bệnh quá mờ hoặc khó xác định;
4. giữ ảnh ở kích thước `640×640`;
5. bổ sung augmentation offline bằng **horizontal flip** và **rotate 90°**.

Mục tiêu là làm cho một bounding box đại diện cho một vùng tổn thương có ý nghĩa thị giác hơn, giảm annotation noise và giúp YOLO học localization rõ ràng hơn.

---

## 3. Dataset Statistics

| Split | Images | Boxes | Anthracnose | Leaf Miner | Red Rust |
|---|---:|---:|---:|---:|---:|
| Train | 347 | 1174 | 403 | 193 | 578 |
| Validation | 33 | 105 | 47 | 24 | 34 |
| Test | 16 | 53 | 6 | 8 | 39 |

### So với Take 02

| Split | Take 02 boxes | Take 03 boxes | Thay đổi |
|---|---:|---:|---:|
| Train | 4796 | 1174 | **-75.5%** |
| Validation | 466 | 105 | **-77.5%** |
| Test | 202 | 53 | **-73.8%** |

Đây là một thay đổi rất lớn về **annotation policy**. Vì vậy Take 03 **không phải ablation trực tiếp** với Take 01/02. Metric giữa các take không thể được hiểu đơn giản là “annotation mới tốt hơn/xấu hơn” vì ground truth đã thay đổi đáng kể.

---

## 4. Kết quả Test tổng thể

| Metric | Take 03 |
|---|---:|
| Precision | **0.3111** |
| Recall | **0.4295** |
| F1-score | **0.3608** |
| mAP@0.50 | **0.3522** |
| mAP@0.75 | **0.2544** |
| mAP@0.50:0.95 | **0.2148** |
| Mean IoU (matched TP) | **0.8163** |
| Median IoU (matched TP) | **0.8288** |

Custom matching tại `confidence=0.25`, `IoU≥0.50`:

| Metric | Value |
|---|---:|
| TP | 14 |
| FP | 37 |
| FN | 39 |
| Precision | 0.2745 |
| Recall | 0.2642 |
| F1 | 0.2692 |

---

## 5. Kết quả theo từng class

| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Anthracnose | 0.1581 | 0.3333 | 0.2145 | 0.2378 | 0.1462 |
| Leaf Miner | 0.4401 | 0.7500 | 0.5547 | 0.6126 | 0.3877 |
| Red Rust | 0.3350 | 0.2051 | 0.2544 | 0.2063 | 0.1106 |

Custom IoU matching:

| Class | TP | FP | FN | Mean IoU TP |
|---|---:|---:|---:|---:|
| Anthracnose | 2 | 11 | 4 | 0.8467 |
| Leaf Miner | 6 | 10 | 2 | 0.8356 |
| Red Rust | 6 | 16 | 33 | 0.7870 |

---

## 6. Điểm tích cực của Take 03

### 6.1. Localization của các True Positive tốt hơn rõ rệt

Take 03 đạt:

- Mean IoU TP = **0.8163**
- Median IoU TP = **0.8288**

Đây là mức cao nhất trong ba take hiện tại:

| Take | Mean IoU | Median IoU |
|---|---:|---:|
| Take 01 | 0.7505 | 0.7565 |
| Take 02 | 0.7622 | 0.7707 |
| **Take 03** | **0.8163** | **0.8288** |

Tín hiệu này phù hợp với mục tiêu annotation mới: khi model phát hiện đúng một vùng bệnh, box có xu hướng khớp ground truth tốt hơn.

### 6.2. Leaf Miner vẫn là class ổn định nhất

`leaf_miner` đạt:

- Recall = **0.7500**
- mAP50 = **0.6126**
- mAP50-95 = **0.3877**
- Mean IoU TP = **0.8356**

Tuy nhiên Test chỉ có **8 box Leaf Miner**, nên metric class này còn nhạy với từng mẫu đơn lẻ.

### 6.3. Experiment ID đã được quản lý đúng

Khác Take 02, metadata, checkpoint và archive của Take 03 đều dùng đúng:

`EXP-Y26S-SMALL-003`

Điều này giúp experiment có thể truy vết đúng.

---

## 7. Failure Analysis

### 7.1. Detection tổng thể vẫn chưa đạt

Dù localization TP tốt hơn, detection tổng thể vẫn thấp:

- Precision = **0.3111**
- Recall = **0.4295**
- mAP50 = **0.3522**
- mAP50-95 = **0.2148**

Custom matching còn cho thấy:

- 14 TP
- 37 FP
- 39 FN

Nghĩa là model vẫn vừa **dự đoán dư** vừa **bỏ sót nhiều lesion**.

### 7.2. Red Rust là bottleneck lớn nhất ở Test

Test có 39 Red Rust boxes nhưng custom matching chỉ có:

- TP = 6
- FN = 33
- Recall = 0.1538

Mặc dù Mean IoU của các Red Rust TP đạt 0.7870, model chưa tìm được phần lớn vùng Red Rust thật.

### 7.3. Anthracnose Test quá nhỏ để kết luận ổn định

Test chỉ có **6 Anthracnose boxes**. Vì vậy các metric Anthracnose hiện có độ bất định cao. Không nên dùng 6 box này để kết luận rằng annotation strategy mới làm Anthracnose tốt hơn hay xấu hơn.

### 7.4. Test distribution đang lệch mạnh

Trong 53 ground-truth boxes của Test:

- Anthracnose: 6
- Leaf Miner: 8
- Red Rust: 39

Red Rust chiếm khoảng **73.6%** toàn bộ Test boxes. Test set này chưa cân bằng đủ để dùng như một benchmark ổn định theo class.

### 7.5. Có dấu hiệu overfitting sau khoảng epoch 30

Validation đạt đỉnh khoảng epoch 30:

- best validation mAP50 ≈ **0.3463**
- best validation mAP50-95 ≈ **0.1729**

Sau đó:

- train losses tiếp tục giảm;
- validation box/class/L1 loss có xu hướng tăng trở lại;
- validation mAP không cải thiện thêm.

Do đó, model bắt đầu có dấu hiệu overfitting sau vùng epoch ~30.

### 7.6. Offline augmentation không tạo thêm thông tin độc lập

Horizontal flip và Rotate 90° giúp tăng số mẫu quan sát, nhưng chúng vẫn được sinh từ cùng ảnh nguồn.

Đồng thời training config đã có:

- `fliplr=0.5`
- `mosaic=0.5`

Do đó có khả năng augmentation offline và augmentation runtime đang bị **trùng vai trò**. Đây là một giả thuyết cần kiểm chứng bằng ablation, không phải kết luận chắc chắn từ Take 03.

---

## 8. So sánh ba Take

> **Cảnh báo:** Take 03 dùng annotation policy và ground truth khác mạnh so với Take 01/02. Bảng dưới chỉ dùng để theo dõi lịch sử experiment, **không phải so sánh apples-to-apples**.

| Metric | Take 01 | Take 02 | Take 03 |
|---|---:|---:|---:|
| Precision | 0.6196 | 0.5811 | **0.3111** |
| Recall | 0.5637 | 0.5099 | **0.4295** |
| F1 | 0.5903 | 0.5432 | **0.3608** |
| mAP50 | 0.5589 | 0.5336 | **0.3522** |
| mAP50-95 | 0.2988 | 0.3073 | **0.2148** |
| Mean IoU TP | 0.7505 | 0.7622 | **0.8163** |

Điểm quan trọng nhất:

> **Take 03 cải thiện chất lượng localization của box đúng, nhưng chưa cải thiện khả năng phát hiện tổng thể.**

Điều này gợi ý annotation mới có thể giúp box “sạch” hơn, nhưng số lượng supervision giảm rất mạnh và dataset nhỏ vẫn chưa đủ để model học ổn định tất cả class.

---

## 9. Đánh giá augmentation Horizontal Flip + Rotate 90°

### Horizontal Flip

Giữ lại được vì hình thái lesion trên lá không phụ thuộc hướng trái/phải.

### Rotate 90°

Có thể sử dụng trong bài toán này vì hướng của chiếc lá không quyết định class bệnh. Tuy nhiên:

- không làm tăng đa dạng bệnh thật;
- không giải quyết class imbalance;
- không giải quyết thiếu ảnh nguồn;
- có thể tạo nhiều mẫu tương quan cao với ảnh gốc.

Do đó Rotate 90° chỉ nên là **augmentation hỗ trợ**, không nên được xem là giải pháp chính để bù thiếu dữ liệu.

---

## 10. Kết luận Take 03

Take 03 **không nên bị xem là thất bại hoàn toàn về annotation**.

Kết quả cho thấy hai tín hiệu khác nhau:

**Tích cực**
- box TP khớp ground truth tốt hơn;
- Mean IoU tăng lên 0.8163;
- annotation đã giảm mạnh các box nhỏ/chồng đè/mờ;
- traceability của experiment được sửa đúng.

**Chưa đạt**
- Precision thấp;
- mAP50 và mAP50-95 chưa đủ tốt;
- FP và FN vẫn nhiều;
- Red Rust Recall rất thấp;
- Test set quá nhỏ và lệch;
- có dấu hiệu overfitting sau epoch 30.

Vì vậy trạng thái phù hợp là:

`FAILURE FOR FINAL MODEL — USEFUL DATA/ANNOTATION ITERATION`

---

## 11. Khuyến nghị cho Take 04

### Ưu tiên 1 — Giữ annotation policy mới

Không quay lại hàng nghìn box li ti/chồng đè.

Giữ các nguyên tắc:

- lesion rõ ràng;
- box sát vùng bệnh;
- gom cụm hợp lý;
- bỏ vùng quá mờ;
- không box chồng đè không cần thiết.

### Ưu tiên 2 — Tăng **ảnh nguồn độc lập**, không chỉ augmentation

Hiện số box đã giảm rất mạnh. Cần bù bằng:

- thêm ảnh thật;
- thêm lá khác;
- nhiều mức độ bệnh;
- nhiều background/lighting;
- đặc biệt tăng Anthracnose và Red Rust.

### Ưu tiên 3 — Khóa Test set đủ lớn

Nên tạo Test cố định có đủ đại diện từng class. Không thay Test annotation/split giữa các experiment sau khi đã khóa benchmark.

### Ưu tiên 4 — Tạo một ablation sạch cho augmentation

Experiment tiếp theo nên giữ nguyên:

- images;
- boxes;
- split;
- seed;
- YOLO26s;
- hyperparameters.

Chỉ so sánh:

`No offline augmentation` vs `Horizontal Flip + Rotate 90°`

hoặc ưu tiên **chỉ dùng augmentation runtime của YOLO** trước.

### Ưu tiên 5 — Giảm overfitting

Dựa trên Take 03, có thể thử:

- `epochs=35–40`, hoặc
- EarlyStopping với patience ngắn hơn;

nhưng chỉ sau khi khóa dataset để không thay đồng thời quá nhiều yếu tố.

---

## 12. Decision

- [x] Giữ Take 03 trong Failure Archive.
- [x] Giữ annotation policy mới làm hướng phát triển.
- [x] Không dùng model này làm final detector.
- [x] Không kết luận augmentation offline giúp model dựa trên Take 03.
- [ ] Tăng dữ liệu nguồn độc lập.
- [ ] Khóa Train/Val/Test.
- [ ] Tạo augmentation ablation sạch.
- [ ] Train `EXP-Y26S-SMALL-004`.
