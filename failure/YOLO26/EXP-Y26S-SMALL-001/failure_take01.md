# FAILURE TAKE 01 — YOLO26s Small Dataset Baseline

## 1. Experiment Information

| Field | Value |
|---|---|
| Experiment ID | `EXP-Y26S-SMALL-001` |
| Failure Take | `failure_take01` |
| Model | YOLO26s |
| Task | Direct disease object detection |
| Classes | `anthracnose`, `leaf_miner`, `red_rust` |
| Failure Type | `PERFORMANCE_FAILURE` + suspected `DATA_FAILURE` |
| Status | **FAILED FOR FINAL — KEEP AS BASELINE** |

Experiment này **không bị lỗi khi train**. Pipeline đã chạy thành công từ training đến test evaluation. Experiment được đưa vào `failure/` vì chất lượng hiện tại chưa phù hợp để chọn làm mô hình cuối, đặc biệt do Recall thấp, số False Negative lớn và hiệu năng giữa các class không đồng đều.

---

## 2. Objective

Mục tiêu của `EXP-Y26S-SMALL-001` là sử dụng một dataset nhỏ tách từ Master Dataset để:

- kiểm tra pipeline YOLO26 trên Kaggle;
- xác nhận định dạng annotation YOLO hoạt động;
- kiểm tra khả năng YOLO26s học vùng bệnh;
- lấy baseline ban đầu cho Precision, Recall, F1, IoU và mAP;
- tìm các vấn đề của dataset/annotation trước khi huấn luyện Master Dataset.

Do đó, dù kết quả không đủ tốt cho final model, experiment vẫn đạt giá trị như một **pilot/baseline failure experiment**.

---

## 3. Training Configuration

| Parameter | Value |
|---|---:|
| Model | `yolo26s.pt` |
| Epochs | 50 |
| Patience | 20 |
| Image Size | 640 |
| Batch Size | 8 |
| Optimizer | AdamW |
| Learning Rate | 0.001 |
| Weight Decay | 0.0005 |
| Seed | 42 |
| Mosaic | 0.5 |
| MixUp | 0.0 |
| Copy Paste | 0.0 |
| Horizontal Flip | 0.5 |
| Custom prediction confidence | 0.25 |
| NMS IoU | 0.70 |
| Custom matching IoU | 0.50 |

Điểm cần lưu ý: dataset thử nghiệm đã được preprocess ở độ phân giải thấp trước khi YOLO nhận ảnh đầu vào. Vì lesion của Anthracnose và Red Rust có nhiều vùng nhỏ, việc mất chi tiết trước khi train có thể ảnh hưởng tới localization và Recall. Đây cần được xác minh bằng experiment dataset resolution tiếp theo, không xem là kết luận duy nhất từ Take 01.

---

## 4. Dataset Summary

### 4.1. Number of images

| Split | Images |
|---|---:|
| Train | 355 |
| Validation | 34 |
| Test | 16 |

Đây là dataset pilot nhỏ nên các chỉ số Test, đặc biệt metric theo class có ít mẫu, có thể dao động mạnh.

### 4.2. Training bounding boxes

| Class | Train Boxes |
|---|---:|
| Anthracnose | 1,414 |
| Leaf Miner | 243 |
| Red Rust | 3,171 |

Dataset có sự khác biệt lớn về số lượng bounding box giữa các class. Tuy nhiên kết quả cho thấy class có ít box nhất không nhất thiết có hiệu năng thấp nhất, vì vậy imbalance không phải nguyên nhân duy nhất.

### 4.3. Test bounding boxes

| Class | Test Boxes |
|---|---:|
| Anthracnose | 73 |
| Leaf Miner | 11 |
| Red Rust | 118 |
| **Total** | **202** |

`leaf_miner` chỉ có 11 box trong Test, nên metric rất cao của class này chưa đủ mạnh để kết luận class này luôn dễ nhận diện trên dữ liệu lớn.

---

## 5. Main Test Metrics

| Metric | Result |
|---|---:|
| Precision | **0.6196** |
| Recall | **0.5637** |
| F1-score | **0.5903** |
| mAP@0.50 | **0.5589** |
| mAP@0.75 | **0.3215** |
| mAP@0.50:0.95 | **0.2988** |
| Mean IoU of matched TP | **0.7505** |
| Median IoU of matched TP | **0.7565** |

### Nhận xét chính

`Mean IoU ≈ 0.75` cho thấy khi YOLO phát hiện đúng lesion, bounding box thường khớp tương đối tốt với ground truth. Vấn đề chính của model hiện tại không phải chỉ là box lệch vị trí, mà là **không phát hiện đủ các lesion cần tìm**.

---

## 6. Per-Class Performance

| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Anthracnose | 0.378 | 0.350 | 0.363 | **0.302** | **0.122** |
| Leaf Miner | 0.869 | 0.909 | 0.888 | **0.940** | **0.590** |
| Red Rust | 0.612 | 0.432 | 0.507 | **0.435** | **0.185** |

Hiệu năng giữa ba class không đồng đều. `anthracnose` là class yếu nhất trong Take 01. `red_rust` có kết quả tốt hơn Anthracnose nhưng Recall vẫn thấp. `leaf_miner` đạt metric rất cao nhưng Test support quá nhỏ để xem đây là kết luận cuối.

---

## 7. Custom IoU Matching Analysis

Tại:

```text
Prediction confidence = 0.25
Matching IoU          = 0.50
```

kết quả custom matching ghi nhận:

| Metric | Value |
|---|---:|
| True Positive | **67** |
| False Positive | **53** |
| False Negative | **135** |
| Custom Precision | ~0.558 |
| Custom Recall | ~0.332 |
| Mean IoU của TP | ~0.751 |

### Failure symptom quan trọng nhất

```text
Model phát hiện được một số lesion
            ↓
Box của các TP tương đối tốt
            ↓
Nhưng bỏ sót rất nhiều lesion
            ↓
False Negative cao
            ↓
Recall thấp
```

135 False Negative trên 202 ground-truth boxes ở cấu hình custom threshold là dấu hiệu cần ưu tiên xử lý trước khi chuyển sang Master Dataset.

---

## 8. Observed Failure Symptoms

### 8.1. Nhiều lesion bị bỏ sót

Recall thấp hơn Precision và custom matching có số FN rất lớn. Đây là failure chính của Take 01.

### 8.2. Anthracnose yếu

Anthracnose đạt:

```text
Recall      ≈ 0.350
mAP50       ≈ 0.302
mAP50-95    ≈ 0.122
```

Điều này cho thấy model chưa học tốt các vùng Anthracnose nhỏ/phức tạp trong pilot dataset.

### 8.3. Red Rust chưa ổn định

Red Rust có nhiều annotation nhất nhưng Recall chỉ khoảng 0.432. Điều này chứng minh số lượng box lớn không tự động tạo ra detection tốt.

### 8.4. Nhiều box chồng lấn trên cùng vùng bệnh

Một số prediction cho thấy nhiều bounding box cùng class xuất hiện sát hoặc chồng lên nhau trên một vùng lá. Có hai nhóm nguyên nhân cần kiểm tra:

- annotation có quá nhiều box nhỏ nằm sát nhau;
- NMS/inference threshold giữ lại nhiều box gần nhau.

Không nên chỉ giảm NMS để che vấn đề. Trước hết phải review annotation.

### 8.5. Lesion nhỏ là trường hợp khó

Anthracnose và Red Rust có nhiều dấu hiệu dạng đốm/cụm nhỏ. Nếu mỗi đốm rất nhỏ được annotate riêng hoặc ảnh đã mất chi tiết do resize, YOLO sẽ khó học và khó đạt Recall cao.

---

## 9. Root Cause Analysis

### 9.1. Data / Annotation — ưu tiên cao nhất

Các nguyên nhân nghi ngờ:

1. **Bounding box quá vụn**: nhiều đốm bệnh nhỏ gần nhau có thể đang được tách thành quá nhiều box.
2. **Annotation style chưa đồng nhất**: cùng kiểu lesion có thể được annotate theo nhiều mức độ chi tiết khác nhau.
3. **Small-object information bị giảm**: ảnh đầu vào của dataset pilot có thể đã mất chi tiết trước khi train.
4. **Dataset Test nhỏ**: chỉ 16 ảnh làm metric theo class chưa ổn định, đặc biệt Leaf Miner.
5. **Class distribution không cân bằng**: có thể ảnh hưởng nhưng không giải thích toàn bộ kết quả.

### 9.2. Model / Training

Training curve cho thấy model có học và metric tăng trong quá trình train. Không có dấu hiệu cho thấy nguyên nhân chính là training crash hoặc model hoàn toàn không hội tụ.

Vì vậy **không ưu tiên tăng epochs hoặc đổi model ngay**.

### 9.3. Inference Threshold

F1-confidence curve của experiment cho thấy F1 tổng đạt vùng tốt hơn ở confidence thấp hơn 0.25. Tuy nhiên threshold không được chọn dựa trên Test.

Cần:

```text
Validation
   ↓
chọn confidence/NMS phù hợp
   ↓
khóa threshold
   ↓
Test cuối cùng
```

---

## 10. What Worked

Take 01 vẫn có những kết quả tích cực:

- Kaggle training pipeline chạy hoàn chỉnh;
- YOLO26s load/train/evaluate thành công;
- annotation YOLO đọc đúng;
- model đã học được đặc trưng lesion;
- Mean IoU của TP đạt khoảng 0.75;
- Leaf Miner được detect tốt trên số mẫu hiện có;
- các metric, curves, confusion matrix và prediction artifacts đã xuất được;
- experiment đã chỉ ra rõ bottleneck cần cải thiện trước khi train Master Dataset.

---

## 11. Why This Experiment Is Not Selected as Final

Không chọn `EXP-Y26S-SMALL-001` làm final model vì:

1. Recall tổng còn hạn chế.
2. Custom analysis cho thấy False Negative lớn.
3. Anthracnose và Red Rust có AP thấp.
4. Hiệu năng giữa class không đồng đều.
5. Test dataset quá nhỏ để đưa ra kết luận cuối.
6. Dataset/annotation cần được review trước khi tiếp tục tối ưu model.

Experiment được giữ lại làm **baseline so sánh**.

---

## 12. Corrective Action Plan

Thứ tự xử lý được chốt như sau.

### Action 1 — Review annotation

Lấy khoảng 30–50 ảnh `anthracnose` và `red_rust`, ưu tiên ảnh có nhiều lesion nhỏ.

Kiểm tra:

- box có quá nhỏ không;
- cùng một cụm có bị chia thành quá nhiều box không;
- box có sát lesion không;
- annotation style giữa các ảnh có nhất quán không;
- có lesion rõ ràng bị bỏ annotation không.

Theo Annotation Guideline của project, nhiều đốm nhỏ nằm sát nhau và tạo thành một cụm bệnh rõ ràng nên cân nhắc dùng một cluster box hợp lý thay vì hàng chục box cực nhỏ.

### Action 2 — Chuẩn hóa lại dataset pilot

Sau review:

```text
Dataset Pilot hiện tại
        ↓
Review / Fix Annotation
        ↓
Dataset Pilot v2
```

Không thay hyperparameter trong bước này để có thể đo trực tiếp tác động của data fix.

### Action 3 — Giữ nguyên YOLO26s và train lại

Experiment kế tiếp đề xuất:

```text
EXP-Y26S-DATAFIX-002
```

Giữ cố định:

```text
Model       : YOLO26s
Epochs      : 50
Batch       : 8
Optimizer   : AdamW
LR          : 0.001
Seed        : 42
```

Thay đổi chính:

```text
Annotation / Dataset Quality
```

### Action 4 — Resolution ablation sau khi data sạch

Khi `DATAFIX-002` tốt hơn baseline, mới thử:

```text
EXP-Y26S-RES640-003
EXP-Y26S-RES960-004
```

Mục tiêu là kiểm tra ảnh độ phân giải cao hơn có cải thiện small-lesion detection hay không.

### Action 5 — Threshold chỉ chọn bằng Validation

Tối ưu confidence và NMS trên Validation. Không sử dụng Test để lựa chọn threshold.

### Action 6 — Chưa dùng class weighting ngay

Leaf Miner đang ít dữ liệu nhất nhưng metric cao nhất trong pilot. Do đó chưa có đủ bằng chứng để ưu tiên class weighting. Chỉ thử weighting/oversampling như một experiment riêng sau khi data quality và resolution đã được xử lý.

---

## 13. Success Criteria for the Next Take

Không đặt một ngưỡng tuyệt đối mới khi chưa có thêm dữ liệu. `EXP-Y26S-DATAFIX-002` được xem là cải thiện nếu trên cùng protocol đánh giá:

- Recall tăng so với Take 01;
- FN giảm;
- mAP50-95 tăng;
- Anthracnose AP tăng;
- Red Rust Recall/AP tăng;
- Mean IoU không suy giảm đáng kể;
- không tạo thêm quá nhiều FP để đổi lấy Recall.

Mục tiêu chính không phải chỉ tăng mAP tổng mà là **giảm bỏ sót lesion trong khi duy trì localization tốt**.

---

## 14. Next Experiment

```text
Experiment ID : EXP-Y26S-DATAFIX-002
Model         : YOLO26s
Main Change   : Review + sửa annotation/dataset pilot
Keep Fixed    : epochs, batch, optimizer, LR, seed
Primary Goal  : tăng Recall, giảm FN, tăng AP Anthracnose/Red Rust
Comparison    : EXP-Y26S-SMALL-001
```

Sau đó mới thực hiện resolution ablation `640 vs 960`.

---

## 15. Final Decision

- [ ] Keep as final candidate
- [x] **Keep as baseline only**
- [x] **Re-run after data/annotation fix**
- [ ] Discard all artifacts

`EXP-Y26S-SMALL-001` được lưu trong `failure/` vì nó là một failure có giá trị nghiên cứu: mô hình đã chạy thành công nhưng kết quả chỉ ra rõ vấn đề về Recall, small-lesion detection và chất lượng/độ chi tiết annotation cần giải quyết trước khi train Master Dataset.
