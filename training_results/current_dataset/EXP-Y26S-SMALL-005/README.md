# EXP-Y26S-SMALL-005 — YOLO26s Dense Small-Lesion Detection

> **Status:** ⚠️ Current-dataset experiment — **chưa chọn làm final detector**  
> **Model:** YOLO26s pretrained  
> **Task:** Direct disease object detection  
> **Classes:** `anthracnose`, `leaf_miner`, `red_rust`

## 1. Annotation policy của experiment

Take 005 sử dụng chiến lược bounding rất chi tiết:

- `anthracnose`: gán cả các chấm đen nhỏ nếu nhìn thấy rõ;
- `leaf_miner`: gán các vùng tổn thương/đường đục rõ;
- `red_rust`: do số lượng chấm rất nhiều và nhỏ nên **không thể gán hết toàn bộ dấu hiệu**.

Điểm này rất quan trọng khi diễn giải metric vì mức độ đầy đủ của annotation đang khác nhau giữa các class.

---

## 2. Training setup

| Field | Value |
|---|---|
| Experiment ID | `EXP-Y26S-SMALL-005` |
| Model | `yolo26s.pt` |
| Seed | `42` |
| Input size | `640 × 640` |
| Epochs | `50` |
| Batch size | `8` |
| Optimizer | AdamW |
| Initial learning rate | `0.001` |
| Weight decay | `0.0005` |
| Mosaic | `0.5` |
| MixUp | `0.0` |
| Copy-Paste | `0.0` |
| Horizontal flip | `0.5` |
| Custom eval confidence | `0.25` |
| NMS IoU | `0.7` |
| Matching IoU | `0.5` |
| Training time | `570.66 s` (~9.51 min) |

---

## 3. Dataset statistics

| Split | Images | Boxes | Boxes/Image | Anthracnose | Leaf Miner | Red Rust |
|---|---:|---:|---:|---:|---:|---:|
| Train | 429 | 13,999 | **32.63** | 4,446 | 519 | 9,034 |
| Validation | 44 | 1,402 | **31.86** | 406 | 61 | 935 |
| Test | 21 | 531 | **25.29** | 159 | 31 | 341 |

Tỷ lệ box trong Train:

- `anthracnose`: **31.8%**
- `leaf_miner`: **3.7%**
- `red_rust`: **64.5%**

Dataset detection vẫn mất cân bằng mạnh. `leaf_miner` rất ít box, còn `red_rust` chiếm gần 2/3 toàn bộ annotation.

Ngoài ra `images_with_boxes = images` ở cả Train / Validation / Test, tức bộ detection này **không có negative image rõ ràng**. Đây là một nguyên nhân có thể làm False Positive cao.

---

## 4. Test results

| Metric | Result |
|---|---:|
| Precision | **0.6929** |
| Recall | **0.6506** |
| F1-score | **0.6711** |
| mAP@0.50 | **0.6609** |
| mAP@0.75 | **0.2793** |
| mAP@0.50:0.95 | **0.3405** |
| Mean IoU (matched TP) | **0.7629** |
| Median IoU (matched TP) | **0.7741** |

### Custom matching — `conf=0.25`, `IoU≥0.50`

| Metric | Value |
|---|---:|
| TP | **357** |
| FP | **246** |
| FN | **174** |
| Precision | **0.5920** |
| Recall | **0.6723** |
| F1 | **0.6296** |

Khoảng **40.8%** detections tại operating point này là False Positive (`246 / (357 + 246)`). Vì vậy model vẫn chưa đủ sạch để dùng làm detector cuối.

---

## 5. Per-class performance

| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Anthracnose | **0.7233** | 0.5425 | 0.6200 | 0.6203 | 0.2636 |
| Leaf Miner | **0.8229** | 0.6000 | **0.6940** | 0.6584 | **0.4012** |
| Red Rust | 0.5326 | **0.8094** | 0.6424 | **0.7040** | 0.3566 |

### Custom IoU matching theo class

| Class | TP | FP | FN | Precision | Recall | Mean IoU TP |
|---|---:|---:|---:|---:|---:|---:|
| Anthracnose | 79 | 29 | 80 | **0.7315** | 0.4969 | 0.7367 |
| Leaf Miner | 18 | 3 | 13 | **0.8571** | 0.5806 | **0.8229** |
| Red Rust | 260 | **214** | 81 | 0.5485 | **0.7625** | 0.7667 |

### Anthracnose

- Precision tốt nhưng Recall chỉ khoảng `0.54`.
- Chính sách gán cả chấm đen rất nhỏ làm số object tăng mạnh.
- Sau resize `640×640`, nhiều chấm có thể chỉ còn vài pixel nên localization khó và Recall giảm.
- Sample prediction cho thấy model có xu hướng sinh nhiều bbox Anthracnose nhỏ trên texture / vùng lá không chắc chắn.

### Leaf Miner

- Metric đẹp nhất về Precision và mAP50-95.
- Tuy nhiên Test chỉ có **31 box**, nên kết quả còn nhạy với sample size.
- Chưa nên kết luận đây là class mạnh nhất chỉ từ test set nhỏ này.

### Red Rust

- Recall rất cao (`0.8094`) nhưng Precision thấp (`0.5326`).
- Train có **9,034** box Red Rust nên supervision cho Recall rất mạnh.
- Tuy nhiên annotation Red Rust **không exhaustive**: không gán hết các chấm nhỏ. Một phần prediction có thể đúng về mặt triệu chứng nhưng không có GT tương ứng và bị tính là False Positive.
- Vì vậy Precision Red Rust không thể diễn giải hoàn toàn như “model dự đoán sai”.

---

## 6. Annotation-policy issue — điểm quan trọng nhất

Take 005 đang có **class-dependent annotation completeness**:

- Anthracnose: rất chi tiết, đến mức từng chấm nhỏ có thể là một object.
- Red Rust: chỉ gán một phần dấu hiệu do số chấm quá nhiều.
- Leaf Miner: object thường lớn và rõ hơn.

Hệ quả:

1. Model Anthracnose được khuyến khích phát hiện object cực nhỏ và dễ trở nên nhạy quá mức với texture/chấm tối.
2. Model Red Rust có thể phát hiện đúng một dấu hiệu chưa được gán GT nhưng vẫn bị tính FP.
3. Precision/Recall giữa các class không còn phản ánh hoàn toàn cùng một annotation unit.
4. Nếu dùng YOLO crop cho DenseNet/ResNet/ViT, các bbox quá nhỏ có thể không còn đủ thông tin thị giác sau resize.

**Kết luận:** không nên dùng nguyên policy “bounding càng chi tiết càng tốt” làm guideline cuối cùng.

---

## 7. Qualitative findings

Các sample prediction trong archive cho thấy:

- lesion lớn / rõ thường được localization hợp lý;
- một số lá xuất hiện rất nhiều bbox Anthracnose nhỏ với confidence khoảng `0.3–0.7`;
- có ảnh gần như healthy vẫn xuất hiện nhiều Anthracnose box confidence thấp;
- Leaf Miner thường tạo box ổn định hơn do tổn thương lớn và rõ hơn.

Điều này phù hợp với Custom FP = **246** và cho thấy dense tiny-box annotation có trade-off lớn về False Positive.

---

## 8. Training / validation behavior

- Best validation mAP50 ≈ **0.7062** tại epoch **14**.
- Best validation mAP50-95 ≈ **0.3774** tại epoch **42**.
- Test mAP50 = **0.6609** và mAP50-95 = **0.3405**.
- Không có dấu hiệu generalization collapse nghiêm trọng; Test thấp hơn Validation nhưng vẫn cùng vùng hiệu năng.
- 50 epoch đã đủ để thấy plateau; tăng epoch đơn thuần không phải ưu tiên tiếp theo.

F1 curve trên Test đạt khoảng **0.65 tại confidence ≈ 0.238**. **Không dùng Test để chọn threshold cuối**; threshold phải được chọn trên Validation rồi khóa trước khi đánh giá Test.

---

## 9. Historical reference — Take 004 vs Take 005

> **Chỉ dùng để theo dõi tiến trình. Không phải ablation khoa học**, vì dataset và annotation policy đã thay đổi.

| Metric | Take 004 | Take 005 | Delta |
|---|---:|---:|---:|
| Precision | 0.4830 | **0.6929** | **+0.2099** |
| Recall | 0.5360 | **0.6506** | **+0.1146** |
| F1 | 0.5081 | **0.6711** | **+0.1630** |
| mAP50 | 0.4929 | **0.6609** | **+0.1680** |
| mAP50-95 | 0.2588 | **0.3405** | **+0.0817** |
| Mean IoU TP | **0.7910** | 0.7629 | **-0.0281** |

Take 005 tốt hơn rõ về detection metrics, nhưng IoU localization giảm nhẹ, phù hợp với việc dataset chứa nhiều bbox nhỏ và khó hơn.

---

## 10. Decision

**Không xếp Take 005 vào Failure Archive.** Đây là experiment hợp lệ trên dataset hiện tại và YOLO26s đã học được tín hiệu bệnh tương đối tốt.

Tuy nhiên trạng thái vẫn là:

> ⚠️ **CURRENT / EXPERIMENTAL — chưa chọn làm final detector**

Lý do:

- False Positive vẫn cao.
- Detection dataset chưa có negative images.
- Annotation unit chưa nhất quán giữa Anthracnose và Red Rust.
- Leaf Miner thiếu dữ liệu.
- Một số bbox cực nhỏ không phù hợp với downstream classifier.

---

## 11. Recommended next experiment — EXP-Y26S-SMALL-006

Giữ nguyên YOLO26s và phần lớn hyperparameter để isolate ảnh hưởng của data policy:

1. Chuẩn hóa annotation unit theo **lesion/cluster có ý nghĩa**, không phải “mọi chấm đều là một object”.
2. Anthracnose: chấm nhỏ sát nhau → gom thành cluster; box riêng chỉ khi lesion rõ và đủ lớn sau resize.
3. Red Rust: dùng cùng logic cluster-level; tránh policy “gán một phần ngẫu nhiên các chấm”.
4. Bổ sung negative images:
   - `healthy` cashew leaves với empty label;
   - `not_cashew_leaf` nếu detector cần xử lý OOD/background.
5. Ưu tiên tăng dữ liệu `leaf_miner`.
6. Sau khi annotation policy ổn định mới làm ablation resolution `640 / 960 / 1280`.

Nếu vẫn muốn detect từng lesion cực nhỏ, resolution cao hơn có thể hữu ích, nhưng chỉ nên thử sau khi annotation rule đã nhất quán.

---

## 12. Repository artifacts

Repo lưu các file nhẹ phục vụ tái lập và báo cáo:

- `experiment_config.json`
- `metrics_summary.json`
- `dataset_statistics.csv`
- `per_class_metrics.csv`
- `custom_iou_per_class.csv`
- `custom_iou_per_image.csv`
- `final_summary.csv`
- `results.csv`

Checkpoint `.pt`, ảnh prediction và full ZIP không commit trực tiếp để tránh làm repository phình lớn. Full archive `EXP-Y26S-SMALL-005.zip` được giữ riêng.
