# EXP-Y26S-SMALL-007 — YOLO26s Detailed Clear-Lesion Annotation

> **Source archive:** `EXP-Y26S-SMALL-007.zip`  
> **Model:** YOLO26s  
> **Task:** Direct disease object detection  
> **Classes:** `anthracnose`, `leaf_miner`, `red_rust`  
> **Status:** Failure Archive / data-annotation iteration, **not final detector**

## 1. Mục tiêu của Take 007

Take 007 tiếp tục chiến lược annotation theo hướng:

- bounding **khá chi tiết** các dấu hiệu bệnh nhìn rõ;
- vẫn **bỏ qua các chi tiết quá nhỏ, quá mờ hoặc không chắc chắn**;
- không quay lại policy “mọi chấm bệnh đều phải có box” của Take 005;
- giữ YOLO26s, input `640×640`, seed `42` và các hyperparameter chính tương tự Take 006.

Điểm tích cực về traceability: metadata bên trong archive đã ghi đúng `EXP-Y26S-SMALL-007`.

> **Lưu ý quan trọng:** Take 006 và Take 007 không phải controlled ablation sạch vì số ảnh, số box và đặc biệt Validation/Test set thay đổi mạnh. Vì vậy phần so sánh dưới đây chỉ dùng để theo dõi tiến trình, không được quy toàn bộ delta metric cho annotation policy.

## 2. Cấu hình huấn luyện

| Field | Value |
|---|---|
| Model | `yolo26s.pt` |
| Seed | `42` |
| Epochs | `50` |
| Patience | `20` |
| Input | `640×640` |
| Batch size | `8` |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `0.0005` |
| Mosaic | `0.5` |
| MixUp | `0.0` |
| Copy-Paste | `0.0` |
| Horizontal flip | `0.5` |
| Custom prediction confidence | `0.25` |
| NMS IoU | `0.7` |
| Matching IoU | `0.5` |

## 3. Dataset statistics

| Split | Images | Images with boxes | Boxes | Anthracnose | Leaf Miner | Red Rust |
|---|---:|---:|---:|---:|---:|---:|
| Train | 496 | 494 | 14,261 | 3,465 | 732 | 10,064 |
| Validation | 80 | 80 | 2,178 | 676 | 109 | 1,393 |
| Test | 39 | 39 | 1,335 | 420 | 91 | 824 |

Annotation density:

- Train: **28.75 boxes/image**
- Validation: **27.23 boxes/image**
- Test: **34.23 boxes/image**

Train distribution:

- Anthracnose: **24.30%**
- Leaf Miner: **5.13%**
- Red Rust: **70.57%**

Class imbalance vẫn rất lớn; Red Rust có khoảng **13.75 lần** số box của Leaf Miner.

Negative supervision vẫn gần như chưa có: Train chỉ có **2/496** ảnh không box; Validation và Test đều không có ảnh 0-box.

## 4. Dataset thay đổi so với Take 006

| Split | Take 006 images | Take 007 images | Take 006 boxes | Take 007 boxes |
|---|---:|---:|---:|---:|
| Train | 441 | **496** | 12,781 | **14,261** |
| Validation | 42 | **80** | 1,108 | **2,178** |
| Test | 20 | **39** | 491 | **1,335** |

Đặc biệt Test tăng từ **20 → 39 ảnh** và **491 → 1,335 GT boxes**. Do đó Take 007 được đánh giá trên một test set lớn hơn và dày annotation hơn nhiều; metric không thể so trực tiếp như một ablation chỉ thay annotation rule.

## 5. Kết quả Test tổng thể

| Metric | Take 006 | Take 007 | Delta |
|---|---:|---:|---:|
| Precision | **0.7473** | 0.6945 | -0.0528 |
| Recall | **0.6648** | 0.6598 | -0.0050 |
| F1 | **0.7036** | 0.6767 | -0.0269 |
| mAP@0.50 | **0.7298** | 0.6993 | -0.0304 |
| mAP@0.75 | **0.3577** | 0.2912 | -0.0665 |
| mAP@0.50:0.95 | **0.3877** | 0.3517 | -0.0360 |
| Mean IoU (matched TP) | **0.7603** | 0.7471 | -0.0132 |
| Median IoU (matched TP) | **0.7720** | 0.7549 | -0.0171 |

Take 007 **không vượt Take 006 trên aggregate YOLO metrics**. Mức giảm lớn nhất nằm ở `mAP75`, cho thấy localization ở ngưỡng IoU chặt hơn đang khó hơn.

Tuy nhiên do Test set thay đổi rất mạnh, không thể kết luận rằng policy annotation của Take 007 kém hơn Take 006 chỉ từ bảng này.

## 6. Custom IoU evaluation tại conf=0.25, IoU≥0.50

Take 007:

- TP = **825**
- FP = **389**
- FN = **510**
- Precision = **0.6796**
- Recall = **0.6180**
- F1 = **0.6473**
- Mean IoU TP = **0.7471**

So với Take 006:

| Metric | Take 006 | Take 007 | Delta |
|---|---:|---:|---:|
| Custom Precision | 0.6075 | **0.6796** | **+0.0721** |
| Custom Recall | **0.7251** | 0.6180 | -0.1071 |
| Custom F1 | **0.6611** | 0.6473 | -0.0138 |

Ở operating point cố định `conf=0.25`, Take 007 tạo prediction **sạch hơn về precision**, nhưng bỏ sót nhiều GT hơn. Đây là trade-off rõ ràng giữa Precision và Recall.

Raw TP/FP/FN không được so trực tiếp vì Take 007 Test có 1,335 boxes, lớn hơn rất nhiều so với 491 boxes của Take 006.

## 7. Kết quả theo từng class

| Class | Metric | Take 006 | Take 007 | Delta |
|---|---|---:|---:|---:|
| Anthracnose | Precision | 0.6934 | **0.7771** | +0.0837 |
|  | Recall | **0.5949** | 0.5230 | -0.0719 |
|  | F1 | **0.6404** | 0.6252 | -0.0152 |
|  | mAP50 | **0.6666** | 0.6296 | -0.0370 |
|  | mAP50-95 | **0.2847** | 0.2515 | -0.0332 |
| Leaf Miner | Precision | **0.8571** | 0.6651 | -0.1920 |
|  | Recall | 0.7198 | **0.8022** | +0.0824 |
|  | F1 | **0.7825** | 0.7272 | -0.0552 |
|  | mAP50 | 0.7839 | **0.8008** | +0.0170 |
|  | mAP50-95 | **0.5108** | 0.4875 | -0.0233 |
| Red Rust | Precision | **0.6916** | 0.6414 | -0.0502 |
|  | Recall | **0.6796** | 0.6541 | -0.0255 |
|  | F1 | **0.6855** | 0.6477 | -0.0378 |
|  | mAP50 | **0.7388** | 0.6676 | -0.0712 |
|  | mAP50-95 | **0.3675** | 0.3162 | -0.0513 |

### Diễn giải

**Anthracnose:** Precision tăng mạnh nhưng Recall giảm. Model ít prediction sai hơn trên class này, đổi lại bỏ sót nhiều lesion hơn. Sample prediction vẫn cho thấy ở một số lá có rất nhiều box Anthracnose nhỏ, nên cần tiếp tục kiểm soát mức độ “chi tiết” để không quay lại trạng thái quá dày.

**Leaf Miner:** Recall và mAP50 tăng, nhưng Precision giảm mạnh. Đây là dấu hiệu model nhạy hơn với Leaf Miner nhưng có nhiều nhầm lẫn hơn. Test Take 007 có 91 Leaf Miner GT boxes, lớn hơn đáng kể Take 006 (25), nên đánh giá này đáng tin hơn về cỡ mẫu nhưng vẫn không phải cùng test set.

**Red Rust:** cả Precision, Recall và mAP đều giảm. Red Rust vẫn chiếm hơn 70% Train boxes nên vấn đề không đơn thuần là thiếu số lượng; cần kiểm tra consistency của box, hard negatives và sự đa dạng ảnh nguồn.

## 8. Confusion matrix

Normalized confusion matrix cho thấy:

- Anthracnose: khoảng **0.78** true Anthracnose được nhận đúng;
- Leaf Miner: khoảng **0.62** được nhận đúng, khoảng **0.30** bị nhầm thành Anthracnose;
- Red Rust: khoảng **0.88** được nhận đúng;
- background false-positive entries vẫn tập trung nhiều ở Red Rust (~`0.43`) và Anthracnose (~`0.38`).

Leaf Miner ↔ Anthracnose vẫn là cặp cần audit thêm, đặc biệt các vùng hoại tử/đốm nhỏ có hình thái gần nhau.

## 9. F1–Confidence analysis

Test F1 curve đạt cực đại khoảng:

- **F1 ≈ 0.66**
- tại confidence ≈ **0.248**

Con số này gần threshold `0.25` đang dùng trong custom evaluation. Tuy nhiên threshold cuối vẫn phải được chọn trên **Validation**, không dùng Test để tuning.

## 10. Training behavior

Training chạy đủ **50 epochs**.

- Best validation `mAP50 ≈ 0.6552` tại epoch **36**
- Best validation `mAP50-95 ≈ 0.3341` tại epoch **36**
- Sau epoch 36, validation metric chủ yếu dao động/plateau
- Test `mAP50 = 0.6993`, `mAP50-95 = 0.3517`

Không có dấu hiệu collapse rõ rệt, nhưng tăng epoch đơn thuần khó có khả năng giải quyết bottleneck hiện tại.

## 11. Điểm tích cực của Take 007

1. Annotation policy vẫn tránh các chi tiết quá nhỏ/mờ, hợp lý hơn policy quá dày của Take 005.
2. Traceability đã được sửa: Experiment ID trong artifact là đúng `EXP-Y26S-SMALL-007`.
3. Test set lớn hơn đáng kể, cho phép quan sát lỗi trên nhiều lesion hơn.
4. Custom Precision tăng từ `0.6075 → 0.6796`.
5. Leaf Miner Recall và mAP50 tăng.
6. Annotation density Train (~28.75 boxes/image) không tăng mất kiểm soát dù số ảnh/box tăng.

## 12. Hạn chế / Failure analysis

### 12.1. Không vượt Take 006 trên metric tổng thể

F1, mAP50, mAP50-95 và Mean IoU đều thấp hơn Take 006.

### 12.2. Split và ground truth thay đổi quá mạnh

Đây là hạn chế phương pháp lớn nhất. Test tăng gần gấp đôi số ảnh và hơn 2.7 lần số GT boxes. Vì vậy Take 006 ↔ 007 không thể dùng như controlled annotation ablation.

### 12.3. Class imbalance vẫn nghiêm trọng

Red Rust = **70.57%** Train boxes, Leaf Miner chỉ **5.13%**.

### 12.4. Negative images gần như chưa có

Chỉ **2/496 Train images** không có box. Điều này tiếp tục hạn chế khả năng học background và có thể góp phần vào false positives.

### 12.5. Chi tiết Anthracnose vẫn có nguy cơ quá dày

Một số sample prediction cho thấy model tạo hàng loạt box Anthracnose nhỏ trên cùng một lá. Cần phân biệt rõ:
- lesion nhỏ nhưng rõ/có ý nghĩa → annotate;
- texture/chấm cực nhỏ, mơ hồ hoặc không đủ tín hiệu sau resize → bỏ hoặc gom cluster theo guideline.

### 12.6. Red Rust vẫn chiếm ưu thế dữ liệu nhưng metric không tăng

Điều này gợi ý cần tập trung vào **chất lượng/đa dạng/consistency**, không chỉ tăng số box.

## 13. Kết luận

Take 007 là một iteration dữ liệu có giá trị, nhưng **chưa vượt Take 006 về metric tổng thể**:

- F1 = **0.6767**
- mAP50 = **0.6993**
- mAP50-95 = **0.3517**
- Mean IoU = **0.7471**

Điểm tích cực là custom Precision tăng và Test set lớn hơn nhiều, nhưng Recall ở operating point `0.25` giảm và class imbalance/negative supervision vẫn chưa được giải quyết.

**Quan trọng nhất cho Take tiếp theo:** không nên tiếp tục thay đổi split cùng lúc với annotation. Hãy khóa một phiên bản Train/Val/Test + GT, sau đó mới kiểm tra từng thay đổi annotation/hyperparameter để có ablation hợp lệ.

## 14. Đề xuất Take 008

- Giữ policy: **chi tiết nhưng bỏ lesion quá nhỏ/mờ**.
- Khóa Train/Val/Test và không thay Test nữa.
- Bổ sung negative images `healthy` và, nếu phù hợp pipeline, `not_cashew_leaf` dưới dạng 0-box.
- Audit riêng Leaf Miner ↔ Anthracnose confusion.
- Giảm phụ thuộc vào việc tăng số box Red Rust; ưu tiên đa dạng ảnh nguồn.
- Giữ YOLO26s + 640 trước khi đổi model/resolution.
- Tune confidence threshold trên Validation và khóa trước khi đánh giá Test.
