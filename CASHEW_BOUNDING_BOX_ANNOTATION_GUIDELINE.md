# CASHEW DISEASE BOUNDING BOX ANNOTATION GUIDELINE

## 1. Mục đích

Tài liệu này quy định **phương pháp gán nhãn Bounding Box thống nhất cho toàn bộ team** trong dự án phát hiện và phân loại bệnh trên lá cây điều.

Mục tiêu:

- đảm bảo tất cả thành viên annotate theo cùng một tiêu chuẩn;
- giảm sai lệch giữa các annotator;
- tạo Master Dataset có chất lượng tốt cho YOLO;
- hỗ trợ cả bài toán YOLO Direct và YOLO + Classifier;
- hạn chế label noise và bounding box không nhất quán;
- tạo dữ liệu có thể sử dụng trực tiếp cho huấn luyện, đánh giá và báo cáo khóa luận.

---

# 2. Các class sử dụng khi Bounding Box

Master Object Detection Dataset chỉ sử dụng 3 class bệnh:

```text
anthracnose
leaf_miner
red_rust
```

Hai nhóm dữ liệu sau vẫn được giữ trong dataset nhưng **không tạo Bounding Box**:

```text
healthy
not_cashew_leaf
```

Lý do:

- `healthy`: không có vùng tổn thương bệnh;
- `not_cashew_leaf`: không phải lá điều, không có vùng bệnh cần detect.

---

# 3. Nguyên tắc tổng quát

Một Bounding Box phải trả lời được câu hỏi:

> **Vùng tổn thương nào trên ảnh đang thể hiện bệnh gì?**

Không khoanh:

```text
toàn bộ ảnh        ❌
toàn bộ chiếc lá   ❌
cành cây           ❌
background         ❌
vùng lá khỏe       ❌
```

Chỉ khoanh:

```text
vùng tổn thương có dấu hiệu bệnh rõ ràng ✅
```

---

# 4. Quy tắc Bounding Box cơ bản

| Trường hợp | Cách annotate |
|---|---|
| Một vùng bệnh rõ ràng | 1 box sát vùng bệnh |
| Hai vùng bệnh tách biệt | 2 box |
| Nhiều đốm nhỏ nằm sát nhau thành cụm | 1 box bao cụm |
| Nhiều vùng nhỏ cách xa nhau | Chia thành nhiều box |
| Một vùng bệnh lớn, liên tục | 1 box lớn bao vùng tổn thương |
| Hai bệnh khác nhau trên cùng lá | Box riêng cho từng bệnh |
| Hai vùng bệnh khác nhau chồng lên nhau | Có thể để box overlap |
| Healthy | 0 box |
| Not Cashew Leaf | 0 box |
| Không chắc chắn | Đưa vào Review, không tự đoán |

---

# 5. Bounding Box phải sát vùng tổn thương

Bounding Box phải bao vùng bệnh nhưng **hạn chế tối đa phần lá khỏe nằm bên trong box**.

## Sai

```text
┌────────────────────────────┐
│                            │
│       healthy tissue       │
│                            │
│          lesion            │
│                            │
│       healthy tissue       │
│                            │
└────────────────────────────┘
```

Box quá rộng có thể làm mô hình học thêm:

- màu lá;
- gân lá;
- background;
- vùng healthy;

thay vì tập trung vào đặc trưng bệnh.

## Đúng

```text
       ┌─────────────┐
       │   lesion    │
       └─────────────┘
```

Không cần sát từng pixel như segmentation, nhưng box phải đủ chặt để đại diện đúng vùng tổn thương.

---

# 6. Một vùng tổn thương rõ ràng = một Bounding Box

Ví dụ ảnh có ba vùng bệnh tách biệt:

```text
        [A]

                         [B]


              [C]
```

Annotate:

```text
box_01 → A
box_02 → B
box_03 → C
```

Không tạo một box lớn bao cả A, B, C nếu giữa các vùng đó có nhiều vùng lá khỏe.

---

# 7. Quy tắc đối với nhiều đốm bệnh nhỏ

Đây là trường hợp dễ gây khác biệt giữa các annotator nhất.

## 7.1. Các đốm nhỏ nằm sát nhau

Ví dụ:

```text
. . . .
.. . ..
. . ...
```

Nếu các đốm tạo thành một **cụm tổn thương rõ ràng**, có thể tạo:

```text
┌───────────────┐
│ . . . .       │
│ .. . ..       │
│ . . ...       │
└───────────────┘
```

→ **1 Bounding Box cho cả cụm.**

## 7.2. Các vùng tổn thương cách xa nhau

Ví dụ:

```text
...                    ...



           ...


...                         ...
```

→ Tạo các Bounding Box riêng.

## 7.3. Không annotate quá vụn

Không nên:

```text
1 đốm rất nhỏ = 1 box
```

nếu ảnh có hàng chục đốm li ti gần nhau.

Cũng không nên:

```text
1 box bao gần hết chiếc lá
```

chỉ vì trên lá có nhiều đốm bệnh.

Mục tiêu là tạo box ở mức **vùng tổn thương có ý nghĩa về mặt thị giác**.

---

# 8. Không tạo Bounding Box quá nhỏ không có ý nghĩa

Nếu một dấu hiệu bệnh:

- chỉ vài pixel;
- rất khó quan sát;
- gần như mất hoàn toàn sau resize;
- không thể xác định class đáng tin cậy;

thì không nên cố tạo một box riêng.

Ưu tiên:

```text
các đốm nhỏ nằm gần nhau
        ↓
gom thành một cluster hợp lý
        ↓
tạo 1 Bounding Box cho cluster
```

Không tạo hàng chục box cực nhỏ nếu chúng không mang ý nghĩa học tập rõ ràng cho mô hình.

---

# 9. Một ảnh có thể có nhiều bệnh

Không áp dụng quy tắc:

```text
1 ảnh = 1 disease
```

Một lá hoàn toàn có thể xuất hiện nhiều bệnh.

Ví dụ:

```text
┌─────────────────────────────────┐
│                                 │
│ [Anthracnose]                   │
│                                 │
│                   [Red Rust]    │
│                                 │
│        [Leaf Miner]             │
│                                 │
└─────────────────────────────────┘
```

Annotation:

```text
box_01 → anthracnose
box_02 → red_rust
box_03 → leaf_miner
```

Do đó:

```text
1 image
→ N Bounding Boxes
→ có thể chứa M disease classes
```

---

# 10. Hai bệnh khác nhau nằm gần nhau

Nếu hai vùng bệnh khác class nằm cạnh nhau:

```text
Anthracnose + Red Rust
```

không được gom thành:

```text
1 box → anthracnose + red_rust ❌
```

Phải tách:

```text
box_A → anthracnose
box_B → red_rust
```

Bounding Box có thể overlap nếu cần.

---

# 11. Quy tắc cho từng class

## 11.1. Anthracnose

Label:

```text
anthracnose
```

Quy tắc:

- vùng tổn thương liên tục → 1 box;
- nhiều vùng tách biệt → nhiều box;
- nhiều đốm nằm sát nhau thành một cụm → có thể gom cluster;
- không khoanh toàn bộ lá chỉ vì lá có nhiều dấu hiệu Anthracnose.

---

## 11.2. Leaf Miner

Label:

```text
leaf_miner
```

Quy tắc:

- khoanh vùng đường hầm/vệt tổn thương do Leaf Miner gây ra;
- nếu đường tổn thương liên tục → ưu tiên 1 box bao toàn vùng;
- không chia một đường tổn thương liên tục thành nhiều box nhỏ không cần thiết.

---

## 11.3. Red Rust

Label:

```text
red_rust
```

Quy tắc:

- khoanh vùng các dấu hiệu Red Rust rõ ràng;
- nếu nhiều đốm nhỏ tập trung thành cụm → dùng 1 box theo cluster;
- nếu các vùng cách xa nhau → tách nhiều box.

---

# 12. Healthy

Ảnh thuộc nhóm:

```text
healthy
```

phải có:

```text
0 Bounding Boxes
```

Không tạo class:

```text
healthy_box ❌
```

Không khoanh toàn bộ chiếc lá rồi gắn label `healthy`.

Healthy đóng vai trò **negative image** đối với object detection.

---

# 13. Not Cashew Leaf

Ảnh thuộc nhóm:

```text
not_cashew_leaf
```

phải có:

```text
0 Bounding Boxes
```

Không tạo:

```text
not_cashew_leaf_box ❌
```

Class này vẫn được sử dụng trong bài toán classification 5 lớp, nhưng không phải class Bounding Box của Object Detection.

---

# 14. Ảnh không chắc chắn

Nếu annotator không chắc về vùng bệnh hoặc class:

```text
Không chắc
    ↓
Không tự đoán
    ↓
Đưa ảnh vào REVIEW / UNCERTAIN
```

Sau đó:

```text
Annotator
    ↓
Reviewer
    ↓
Team Discussion
    ↓
Nếu cần → GVHD / chuyên gia
    ↓
Final Label
```

Một annotation sai class có thể gây ảnh hưởng lớn hơn việc tạm thời chưa annotate.

---

# 15. Consistency giữa các thành viên

Dataset không chỉ cần box đúng mà còn phải có **annotation style nhất quán**.

Ví dụ cùng một dạng tổn thương:

```text
. . . .
.. . ..
. . ...
```

Không được xảy ra:

```text
Member A → 1 box cluster
Member B → 12 box nhỏ
Member C → 1 box bao toàn bộ lá
```

Team phải thống nhất cách hiểu về:

- thế nào là một lesion;
- khi nào được gom cluster;
- box được phép rộng đến mức nào;
- lesion nhỏ đến mức nào thì bỏ qua;
- khi nào cần tách nhiều box.

---

# 16. Không chia annotation theo disease class

Không nên phân công:

```text
Member A → chỉ Anthracnose
Member B → chỉ Leaf Miner
Member C → chỉ Red Rust
```

Vì sẽ tạo ra nhiều annotation style khác nhau.

Nên chia theo **batch ảnh hỗn hợp**:

```text
Batch_001 → Member A
Batch_002 → Member B
Batch_003 → Member C
```

Mỗi batch nên có nhiều class khác nhau.

---

# 17. Pilot Annotation trước khi làm toàn bộ dataset

Không nên annotate toàn bộ dataset ngay từ đầu.

## Giai đoạn Pilot

Chọn khoảng:

```text
100–200 images
```

Trong đó chọn khoảng:

```text
30–50 images
```

để tất cả thành viên annotate độc lập cùng một bộ ảnh.

Sau đó so sánh:

- vị trí Bounding Box;
- kích thước box;
- disease class;
- cách xử lý cluster;
- cách xử lý lesion nhỏ;
- cách xử lý nhiều bệnh;
- các trường hợp không chắc chắn.

Sau khi thống nhất mới khóa:

```text
Annotation Guideline v1.0
```

và bắt đầu annotate toàn bộ dataset.

---

# 18. Cross Review

Với team 3 người, có thể sử dụng:

```text
Member A annotate
        ↓
Member B review

Member B annotate
        ↓
Member C review

Member C annotate
        ↓
Member A review
```

Reviewer cần kiểm tra:

```text
□ Sai class?
□ Thiếu vùng bệnh?
□ Box quá rộng?
□ Box quá chật?
□ Hai lesion tách biệt bị gom nhầm?
□ Một cluster bị chia quá vụn?
□ Có box vào background?
□ Có annotate Healthy không?
□ Có annotate Not Cashew Leaf không?
□ Có lesion nghi ngờ cần review lại không?
```

Nếu annotator và reviewer không thống nhất:

```text
Disagreement
    ↓
Team Discussion
    ↓
Final Decision
```

---

# 19. Quy tắc Train / Validation / Test

Cả ba tập đều cần Ground Truth Bounding Box:

```text
Train → Annotate
Val   → Annotate
Test  → Annotate
```

Tuy nhiên, **không được thay đổi split sau khi dataset đã được khóa**.

Ví dụ:

```text
image_001 → Train
image_002 → Val
image_003 → Test
```

Sau Roboflow vẫn phải giữ:

```text
image_001 → Train
image_002 → Val
image_003 → Test
```

Không random split lại.

## Test Set

Test được annotate để tính các metric như:

```text
Precision
Recall
mAP@0.5
mAP@0.5:0.95
```

Nhưng Test không được sử dụng để:

- chọn learning rate;
- chọn augmentation;
- chọn epochs;
- chọn architecture;
- chọn threshold;
- tune hyperparameter.

---

# 20. Master Dataset

Master Dataset luôn giữ **disease label cụ thể**:

```text
anthracnose
leaf_miner
red_rust
```

Không annotate Master Dataset trực tiếp thành:

```text
lesion
```

Lý do: từ một Master Dataset có thể sinh ra hai bài toán.

## YOLO Direct

```text
anthracnose
leaf_miner
red_rust
```

Output:

```text
Bounding Box + Disease Class
```

## YOLO Lesion

Map toàn bộ các disease class thành:

```text
lesion
```

Output:

```text
Bounding Box vùng tổn thương
```

Sau đó crop vùng tổn thương và chuyển sang classifier.

Ví dụ Master Annotation:

```text
box_01 → anthracnose
box_02 → anthracnose
box_03 → red_rust
```

YOLO Direct:

```text
anthracnose
anthracnose
red_rust
```

YOLO Lesion:

```text
lesion
lesion
lesion
```

Như vậy team chỉ cần annotate một lần.

---

# 21. Quy trình Annotation chính thức

```text
RAW / SPLIT DATASET
        │
        ↓
UPLOAD ROBOFLOW
        │
        ↓
PILOT ANNOTATION
100–200 images
        │
        ↓
30–50 images
cả team annotate độc lập
        │
        ↓
COMPARE & DISCUSS
        │
        ↓
LOCK GUIDELINE v1.0
        │
        ↓
SPLIT INTO MIXED BATCHES
        │
        ↓
FULL ANNOTATION
        │
        ↓
CROSS REVIEW
        │
        ↓
FIX DISAGREEMENTS
        │
        ↓
FINAL QUALITY CHECK
        │
        ↓
MASTER ANNOTATED DATASET
```

---

# 22. Checklist trước khi Submit một ảnh

Trước khi hoàn thành annotation của một ảnh, annotator cần kiểm tra:

```text
□ Đã tìm hết các vùng bệnh rõ ràng chưa?

□ Bounding Box có sát vùng tổn thương không?

□ Box có chứa quá nhiều healthy tissue không?

□ Các lesion tách biệt có bị gom chung không?

□ Các đốm sát nhau có nên gom thành cluster không?

□ Có tạo quá nhiều box cực nhỏ không?

□ Disease class có chắc chắn đúng không?

□ Nếu có nhiều bệnh, đã label riêng từng bệnh chưa?

□ Có vô tình khoanh toàn bộ chiếc lá không?

□ Healthy có 0 box không?

□ Not Cashew Leaf có 0 box không?

□ Nếu không chắc, đã đưa ảnh vào Review chưa?
```

---

# 23. Checklist dành cho Reviewer

```text
□ Kiểm tra tất cả vùng lesion có được annotate.

□ Kiểm tra class của từng Bounding Box.

□ Kiểm tra box không quá rộng.

□ Kiểm tra box không cắt mất vùng lesion quan trọng.

□ Kiểm tra cách gom cluster có đúng guideline.

□ Kiểm tra lesion tách biệt có được tách box.

□ Kiểm tra multiple diseases.

□ Kiểm tra ảnh Healthy không có box.

□ Kiểm tra ảnh Not Cashew Leaf không có box.

□ Kiểm tra các case UNCERTAIN.

□ Đảm bảo annotation style thống nhất với toàn bộ dataset.
```

---

# 24. Các lỗi annotation cần tránh

## Lỗi 1 — Khoanh toàn bộ lá

```text
Whole Leaf → Disease ❌
```

YOLO cần học vùng tổn thương, không phải chỉ vị trí chiếc lá.

---

## Lỗi 2 — Box quá rộng

```text
Lesion + rất nhiều vùng healthy ❌
```

---

## Lỗi 3 — Box quá vụn

```text
1 cụm 20 đốm → 20 box cực nhỏ ❌
```

nếu các đốm tạo thành một cluster rõ ràng.

---

## Lỗi 4 — Gom hai bệnh thành một box

```text
Anthracnose + Red Rust → 1 box ❌
```

---

## Lỗi 5 — Bỏ sót disease thứ hai

Một ảnh có nhiều disease thì phải annotate đầy đủ các bệnh nhìn thấy rõ ràng.

---

## Lỗi 6 — Tự đoán label

Nếu không chắc:

```text
Review ✅
Guess ❌
```

---

## Lỗi 7 — Annotate Healthy thành object

```text
Healthy Leaf → healthy bounding box ❌
```

---

## Lỗi 8 — Annotate Not Cashew Leaf thành object

```text
Not Cashew Leaf → not_cashew bounding box ❌
```

---

# 25. Quy tắc ngắn gọn để cả team ghi nhớ

> **Khoanh sát vùng tổn thương có ý nghĩa; một vùng bệnh rõ ràng là một box; các đốm sát nhau có thể gom thành cluster; bệnh khác nhau luôn tách label; Healthy và Not Cashew Leaf không có box; không chắc thì Review, không đoán.**

---

# 26. Class Mapping

| Dataset Class | Classification | Object Detection Master | YOLO Lesion |
|---|---|---|---|
| Anthracnose | `anthracnose` | `anthracnose` | `lesion` |
| Leaf Miner | `leaf_miner` | `leaf_miner` | `lesion` |
| Red Rust | `red_rust` | `red_rust` | `lesion` |
| Healthy | `healthy` | No Bounding Box | No Bounding Box |
| Not Cashew Leaf | `not_cashew_leaf` | No Bounding Box | No Bounding Box |

---

# 27. Versioning Guideline

Khi thay đổi quy tắc annotation, không sửa âm thầm.

Sử dụng version:

```text
Annotation Guideline v1.0
Annotation Guideline v1.1
Annotation Guideline v1.2
Annotation Guideline v2.0
```

Ghi lại:

```text
Version:
Date:
Changed By:
Change:
Reason:
Affected Images:
Need Re-review: Yes / No
```

Ví dụ:

```text
Version: v1.1
Change: Điều chỉnh quy tắc gom Red Rust cluster
Reason: Pilot cho thấy annotator chia box không nhất quán
Affected Images: Red Rust
Need Re-review: Yes
```

---

# 28. Definition of Done cho Annotation Dataset

Dataset chỉ được xem là sẵn sàng cho huấn luyện khi:

```text
Pilot Annotation hoàn thành
+
Guideline đã khóa
+
Toàn bộ ảnh đã được xử lý
+
Healthy / Not Cashew Leaf được kiểm tra
+
Cross Review hoàn thành
+
Các disagreement đã được giải quyết
+
Train / Val / Test giữ nguyên
+
Không có random re-split
+
Class names thống nhất
+
Master Dataset được version hóa
```

---

# 29. Kết luận

Quy trình annotation của dự án được chuẩn hóa theo hướng:

```text
Consistent
+
Reproducible
+
Reviewable
+
Suitable for YOLO
+
Reusable for Two-stage Pipeline
```

Master Annotation luôn giữ:

```text
anthracnose
leaf_miner
red_rust
```

Sau đó mới chuyển đổi sang:

```text
lesion
```

khi tạo dataset cho pipeline YOLO Lesion + Classifier.

---

## Project

**Đề tài:** Ứng dụng mô hình học sâu trong phân loại bệnh trên lá cây điều

**Annotation Platform:** Roboflow

**Task:** Object Detection

**Master Classes:**

```text
anthracnose
leaf_miner
red_rust
```
