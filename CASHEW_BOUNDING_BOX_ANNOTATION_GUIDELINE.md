# CASHEW DISEASE BOUNDING BOX ANNOTATION GUIDELINE

**Version:** 1.1  
**Status:** Current annotation policy  
**Applies to:** Cashew leaf disease object detection / YOLO

## 1. Mục đích

Tài liệu quy định cách gán nhãn Bounding Box thống nhất cho toàn bộ team nhằm:

- giảm khác biệt giữa annotator;
- giảm label noise;
- tạo ground truth ổn định cho YOLO Direct và YOLO + Classifier;
- tránh tạo quá nhiều object cực nhỏ khó học;
- đảm bảo Train/Validation/Test có cùng annotation policy.

Policy hiện tại được điều chỉnh sau chuỗi YOLO Take 01–06. Kết quả Take 05 → Take 06 cho thấy **selective clear-lesion annotation** cải thiện Precision, F1 và mAP trong khi Mean IoU của matched true positives gần như không đổi. Vì vậy v1.1 ưu tiên lesion rõ, đủ lớn và có ý nghĩa thị giác thay vì cố bounding mọi chấm cực nhỏ.

## 2. Classes

Bounding-box classes:

```text
anthracnose
leaf_miner
red_rust
```

Negative images:

```text
healthy
not_cashew_leaf
```

`healthy` và `not_cashew_leaf` **không tạo box**. Chúng phải được giữ có kiểm soát trong detection dataset dưới dạng **0-box images** để cung cấp negative supervision.

## 3. Nguyên tắc cốt lõi

Chỉ annotate:

> vùng tổn thương có dấu hiệu bệnh đủ rõ để một người review độc lập có thể nhận ra và gán class đáng tin cậy.

Không annotate:

```text
toàn bộ ảnh          ❌
toàn bộ chiếc lá     ❌
background           ❌
healthy tissue       ❌
chấm vài pixel mơ hồ ❌
vùng không chắc class ❌
```

## 4. Selective clear-lesion policy

### Annotate

- lesion rõ ràng;
- lesion đủ lớn để còn ý nghĩa sau resize;
- cluster nhiều đốm nhỏ tạo thành một vùng bệnh có ý nghĩa;
- vùng bệnh liên tục;
- các lesion tách biệt có khoảng healthy tissue rõ ràng.

### Không cố annotate

- từng chấm li ti nếu ảnh có hàng chục/hàng trăm chấm;
- box chỉ vài pixel;
- dấu hiệu mờ không phân biệt được bệnh với noise/texture;
- mọi chấm Red Rust chỉ để tăng số box;
- mọi chấm đen Anthracnose nếu bản thân chúng quá nhỏ hoặc mơ hồ.

Mục tiêu không phải **tối đa số box**, mà là **tối đa chất lượng supervision**.

## 5. Quy tắc box

| Trường hợp | Cách annotate |
|---|---|
| Một vùng bệnh rõ | 1 box sát vùng bệnh |
| Hai vùng bệnh tách biệt | 2 box |
| Nhiều đốm nhỏ sát nhau thành cụm | 1 box bao cluster |
| Nhiều vùng cách xa nhau | Nhiều box |
| Một vùng bệnh lớn liên tục | 1 box |
| Hai bệnh trên cùng lá | Box riêng theo từng class |
| Box khác class cần overlap | Cho phép overlap |
| Healthy | 0 box |
| Not Cashew Leaf | 0 box |
| Không chắc chắn | Đưa Review / Uncertain |

Bounding box phải đủ chặt, hạn chế healthy tissue nằm trong box. Không cần sát từng pixel như segmentation.

## 6. Quy tắc theo class

### Anthracnose

- vùng hoại tử/cháy rõ → box;
- nhiều đốm đen nhỏ sát nhau → ưu tiên cluster;
- không cố tạo một box cho từng chấm đen cực nhỏ;
- không khoanh toàn bộ lá chỉ vì nhiều dấu hiệu rải rác.

### Leaf Miner

- khoanh vùng đường hầm/vệt tổn thương rõ;
- đường tổn thương liên tục → ưu tiên một box đại diện;
- các vùng tách biệt → tách box;
- không chia một đường liên tục thành nhiều box vụn không cần thiết.

### Red Rust

- khoanh các vùng đỏ/cam/rỉ sắt rõ;
- cụm nhiều chấm sát nhau → cluster box;
- vùng tách biệt → box riêng;
- không cố annotation exhaustive toàn bộ hàng loạt chấm rất nhỏ.

## 7. Một ảnh có nhiều bệnh

Không áp dụng:

```text
1 image = 1 disease
```

Một ảnh có thể có nhiều class:

```text
box_01 → anthracnose
box_02 → red_rust
box_03 → leaf_miner
```

Không gộp hai disease class thành một label.

## 8. Negative images

Detection dataset cần ảnh không có lesion target.

### Healthy

```text
healthy image → 0 box
```

### Not Cashew Leaf

```text
not_cashew_leaf image → 0 box
```

Không tạo `healthy_box` hoặc `not_cashew_leaf_box`.

Negative image phải được phân bổ vào Train/Validation/Test theo split đã khóa; không chỉ bổ sung vào Test.

## 9. Consistency giữa annotator

Không được xảy ra với cùng kiểu lesion:

```text
Annotator A → 1 cluster box
Annotator B → 15 tiny boxes
Annotator C → 1 box gần hết lá
```

Team phải thống nhất:

- thế nào là lesion đủ rõ;
- khi nào gom cluster;
- khi nào bỏ qua lesion nhỏ;
- box được phép chứa bao nhiêu healthy tissue;
- khi nào lesion phải tách riêng.

## 10. Review workflow

```text
Annotator
   ↓
Cross Review
   ↓
Fix disagreement
   ↓
Final QC
```

Reviewer kiểm tra:

```text
□ đúng class?
□ còn thiếu lesion rõ ràng?
□ có box cho dấu hiệu quá nhỏ/mơ hồ?
□ box quá rộng?
□ box quá chật?
□ cluster bị chia quá vụn?
□ lesion tách biệt bị gom sai?
□ có box vào healthy/background?
□ negative image có đúng 0 box?
□ có trường hợp cần expert review?
```

Nếu không chắc:

```text
Không tự đoán → REVIEW / UNCERTAIN
```

## 11. Train / Validation / Test

Cả ba split cần ground truth theo **cùng một annotation policy**.

Không random split lại sau khi dataset đã khóa.

Test không dùng để:
- chọn learning rate;
- chọn augmentation;
- chọn model size;
- chọn epochs;
- chọn confidence threshold;
- điều chỉnh annotation policy.

Confidence threshold cuối phải chọn trên Validation, khóa lại, rồi mới đánh giá Test.

## 12. Object detection master dataset

Master annotation giữ disease class:

```text
anthracnose
leaf_miner
red_rust
```

Từ đó có thể sinh:

### YOLO Direct

```text
Bounding Box + Disease Class
```

### YOLO Lesion + Classifier

Map ba disease classes thành:

```text
lesion
```

YOLO phát hiện ROI → crop → classifier phân loại.

Không cần annotate hai lần.

## 13. Checklist trước khi submit ảnh

```text
□ Chỉ annotate lesion rõ và có ý nghĩa?
□ Box có sát vùng tổn thương?
□ Có quá nhiều healthy tissue trong box?
□ Có tạo box cực nhỏ không cần thiết?
□ Các đốm gần nhau có nên gom cluster?
□ Các lesion xa nhau có được tách đúng?
□ Class đúng?
□ Ảnh nhiều bệnh đã có box riêng theo class?
□ Healthy / Not Cashew Leaf có đúng 0 box?
□ Có trường hợp uncertain cần review?
```

## 14. Version history

### v1.0
Policy ban đầu: quy tắc box sát lesion, cluster, multi-disease, cross review.

### v1.1 — Current
Bổ sung từ kết quả Failure Analysis Take 05/06:

- ưu tiên **selective clear-lesion annotation**;
- không bounding exhaustive các chấm cực nhỏ;
- nhấn mạnh negative images;
- khóa threshold bằng Validation;
- giữ annotation policy cố định trước final detector experiment.

Xem bằng chứng thực nghiệm tại:

[`failure/YOLO26/README.md`](./failure/YOLO26/README.md)
