# YOLO26 Failure Experiments

Thư mục này lưu các YOLO26 experiment đã chạy thành công về mặt kỹ thuật nhưng **chưa được chọn làm detector cuối cùng**. Toàn bộ Take 01 → Take 07 hiện được quản lý tại đây để tách khỏi các baseline classification hợp lệ trong `training_results/current_dataset/`.

## Experiment Index

| Take | Experiment | Dataset / Change | Precision | Recall | mAP50 | mAP50-95 | Mean IoU | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| 01 | [`EXP-Y26S-SMALL-001`](./EXP-Y26S-SMALL-001/failure_take01.md) | Baseline dataset nhỏ | 0.6196 | 0.5637 | 0.5589 | 0.2988 | 0.7505 | Failure baseline |
| 02 | [`EXP-Y26S-SMALL-002`](./EXP-Y26S-SMALL-002/failure_take02.md) | Dataset resize `640×640` | 0.5811 | 0.5099 | 0.5336 | 0.3073 | 0.7622 | Mixed / Inconclusive |
| 03 | [`EXP-Y26S-SMALL-003`](./EXP-Y26S-SMALL-003/failure_take03.md) | Clean annotation + Horizontal Flip + Rotate 90° | 0.3111 | 0.4295 | 0.3522 | 0.2148 | **0.8163** | Failure for final model / useful data iteration |
| 04 | [`EXP-Y26S-SMALL-004`](./EXP-Y26S-SMALL-004/failure_take04.md) | Thêm lesion nhỏ có ý nghĩa + tăng ảnh annotation | 0.4830 | 0.5360 | 0.4929 | 0.2588 | 0.7910 | Promising data iteration / not final |
| 05 | [`EXP-Y26S-SMALL-005`](./EXP-Y26S-SMALL-005/README.md) | Dense small-lesion annotation | 0.6929 | 0.6506 | 0.6609 | 0.3405 | 0.7629 | Improved, but annotation policy still noisy / not final |
| 06 | [`EXP-Y26S-SMALL-006`](./EXP-Y26S-SMALL-006/README.md) | Selective clear-lesion annotation | **0.7473** | **0.6648** | **0.7298** | **0.3877** | 0.7603 | Best aggregate YOLO metrics so far, still not final |
| 07 | [`EXP-Y26S-SMALL-007`](./EXP-Y26S-SMALL-007/README.md) | Detailed clear-lesion annotation + larger dataset/test | 0.6945 | 0.6598 | 0.6993 | 0.3517 | 0.7471 | Larger evaluation; below Take 006 aggregate metrics / not final |

## Take 005 → Take 006

Take 005 bounding rất dày, kể cả nhiều lesion/chấm rất nhỏ. Take 006 chuyển sang ưu tiên các vùng bệnh **rõ, đủ lớn và có ý nghĩa hơn**.

Kết quả Take 006 tăng rõ so với Take 005:

- Precision: `0.6929 → 0.7473`;
- Recall: `0.6506 → 0.6648`;
- F1: `0.6711 → 0.7036`;
- mAP50: `0.6609 → 0.7298`;
- mAP50-95: `0.3405 → 0.3877`;
- Mean IoU gần như giữ nguyên: `0.7629 → 0.7603`.

Điều này ủng hộ annotation policy chọn lọc lesion rõ hơn thay vì cố bounding mọi chấm cực nhỏ.

## Take 006 → Take 007

Take 007 tiếp tục giữ nguyên tinh thần **chi tiết nhưng bỏ lesion quá nhỏ/mờ**, đồng thời mở rộng dataset:

- Train images: `441 → 496`;
- Validation images: `42 → 80`;
- Test images: `20 → 39`;
- Test GT boxes: `491 → 1,335`.

Aggregate metrics Take 007 thấp hơn Take 006:

- Precision: `0.7473 → 0.6945`;
- Recall: `0.6648 → 0.6598`;
- F1: `0.7036 → 0.6767`;
- mAP50: `0.7298 → 0.6993`;
- mAP50-95: `0.3877 → 0.3517`;
- Mean IoU: `0.7603 → 0.7471`.

Tuy nhiên đây **không phải controlled ablation** vì Validation/Test thay đổi rất mạnh. Take 007 được đánh giá trên một Test set lớn hơn nhiều và dày GT hơn, nên không được kết luận đơn giản rằng annotation policy 007 kém hơn.

Ở custom evaluation `conf=0.25`, Take 007 có Precision `0.6796` cao hơn Take 006 `0.6075`, nhưng Recall giảm `0.7251 → 0.6180`. Điều này cho thấy prediction ở operating point cố định sạch hơn nhưng bỏ sót nhiều GT hơn.

## Nhận xét per-class Take 007

- **Anthracnose:** Precision cao (`0.7771`) nhưng Recall thấp hơn (`0.5230`); một số sample vẫn cho thấy nhiều box rất nhỏ trên cùng một lá.
- **Leaf Miner:** Recall cao (`0.8022`) và mAP50 tốt (`0.8008`), nhưng Precision giảm xuống `0.6651`; confusion với Anthracnose vẫn đáng chú ý.
- **Red Rust:** Precision `0.6414`, Recall `0.6541`, mAP50 `0.6676`; dù chiếm hơn 70% Train boxes, metric không vượt Take 006, cho thấy bottleneck không chỉ là số lượng.

## Vì sao Take 007 vẫn ở Failure Archive?

1. Aggregate metric chưa vượt Take 006.
2. Class imbalance vẫn lớn: Train Take 007 có `anthracnose=3,465`, `leaf_miner=732`, `red_rust=10,064`.
3. Negative images gần như chưa có: chỉ `2/496` Train images là 0-box; Validation/Test đều có box.
4. Split và ground truth thay đổi giữa Take 006/007, nên không thể dùng như controlled ablation.
5. Chưa có detector được khóa cấu hình, khóa split, khóa threshold trên Validation và kiểm tra final theo protocol hoàn chỉnh.

## Bottleneck hiện tại

- **Class imbalance:** Red Rust vẫn chiếm khoảng 70.6% Train boxes, Leaf Miner khoảng 5.1%.
- **Negative supervision:** cần thêm `healthy` và nếu phù hợp pipeline thì `not_cashew_leaf` dưới dạng ảnh 0 box.
- **Annotation consistency:** vẫn cần quy định rõ mức chi tiết tối thiểu, khi nào gộp cluster và khi nào bỏ lesion quá nhỏ/mờ.
- **Leaf Miner ↔ Anthracnose confusion:** cần audit riêng các vùng hoại tử/đốm nhỏ dễ nhầm.
- **Ablation protocol:** phải khóa Train/Val/Test và GT trước khi so thay đổi annotation/hyperparameter.

## Hướng tiếp theo

1. Giữ policy Take 007: lesion rõ → annotate; lesion quá nhỏ/mờ/không chắc → bỏ hoặc gom cluster hợp lý.
2. Khóa một phiên bản Train/Val/Test và không thay Test nữa.
3. Bổ sung negative images có kiểm soát.
4. Tăng đa dạng ảnh nguồn Leaf Miner; không chỉ tăng thêm box Red Rust.
5. Giữ YOLO26s + `640×640` trước khi đổi resolution/model size.
6. Tune confidence threshold trên Validation rồi khóa threshold trước khi đánh giá Test.
7. Chỉ đưa YOLO ra khỏi `failure/` khi đã chốt một detector cuối theo protocol này.
