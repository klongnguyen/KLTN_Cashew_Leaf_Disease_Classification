# YOLO26 Failure Experiments

Thư mục này lưu các YOLO26 experiment đã chạy thành công về mặt kỹ thuật nhưng **chưa được chọn làm detector cuối cùng**. Toàn bộ Take 01 → Take 06 hiện được quản lý tại đây để tách khỏi các baseline classification hợp lệ trong `training_results/current_dataset/`.

## Experiment Index

| Take | Experiment | Dataset / Change | Precision | Recall | mAP50 | mAP50-95 | Mean IoU | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| 01 | [`EXP-Y26S-SMALL-001`](./EXP-Y26S-SMALL-001/failure_take01.md) | Baseline dataset nhỏ | 0.6196 | 0.5637 | 0.5589 | 0.2988 | 0.7505 | Failure baseline |
| 02 | [`EXP-Y26S-SMALL-002`](./EXP-Y26S-SMALL-002/failure_take02.md) | Dataset resize `640×640` | 0.5811 | 0.5099 | 0.5336 | 0.3073 | 0.7622 | Mixed / Inconclusive |
| 03 | [`EXP-Y26S-SMALL-003`](./EXP-Y26S-SMALL-003/failure_take03.md) | Clean annotation + Horizontal Flip + Rotate 90° | 0.3111 | 0.4295 | 0.3522 | 0.2148 | **0.8163** | Failure for final model / useful data iteration |
| 04 | [`EXP-Y26S-SMALL-004`](./EXP-Y26S-SMALL-004/failure_take04.md) | Thêm lesion nhỏ có ý nghĩa + tăng ảnh annotation | 0.4830 | 0.5360 | 0.4929 | 0.2588 | 0.7910 | Promising data iteration / not final |
| 05 | [`EXP-Y26S-SMALL-005`](./EXP-Y26S-SMALL-005/README.md) | Dense small-lesion annotation | 0.6929 | 0.6506 | 0.6609 | 0.3405 | 0.7629 | Improved, but annotation policy still noisy / not final |
| 06 | [`EXP-Y26S-SMALL-006`](./EXP-Y26S-SMALL-006/README.md) | Selective clear-lesion annotation | **0.7473** | **0.6648** | **0.7298** | **0.3877** | 0.7603 | Best YOLO iteration so far, still archived as failure / not final |

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

## Vì sao Take 006 vẫn ở Failure Archive?

Take 006 hiện là kết quả YOLO tốt nhất nhưng **chưa đủ điều kiện để gọi là final detector** vì:

1. Class imbalance vẫn lớn. Train Take 006 có 12,781 boxes: `anthracnose=3,169`, `leaf_miner=643`, `red_rust=8,969`.
2. Negative images còn thiếu. Train Take 006 có `441/441` ảnh có box; Validation chỉ có 2 ảnh không box; Test `20/20` ảnh có box.
3. Annotation policy qua các take thay đổi mạnh, nên không phải toàn bộ so sánh đều là controlled ablation.
4. Take 006 còn lỗi traceability: một số metadata/checkpoint vẫn ghi ID `EXP-Y26S-SMALL-005`.
5. Chưa có detector được khóa cấu hình, khóa split, khóa threshold trên Validation và kiểm tra final theo protocol hoàn chỉnh.

## Bottleneck hiện tại

- **Class imbalance:** Red Rust vẫn chiếm phần lớn box, Leaf Miner ít hơn rõ rệt.
- **Negative supervision:** cần thêm `healthy` và nếu phù hợp với pipeline thì `not_cashew_leaf` dưới dạng ảnh 0 box.
- **Annotation consistency:** cần giữ một policy ổn định giữa các class và các split.
- **Traceability:** Experiment ID phải đồng nhất giữa folder, config, metrics, checkpoint và archive.

## Hướng tiếp theo

1. Giữ annotation policy của Take 006 làm baseline mới: ưu tiên lesion rõ, đủ lớn, có ý nghĩa; tránh bounding mọi chấm cực nhỏ.
2. Bổ sung negative images có kiểm soát.
3. Tăng ảnh nguồn Leaf Miner thay vì tiếp tục tăng mạnh Red Rust.
4. Khóa Train/Val/Test + ground truth trước lần train tiếp theo.
5. Chọn confidence threshold trên Validation, sau đó khóa threshold trước khi đánh giá Test.
6. Chỉ đưa YOLO ra khỏi `failure/` khi đã chốt một detector cuối theo protocol này.
