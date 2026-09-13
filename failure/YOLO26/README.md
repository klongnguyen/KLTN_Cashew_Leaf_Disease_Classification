# YOLO26 Failure Experiments

Thư mục này lưu các YOLO26 experiment đã chạy thành công về mặt kỹ thuật nhưng chưa đạt tiêu chí để trở thành mô hình cuối cùng.

## Experiment Index

| Take | Experiment | Dataset / Change | Precision | Recall | mAP50 | mAP50-95 | Mean IoU | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| 01 | [`EXP-Y26S-SMALL-001`](./EXP-Y26S-SMALL-001/failure_take01.md) | Baseline dataset nhỏ | 0.6196 | 0.5637 | 0.5589 | 0.2988 | 0.7505 | Failure baseline |
| 02 | [`EXP-Y26S-SMALL-002`](./EXP-Y26S-SMALL-002/failure_take02.md) | Dataset resize `640×640` | 0.5811 | 0.5099 | 0.5336 | 0.3073 | 0.7622 | Mixed / Inconclusive |
| 03 | [`EXP-Y26S-SMALL-003`](./EXP-Y26S-SMALL-003/failure_take03.md) | Clean annotation + Horizontal Flip + Rotate 90° | 0.3111 | 0.4295 | 0.3522 | 0.2148 | **0.8163** | Failure for final model / useful data iteration |
| 04 | [`EXP-Y26S-SMALL-004`](./EXP-Y26S-SMALL-004/failure_take04.md) | Thêm lesion nhỏ có ý nghĩa + tăng ảnh annotation | **0.4830** | **0.5360** | **0.4929** | **0.2588** | 0.7910 | Promising data iteration / not final |

## Nhận xét hiện tại

Take 03 giảm mạnh số box nhỏ/chồng đè/mờ nên localization của các TP rất tốt nhưng supervision quá thưa. Take 04 đưa lại các lesion nhỏ **rõ ràng và có ý nghĩa**, đồng thời tăng số ảnh annotation. Kết quả là Precision, Recall, F1 và mAP phục hồi rõ rệt trong khi Mean IoU vẫn giữ ở mức tốt ~0.79.

Bottleneck hiện tại đã chuyển sang:

1. **False Positive cao** — custom evaluation có 353 TP nhưng 354 FP.
2. **Class imbalance lớn** — Leaf Miner chỉ chiếm khoảng 2.9% số box Train.
3. **Thiếu negative images** — thống kê hiện tại cho thấy mọi ảnh ở Train/Val/Test đều có box.
4. **Traceability** — archive Take 04 là `EXP-Y26S-SMALL-004` nhưng metadata/checkpoint bên trong vẫn là `EXP-Y26S-SMALL-003`.

## Hướng tiếp theo

1. Khóa annotation policy của Take 04: lesion nhỏ nhưng rõ → annotate; lesion quá mờ/không chắc → bỏ; không bounding tất cả dấu hiệu trên lá.
2. Tăng **ảnh nguồn Leaf Miner** thay vì tiếp tục tăng Red Rust.
3. Bổ sung `healthy` và `not_cashew_leaf` dưới dạng negative images (0 box) cho YOLO Direct.
4. Khóa Train/Val/Test và ground truth trước khi thực hiện ablation tiếp theo.
5. Tune confidence threshold trên Validation, không trên Test.
6. Sửa `EXPERIMENT_ID` thành `EXP-Y26S-SMALL-005` trước lần train tiếp theo.
