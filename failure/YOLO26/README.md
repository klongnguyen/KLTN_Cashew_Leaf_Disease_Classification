# YOLO26 Failure Experiments

Thư mục này lưu các YOLO26 experiment đã chạy thành công về mặt kỹ thuật nhưng chưa đạt tiêu chí để trở thành mô hình cuối cùng.

## Experiment Index

| Take | Experiment | Dataset / Change | Precision | Recall | mAP50 | mAP50-95 | Mean IoU | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| 01 | [`EXP-Y26S-SMALL-001`](./EXP-Y26S-SMALL-001/failure_take01.md) | Baseline dataset nhỏ | 0.6196 | 0.5637 | 0.5589 | 0.2988 | 0.7505 | Failure baseline |
| 02 | [`EXP-Y26S-SMALL-002`](./EXP-Y26S-SMALL-002/failure_take02.md) | Dataset resize `640×640` | 0.5811 | 0.5099 | 0.5336 | 0.3073 | 0.7622 | Mixed / Inconclusive |

## Nhận xét hiện tại

Take 02 cho thấy resolution cao hơn có tín hiệu cải thiện localization (`Mean IoU`, `mAP50-95`) nhưng không cải thiện detection tổng thể. Precision, Recall, F1 và mAP50 đều giảm so với Take 01.

Ngoài ra, Train set giữa hai take không hoàn toàn giống nhau nên chưa thể xem đây là một resolution ablation sạch.

## Hướng tiếp theo

1. Khóa một Master Pilot Dataset cố định.
2. Review annotation `anthracnose` và `red_rust`.
3. Đảm bảo số ảnh/box/split giống nhau giữa các experiment.
4. Sửa `EXPERIMENT_ID` đúng trước khi train.
5. Chỉ thay một biến trong mỗi ablation.
6. Dùng Validation để chọn threshold/hyperparameter; Test chỉ dùng đánh giá cuối.
