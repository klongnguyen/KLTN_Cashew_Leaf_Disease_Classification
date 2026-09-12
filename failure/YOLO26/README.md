# YOLO26 Failure Experiments

Thư mục này lưu các YOLO26 experiment đã chạy thành công về mặt kỹ thuật nhưng chưa đạt tiêu chí để trở thành mô hình cuối cùng.

## Experiment Index

| Take | Experiment | Dataset / Change | Precision | Recall | mAP50 | mAP50-95 | Mean IoU | Status |
|---|---|---|---:|---:|---:|---:|---:|---|
| 01 | [`EXP-Y26S-SMALL-001`](./EXP-Y26S-SMALL-001/failure_take01.md) | Baseline dataset nhỏ | 0.6196 | 0.5637 | 0.5589 | 0.2988 | 0.7505 | Failure baseline |
| 02 | [`EXP-Y26S-SMALL-002`](./EXP-Y26S-SMALL-002/failure_take02.md) | Dataset resize `640×640` | 0.5811 | 0.5099 | 0.5336 | 0.3073 | 0.7622 | Mixed / Inconclusive |
| 03 | [`EXP-Y26S-SMALL-003`](./EXP-Y26S-SMALL-003/failure_take03.md) | Clean annotation + Horizontal Flip + Rotate 90° | 0.3111 | 0.4295 | 0.3522 | 0.2148 | **0.8163** | Failure for final model / useful data iteration |

## Nhận xét hiện tại

Take 03 đã giảm mạnh số box nhỏ, box chồng đè và vùng bệnh quá mờ. Số annotation giảm khoảng 74–78% tùy split. Do ground truth thay đổi mạnh, Take 03 không thể so sánh trực tiếp với Take 01/02 như một ablation thuần túy.

Tín hiệu tích cực nhất của Take 03 là **Mean IoU của matched true positives tăng lên 0.8163**, cho thấy box đúng có chất lượng localization tốt hơn. Tuy nhiên Precision, Recall, mAP50 và mAP50-95 vẫn thấp; Red Rust có nhiều false negative và validation có dấu hiệu overfitting sau khoảng epoch 30.

## Hướng tiếp theo

1. Giữ annotation policy mới, không quay lại hàng nghìn box li ti/chồng đè.
2. Tăng ảnh nguồn độc lập thay vì chỉ tăng augmentation.
3. Khóa Train/Val/Test và không đổi ground truth giữa các experiment.
4. Tạo ablation sạch cho offline augmentation.
5. Dùng Validation để chọn threshold/hyperparameter; Test chỉ dùng đánh giá cuối.
6. Theo dõi overfitting và cân nhắc dừng quanh 35–40 epochs sau khi dataset đã được khóa.
