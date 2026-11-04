# Negative Image Evaluation — V003 so với NEG V02

Bốn checkpoint `best.pt` được chạy trên cùng **951 ảnh Validation NEG V02**: 893 ảnh có GT box và 58 ảnh âm (38 healthy, 20 not_cashew_leaf). Không train lại và không dùng Test.

## Cấu hình và cách tính

- Ultralytics 8.4.197; ảnh 640 px; batch 4; IoU 0.7; FP16 tắt; `rect=False`.
- Báo nhầm ảnh âm: ít nhất một box bệnh ở `conf ≥ 0.25`. Mọi box trên ảnh 0 GT là false positive theo nhãn hiện có.
- P/R/AP: `model.val(split='val', conf=0.001, iou=0.7)` trên cả 951 ảnh để tính đường Precision–Recall chuẩn. Ngưỡng conf thấp này được giữ giống nhau cho cả bốn checkpoint.
- Dấu vân tay tập Validation SHA256: `9a502be84e2d619a6d75bb76b979c7d40eabc8d6688124d05d013965555ee78f`.

## Báo nhầm trên ảnh âm

| Model | Run | Healthy ảnh báo nhầm | Not cashew ảnh báo nhầm | Tổng ảnh báo nhầm | Box sai / ảnh âm |
|---|---|---:|---:|---:|---:|
| yolov8s | v003 | 8/38 (21.1%) | 5/20 (25.0%) | 13/58 (22.4%) | 0.328 |
| yolov8s | neg_v02 | 8/38 (21.1%) | 6/20 (30.0%) | 14/58 (24.1%) | 0.328 |
| yolo26s | v003 | 6/38 (15.8%) | 9/20 (45.0%) | 15/58 (25.9%) | 0.414 |
| yolo26s | neg_v02 | 2/38 (5.3%) | 1/20 (5.0%) | 3/58 (5.2%) | 0.052 |

YOLO26s giảm từ 15 xuống 3 ảnh âm bị báo nhầm; YOLOv8s tăng từ 13 lên 14 ảnh ở ngưỡng đã định.

## Chuyển trạng thái theo cùng từng ảnh

| Model | V003 báo nhầm, NEG V02 không | NEG V02 báo nhầm, V003 không | McNemar exact p (thăm dò) |
|---|---:|---:|---:|
| yolov8s | 5 | 6 | 1.0000 |
| yolo26s | 14 | 2 | 0.0042 |

Giá trị p là phân tích thăm dò trên 58 ảnh của một seed, không thay thế đánh giá qua nhiều seed hoặc tập dữ liệu độc lập.

## P/R/AP trên cùng Validation

| Model | Run | Precision | Recall | mAP50 | mAP50–95 |
|---|---|---:|---:|---:|---:|
| yolov8s | v003 | 0.665 | 0.642 | 0.690 | 0.374 |
| yolov8s | neg_v02 | 0.703 | 0.643 | 0.698 | 0.385 |
| yolo26s | v003 | 0.696 | 0.653 | 0.705 | 0.398 |
| yolo26s | neg_v02 | 0.677 | 0.665 | 0.707 | 0.397 |

## Class dự đoán nhầm trên ảnh âm (số box)

| Model | Run | Anthracnose | Leaf Miner | Red Rust |
|---|---|---:|---:|---:|
| yolov8s | v003 | 17 | 1 | 1 |
| yolov8s | neg_v02 | 15 | 1 | 3 |
| yolo26s | v003 | 17 | 6 | 1 |
| yolo26s | neg_v02 | 1 | 0 | 2 |

Chi tiết theo nhóm và class nằm trong từng file `negative_per_class.csv`; danh sách từng ảnh ở `negative_per_image.csv`, từng box ở `negative_boxes.csv`.

## Giới hạn

- Kết luận false positive dựa trên nhãn 0 box của ảnh âm; ảnh bệnh chưa gán nhãn sẽ làm sai kết luận.
- Chỉ một seed và 58 ảnh âm Validation. Không suy ra hiệu quả tổng quát cho mọi dữ liệu mới.
- YOLO26 là kiến trúc end-to-end; tham số IoU NMS có thể tác động khác YOLOv8, nhưng trong từng cặp cùng kiến trúc, cấu hình suy luận giống nhau.
- Các checkpoint V003 và NEG V02 được đánh giá bằng cùng Ultralytics 8.4.197; metric cũ trong ANALYSIS ZIP không được dùng để so sánh.

[Tài liệu Ultralytics về Predict](https://docs.ultralytics.com/modes/predict) · [Tài liệu về Validation](https://docs.ultralytics.com/modes/val)
