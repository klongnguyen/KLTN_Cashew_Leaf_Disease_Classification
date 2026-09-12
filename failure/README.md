# FAILURE EXPERIMENT ARCHIVE

Thư mục `failure/` lưu các experiment đã chạy được nhưng **chưa đạt yêu cầu để trở thành kết quả cuối**, hoặc các experiment có vấn đề về dữ liệu, annotation, cấu hình, pipeline hay khả năng tổng quát hóa.

Mục tiêu là giữ lại bằng chứng thực nghiệm, tránh lặp lại cấu hình không hiệu quả và phục vụ phần **Failure Analysis / Discussion** của khóa luận.

## Failure Index

| Take | Model | Experiment | Thay đổi chính | Kết luận | Báo cáo |
|---|---|---|---|---|---|
| 01 | YOLO26s | `EXP-Y26S-SMALL-001` | Baseline dataset nhỏ | Recall thấp, nhiều lesion bị bỏ sót | [failure_take01.md](./YOLO26/EXP-Y26S-SMALL-001/failure_take01.md) |
| 02 | YOLO26s | `EXP-Y26S-SMALL-002` | Resize dataset lên `640×640` | Localization tăng nhẹ nhưng Precision/Recall/mAP50 không cải thiện; ablation chưa sạch | [failure_take02.md](./YOLO26/EXP-Y26S-SMALL-002/failure_take02.md) |
| 03 | YOLO26s | `EXP-Y26S-SMALL-003` | Giảm box nhỏ/chồng đè/mờ + Horizontal Flip + Rotate 90° | Mean IoU TP tăng mạnh nhưng detection tổng thể còn yếu; annotation policy thay đổi nên không so sánh trực tiếp với Take 01/02 | [failure_take03.md](./YOLO26/EXP-Y26S-SMALL-003/failure_take03.md) |

> Take 02 có lỗi traceability: archive bên ngoài là `EXP-Y26S-SMALL-002` nhưng artifact bên trong vẫn ghi `EXP-Y26S-SMALL-001`.

> Take 03 đã sửa đúng traceability: archive, metadata và checkpoint đều dùng `EXP-Y26S-SMALL-003`.

## Cấu trúc hiện tại

```text
failure/
├── README.md
├── templates/
│   └── failure_report_template.md
└── YOLO26/
    ├── README.md
    ├── EXP-Y26S-SMALL-001/
    │   ├── failure_take01.md
    │   ├── config/
    │   └── metrics/
    ├── EXP-Y26S-SMALL-002/
    │   ├── failure_take02.md
    │   ├── config/
    │   └── metrics/
    └── EXP-Y26S-SMALL-003/
        ├── failure_take03.md
        ├── config/
        │   └── experiment_config.json
        └── metrics/
            ├── metrics_summary.json
            ├── final_summary.csv
            ├── per_class_metrics.csv
            ├── custom_iou_per_class.csv
            └── dataset_statistics.csv
```

## Cấu trúc chuẩn cho failed experiment

```text
EXP-.../
├── failure_takeXX.md
├── config/
│   └── experiment_config.json
├── metrics/
│   ├── metrics_summary.json
│   ├── final_summary.csv
│   ├── per_class_metrics.csv
│   └── dataset_statistics.csv
├── figures/                  # optional
├── predictions/              # optional
└── checkpoints/              # optional
```

Không cần lưu toàn bộ dataset trong `failure/`. Checkpoint lớn chỉ nên lưu khi cần tái kiểm tra; có thể dùng Git LFS hoặc Drive.

## Phân loại failure

| Failure Type | Ý nghĩa |
|---|---|
| `PIPELINE_FAILURE` | Notebook/code không chạy được hoặc lỗi kết nối pipeline |
| `DATA_FAILURE` | Dữ liệu, split, annotation hoặc preprocessing có vấn đề |
| `DATA_POLICY_CHANGE` | Quy tắc annotation/ground truth thay đổi đáng kể nên metric không còn so sánh trực tiếp |
| `TRAINING_FAILURE` | Loss bất thường, divergence, training không hội tụ |
| `PERFORMANCE_FAILURE` | Train thành công nhưng metric chưa đủ tốt để chọn làm final model |
| `GENERALIZATION_FAILURE` | Validation/Test/Real Holdout giảm mạnh |
| `DEPLOYMENT_FAILURE` | Mô hình quá chậm, quá lớn hoặc không phù hợp triển khai |

## Quy tắc experiment

1. Mỗi experiment có `Experiment ID` riêng.
2. Không ghi đè `failure_takeXX.md` cũ.
3. Giữ config, metrics và các figure quan trọng nếu cần.
4. Test set không dùng để tuning.
5. Khi làm ablation phải giữ cố định dataset, split, labels, seed, model và hyperparameters; chỉ thay **một biến đang khảo sát**.
6. Nếu annotation/ground truth thay đổi mạnh, đánh dấu `DATA_POLICY_CHANGE` hoặc `INCONCLUSIVE`, không diễn giải như một ablation sạch.
