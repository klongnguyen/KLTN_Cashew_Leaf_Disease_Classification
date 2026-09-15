# FAILURE EXPERIMENT ARCHIVE

Thư mục `failure/` lưu các experiment đã chạy được nhưng **chưa đạt yêu cầu để trở thành kết quả cuối**, hoặc các experiment có vấn đề về dữ liệu, annotation, cấu hình, pipeline hay khả năng tổng quát hóa.

Mục tiêu là giữ lại bằng chứng thực nghiệm, tránh lặp lại cấu hình không hiệu quả và phục vụ phần **Failure Analysis / Discussion** của khóa luận.

## Failure Index

| Take | Model | Experiment | Thay đổi chính | Kết luận | Báo cáo |
|---|---|---|---|---|---|
| 01 | YOLO26s | `EXP-Y26S-SMALL-001` | Baseline dataset nhỏ | Recall thấp, nhiều lesion bị bỏ sót | [failure_take01.md](./YOLO26/EXP-Y26S-SMALL-001/failure_take01.md) |
| 02 | YOLO26s | `EXP-Y26S-SMALL-002` | Resize dataset lên `640×640` | Localization tăng nhẹ nhưng Precision/Recall/mAP50 không cải thiện; ablation chưa sạch | [failure_take02.md](./YOLO26/EXP-Y26S-SMALL-002/failure_take02.md) |
| 03 | YOLO26s | `EXP-Y26S-SMALL-003` | Giảm box nhỏ/chồng đè/mờ + Horizontal Flip + Rotate 90° | Mean IoU TP tăng mạnh nhưng detection tổng thể còn yếu; annotation policy thay đổi | [failure_take03.md](./YOLO26/EXP-Y26S-SMALL-003/failure_take03.md) |
| 04 | YOLO26s | `EXP-Y26S-SMALL-004` | Bounding thêm lesion nhỏ có ý nghĩa + tăng ảnh annotation | Precision/Recall/mAP phục hồi rõ so với Take 03; FP và imbalance vẫn lớn | [failure_take04.md](./YOLO26/EXP-Y26S-SMALL-004/failure_take04.md) |
| 05 | YOLO26s | `EXP-Y26S-SMALL-005` | Dense small-lesion annotation, bounding nhiều chi tiết nhỏ | Metric tăng mạnh nhưng annotation quá dày, class imbalance và thiếu negative images vẫn rõ | [README.md](./YOLO26/EXP-Y26S-SMALL-005/README.md) |
| 06 | YOLO26s | `EXP-Y26S-SMALL-006` | Selective clear-lesion annotation, ưu tiên lesion rõ và đủ lớn | Tốt hơn Take 005 về Precision/F1/mAP nhưng vẫn chưa đạt final detector; còn imbalance, thiếu negative images và traceability issue | [README.md](./YOLO26/EXP-Y26S-SMALL-006/README.md) |

> Take 02 có lỗi traceability: archive bên ngoài là `EXP-Y26S-SMALL-002` nhưng artifact bên trong vẫn ghi `EXP-Y26S-SMALL-001`.

> Take 04 có lỗi traceability: archive bên ngoài là `EXP-Y26S-SMALL-004` nhưng metadata/checkpoint bên trong vẫn ghi `EXP-Y26S-SMALL-003`.

> Take 06 có lỗi traceability: archive bên ngoài là `EXP-Y26S-SMALL-006` nhưng một số metadata/checkpoint bên trong vẫn ghi `EXP-Y26S-SMALL-005`.

## Cấu trúc hiện tại

```text
failure/
├── README.md
├── templates/
│   └── failure_report_template.md
└── YOLO26/
    ├── README.md
    ├── EXP-Y26S-SMALL-001/
    ├── EXP-Y26S-SMALL-002/
    ├── EXP-Y26S-SMALL-003/
    ├── EXP-Y26S-SMALL-004/
    ├── EXP-Y26S-SMALL-005/
    └── EXP-Y26S-SMALL-006/
```

## Cấu trúc chuẩn cho failed experiment

```text
EXP-.../
├── failure_takeXX.md hoặc README.md
├── config / experiment_config.json
├── metrics / CSV / JSON
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

1. Mỗi experiment có `Experiment ID` riêng và phải đồng nhất giữa notebook, archive, config, metrics, `args.yaml` và checkpoint.
2. Không ghi đè báo cáo experiment cũ.
3. Giữ config, metrics và các figure quan trọng nếu cần.
4. Test set không dùng để tuning.
5. Khi làm ablation phải giữ cố định dataset, split, labels, seed, model và hyperparameters; chỉ thay **một biến đang khảo sát**.
6. Nếu annotation/ground truth thay đổi mạnh, đánh dấu `DATA_POLICY_CHANGE` hoặc `INCONCLUSIVE`, không diễn giải như một ablation sạch.
7. Toàn bộ các lần train YOLO đang ở giai đoạn phát triển/iteration được lưu trong `failure/YOLO26/` cho đến khi có một detector được chốt làm final model.
