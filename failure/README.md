# FAILURE EXPERIMENT ARCHIVE

Thư mục `failure/` dùng để lưu các experiment **đã chạy được nhưng không đạt yêu cầu để trở thành kết quả cuối**, hoặc các experiment gặp vấn đề về dữ liệu, cấu hình, pipeline hay chất lượng mô hình.

Mục tiêu của thư mục này là giữ lại bằng chứng thực nghiệm, tránh lặp lại các cấu hình đã thất bại và phục vụ phần **Failure Analysis / Discussion** của khóa luận.

---

## Failure Index

| Take | Model | Experiment | Thay đổi chính | Kết luận | Báo cáo |
|---|---|---|---|---|---|
| 01 | YOLO26s | `EXP-Y26S-SMALL-001` | Baseline dataset nhỏ | Model học được nhưng Recall thấp, nhiều lesion bị bỏ sót | [failure_take01.md](./YOLO26/EXP-Y26S-SMALL-001/failure_take01.md) |
| 02 | YOLO26s | `EXP-Y26S-SMALL-002` | Resize dataset lên `640×640` | Localization tăng nhẹ nhưng Precision/Recall/mAP50 không cải thiện; ablation chưa sạch | [failure_take02.md](./YOLO26/EXP-Y26S-SMALL-002/failure_take02.md) |

> `EXP-Y26S-SMALL-002` là tên archive bên ngoài. Artifact bên trong vẫn ghi nhầm `EXP-Y26S-SMALL-001`; lỗi traceability này được ghi rõ trong Take 02.

---

## Cấu trúc thư mục

```text
failure/
├── README.md
├── templates/
│   └── failure_report_template.md
│
├── YOLO26/
│   ├── README.md
│   ├── EXP-Y26S-SMALL-001/
│   │   ├── failure_take01.md
│   │   ├── config/
│   │   │   └── experiment_config.json
│   │   └── metrics/
│   │       └── metrics_summary.json
│   │
│   └── EXP-Y26S-SMALL-002/
│       ├── failure_take02.md
│       ├── config/
│       │   └── experiment_config.json
│       └── metrics/
│           ├── metrics_summary.json
│           ├── final_summary.csv
│           ├── per_class_metrics.csv
│           └── dataset_statistics.csv
│
├── CNN/
├── RESNET50/
├── DENSENET121/
└── VIT/
```

> Git không lưu thư mục rỗng. Các thư mục `figures/`, `predictions/`, `checkpoints/` chỉ cần tạo khi thực sự có artifact tương ứng.

---

## Cấu trúc chuẩn cho một failed experiment

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
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── BoxPR_curve.png
│   └── BoxF1_curve.png
├── predictions/              # optional
│   └── sample_predictions/
└── checkpoints/              # optional
    └── best.pt
```

Không cần lưu toàn bộ dataset trong `failure/`.

Checkpoint chỉ nên lưu khi cần tái kiểm tra. Với file lớn, cân nhắc Git LFS hoặc lưu trên Drive và ghi link/tham chiếu trong report thay vì làm repository quá nặng.

---

## Quy tắc đặt tên

Mỗi failed experiment phải giữ nguyên `Experiment ID` đã dùng khi train.

Ví dụ:

```text
EXP-Y26S-SMALL-001
EXP-Y26S-SMALL-002
EXP-Y26S-DATAFIX-003
EXP-VIT-SCRATCH-001
```

Báo cáo failure dùng format:

```text
failure_take01.md
failure_take02.md
failure_take03.md
```

Nếu cùng một experiment được phân tích lại nhiều lần, tăng số `take` thay vì ghi đè báo cáo cũ.

---

## Phân loại failure

| Failure Type | Ý nghĩa |
|---|---|
| `PIPELINE_FAILURE` | Notebook/code không chạy được hoặc lỗi kết nối pipeline |
| `DATA_FAILURE` | Dữ liệu, split, annotation hoặc preprocessing có vấn đề |
| `TRAINING_FAILURE` | Loss bất thường, divergence, training không hội tụ |
| `PERFORMANCE_FAILURE` | Train thành công nhưng metric chưa đủ tốt để chọn làm final model |
| `GENERALIZATION_FAILURE` | Validation/Test/Real Holdout giảm mạnh |
| `DEPLOYMENT_FAILURE` | Mô hình quá chậm, quá lớn hoặc không phù hợp triển khai |

---

## Artifact bắt buộc nên giữ

Đối với một failed experiment có giá trị nghiên cứu, nên giữ tối thiểu:

```text
failure report
experiment config
metrics summary
learning curves
confusion matrix
PR/F1 curves nếu là detection
một số prediction đúng/sai đại diện
```

Trong giai đoạn đầu có thể chỉ commit `report + config + metrics` để repository gọn. Figure hoặc checkpoint lớn có thể bổ sung khi cần trình bày báo cáo.

---

## Nguyên tắc đánh giá

Một experiment nằm trong `failure/` không có nghĩa là vô ích. Nó được giữ lại nếu giúp trả lời một trong các câu hỏi:

- Model sai ở đâu?
- Data/annotation có vấn đề gì?
- Hyperparameter nào không phù hợp?
- Class nào khó nhất?
- Lesion nhỏ có bị bỏ sót không?
- Resolution có ảnh hưởng không?
- Có dấu hiệu overfitting hay không?
- Cần thay đổi gì ở experiment tiếp theo?

---

## Quy trình sau một failed experiment

```text
Train / Evaluate
      ↓
Không đạt yêu cầu final
      ↓
Lưu config + metrics + figures
      ↓
Viết failure_takeXX.md
      ↓
Xác định root cause
      ↓
Đề xuất đúng một nhóm thay đổi chính
      ↓
Tạo Experiment ID mới
      ↓
So sánh với failed baseline
```

Không nên thay đồng thời quá nhiều yếu tố ở experiment tiếp theo vì sẽ khó xác định yếu tố nào thực sự tạo ra cải thiện.

---

## Quy tắc quan trọng cho ablation

Khi muốn chứng minh ảnh hưởng của một yếu tố như resolution, augmentation hoặc learning rate, phải giữ cố định các yếu tố còn lại:

```text
same dataset
same split
same labels
same boxes
same seed
same model
same hyperparameters
```

Chỉ thay **một biến đang khảo sát**. Nếu dataset hoặc annotation thay đổi đồng thời, experiment phải được đánh dấu là `INCONCLUSIVE` thay vì dùng để kết luận nguyên nhân - kết quả.
