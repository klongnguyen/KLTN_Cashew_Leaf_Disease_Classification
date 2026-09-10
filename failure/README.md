# FAILURE EXPERIMENT ARCHIVE

Thư mục `failure/` dùng để lưu các experiment **đã chạy được nhưng không đạt yêu cầu để trở thành kết quả cuối**, hoặc các experiment gặp vấn đề về dữ liệu, cấu hình, pipeline hay chất lượng mô hình.

Mục tiêu của thư mục này là giữ lại bằng chứng thực nghiệm, tránh lặp lại các cấu hình đã thất bại và phục vụ phần **Failure Analysis / Discussion** của khóa luận.

## Cấu trúc đề xuất

```text
failure/
├── README.md
├── templates/
│   └── failure_report_template.md
│
├── YOLO26/
│   └── EXP-Y26S-SMALL-001/
│       ├── failure_take01.md
│       ├── config/
│       │   └── experiment_config.json
│       ├── metrics/
│       │   └── metrics_summary.json
│       ├── figures/
│       │   ├── results.png
│       │   ├── confusion_matrix.png
│       │   ├── confusion_matrix_normalized.png
│       │   ├── BoxPR_curve.png
│       │   └── BoxF1_curve.png
│       ├── predictions/
│       │   └── sample_predictions/
│       └── checkpoints/
│           └── best.pt          # optional
│
├── CNN/
├── RESNET50/
├── DENSENET121/
└── VIT/
```

> Git không lưu thư mục rỗng. Các thư mục `figures/`, `predictions/`, `checkpoints/` chỉ cần tạo khi thực sự có artifact tương ứng.

## Quy tắc đặt tên

Mỗi failed experiment phải giữ nguyên `Experiment ID` đã dùng khi train.

Ví dụ:

```text
EXP-Y26S-SMALL-001
EXP-Y26S-DATAFIX-002
EXP-VIT-SCRATCH-001
```

Báo cáo failure dùng format:

```text
failure_take01.md
failure_take02.md
failure_take03.md
```

Nếu cùng một experiment được phân tích lại nhiều lần, tăng số `take` thay vì ghi đè báo cáo cũ.

## Phân loại failure

Có thể ghi một hoặc nhiều loại trong báo cáo:

| Failure Type | Ý nghĩa |
|---|---|
| `PIPELINE_FAILURE` | Notebook/code không chạy được hoặc lỗi kết nối pipeline |
| `DATA_FAILURE` | Dữ liệu, split, annotation hoặc preprocessing có vấn đề |
| `TRAINING_FAILURE` | Loss bất thường, divergence, training không hội tụ |
| `PERFORMANCE_FAILURE` | Train thành công nhưng metric chưa đủ tốt để chọn làm final model |
| `GENERALIZATION_FAILURE` | Validation/Test/Real Holdout giảm mạnh |
| `DEPLOYMENT_FAILURE` | Mô hình quá chậm, quá lớn hoặc không phù hợp triển khai |

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

Không cần lưu lại toàn bộ dataset trong `failure/`.

Checkpoint chỉ nên lưu khi cần tái kiểm tra. Với file lớn, cân nhắc Git LFS hoặc lưu trên Drive và ghi link/tham chiếu trong report thay vì làm repository quá nặng.

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
