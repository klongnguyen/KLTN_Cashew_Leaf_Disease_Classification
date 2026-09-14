# Legacy Dataset Experiment Archive

Thư mục này lưu các experiment được huấn luyện bằng **dataset cũ** trước khi Master Dataset được thay đổi.

Các kết quả tại đây chỉ có giá trị:

- tham khảo lịch sử thử nghiệm;
- đối chiếu cách cấu hình notebook/model;
- xem lại lỗi, learning curves và cách báo cáo metrics;
- không dùng làm kết quả benchmark chính thức cho dataset hiện tại.

## Archived experiments

### EXP-DENSENET121-SCRATCH-001
DenseNet121 huấn luyện từ scratch trên dataset cũ.

### EXP-VIT-SCRATCH-001
Vision Transformer huấn luyện từ scratch trên dataset cũ.

## Lưu ý

Do dataset hiện tại khác dataset đã dùng cho hai experiment này, các chỉ số Accuracy, Macro-F1, Balanced Accuracy, Confusion Matrix và các metrics khác **không được so sánh trực tiếp** với kết quả huấn luyện mới.

Các experiment mới phải được lưu tại:

`../current_dataset/`
