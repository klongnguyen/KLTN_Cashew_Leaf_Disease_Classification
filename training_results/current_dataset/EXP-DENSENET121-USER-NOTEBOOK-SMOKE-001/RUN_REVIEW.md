# Kiểm tra notebook DenseNet121 do người dùng cung cấp

Nguồn: `C:\Users\KIM LONG\Downloads\densenet121_scratch_cashewdata_split7_5class_001 (1).ipynb`.

## Phạm vi đã chạy lại

- Thực thi nguyên mã các cell 19, 21 và 23 (nạp `tf.data`, dựng và biên dịch DenseNet121, callback) với thư mục dữ liệu cục bộ; huấn luyện 2 batch Train, đánh giá 1 batch Validation trong 1 epoch.
- Hoàn thành `fit`, lưu checkpoint `.keras`, nạp lại checkpoint và chạy suy luận cùng batch hai lần. Chênh lệch xác suất lớn nhất giữa hai lần suy luận là 0.
- TensorFlow 2.21 trên Windows trong phiên này báo `GPU=[]`; phép thử dùng CPU, mất 65,23 giây, nên accuracy 21,88% Train / 6,25% Validation của **32/16 ảnh** không phải phép đo chất lượng mô hình.
- Thư mục cục bộ có đủ 4825 Train và 1395 Validation nhưng chỉ có 686 Test, thiếu `test/not_cashew_leaf/not_cashew_leaf_1514.JPG` so với ZIP gốc 687 Test. Phép thử ngắn không đánh giá Test. Để tái lập kết quả gốc cần dùng ZIP có SHA-256 `6983b9ab8339489f18edb00f773b319ee6c62c2a3076c199b078ef52b464a3bf`.

## Kết quả đã có trong notebook đính kèm (chưa huấn luyện lại toàn bộ)

- 49 epoch, 302 batch Train/epoch, khoảng 14.798 lần cập nhật trọng số; checkpoint tốt nhất theo `val_loss` ở epoch 39 (khoảng 11.778 lần cập nhật).
- Test 639/687 đúng = 93,01%, macro F1 = 92,50%. Nhóm tên `real_`: 19/28 đúng = 67,86%; nhóm còn lại: 620/659 đúng = 94,08%. Validation không có tên tệp `real_`, nên không chọn checkpoint theo hiệu quả nhóm này được.
- Mã sử dụng `preprocess_input`, cùng thứ tự lớp cho các split, `shuffle=False` cho Validation/Test và nạp checkpoint tốt nhất để đánh giá. Phép thử ngắn không phát hiện lỗi chạy trong các cell này. Cảnh báo `shuffle=True` khi gọi `fit` với `tf.data.Dataset` không tắt việc xáo trộn của `image_dataset_from_directory` ở Train.

## Vấn đề dữ liệu cần xử lý

- Kiểm tra trùng lặp trong notebook chỉ chặn SHA-256 giống nhau. Cặp `test/not_cashew_leaf/not_cashew_leaf_1514.JPG` và `train/not_cashew_leaf/not_cashew_leaf_42.jpg` khác byte nhưng có pHash bằng nhau và thumbnail grayscale 64×64 giống hệt (MAD=0); cần mở ảnh gốc để xác nhận trước khi công bố Test là độc lập. 39 cặp pHash khác split trong audit của notebook chỉ là ứng viên, không phải 39 lỗi đã xác nhận.
- Việc phân nhóm chỉ dựa trên tiền tố tên `real_` chưa chứng minh ảnh đó thực sự được chụp ngoài hiện trường hoặc độc lập theo lá/cây/buổi chụp. Test `real_` chỉ có 28 ảnh, không có lớp `not_cashew_leaf`; kết quả nhóm có độ bất định lớn.
- Một seed với nhiều bước cập nhật hơn có thể giải thích một phần chênh lệch với thí nghiệm 30 epoch; chưa thể quy toàn bộ chênh lệch cho dataset. Cần so sánh cùng số bước, cùng split sạch, nhiều seed và holdout ảnh hiện trường độc lập.

## Trạng thái

Đã xác nhận pipeline và checkpoint chạy được; **chưa huấn luyện lại đủ 49 epoch trên GPU trong phiên này** vì không có runtime GPU nào được gắn vào môi trường thực thi của trợ lý. GPU G4 mà người dùng đã kết nối cần được chia sẻ qua notebook/runtime có thể truy cập trong phiên để chạy lại đầy đủ.
