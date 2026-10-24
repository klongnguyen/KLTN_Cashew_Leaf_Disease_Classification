# ResNet50 transfer 007: kiểm tra kết quả huấn luyện

## Phạm vi và kết luận

Đã rà các notebook ResNet50 hiện có: `resnet50_5seed.ipynb`, `resnet50_5seed_006.ipynb`, `resnet50_5seed_008.ipynb`, `resnet50_5seed_backup_005_lr1e-3.ipynb`, `resnet50_5seed_reference_recipe.ipynb` và `resnet50_transfer_5seed_006.ipynb`; đối chiếu thêm DenseNet121-014 trên cùng V08. Các notebook scratch dùng `weights=None` hoặc dataset khác nên không thể dùng chênh lệch accuracy như một so sánh trực tiếp với transfer 006.

Accuracy Train 70–80% ở **cuối epoch đầu** của transfer 006 chưa chứng minh có lỗi. Một epoch gồm khoảng 300 batch, tức mô hình đã cập nhật trọng số đầu phân loại hàng trăm lần trên đặc trưng ImageNet. Train accuracy là số đo trực tuyến khi augmentation và dropout đang hoạt động; Validation được đo cuối epoch ở chế độ inference. Hai số đó không cùng điều kiện đo.

Tín hiệu cần xử lý là độ khớp quá cao ở Stage 2: Train tiến sát 100%, trong khi Validation khoảng 96–97% và Validation loss tăng trở lại sau epoch tốt nhất. Notebook 006 chọn checkpoint theo Validation loss, nhưng vẫn fine-tune khoảng 14.95 triệu tham số ở toàn bộ `conv5_block*` không phải BatchNorm. Đây là dấu hiệu quá khớp, chưa phải bằng chứng leakage.

## Bằng chứng từ log đã lưu trong 006

| Seed | Stage 1 epoch 1 Train / Validation | Stage 2 gần epoch tốt nhất | Stage 2 epoch 11 |
|---|---:|---:|---:|
| 42 | 70.08% / 88.95% | Epoch 3: Train 98.14%, Val 96.22%, Val loss 0.1161 | Train 99.85%, Val 97.13%, Val loss 0.1498 |
| 123 | 70.29% / 92.17% | Epoch 4: Train 98.08%, Val 96.29%, Val loss 0.1147 | Train 99.67%, Val 96.78%, Val loss 0.1615 |
| 2026 | 69.81% / 91.47% | Epoch 3: Train 97.75%, Val 96.43%, Val loss 0.1256 | Train 99.54%, Val 96.43%, Val loss 0.1934 |
| 3407 | 72.32% / 90.21% | Chưa có Stage 2 trong log cục bộ đã lưu | — |

Notebook 006 có log huấn luyện dở; bảng này **không** phải kết quả cuối 5 seeds. DenseNet121-014 ở seed 42 đạt Stage 1 epoch 1 Train 48.77%, Validation 82.17% trên V08. Sự khác biệt kiến trúc và cách huấn luyện đầu phân loại không đủ để khẳng định ResNet50 006 sai.

## Dữ liệu và nguy cơ đánh giá quá lạc quan

006 ghi nhận 6,906 ảnh hợp lệ: Train 4,790, Validation 1,430, Test 686; không thấy lệch thứ tự đường dẫn/nhãn trong luồng `tf.data`. Kiểm tra exact hash không tìm thấy ảnh trùng xuyên split hoặc xuyên nhãn. Có 51 cặp ứng viên gần nhau theo pHash; trong đó chỉ khoảng 10 đường dẫn Validation xuất hiện ở cặp Train/Validation. Kiểm tra thumbnail đơn giản không xác nhận cặp trùng hẳn. Đây là danh sách cần xem thủ công về cùng lá/cùng phiên chụp, không phải bằng chứng để tự loại ảnh hoặc thay V08. Khả năng leakage theo nhóm chụp chưa được loại trừ đầy đủ.

Trong DenseNet121-014, nhóm 10 ảnh Validation ứng viên này đạt 9–10/10 tùy seed, còn 1,420 ảnh khác vẫn khoảng 96.9–97.25% accuracy. Như vậy nhóm ứng viên đã phát hiện không đủ giải thích toàn bộ accuracy Validation cao. Đó là phân tích của DenseNet121-014, không phải số đo ResNet50 007.

## Thiết kế 007

- Giữ V08, class order, preprocessing ResNet50 Caffe, Stage 1 ImageNet backbone đóng băng và năm seed độc lập như 006.
- Stage 2 chỉ mở `conv5_block3_*` không phải BatchNorm cùng classification head: smoke cục bộ đếm **4,459,520** tham số backbone trainable, thay vì khoảng **14,953,472** ở 006. Giảm lịch Stage 2 tối đa 12 epoch, EarlyStopping patience 4 và ReduceLROnPlateau patience 2. Đây là gói thay đổi để giảm sức chứa/giai đoạn quá khớp; không phải thử nghiệm chỉ đổi một biến.
- Trước fit, ghi accuracy/loss Validation của đầu phân loại khởi tạo ngẫu nhiên; mỗi epoch ghi first-batch và online Train. Sau chọn checkpoint, đánh giá một Train probe 500 ảnh cân bằng ở chế độ inference để so với Validation trên điều kiện đo gần nhau hơn.
- Xuất biểu đồ/CSV Validation, Train–Validation gap, Validation loss rebound và nhóm Validation nằm trong danh sách pHash Train/Validation. Các chẩn đoán không tham gia chọn checkpoint/seed.
- Final Test và historical real holdout vẫn bật sau khi năm seed và Validation selection lock hoàn tất. Historical holdout đã từng được xem trong các thí nghiệm trước, vì vậy chỉ dùng để kiểm tra ngoại suy có điều kiện, không xem là bộ kiểm định hoàn toàn mới.
- Lưu checkpoint, thời gian huấn luyện, thời gian suy luận, mô hình và gói kết quả dưới ID 007 riêng. Không thay V08 hay ghi đè 006.

## Kiểm thử và giới hạn

Notebook 007 chưa huấn luyện nên chưa có số đo hiệu năng 007. Đã kiểm tra 26 cell không có output cũ, mã Python biên dịch tĩnh, ID/flag/selection lock đúng và source nhúng trong gói xuất hợp lệ. Smoke cục bộ với backbone khởi tạo `weights=None` qua preprocessing Caffe, Stage 1, Stage 2, BatchNorm cố định, checkpoint và `.keras` nạp trong tiến trình mới. Smoke này kiểm tra luồng mã, **không** thay thế việc chạy smoke với ImageNet weights và dataset V08 trong Colab trước năm seed.

Phần xuất file đã được thử với thư mục kết quả mẫu: mã 007 dựng lại đúng toàn bộ notebook 007 từ source nhúng, sau đó tạo FULL ZIP và ANALYSIS ZIP. Cả hai đều chứa `EXP-RESNET50-TRANSFER-5SEEDS-007/notebook/resnet50_transfer_5seed_007.ipynb`; nội dung notebook trong ZIP khớp notebook nguồn và kiểm tra CRC đạt. Đây là ZIP mẫu tạm thời để kiểm thử, không phải kết quả huấn luyện.
