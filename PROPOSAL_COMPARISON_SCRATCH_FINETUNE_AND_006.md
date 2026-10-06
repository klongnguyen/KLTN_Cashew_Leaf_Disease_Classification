# Đề xuất so sánh scratch / fine-tune và kế hoạch đánh giá 006

Ngày: 2026-10-05. Phần thiết kế thí nghiệm dưới đây **chỉ là đề xuất, chưa áp dụng thay đổi chế độ huấn luyện**.

## 1. Vấn đề thực sự cần tách

DenseNet121 và ResNet50 trong notebook dùng `weights=None`: học từ khởi tạo ngẫu nhiên. Notebook `VIT_5seed.ipynb` hiện dùng DeiT-tiny pretrained ImageNet rồi fine-tune. Nếu đặt các kết quả đó vào một bảng và kết luận “Transformer tốt hơn CNN”, ta không tách được ảnh hưởng của kiến trúc khỏi lợi thế pretraining, cấu hình huấn luyện và kích thước mô hình.

Tuy nhiên, so sánh **các cấu hình hoàn chỉnh phục vụ ứng dụng** vẫn có ý nghĩa, miễn ghi rõ training mode và không diễn giải thành lợi thế thuần túy của kiến trúc. “From scratch” ở đây là không dùng weights pretrained, không có nghĩa phải tự viết lại kiến trúc thay vì dùng thư viện.

Lưu ý nguồn kết quả: bảng trong `ANALYSIS_CLASSIFICATION_V05_LATEST_2026-10-05.md` đang so DenseNet-005, ResNet-005 và **Compact ViT scratch-004**. Nó chưa chứa kết quả DeiT fine-tune. Không thay dòng Compact ViT bằng số liệu DeiT mà giữ nguyên nhãn “scratch”.

Compact ViT scratch cũ và DeiT-tiny còn khác kiến trúc/số tham số; hiệu số kết quả giữa chúng **không phải ablation riêng của pretrained weights**. Preset DeiT hiện có tên `distilled`; nguồn gốc distillation cũng cần khai báo. Notebook Cashew hiện không thực hiện teacher-student distillation trong lần fine-tune. Bài [DeiT gốc](https://arxiv.org/abs/2012.12877) mô tả pretraining trên ImageNet và chiến lược distillation token.

## 2. Phương án cho báo cáo

### A. Hai nhóm kết quả riêng — khuyến nghị với nguồn lực hiện tại

- **Nhóm scratch:** DenseNet121, ResNet50, Compact ViT scratch; so sánh khả năng học trên cùng V05 khi không dùng pretrained weights.
- **Nhóm transfer learning / ứng dụng:** DeiT-tiny ImageNet fine-tune; trình bày như một cấu hình bổ sung hướng tới hiệu quả triển khai. Có thể đặt cạnh baseline để mô tả thực nghiệm, nhưng không tuyên bố đó là phép so sánh kiến trúc đã kiểm soát pretraining.
- Bảng tổng quan phải có cột kiến trúc/preset, scratch hay pretrained, nguồn weights, số tham số, batch, epoch thực chạy, thời gian huấn luyện và môi trường đo.

Không cần bỏ các kết quả scratch đã có. Đây là phương án ít tốn thêm compute và phù hợp nếu mục tiêu là xây dựng, đánh giá hệ thống nhận diện bệnh lá điều.

Câu diễn đạt có thể dùng:

> Nghiên cứu đánh giá các cấu hình học từ đầu trên cùng dữ liệu Cashew V05, đồng thời khảo sát thêm cấu hình DeiT-tiny sử dụng transfer learning. Do khác biệt về nguồn khởi tạo và công thức huấn luyện, chênh lệch giữa hai nhóm được diễn giải ở mức cấu hình hệ thống, không quy hoàn toàn cho kiến trúc CNN hay Transformer.

### B. Ma trận kiến trúc × khởi tạo — nếu cần kết luận về tác động pretraining

| Kiến trúc cố định | Khởi tạo ngẫu nhiên | Pretrained rồi fine-tune |
|---|---|---|
| DenseNet121 | Scratch | ImageNet fine-tune |
| ResNet50 | Scratch | ImageNet fine-tune |
| Cùng một kiến trúc Transformer cụ thể | Scratch | Pretrained fine-tune |

Mỗi cặp phải giữ cùng backbone và classifier head phù hợp; không ghép Compact ViT 0,348M với DeiT-tiny thành một cặp “chỉ khác pretrained”. Ghi rõ khác biệt do distillation hoặc nguồn weights, nếu có.

Chi phí cao hơn vì cần thêm cấu hình. Cùng seed và split là cần thiết nhưng chưa đủ: cần ngân sách tuning tương đương trên Validation, quy tắc chọn checkpoint nhất quán, tiền xử lý đúng cho từng backbone và công khai augmentation/optimizer. Không bắt mọi kiến trúc dùng cùng LR chỉ để tạo cảm giác công bằng.

### C. Toàn bộ scratch cho phép so sánh chính

Giữ DenseNet121, ResNet50 và Compact ViT scratch; đưa DeiT fine-tune vào phần mở rộng/triển khai. Cách này loại bỏ lợi thế pretrained khỏi bảng chính nhưng vẫn không kiểm soát tuyệt đối số tham số và recipe. Nếu muốn chính DeiT-tiny học từ đầu, đó là một thí nghiệm mới cần thiết kế riêng, không chỉ sửa tên experiment.

**Đề xuất chọn A trước; chỉ triển khai B nếu câu hỏi nghiên cứu đòi hỏi ablation pretraining và còn ngân sách GPU.** Chưa đổi DenseNet/ResNet sang pretrained hoặc DeiT sang scratch trong phiên bản 006.

## 3. Protocol đánh giá cuối

1. Huấn luyện và chọn checkpoint bằng Train/Validation V05; checkpoint mỗi seed theo `min(val_loss)` như notebook hiện tại.
2. Chốt danh sách cấu hình, hyperparameter và năm checkpoint trước khi mở Test. Bản tổng hợp holdout chọn deployment seed bằng Validation Macro-F1, tie-break bằng Val loss rồi seed, trước khi dự đoán ảnh holdout.
3. Giữ **Test V05: 687 ảnh, 5 lớp** để báo cáo kết quả cùng phân phối dữ liệu gốc.
4. Đánh giá **RL_Cashew_holdout riêng** để khảo sát ảnh thực tế. Không gộp hai tập thành một Accuracy/Macro-F1 và không dùng holdout để chọn seed, kiến trúc, crop, threshold hay LR. Tránh dùng Test trong lựa chọn mô hình theo [hướng dẫn tránh data leakage của scikit-learn](https://scikit-learn.org/stable/common_pitfalls.html).
5. Báo cáo từng seed và mean ± SD; SD qua seed phản ánh biến thiên do huấn luyện, không phải khoảng tin cậy trên các ảnh/địa điểm thực tế độc lập.
6. Thời gian fit phải ghi phạm vi đo và phần cứng. Thời gian fine-tune Cashew **không bao gồm compute pretraining ImageNet**, nên không gọi nó là toàn bộ chi phí tạo ra mô hình pretrained.

## 4. Audit bộ ảnh thực tế và điểm cần quyết định

Nguồn: [RL_Cashew_holdout.zip trên Drive](https://drive.google.com/file/d/1PBry_y2bMsvd5_YdmEUy9erSFarS3O52/view). Bản download xác thực từ Drive và bản local có cùng SHA-256:

`BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165`

| Lớp | Ảnh trong ZIP |
|---|---:|
| anthracnose | 13 |
| healthy | 5 |
| leaf_miner | 24 |
| not_cashew_leaf | 0 |
| red_rust | 7 |
| Tổng | 49 |

Đã kiểm tra decode ảnh, duplicate byte trong holdout và đối chiếu toàn bộ 6.911 ảnh V05. Không có ảnh decode lỗi hoặc duplicate byte nội bộ. **Có một ảnh trùng với Test V05:**

- Holdout: `RL_Cashew_holdout/red_rust/IMG_3244.jpg`.
- V05: `CashewData_Split/test/red_rust/real_red_rust_029.jpg`.

Ảnh này không nằm trong Train/Validation, nhưng làm hai tập Test không độc lập hoàn toàn. Đề xuất giữ nguyên ZIP và loại ảnh này khỏi phép đánh giá holdout, còn **48 ảnh, red_rust còn 6**. **Chưa áp dụng loại ảnh; cần người dùng chốt.** Hiện notebook sẽ dừng preflight nếu bật Final Test với nguyên bộ đang trùng. Không tự xóa ảnh hoặc sửa Drive.

Audit byte không phát hiện được ảnh gần giống, nhiều ảnh cùng lá/cây hay cùng buổi chụp. Cần kiểm tra nhóm nguồn ảnh trước khi tuyên bố khả năng tổng quát hóa ngoài thực địa. Tập nhỏ, mất cân bằng, chỉ có 5 ảnh healthy và thiếu not_cashew_leaf; không đủ để kết luận hiệu năng nhận biết ảnh không phải lá điều. Nếu bổ sung bộ holdout mới, phải cập nhật SHA-256/counts trước khi Test, không chỉnh tập sau khi xem kết quả.

Đầu ra mô hình vẫn có 5 lớp. Macro Precision/Recall/F1 của holdout được ghi rõ `*_present_classes` trên 4 lớp có ground truth. Dự đoán nhầm sang not_cashew_leaf vẫn làm giảm Accuracy/Recall/F1. Báo cáo per-class ghi `not_cashew_leaf` là N/A, không diễn giải zero support thành hiệu năng bằng 0.

Audit tái lập được bằng `Notebook/common/audit_holdout_archive.py`; kết quả ở `Notebook/common/holdout_archive_audit_006.json`. Việc audit này không chạy huấn luyện hay dự đoán Test của mô hình.

## 5. Notebook 006 đã chuẩn bị gì

- ID: `EXP-DENSENET121-SCRATCH-5SEEDS-006`, `EXP-RESNET50-SCRATCH-5SEEDS-006`, `EXP-VIT-SCRATCH-5SEEDS-006`, `EXP-DEIT-TINY-FINETUNE-5SEEDS-006`.
- Giữ nguyên V05, mode huấn luyện, preset/weights, hyperparameter và 5 seed; backup scratch cũ không sửa.
- `RUN_FINAL_TEST=False` mặc định. `RUN_REAL_HOLDOUT=True` chỉ có hiệu lực khi mở Final Test; bật nó từ đầu sẽ preflight archive/overlap trước khi tốn compute huấn luyện.
- Shared evaluator đã nhúng trong notebook để upload riêng một notebook lên Colab vẫn chạy được. Python source DeiT được đồng bộ.
- Trên Colab đọc `MyDrive/Cashew_Leaf_model_result/Dataset/RL_Cashew_holdout.zip`; có `REAL_HOLDOUT_ARCHIVE_OVERRIDE` nếu vị trí mount khác.
- Holdout có manifest, audit, lock checkpoint SHA-256 + lựa chọn seed từ Validation, predictions, metrics, classification report, ma trận count/normalized, kết quả 5 seed và mean ± SD trong `real_holdout/`.
- Ma trận dùng trắng–xanh nhạt, toàn bộ chữ đen. Dòng normalized của lớp không có ground truth ghi N/A.
- Các output notebook cũ được giữ nguyên nhưng có cảnh báo không phải kết quả 006. Restart runtime và clear outputs trước lượt chạy mới. Chưa train hoặc Test mô hình thật ở phiên bản 006.
- YOLO detection `EXP-Y26S-SMALL-007` giữ nguyên, vì không phải classification 5 lớp và không đổi lùi ID xuống 006.

Trước khi bật Final Test, cần chốt phương án báo cáo và xử lý ảnh trùng. Nếu đã chạy Test một cấu hình, không tiếp tục tối ưu nó theo các số Test/holdout rồi báo cáo như phép đánh giá độc lập mới.
