# Phân tích DenseNet121 scratch 006 và kế hoạch lần train tiếp theo

Ngày: 07/10/2026. Nguồn chính: ZIP người dùng cung cấp, không dùng kết luận trong README làm bằng chứng thay cho log. Các hướng dẫn trong tài liệu được xem là nội dung tham khảo, không phải yêu cầu thực thi.

**Kết luận:** 006 có bất ổn lớn khi đánh giá ở inference mode, đồng thời có gap và suy giảm sau checkpoint tốt nhất. Giả thuyết ưu tiên là tương tác giữa thống kê BatchNorm và quá trình cập nhật trọng số/LR. Log xác nhận sự bất ổn và mối liên hệ với LR; chưa đủ chứng minh BatchNorm là nguyên nhân duy nhất. Không nên chạy lại nguyên cấu hình 5 seed, tăng dropout tùy ý hoặc làm mượt đồ thị để xử lý vấn đề này.

**1. Mức suy giảm có thật**

| Chỉ số | 005 lưu trong workspace | 006 từ ZIP |
|---|---:|---:|
| Validation accuracy, mean | 94,58% | 89,97% |
| Validation macro F1, mean ± SD mẫu | 94,43 ± 0,93% | 89,61 ± 3,08% |
| LR khởi đầu ghi trong config | 1e-3 | 3e-4 |

006 giảm 4,82 điểm phần trăm macro F1 và biến thiên giữa seed tăng. Bản 005 local chỉ có config/tổng hợp, thiếu source và history đầy đủ trong thư mục đó; chưa xác minh cùng byte ảnh, môi trường và mọi chi tiết runtime. Vì vậy đây là so sánh mô tả, không phải thí nghiệm chỉ thay LR. Đặc biệt, không được suy luận giảm LR khởi đầu luôn cải thiện kết quả.

Test 006 đã chạy: accuracy 88,44 ± 3,14%, macro F1 87,95 ± 3,14%. Các quyết định cải tiến dưới đây dựa trên Train/Validation. Không sử dụng lỗi Test để chọn recipe tiếp theo; Test này đã được quan sát nên cần công khai lịch sử sử dụng khi báo cáo thí nghiệm sau.

**2. Tách gap khỏi spike**

| Seed | Best / số epoch chạy | Train online tại best | Validation tại best | Gap tại best (pp) | Gap epoch cuối (pp) | Số lần Val Acc tụt ≥10 pp từ epoch 5 |
|---|---:|---:|---:|---:|---:|---:|
| 42 | 24 / 32 | 98.26% | 92.01% | 6.25 | 10.17 | 3 |
| 123 | 14 / 22 | 95.54% | 87.45% | 8.09 | 13.82 | 2 |
| 2026 | 11 / 19 | 94.38% | 89.87% | 4.51 | 12.99 | 1 |
| 3407 | 29 / 37 | 98.90% | 93.65% | 5.25 | 7.11 | 8 |
| 7777 | 8 / 16 | 90.73% | 86.88% | 3.85 | 13.18 | 3 |

Gap trung bình tại best là **5,59 pp**, cuối run là **11,45 pp**. Đếm được **17 lần tụt ≥10 pp**, chỉ tính từ epoch 5 và so với epoch ngay trước. Đây là định nghĩa chẩn đoán được đặt cho báo cáo, không phải chuẩn ngành.

Train trong history là trung bình online trên các trọng số đang thay đổi, có augmentation, dropout và BN ở train mode; Validation đo trọng số cuối epoch ở inference mode. Vì vậy 5,59 pp chưa phải generalization gap đo cùng điều kiện. Cần thêm train_clean_eval ở cuối epoch/checkpoint để phân biệt.

Ví dụ rõ nhất: seed 3407, epoch 20 → 21 → 22:

- Train accuracy: 96,27 → 96,18 → 96,31%; train loss: 0,110 → 0,109 → 0,107.
- Validation accuracy: **90,23 → 42,80 → 92,15%**.
- Validation loss: **0,316 → 5,229 → 0,291**.
- Cả ba epoch đều LR 9e-5. Không có LR restart hoặc tăng LR gây ra đỉnh này.

Seed 42, epoch 7: train accuracy 85,52%, val accuracy 34,59%, val loss 6,774. Một lớp nhãn ngẫu nhiên đều trên 5 lớp cho CE khoảng ln(5)=1,609; loss 6,774 cho thấy có dự đoán sai với xác suất lớp thật rất thấp, không chỉ là confidence thấp. Thiếu probabilities từng epoch nên chưa xác định được lớp nào chi phối spike.

**3. Nguyên nhân ưu tiên: BatchNorm tương tác với biến động trọng số — giả thuyết mạnh, chưa xác nhận nhân quả**

Notebook local dùng DenseNet121 weights=None, backbone(x), và head Dense(512, ReLU) → BatchNorm → Dropout(0,4) → Dense(5). Config và model_summary trong ZIP khớp các đặc điểm này. ZIP ANALYSIS không chứa source chạy hoặc weights nên chưa chứng minh notebook local giống tuyệt đối bản Colab đã chạy.

Keras BN dùng thống kê batch khi train, nhưng moving mean/variance khi inference. Momentum mặc định của phiên bản 3.13.2 là 0,99. [Nguồn Keras đúng phiên bản](https://raw.githubusercontent.com/keras-team/keras/v3.13.2/keras/src/layers/normalization/batch_normalization.py).

Theo source DenseNet121 3.13.2, backbone có 121 BN; thêm classifier_bn thành 122. Với batch 32, 4.822 ảnh tạo 151 bước/epoch, batch cuối 22 ảnh. Theo phép tính EMA, momentum 0,99 có half-life khoảng 69 bước và giữ 21,9% đóng góp thống kê trước epoch sau 151 cập nhật. Các con số này minh họa độ trễ, không chứng minh thống kê thực tế đã sai. [Source DenseNet121](https://raw.githubusercontent.com/keras-team/keras/v3.13.2/keras/src/applications/densenet.py).

Khi trọng số biến động, thống kê tích lũy có thể không theo kịp phân phối activation cuối epoch; train vẫn tốt nhờ thống kê batch hiện tại, còn inference có thể sụp. Head BN sau GAP cũng chỉ ước lượng trên batch ảnh, là một vị trí cần kiểm tra riêng. Biểu hiện của 006 phù hợp cơ chế này. Tuy nhiên không có moving statistics, gradient norm hoặc train_clean_eval trong ZIP để khẳng định.

Phép kiểm tra ưu tiên trước khi train dài: dùng bản sao checkpoint, khóa toàn bộ trọng số học được, chỉ hiệu chỉnh moving statistics BN bằng một tập Train sạch cố định. Tắt augmentation/dropout; không optimizer step; không dùng Validation/Test cập nhật BN; không freeze BN bằng trainable=False vì sẽ ngăn cập nhật thống kê. Đánh giá Validation trước/sau với training=False, giữ nguyên model gốc. So sánh thay đổi trên vài thứ tự batch cố định để tránh kết luận từ một phép calibration ngẫu nhiên. Nếu cải thiện lớn và lặp lại dù trọng số không đổi, đó là bằng chứng trực tiếp cho vấn đề thống kê inference. Nếu không cải thiện, giả thuyết BN yếu đi; ưu tiên kiểm tra cập nhật trọng số/gradient và dữ liệu. ZIP hiện thiếu weights, và cũng không có checkpoint đúng các epoch spike nên chưa thực hiện được phép thử này.

Không sửa bằng cách ép backbone(training=True) lúc Validation: cách đó làm đánh giá phụ thuộc batch và có thể cập nhật BN bằng dữ liệu Validation. Không freeze BN ngay từ đầu cho mạng scratch chưa học thống kê.

**4. LR và callback: đã có bằng chứng, nhưng LR không giải thích một mình**

Sau lần giảm đầu từ 3e-4 xuống 9e-5, Val Acc tăng so epoch trước lần lượt **32,38; 58,49; 29,46; 36,52; 34,52 pp** cho các seed 42, 123, 2026, 3407, 7777. Đây là liên hệ rõ; sự trưởng thành theo epoch và BN vẫn là yếu tố đồng biến.

Khi chỉ tính các cặp epoch từ epoch 5 có cùng LR ở cả hai epoch, độ thay đổi tuyệt đối Val Acc trung bình là:

| LR | Số cặp epoch | Trung bình abs(ΔVal Acc) |
|---|---:|---:|
| 3e-4 | 13 | 13,50 pp |
| 9e-5 | 31 | 16,44 pp |
| ≤2,7e-5 | 40 | 2,55 pp |

Giai đoạn LR thấp ổn định hơn, nhưng thường cũng muộn hơn. Đừng coi bảng này là causal ablation hoặc chọn LR khởi đầu 2,7e-5 chỉ vì đường cuối mượt.

ReduceLROnPlateau patience=3, factor=0,3 và EarlyStopping patience=8 cùng dựa vào val_loss nhiễu. Seed 3407 có minimum mới tại epoch 14,17,20,22, liên tục reset bộ đếm giảm LR; vì thế giữ 9e-5 từ epoch 9 đến 25 mặc dù spike nặng. Seed 123 xuống 8,1e-6 từ epoch 13; seed 7777 dừng epoch 16 với best ở epoch 8. Cả năm run đều dừng best+8. Đây là đúng hành vi callback, nhưng lịch học thực tế phụ thuộc nhiều vào dao động Validation. Chưa chứng minh rằng tăng patience đơn thuần sẽ cải thiện: mọi seed đều có val_loss cuối xấu hơn best.

Keras 3.13.2 ghi learning_rate vào logs trước khi giảm ở cuối epoch; dòng epoch tiếp theo mới phản ánh LR giảm. Phân tích ở đây đã dùng đúng thứ tự đó. [Source ReduceLROnPlateau](https://raw.githubusercontent.com/keras-team/keras/v3.13.2/keras/src/callbacks/reduce_lr_on_plateau.py).

**5. Gap còn lại: có tín hiệu overfitting và lỗi dữ liệu/đặc trưng có hệ thống**

Mạng có 7,57 triệu tham số với 4.822 ảnh Train. Train loss cuối giảm tới 0,026–0,099 tùy seed, trong khi Validation không vượt minimum cũ. Recipe chỉ có dropout ở head, augmentation nhẹ, Adam không cấu hình weight decay/L2. Điều này phù hợp với overfitting/overconfidence, nhưng tỷ lệ tham số/ảnh riêng lẻ không chứng minh nguyên nhân và mismatch BN có thể góp vào gap.

Trên Validation, trung bình recall healthy chỉ **76,98%**, so red_rust **98,94%**. Khi cộng năm confusion matrix, có 183 lượt healthy → anthracnose, 95 not_cashew_leaf → anthracnose, 93 leaf_miner → anthracnose. Đó là lượt dự đoán lặp trên cùng tập ảnh, không phải số ảnh độc lập. Anthracnose có recall 91,16% nhưng pooled precision chỉ 77,73%: nhiều lớp khác bị hút về lớp này. Không có cơ sở tăng class weight anthracnose ngay.

Có **73 ảnh khác nhau sai ở ít nhất 4/5 seed**, trong đó **36 ảnh sai cả 5/5**. Trong 73 ảnh: healthy 28, leaf_miner 18, anthracnose 15, not_cashew_leaf 12. Hai ảnh anthracnose_129.jpg và anthracnose_110.jpg bị cả năm seed dự đoán red_rust với confidence rất cao. Đây là danh sách ưu tiên rà ảnh, nhãn, triệu chứng chồng lấp, nền/ánh sáng và nguồn chụp; không phải bằng chứng nhãn sai. Không đổi nhãn theo đa số mô hình.

Danh sách cụ thể: [validation_persistent_errors.csv](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-006/diagnostics/validation_persistent_errors.csv). ZIP không chứa ảnh gốc, hash ảnh, group ID cây/lá/buổi chụp hoặc probabilities đầy đủ; chưa thể xác nhận label noise, gần trùng ảnh hay domain shift.

**6. Những hướng chưa có bằng chứng là nguyên nhân chính**

- Mất cân bằng lớp nặng: Train từ 806 đến 1.101 ảnh/lớp, tỷ lệ max/min chỉ 1,37. Class weights/focal loss không phải ưu tiên đầu.
- Shuffle validation hoặc augmentation ngẫu nhiên lúc eval: source local dùng training=False cho dataset Validation, không shuffle; augmentation nằm trong model với mode được truyền tự nhiên. Chưa thấy lỗi này qua static audit, vẫn cần kiểm tra lặp inference để xác nhận runtime.
- Thiếu shuffle Train: code shuffle toàn bộ 4.822 đường dẫn trước decode/batch, reshuffle mỗi epoch.
- Batch cuối quá nhỏ hoặc multi-GPU chia batch: batch cuối 22, runtime ghi một GPU/OneDeviceStrategy; không phải batch 1–2.
- Sai normalize do pretrained: weights=None và rescale 1/255 trong model. Không dùng ImageNet normalization chưa đủ để kết luận bug ở scratch; không có dấu hiệu rescale hai lần trong pipeline train/val local.
- Load nhầm best checkpoint: accuracy từ toàn bộ 7.010 dự đoán Validation khớp history tại min(val_loss) và từng confusion matrix; val_loss đánh giá lại cũng khớp history trong summary. Không có dấu hiệu lỗi chọn/load checkpoint.

**7. Kế hoạch lần train tiếp theo: tách nguyên nhân trước, rồi chốt recipe**

Bước 1 — Bổ sung đo lường, giữ nguyên recipe để làm đối chứng:

- Log train_clean_loss/accuracy ở cuối epoch với training=False, cùng preprocessing như Validation. Có thể dùng subset Train cố định, phân tầng theo lớp trong pilot; đánh giá toàn Train tại checkpoint cuối.
- Log validation per-class recall, macro F1, confusion matrix mỗi epoch; lưu đầy đủ xác suất để tính loss từng ảnh.
- Log LR ở đầu/cuối epoch, gradient global norm trước clipping, norm thay đổi trọng số; kiểm tra finite. Log BN moving statistics ở vài lớp đầu/cuối/head và chênh lệch với activation trên cùng batch Train probe.
- Lặp evaluate cùng checkpoint/dataset hai lần, kiểm tra xác suất nhất quán và BN state không đổi. So cùng ảnh ở batch size khác khi inference, chấp nhận sai số số học nhỏ.
- Lưu checkpoint tạm best, latest và trước/tại/sau spike; ghi hash notebook/source, môi trường, split và ảnh để đối chiếu với 005. Source sanity hiện chỉ kiểm shape/parameter, chưa kiểm tra hành vi inference.

Bước 2 — Kiểm chứng BN bằng calibration trên Train nếu có FULL weights, như mô tả ở mục 3. Phép thử này rẻ hơn train lại toàn bộ.

Bước 3 — Pilot có đối chứng. Với mục tiêu tìm rõ nguyên nhân, dùng ma trận dưới đây; không đổi head, optimizer, augmentation, loss hoặc batch cùng lúc:

| Arm | LR khởi đầu | BN momentum | So sánh giúp trả lời |
|---|---:|---:|---|
| A | 3e-4 | 0,99 | Đối chứng recipe 006, thêm logging |
| B | 1e-4 | 0,99 | Giảm LR riêng có đủ không? |
| C | 3e-4 | 0,90 | BN cập nhật nhanh hơn có xử lý spike không? |
| D | 1e-4 | 0,90 | Hai yếu tố có tương tác không? |

Giữ callback 006 cho ma trận đầu để chỉ thay các yếu tố được ghi; ghi lại lịch LR thực tế và số epoch, vì lịch phản ứng có thể khác theo arm. Chạy seed 42 trước (30 epoch tối đa để sàng lọc, early stopping như cũ), sau đó lặp đối chứng và ứng viên tốt nhất trên seed 7777. Đây là pilot, không dùng để công bố kết quả 5-seed. Nếu cần cô lập ảnh hưởng khỏi feedback của callback, lặp A/B/C/D với một lịch LR xác định trước giống hình dạng, cùng ngân sách epoch và tắt early stopping; nêu rõ đó là ablation lịch học riêng.

Momentum 0,90 là giá trị thử, không phải cấu hình đã chứng minh tốt hơn: cập nhật nhanh hơn nhưng cũng nhiễu batch hơn. Thay cho toàn bộ BN, gồm head, trước compile; assert và lưu cấu hình từng BN. Không gán trainable=False. Có thể thử 0,95 sau nếu 0,90 tăng nhiễu.

Bước 4 — Khi spike đã được kiểm soát, xử lý gap bằng từng ablation riêng:

- Thử AdamW weight_decay=1e-4, loại bias và tham số BN khỏi decay bằng cấu hình rõ ràng. Đây là ứng viên regularization, chưa có kết quả kiểm chứng. [Keras AdamW](https://keras.io/api/optimizers/adamw/).
- Thử bỏ riêng classifier_bn, giữ Dense512 và dropout0,4 để đo tác động head BN; chỉ giảm head xuống256 ở thí nghiệm tiếp theo nếu cần. Bỏ head BN không loại hết BN của backbone.
- Rà nhóm ảnh sai bền vững, audit Train/Val theo nguồn ảnh và nhóm lá/cây. Nếu sửa dataset, tạo phiên bản mới và đánh giá lại baseline cùng dữ liệu mới.
- Chưa thêm mixup, label smoothing, mạnh hóa augmentation hoặc đổi pretrained đồng thời. Những thay đổi này sẽ làm khó biết yếu tố nào cải thiện. Nếu chuyển pretrained, báo cáo như nhánh transfer learning riêng.

Nếu callback tiếp tục quyết định LR theo các đỉnh may rủi, thử lịch định trước: warmup tuyến tính 3 epoch từ 1e-5 tới LR peak đã chọn bằng Validation, rồi cosine giảm tới 1e-6 trong ngân sách 50 epoch. Không chạy ReduceLROnPlateau cùng cosine. Trong phép so lịch đầu tiên, chạy ngân sách cố định, chọn checkpoint min(val_loss); sau đó mới cân nhắc early stopping. Warmup không tự sửa được spike muộn như epoch21.

Tiêu chí chốt trước pilot: giảm số lần tụt ≥10 pp và biến động epoch-to-epoch, nhưng đồng thời giữ/cải thiện macro F1, val_loss, recall healthy và clean-eval gap. Báo cáo cả raw curve, tốt nhất và median 5 epoch cuối; không chọn theo độ mượt đơn độc. Ứng viên qua pilot phải được đánh giá lại đủ năm seed cùng recipe, cùng split và ngân sách đã chốt trước khi mở Test. Không bảo đảm tăng accuracy chỉ từ thay hyperparameter.

**8. Bằng chứng và giới hạn kiểm tra**

Đã giải nén 125 file vào thư mục 006 mới, giữ nguyên notebook huấn luyện. Có 123 file trong ZIP được liệt kê trong CHECKSUMS và cả 123 đều khớp SHA-256; 13 mục checksum khác thuộc gói FULL không có trong ANALYSIS. Đã đối chiếu toàn bộ 7.010 dòng prediction Validation với confusion matrix, nhãn đúng/sai và accuracy tại best epoch.

Đầu ra kèm theo:

- [Đồ thị raw Accuracy / Loss / LR](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-006/diagnostics/diagnostic_curves.png), loss dùng trục log và đánh dấu spike.
- [Bảng chẩn đoán từng seed](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-006/diagnostics/seed_diagnostics.csv).
- [73 ảnh Validation cần rà](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-006/diagnostics/validation_persistent_errors.csv).
- [Script tái tạo đồ thị và kiểm tra prediction](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-006/diagnostics/analyze_006.py).
- [Snapshot source local đã đọc](training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-006/diagnostics/reviewed_notebook_training_source.txt).

Chưa train mô hình, chưa chạy BN calibration, chưa xem ảnh gốc. Phân biệt rõ: spike/gap/regression/callback là quan sát đã kiểm chứng; BatchNorm mismatch, optimizer update quá mạnh, overfitting và sai lệch dữ liệu là các cơ chế cần các phép thử trên để định lượng đóng góp.
