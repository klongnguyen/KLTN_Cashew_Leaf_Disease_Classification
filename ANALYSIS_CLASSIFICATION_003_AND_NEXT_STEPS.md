# Phân tích kết quả classification V04 (bản 003) và việc cần làm tiếp

Ngày rà soát: 2026-09-28.

## Phạm vi và dữ liệu có thể kiểm tra

Thư mục thực tế trong repository là [`training_results/current_dataset/`](training_results/current_dataset/), khác cách viết `training/_results/current/_dataset` trong yêu cầu. Ba experiment mới nhất là `EXP-DENSENET121-SCRATCH-5SEEDS-003`, `EXP-RESNET50-SCRATCH-5SEEDS-003` và `EXP-VIT-SCRATCH-5SEEDS-003`. Cả ba dùng Cashew_dataV04, cùng split 4.822 train / 1.402 validation / 687 test và cùng seed `42, 123, 2026, 3407, 7777` theo các file `experiment_config.json`.

Tôi đã đối chiếu [bảng benchmark](training_results/current_dataset/README.md), từng `aggregate/*.csv`, `README.md`, `experiment_config.json` của ba experiment và [ba notebook](Notebook/). Trong cây thư mục hiện tại **không có ZIP 003**; chỉ có ZIP phân tích ResNet50 002. Các thư mục 003 là bản tóm tắt đã trích ra, không chứa `history.csv`, checkpoint, confusion matrix hay dự đoán từng ảnh của mỗi seed. DenseNet và ViT có thêm SVG learning curves; ResNet50 003 không có SVG này. Vì vậy không thể dựng lại chính xác đường train/validation của ResNet50 003 hoặc xác định nguyên nhân sai từng ảnh chỉ từ bản hiện có.

## Kết quả đo được

| Mô hình | Validation Accuracy | Validation Macro F1 | Test Accuracy | Test Macro F1 |
|---|---:|---:|---:|---:|
| DenseNet121 scratch | 89,87 ± 1,85% | 89,53 ± 1,96% | **91,82 ± 0,86%** | **91,37 ± 0,79%** |
| ResNet50 scratch | 88,22 ± 2,49% | 88,06 ± 2,52% | 90,63 ± 2,11% | 90,17 ± 2,04% |
| Compact ViT scratch | 85,28 ± 1,28% | 84,68 ± 1,38% | 85,30 ± 1,68% | 84,41 ± 1,71% |

Các số là trung bình ± độ lệch chuẩn mẫu của 5 seed, lấy từ `validation_mean_std_summary.csv` và `thesis_test_mean_std_summary.csv` trong từng thư mục `aggregate`. DenseNet121 hiện là baseline tốt nhất cả về điểm trung bình và độ ổn định test. Chênh lệch Test Accuracy trung bình DenseNet121 − ResNet50 là **1,19 điểm phần trăm**; chỉ 5 seed nên không khẳng định sự vượt trội có ý nghĩa thống kê. ResNet50 có 24,64 triệu tham số, DenseNet121 có 7,57 triệu; benchmark thời gian trong repository là trên Kaggle T4×2, chưa phải tốc độ triển khai thực tế.

### ResNet50 cần chú ý về độ ổn định

| Seed | Validation Macro F1 | Test Accuracy | Epoch có `val_loss` tốt nhất |
|---:|---:|---:|---:|
| 42 | 89,67% | 92,58% | 24 |
| 123 | 85,58% | 87,92% | 9 |
| 2026 | 85,19% | 88,79% | 15 |
| 3407 | **90,81%** | 91,85% | 30 |
| 7777 | 89,05% | 91,99% | 31 |

Seed 123 và 2026 xuống thấp ở cả validation lẫn test; `best_epoch` cũng đến sớm hơn. Đây là bằng chứng về độ nhạy theo seed trong cấu hình hiện tại, **chưa đủ để kết luận** do learning rate, BatchNorm hay nhãn ảnh. Seed triển khai 3407 trong metadata đã được chọn bằng **Validation Macro F1**, phù hợp protocol; không nên đổi sang seed 42 chỉ vì Test Accuracy của nó cao hơn.

### Lớp khó và đường học

`anthracnose` có Test F1 thấp nhất ở cả ba mô hình: DenseNet121 **81,30 ± 2,00%**, ResNet50 **79,35 ± 3,84%**, ViT **70,43 ± 4,44%**. Recall tương ứng là 83,11%, 80,66% và 67,87%. ViT còn yếu ở `healthy` (F1 79,72 ± 2,22%). Đây là chỗ nên xem ảnh validation bị nhầm và chất lượng nhãn trước khi tăng độ phức tạp mô hình. Số ảnh train theo lớp nằm trong khoảng 806–1.101; chênh lệch số lượng có tồn tại nhưng bản tóm tắt hiện tại không chứng minh đó là nguyên nhân chính.

[Phân tích đường học DenseNet/ViT 003 đã có](ANALYSIS_TRAIN_VAL_GAP_DENSENET_VIT_003.md) cho thấy ở epoch cuối train cao hơn validation khoảng 7,98 và 7,39 điểm phần trăm. Đó là **khoảng cách tại epoch dừng**, không phải hiệu năng của best checkpoint đã khôi phục. Accuracy của `model.fit()` cũng được đo trong lúc augmentation/dropout đang bật và trọng số đổi theo batch, còn validation được đo sau epoch. Nên đo thêm accuracy trên tập train sạch bằng inference mode trước khi lượng hóa mức overfit. Với ResNet50 003, bản repository hiện không có history nên chưa thể tính khoảng cách này.

Test Accuracy trung bình cao hơn Validation Accuracy 1,95 điểm ở DenseNet121 và 2,41 điểm ở ResNet50; ViT gần bằng nhau. Validation có thể khó hơn đối với hai CNN, nhưng cần kiểm tra ảnh và nguồn ảnh theo split mới xác định được. Không đổi split chỉ để nâng điểm.

## Có cần sửa code không?

**Có, ở phần xuất hình ResNet50.** [Notebook ResNet50](Notebook/ResNet50/resnet50_5seed.ipynb) đã được thêm ô `16A. FIVE-SEED TRAINING AND VALIDATION PANELS` sau vòng huấn luyện. Ô này đọc 5 `seed_*/logs/history.csv`, xuất **một ảnh** `figures/training_curves_5seeds_panel.png` gồm 2 hàng (Accuracy, Loss) × 5 cột seed, đánh dấu epoch có `val_loss` thấp nhất; đồng thời xuất `figures/validation_confusion_matrices_5seeds_panel.png`. Nó dùng lại file đã lưu khi `RUN_TRAINING=False`. Hai hình sẽ nằm trong cả FULL ZIP và ANALYSIS ZIP vì ô đóng gói duyệt toàn bộ `OUT_DIR`.

**Chưa có bằng chứng cần sửa kiến trúc hoặc một lỗi training bắt buộc.** ResNet50 hiện dùng cùng augmentation nhẹ với DenseNet, Adam `1e-3`, checkpoint theo `val_loss`, EarlyStopping patience 8 và ReduceLROnPlateau. Code gọi backbone theo `backbone(x)` và nạp lại checkpoint tốt nhất trước validation. Muốn cải thiện điểm số, nên coi thay đổi LR, regularization hay pretrained là **experiment mới** và kiểm tra bằng validation, không sửa trực tiếp cấu hình của 003.

**Nên bổ sung đo lường ở experiment kế tiếp:** lưu `train_clean_accuracy` bằng `model.evaluate()` trên train dataset không shuffle/không augmentation sau khi nạp best checkpoint; ghi `val_accuracy` tại chính checkpoint đó và `gap_at_best`. Có thể lưu thêm confusion matrix và danh sách ảnh validation sai theo seed trong gói phân tích gọn. Những thay đổi này giúp chẩn đoán đúng trước khi quyết định đổi optimizer hoặc kiến trúc. Notebook ResNet50 hiện ghi `final_train_val_accuracy_gap` từ epoch cuối; con số này chỉ là chỉ báo của quá trình fit, không nên gọi là gap tại checkpoint triển khai.

## Việc nên làm tiếp theo, theo thứ tự

1. **Giữ 003 làm baseline khóa.** Dùng DenseNet121 làm mốc chất lượng chính; giữ ResNet50 và ViT làm đối chứng về kiến trúc/kích thước. Không dùng kết quả test đã xem để chọn hyperparameter hoặc seed cho lần chạy tiếp theo.
2. **Lấy lại gói ANALYSIS/FULL 003 nếu còn trên Kaggle hoặc nơi lưu trữ.** Cần `history.csv`, `validation/confusion_matrix_normalized.csv`, `validation/predictions.csv` cho cả 5 seed, nhất là ResNet50. Nếu có đủ file, có thể chạy riêng ô panel mới trên thư mục kết quả 003 để xuất hình mà không huấn luyện lại; không dùng bản tóm tắt trong repository để vẽ đường giả.
3. **Rà soát lỗi validation theo lớp.** Bắt đầu từ ảnh `anthracnose` bị dự đoán thành `healthy`, `leaf_miner` hoặc `red_rust`, sau đó kiểm tra các ảnh `healthy` mà ViT nhầm. Ghi loại lỗi: triệu chứng nhẹ, nhiều bệnh trên một lá, nền/crop, ảnh mờ, hoặc nhãn đáng nghi. Chỉ chỉnh nhãn theo quy tắc nhất quán và tạo phiên bản dataset mới nếu dữ liệu thay đổi.
4. **Chạy một thí nghiệm đo lường riêng cho ResNet50.** Giữ nguyên split và hyperparameter 003, thêm `train_clean_accuracy` tại best checkpoint và lưu đầy đủ history/prediction. Notebook hiện để ID `EXP-RESNET50-SCRATCH-5SEEDS-004`, phù hợp để không ghi đè 003. Ô panel mới sẽ tạo hai PNG sau khi huấn luyện/validation xong. Lần chạy này xác nhận hiện tượng seed 123/2026 và mức chênh thật tại checkpoint.
5. **Sau khi có log, thử một thay đổi mỗi lần trên validation.** Ưu tiên ResNet50/DenseNet121 giảm LR khởi đầu từ `1e-3` xuống `3e-4`; với ViT thử warm-up ngắn và lịch giảm LR. Sau đó mới thử regularization hoặc augmentation về độ sáng/crop ở mức nhẹ. Mỗi cấu hình có experiment ID mới, cùng split và tiêu chí chọn checkpoint. Chỉ chạy đủ 5 seed cho cấu hình hứa hẹn rồi mới dùng test khóa một lần.
6. **Nếu mục tiêu là ứng dụng thực tế hơn là bắt buộc học từ đầu**, thử ImageNet pretrained + fine-tuning trong nhánh experiment khác. Không gộp điểm pretrained với bảng so sánh scratch như cùng một điều kiện huấn luyện.

Tiêu chí chọn cải tiến: Validation Macro F1 trung bình tăng, độ lệch giữa seed không xấu đi, và F1 `anthracnose` tốt hơn trên validation. Gap train–validation chỉ là chẩn đoán phụ; làm hai đường gần nhau nhưng cả hai cùng thấp không phải cải thiện.

## Giới hạn xác minh

Ô panel mới đã được kiểm tra cấu trúc notebook/nguồn dữ liệu cần đọc, nhưng chưa chạy TensorFlow hoặc vẽ PNG trong máy này: các thư mục 003 hiện không có history và confusion matrix từng seed, còn notebook ResNet50 là bản `004` chưa chạy. Để có ảnh thực, cần chạy ô này trong notebook sau khi đủ 5 seed hoặc cung cấp lại gói ANALYSIS/FULL 003.
