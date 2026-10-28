# Giao thức đánh giá và chuẩn báo cáo

> Áp dụng như checklist cho thực nghiệm mới. Không mặc nhiên tuyên bố mọi thí nghiệm cũ đã tuân thủ hoàn hảo.

## 1. Dataset
Nhiệm vụ phân loại có 5 lớp: anthracnose, healthy, leaf_miner, red_rust và not_cashew_leaf. Phải ghi đúng **label order từ config/model package của từng EXP**. Với từng snapshot, cung cấp Train/Validation/Test, nguồn, checksum ZIP/manifest nếu có và cách phát hiện ảnh lỗi/cùng lá/cùng buổi chụp. Rà soát ảnh giống nhau giữa split bằng cả similarity và kiểm duyệt thủ công. **0 exact duplicate không bảo đảm không có leakage theo nhóm**. Tiền tố real_ không chứng minh provenance field-independent.

## 2. Huấn luyện
Ghi rõ Scratch/Transfer, frozen hoặc trainable layers, preprocess input, augmentation, optimizer/LR, weight decay, batch, early stopping, selection lock và checkpoint. Phân tích 5 seed thường gồm 42, 123, 2026, 3407 và 7777; báo **mean ± sample standard deviation (ddof=1)** với kết quả từng seed. SD giữa các lần train không phải khoảng tin cậy cho địa điểm thu thập mới.

## 3. Dùng Train, Validation, Test và Holdout
| Split | Công dụng | Không nên |
|---|---|---|
| Train | Học weights, augmentation | Trộn ảnh và biến thể liên quan từ Val/Test |
| Validation | Chọn checkpoint, threshold, model | Tuyên bố độ chính xác trên ảnh ngoài vườn |
| Locked Test | Báo cáo kết quả cấu hình đã chọn | Quay lại tuning và gọi là untouched/blind |
| Historical Holdout | Phân tích lỗi, domain gap | Dùng chọn lại seed rồi quảng bá blind |
| Fresh blind field holdout | Đánh giá cuối trên acquisition độc lập | Cho model development tiếp xúc nhãn trước khi khóa |

EXP-019/020/021/025 và ViT-XGBoost-009 **chỉ đánh giá checkpoint có sẵn**, không coi là huấn luyện mới. Holdout02–05 không có true not_cashew_leaf, nên **Macro F1 4 lớp trên Holdout** không thể đọc như **Macro F1 5 lớp trên Test**.

## 4. Chỉ số
Classification: Accuracy, Macro Precision/Recall/F1, Balanced Accuracy, per-class P/R/F1/support, confusion matrix và learning curve. Detection: Precision/Recall/F1, mAP50, mAP50-95, Mean IoU, số GT, ngưỡng và phân tích false negative. Đánh giá triển khai: tham số, checkpoint size, đúng preprocess/label order, latency/hardware. Chú thích mọi đổi split/test; Take006/007 YOLO không có cùng Test/GT.

## 5. Chuẩn README cấp EXP
1. EXP ID, source archive/checksum, trạng thái **TRAINED / EVAL_ONLY / VAL_ONLY / NOT_PROMOTED**.
2. Version dataset và số Train/Val/Test/Holdout; class order, seed, nguồn ảnh.
3. Cấu hình và khác biệt so với đối chứng, nêu rõ nhiều biến thay đổi cùng lúc.
4. Bảng metrics từng seed, mean±SD, per-class report và hình đã commit (nếu có).
5. Quy tắc chọn seed/model **Validation-only**, giới hạn Test/Holdout đã xem.
6. Kết quả, nhận xét khoa học và failure modes (đặc biệt Anthracnose/Healthy).
7. Artifact thực sự có trong Git vs chỉ tồn tại ở ZIP/Drive.

[Dataset changelog](../DATASET_CHANGELOG.md) · [Experiment progress](../training_results/current_dataset/EXPERIMENT_PROGRESS_2026_10.md)
