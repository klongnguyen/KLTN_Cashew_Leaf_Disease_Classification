# EXP-RESNET50-SCRATCH-5SEEDS-006 — ResNet50 Scratch, V05

> **Nguồn:** `EXP-RESNET50-SCRATCH-5SEEDS-006_ANALYSIS (1).zip` do chủ dự án cung cấp; tổng hợp 19/10/2026. `README.md` trong source ghi **Status: INCOMPLETE** vì chưa chạy Final Test. Đây **không** phải thí nghiệm Transfer Learning `EXP-RESNET50-TRANSFER-5SEEDS-006`.

## Dataset & protocol

- Dataset **Cashew_dataV05**: Train **4.822**, Validation **1.402**, Test khóa **687**; tổng **6.911** ảnh; 5 lớp.
- Lớp theo `experiment_config.json`: `anthracnose, healthy, leaf_miner, not_cashew_leaf, red_rust` (khác thứ tự V07!).
- Split fingerprint: Train `7443257702dbba7a392a28b497a49933f5cfac7e187e991043007f8e309a795f`; Val `ea4bba016c9f3abb69c0bf6f7deb3d502929707edfbeafa89f36898b7747e279`; Test `ee49ef59c791a7519c172316afd48554f92fbb1b4f47d11b57c82d6ae1624dc4`.
- 5 seeds: **42, 123, 2026, 3407, 7777**. Mean ± sample standard deviation (`ddof=1`); chọn checkpoint theo **minimum Validation loss**, không chọn bằng Test hoặc Holdout.
- **Final Test V05 chưa chạy cho thí nghiệm 006**, mặc dù benchmark ResNet50 V05 *EXP-005* trước đây đã hoàn thành Test. Không chuyển kết quả Test 005 sang 006.

## Cấu hình đặc trưng của EXP-006

| Hạng mục | Giá trị từ source |
|---|---|
| Model | ResNet50 `weights=None`, `include_top=False`, train backbone từ đầu |
| Head | GAP → Dense(512) → Dropout **0.40** → Dense(5) |
| Trainable / total params | 24.587.269 / **24.641.413** |
| Input | 224×224 RGB; resize bilinear; **Rescaling 1/255** trong model |
| Optimizer / initial LR | Adam / **3e-4** |
| Batch / max epochs | **32 / 50** |
| Augmentation | Flip ngang; rotation **0.04**, zoom **0.05**, translation **0.03**, contrast **0.08** |
| EarlyStopping | Monitor Val Loss, patience **8** |
| ReduceLROnPlateau | factor **0.3**, patience **3**, min LR **1e-6** |
| Runtime | 1 GPU; `OneDeviceStrategy` |

Lưu ý: **ResNet50 Scratch `/255`** không tương đương preprocessing **ResNet50 Caffe** của EXP-007/008. Vì vậy bản sau thay đổi quy trình học, không chỉ thay dataset.

## Training và Validation (5 seeds)

| Seed | Epoch thực tế / best epoch | Val Loss | Accuracy | Macro F1 | Thời gian |
|---:|---|---:|---:|---:|---:|
| 42 | 27 / 19 | 0.354891 | **88.80%** | 88.63% | 1.444,4 s |
| 123 | 16 / 8 | 0.504542 | 82.67% | 81.80% | 870,3 s |
| 2026 | 29 / 21 | 0.427269 | 87.02% | 86.48% | 1.563,6 s |
| **3407** | **17 / 9** | **0.357909** | 86.95% | 86.97% | 913,5 s |
| 7777 | 17 / 9 | 0.372192 | 86.88% | 86.70% | 918,3 s |

**Mean ± sample SD:** Val Accuracy **86.46 ± 2.27%**, Val Macro F1 **86.12 ± 2.56%**, Macro Precision **87.18 ± 2.62%**, Macro Recall **86.03 ± 2.39%**.

**Candidate seed theo minimum Validation Loss trên 5 seeds: 42 (loss 0.354891)**. Bản này không được nâng thành deployment chính thức; đây chỉ là ứng viên lựa chọn theo rule.

## Kết luận và hạn chế

- Chênh lệch seed lớn: seed 123 chỉ **82.67%** trên Val; kiểm tra các lớp nhầm và tốc độ hội tụ trước khi kết luận hiệu quả kiến trúc.
- EXP-006 sử dụng **V05**, EXP-007/008 dùng **V07** khác số ảnh, split, normalization, head, batch/LR. Chênh lệch Validation giữa các phiên bản **không phải** phép thử chỉ thay một hyperparameter.
- Không có Test/Real Holdout của **chính 006** trong ANALYSIS ZIP; các metrics ở đây chỉ là Validation.
- Source ANALYSIS ZIP chứa nhiều log, predictions, biểu đồ PNG hơn các artifact chọn lọc lưu trên GitHub; **binary checkpoint, FULL ZIP không được commit**.

## Artifact trên GitHub

- [Validation per seed](./aggregate/validation_results_5seeds.csv)
- [Validation Mean ± SD](./aggregate/validation_mean_std_summary.csv)
- [Training summary rút gọn](./aggregate/training_summary_5seeds.csv)
- [Config snapshot](./config_snapshot.json)
- [Seed 42 confusion matrix](./seed_42/validation/confusion_matrix.csv) · [Seed 123 confusion matrix](./seed_123/validation/confusion_matrix.csv)
- [Progress tracker](../EXPERIMENT_PROGRESS_2026_10.md)

Source ZIP SHA256: `d1654999d1003f131c25153b705d49798c94c4e2737a15f156f3c7cbb32555e4`.
