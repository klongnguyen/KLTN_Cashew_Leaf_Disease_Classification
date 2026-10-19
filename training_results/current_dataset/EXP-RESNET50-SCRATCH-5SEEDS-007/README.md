# EXP-RESNET50-SCRATCH-5SEEDS-007 — ResNet50 Scratch, V07

> **Nguồn:** `EXP-RESNET50-SCRATCH-5SEEDS-007_ANALYSIS.zip` · Validation completed, **Final Test pending**. **EXP-008 có 5 hàng Validation/predictions/training histories trùng byte EXP-007**; không coi là một lần cải thiện độc lập.

## Dataset

- **Cashew_dataV07_clean_val**, **Train 4.825 / Validation 1.395 / Test 687 = 6.907** ảnh, 5 lớp.
- Manifest SHA256 `f9e7d9a39f33bd631b3133a77d0260055bcdfbcd79cbebbb4d87877b893c6d74`; ZIP SHA256 `6983b9ab8339489f18edb00f773b319ee6c62c2a3076c199b078ef52b464a3bf`.
- Class order: `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`.
- File tên `real_`: Train **170**, Val **0**, Test **28**. Do đó **không có phép đo riêng Real Validation**.
- Dataset audit: 0 exact cross-split duplicate; **52 cặp pHash nghi gần trùng** + 1 candidate thumbnail chưa review thủ công. Không chứng minh không có leakage theo cùng lá/buổi chụp.

## Cấu hình so với EXP-006

| Hạng mục | EXP-006 V05 | EXP-007 V07 |
|---|---|---|
| Weights | ResNet50 `None` | **Giữ Scratch** (`weights=None`) |
| Dataset | V05 / 6.911 | **V07 / 6.907** |
| Head | GAP → Dense(512) → Dropout 0.4 → Dense 5 | **GAP → Dropout 0.3 → Dense 5** |
| Backbone trainable | Toàn bộ | Toàn bộ |
| Input preprocessing | **/255** | **ResNet50 Caffe `preprocess_input` tương đương, nằm trong model** |
| Batch | 32 | **16** |
| Adam LR | 3e-4 | **1e-4** |
| Augmentation | flip/rotation .04/zoom .05/translation .03/contrast .08 | flip/rotation .04/zoom .05; **không contrast** |
| ReduceLR / EarlyStop patience | 3 / 8 | **4 / 10** |
| Total params | 24.641.413 | **23.597.969** |

**Quan trọng:** preprocessing Caffe tương đương ResNet50 nhận **RGB pixel 0–255**, đổi BGR và trừ RGB-channel means nội bộ. **Không chia 255 ngoài pipeline một lần nữa**. Cả hai model đều huấn luyện từ đầu, không có pretrained weights.

## Kết quả Validation 5 seeds

| Seed | Best epoch | Val Loss | Val Accuracy | Val Macro F1 |
|---:|---:|---:|---:|---:|
| 42 | 40 | 0.368771 | **90.97%** | 90.59% |
| **123** | **16** | **0.332464** | 87.81% | 86.91% |
| 2026 | 34 | 0.338971 | **91.83%** | **91.38%** |
| 3407 | 16 | 0.366160 | 88.10% | 87.29% |
| 7777 | 11 | 0.353775 | 87.31% | 86.87% |

**Mean ± sample SD:** Accuracy **89.20 ± 2.04%**, Macro F1 **88.61 ± 2.19%**, Macro Precision **89.73 ± 2.23%**, Macro Recall **88.27 ± 2.10%**.

**Selected deployment candidate seed 123**, theo Val Loss nhỏ nhất `0.332464`; **không được đổi sang 2026** chỉ vì Accuracy cao hơn khi protocol đã khóa `val_loss`.

### Điểm yếu ở Validation

Recall trung bình qua 5 seeds: Anthracnose khoảng **83.48%**, **Healthy 73.80%**, Leaf miner **92.93%**, Red rust **98.51%**, Not cashew **92.61%**. Healthy yếu nhất; nhầm Healthy → Anthracnose thường xuyên. Điều này là baseline trong dataset V07, **chưa suy ra được Recall ảnh ngoài vườn**.

## Giới hạn

- **Final Test V07 chưa chạy**, Holdout chưa chạy trong source.
- **Không so trực tiếp Val V05 của EXP-006 với V07 007 như controlled model improvement**; khác split + nhiều config.
- 008 có source predictions/history/metrics byte-identical; xem [đối chiếu 007/008](../EXP-RESNET50-SCRATCH-5SEEDS-008/README.md).
- Checkpoint binary và original FULL/ANALYSIS ZIP không được commit trong đợt tài liệu này.

## Artifact đã lưu

[Validation results](./aggregate/validation_results_5seeds.csv) · [Mean ± SD](./aggregate/validation_mean_std_summary.csv) · [Training summary](./aggregate/training_summary_5seeds.csv) · [Config snapshot](./config_snapshot.json) · [Dataset audit](./dataset_audit_summary.json) · [Seed 123 confusion matrix](./seed_123/validation/confusion_matrix.csv).

Source ZIP SHA256: `8d4d8d2c02877c9b98481de1c28ad285cf980710c57bfe917aec7e5b7e38824f`.
