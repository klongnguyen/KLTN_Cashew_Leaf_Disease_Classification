# EXP-DENSENET121-TRANSFER-5SEEDS-015 — Historical Real Holdout Evaluation

**Thí nghiệm đánh giá, không huấn luyện mới.** Source archive: `EXP-DENSENET121-TRANSFER-5SEEDS-015_ANALYSIS.zip` (19/10/2026).

- Model/checkpoints: **EXP-014 ImageNet transfer learning**, 5 seeds `[42,123,2026,3407,7777]`.
- **`run_full_training=false`**, **`run_final_test=false`**, **`run_real_holdout=true`**. EXP-015 chỉ infer và chấm điểm những checkpoint đã khóa.
- Deployment seed **123** — lấy từ `EXP-014/validation_selection_lock.json` trước lúc truy cập Holdout; không chọn model bằng Holdout.
- Holdout có 49 ảnh trong archive `RL_Cashew_holdout.zip`; loại **1 ảnh trùng SHA256 với ảnh Test V08** (`red_rust/IMG_3244.jpg` ↔ `test/red_rust/real_red_rust_029.jpg`). Chấm **48 ảnh**: Anthracnose 13, Healthy 5, Leaf miner 24, Red rust 6, **không có ảnh true not_cashew_leaf**.
- Tập này từng được dùng để phân tích EXP-013; **historical previously exposed**, không được gọi là blind holdout mới.

## 1. Kết quả 5 checkpoint gốc EXP-014 trên cùng 48 ảnh

| Seed | Correct | Accuracy | Macro F1 (4 lớp có support) | Ghi chú |
|---:|---:|---:|---:|---|
| 42 | 39/48 | 81.25% | 80.98% | Không chọn deployment |
| **123** | **38/48** | **79.17%** | **80.90%** | **Deployment chọn trước bằng Val Loss** |
| 2026 | 39/48 | 81.25% | 80.78% | Không chọn deployment |
| 3407 | 39/48 | 81.25% | 80.87% | Không chọn deployment |
| 7777 | 39/48 | 81.25% | 80.48% | Không chọn deployment |

**Mean ± sample SD:** Accuracy **80.83 ± 0.93%**, Macro F1 **80.80 ± 0.19%**, Macro Precision **80.08 ± 0.68%**, Macro Recall **85.66 ± 0.52%**. Macro metrics chỉ tính **4 lớp xuất hiện trong ground truth**.

### So sánh lịch sử EXP-013 Scratch → EXP-014 Transfer (được EXP-015 kiểm tra)

| Metric trên cùng 48 ảnh | EXP-013 Scratch | EXP-015 dùng 014 checkpoint | Thay đổi |
|---|---:|---:|---:|
| Holdout Accuracy (5 seeds) | 51.25 ± 8.15% | **80.83 ± 0.93%** | **+29.58 điểm %** |
| Holdout Macro F1 (5 seeds) | 49.62% | **80.80%** | **+31.18 điểm %** |
| Selected deployment seed | EXP-013 seed **3407** | EXP-014 seed **123** | Mỗi seed chọn bằng Validation |
| Deployment Accuracy | 22/48 = **45.83%** | **38/48 = 79.17%** | **+33.34 điểm %** |

Chênh lệch lớn và cùng image/label manifest là bằng chứng **mạnh trên bộ ảnh lịch sử này**, nhưng không có nghĩa có thể ngoại suy độ chính xác 80.83% cho mọi vườn điều.

## 2. Phân tích theo bệnh — deployment seed 123

| Nhãn thật | Correct/Total | Recall | Nhầm lẫn còn lại |
|---|---|---:|---|
| Anthracnose | 8/13 | **61.54%** | 3 → red_rust; 1 → leaf_miner; 1 → not_cashew_leaf |
| Healthy | 5/5 | **100%** | Chỉ 5 mẫu |
| Leaf miner | 19/24 | **79.17%** | 3 → anthracnose; 2 → red_rust |
| Red rust | 6/6 | **100%** | Precision red_rust = **54.55%**, nhiều false positives |

Không có ảnh bệnh nào thành Healthy trong confusion matrix của **seed 123** trên 48 ảnh; không suy rộng kết luận này sang những seed khác hoặc ảnh mới.

**Bốn ảnh cả 5 seeds đều dự đoán sai:**

| Ảnh | Ground truth | Prediction của cả 5 seeds |
|---|---|---|
| `IMG_3188.jpg` | Anthracnose | not_cashew_leaf |
| `IMG_3205.jpg` | Anthracnose | red_rust |
| `IMG_3176.jpg` | Leaf miner | anthracnose |
| `IMG_3197.jpg` | Leaf miner | red_rust |

Đặc biệt seed 123 đoán sai `IMG_3205.jpg` thành red_rust với confidence **91.98%** và `IMG_3197.jpg` với **92.49%**. Cần chuyên gia kiểm tra nhãn, hình thái tổn thương và ảnh cùng phiên chụp; không tự đổi nhãn dựa trên dự đoán mô hình.

## 3. Data provenance & reproducibility

- V08 ZIP SHA256: `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`.
- Source EXP-014 config SHA256: `fcf1ebc509cd5c4dd54f9ad073e5af02ba5b412f72210dfee47a998c921292b2`.
- Source EXP-014 validation lock SHA256: `e2e90a5e450b1c744eefae9584628c2d545885fada2b376ba22cb7b697263ea0`.
- Holdout evaluated manifest SHA256: `8b99e13d574ee915fcbe008b7403a53adadcf55fef5520455b4e1920bec0d750`.
- Holdout byte audit đã loại overlap với Test; **không thay thế audit pHash/cùng cây/cùng lá/cùng buổi chụp**.
- Checkpoint SHA256 metadata liên kết EXP-014 ↔ EXP-015, nhưng **không có binary checkpoint trong ANALYSIS ZIP** để băm và kiểm chứng độc lập.

## 4. File đã lưu trong repo

- [Results 5 seeds](./real_holdout/aggregate/real_holdout_results_5seeds.csv) · [Mean / SD](./real_holdout/aggregate/real_holdout_mean_std.csv) · [Per-class](./real_holdout/aggregate/per_class_real_holdout_mean_std.csv)
- [Deployment result](./real_holdout/aggregate/deployment_seed_result.json) · [Confusion matrix seed 123](./real_holdout/seed_123/confusion_matrix.csv) · [Các lỗi seed 123](./real_holdout/seed_123/error_cases.csv)
- [Evaluation config](./evaluation_config.json) · [Execution status](./execution_status.json) · [Holdout audit](./real_holdout/dataset_audit.json)
- [Biểu đồ so sánh Accuracy 013 vs 015](./figures/holdout_accuracy_comparison.svg) (tái lập từ CSV).
- [Source training report EXP-014](../EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md).
- `Notebook/DenseNet121/EXP-DENSENET121-TRANSFER-5SEEDS-015_MODEL_REPORT.md` là **báo cáo kế hoạch 015 từ trước khi chạy**; README tại đây mới là báo cáo kết quả.

**Chưa commit** toàn bộ original ANALYSIS ZIP, PNG confusion matrices, 48 predictions × 5 seed, checkpoint .h5/.keras, FULL ZIP hoặc notebook gốc. Mọi đường dẫn trong báo cáo chỉ được tuyên bố có trên GitHub nếu đã commit thực tế.

## 5. Kết luận và bước tiếp

- EXP-014 + EXP-015 cho thấy **phương pháp ImageNet pretrained + partial fine-tuning** cải thiện rõ rệt hiệu năng trên **cùng 48 ảnh real holdout lịch sử** so với EXP-013 Scratch.
- **Không retrain** hoặc chọn lại seed theo Holdout. Giữ seed **123** là checkpoint ứng viên theo Validation.
- **Chưa có Final Test V08 686 ảnh cho EXP-014**; chưa có independent blind holdout mới.
- Thu thập test thực địa mới ở nhiều cây/vườn/phiên chụp, bảo đảm label `not_cashew_leaf`, kiểm toán trùng và đánh giá YOLO crop → DenseNet end-to-end.

**Tóm tắt tiến độ:** [EXPERIMENT_PROGRESS_2026_10.md](../EXPERIMENT_PROGRESS_2026_10.md).

