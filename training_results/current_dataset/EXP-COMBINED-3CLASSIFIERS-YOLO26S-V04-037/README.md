# EXP-COMBINED-3CLASSIFIERS-YOLO26S-V04-037

Kết hợp YOLO26s (EXP-Y26S-NEG-V04-001) với ba mô hình phân loại theo **cùng một quy trình** (quy trình EXP-035, mục 3.7.5): gộp 5 seed, một bộ kết hợp hồi quy logistic chọn trên Validation sạch (1361 ảnh, GroupKFold theo nhóm pHash), temperature và ngưỡng từ chối chọn ngoài fold.

| Mô hình phân loại | Thí nghiệm nguồn | ZIP (SHA-256) |
|---|---|---|
| DenseNet121 | EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04 | `EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04_ANALYSIS.zip` (27b5ea684d99…) |
| ResNet50 | EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4 | `EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4_ANALYSIS.zip` (85151ff0dbb9…) |
| ViT (DeiT-S) | EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-256px-1192fd08 | `EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-256px-1192fd08_FULL.zip` (e28040960bab…) |

Tái lập EXP-035 (DenseNet121, gộp): **ĐẠT**. F0 trùng tuyệt đối; F3 trong dung sai 2 ảnh. Bộ kết hợp F3 của DenseNet121 trong notebook này có C = 3, T = 0,9840, ngưỡng = 0,4451; EXP-035/ứng dụng: C = 1, T = 0,9317, ngưỡng = 0,4262. Khác biệt đến từ cách chia fold (xem `aggregate/parity_exp035.json`).

## Bảng 4.13 (gộp 5 seed, Accuracy %)

| Mô hình | Test F0 | Test F3 | Δ Test [CI95] | Thực địa F0 | Thực địa F3 | Δ Thực địa [CI95] |
|---|---|---|---|---|---|---|
| DenseNet121 + YOLO26s | 96,20 | 97,08 | 0,88 [-0,29; 2,04] | 83,33 | 94,44 | 11,11 [6,35; 16,67] |
| ResNet50 + YOLO26s | 95,77 | 97,96 | 2,19 [0,88; 3,65] | 78,57 | 88,89 | 10,32 [4,76; 16,67] |
| ViT (DeiT-S) + YOLO26s | 97,52 | 97,81 | 0,29 [-0,44; 1,17] | 87,30 | 92,86 | 5,56 [1,59; 10,32] |

## Accuracy theo quy tắc (gộp 5 seed, %)

| Mô hình | Tập | F0 | F3a | F3b | F3 |
|---|---|---|---|---|---|
| DenseNet121 | Test | 96,20 | 96,06 | 78,98 | 97,08 |
| DenseNet121 | Thực địa | 83,33 | 80,16 | 76,98 | 94,44 |
| ResNet50 | Test | 95,77 | 95,77 | 78,98 | 97,96 |
| ResNet50 | Thực địa | 78,57 | 79,37 | 76,98 | 88,89 |
| ViT (DeiT-S) | Test | 97,52 | 97,81 | 78,98 | 97,81 |
| ViT (DeiT-S) | Thực địa | 87,30 | 84,13 | 76,98 | 92,86 |

## Theo từng seed (trung bình ± SD, n = 5, %)

| Mô hình | Test F0 | Test F3 | Thực địa F0 | Thực địa F3 |
|---|---|---|---|---|
| DenseNet121 | 96,06 ± 0,18 | 97,08 ± 0,18 | 81,27 ± 1,83 | 94,44 ± 0,56 |
| ResNet50 | 95,59 ± 0,55 | 97,84 ± 0,19 | 76,03 ± 3,04 | 89,68 ± 0,97 |
| ViT (DeiT-S) | 97,14 ± 0,59 | 97,55 ± 0,48 | 85,24 ± 2,14 | 91,90 ± 1,30 |

## So sánh giữa các mô hình (F3, gộp 5 seed)

| Cặp | Tập | Δ Accuracy [CI95] | Chỉ A đúng / chỉ B đúng | p McNemar (Holm) |
|---|---|---|---|---|
| DenseNet121 − ResNet50 | Test | -0,88 [-1,75; -0,15] | 1 / 7 | 0.0703 (0.211) |
| DenseNet121 − ViT (DeiT-S) | Test | -0,73 [-1,75; 0,29] | 4 / 9 | 0.267 (0.534) |
| ResNet50 − ViT (DeiT-S) | Test | 0,15 [-0,73; 1,17] | 6 / 5 | 1 (1) |
| DenseNet121 − ResNet50 | Thực địa | 5,56 [0,79; 11,11] | 9 / 2 | 0.0654 (0.196) |
| DenseNet121 − ViT (DeiT-S) | Thực địa | 1,59 [-2,38; 6,35] | 5 / 3 | 0.727 (0.727) |
| ResNet50 − ViT (DeiT-S) | Thực địa | -3,97 [-7,94; 0,00] | 1 / 6 | 0.125 (0.25) |

## Ghi chú

- F3b chỉ dùng 9 đặc trưng YOLO nên giống nhau cho cả ba mô hình (cùng bằng chứng, cùng Validation sạch).
- DenseNet121 ở đây là 5 mô hình của EXP-035 (giống ứng dụng), khác lần huấn luyện báo cáo ở mục 4.1.1.
- Tập ảnh thực địa (Holdout 09) đã được xem trước: kết quả trên tập này chỉ mang tính mô tả.
- Macro F1 trên tập ảnh thực địa tính trên 4 lớp có nhãn.
- Bootstrap: 10.000 lần, đơn vị là ảnh; không phản ánh biến động do seed.
