# Báo cáo thử nghiệm Targeted Augmentation — ResNet50 V11 (EXP-028/029/030)

Nguồn phân tích: các ANALYSIS ZIP người nghiên cứu cung cấp, ngày 30/10/2026. Không tái huấn luyện. Chỉ sử dụng artifact gốc của từng EXP. Báo cáo này phân biệt rõ tập Validation, Final Test và Holdout lịch sử.

## 1. Phương pháp

| Hạng mục | EXP-028 (đối chứng) | EXP-029 | EXP-030 |
|---|---:|---:|---:|
| Dataset | Cashew V11 Split (11) | Cùng Split | Cùng Split |
| Train ảnh gốc | 4.916 | 4.916 | 4.916 |
| Số ảnh Healthy nguồn có tiền tố real_ | 15 | 15 | 15 |
| Ảnh offline augmented bổ sung | 0 | 15 | 35 |
| Tổng file real_healthy dùng Train | **15** | **30** | **50** |
| Tổng file Train | 4.916 | 4.931 | 4.951 |
| Val / Test gốc | 1.447 / 685 | Giữ nguyên | Giữ nguyên |
| Input, backbone | 256×256, ImageNet ResNet50 | Giữ nguyên | Giữ nguyên |
| Stage 1 / 2 max epoch | 18 / 16 | Giữ nguyên | Giữ nguyên |
| Stage 1 / 2 LR | 1e-4 / 3e-6 | Giữ nguyên | Giữ nguyên |
| Seeds | 42, 123, 2026, 3407, 7777 | Giữ nguyên | Giữ nguyên |
| Chọn checkpoint và seed | Min Validation Loss | Giữ nguyên | Giữ nguyên |

Các ảnh offline augmented là phiên bản **phụ thuộc** của 15 ảnh Healthy nguồn, không phải 35 lá Healthy độc lập mới. Online augmentation vẫn chạy trên ảnh đã biến đổi offline. SHA256 của dataset ZIP gốc: 5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49; SHA256 manifest chung: 91d8279239eaad2007b652f1859667359d0cf01f94af917dd27b0a934a7b7e75.

## 2. Kết quả trung bình 5 seed

| Metric | EXP-028 | EXP-029 | EXP-030 |
|---|---:|---:|---:|
| Validation Accuracy | 95,73% | 95,65% | **95,80%** |
| Validation Macro F1 | 95,57% | 95,48% | **95,64%** |
| Healthy Recall toàn Val | 95,68% | 95,95% | **96,85%** |
| **real_healthy Recall trên Val (7 ảnh)** | **42,86%** | **60,00%** | **77,14%** |
| Val Disease → Healthy (trung bình ảnh/911 bệnh) | 12,6 | 12,6 | 12,4 |
| Final Test Accuracy | 94,07% | **NOT_RUN** | 94,10% |
| Final Test Macro F1 | 93,61% | **NOT_RUN** | 93,62% |
| Holdout 05 Accuracy | **72,50%** | **NOT_RUN** | 71,33% |
| Holdout Macro F1 (4 lớp ground truth) | 67,20% | **NOT_RUN** | 67,27% |

Theo seed, số ảnh real_healthy Validation đúng/7 (EXP-028 → EXP-029 → EXP-030): seed 42: 3→3→5; 123: 3→5→6; 2026: 3→5→6; 3407: 3→4→5; 7777: 3→4→5. Chỉ có 7 ảnh nguồn chung cho cả 5 seed, không được coi là 35 ảnh độc lập.

## 3. Phân tích sự đánh đổi trên Real Holdout 05

| Metric (5-seed mean, 120 ảnh) | EXP-028 | EXP-030 |
|---|---:|---:|
| Healthy Recall (13 ảnh) | 29,23% | **40,00%** |
| Healthy Precision | **66,10%** | 45,79% |
| Anthracnose Recall | **86,96%** | 78,70% |
| Leaf Miner Recall | 78,33% | **80,00%** |
| Red Rust Recall | 60,00% | **61,60%** |
| **Bệnh → Healthy (sai/107 ảnh bệnh)** | **2,0 ảnh/seed** | **6,2 ảnh/seed** |

Bệnh → Healthy theo seed: EXP-028 = [2,3,1,2,2], EXP-030 = [5,5,8,6,7] ứng với [42,123,2026,3407,7777]. Cả 5 seed đều tăng lỗi bỏ sót bệnh. Checkpoint được chọn hợp lệ bằng Validation Loss: **seed 2026**. Trên checkpoint này, Healthy đúng Holdout tăng 4/13 → 6/13, nhưng bệnh → Healthy cũng tăng 1/107 → 8/107 và Holdout Accuracy giảm 71,67% → 66,67%.

Kết quả cho thấy augment nhiều ảnh Healthy có thể làm tăng khuynh hướng dự đoán nhãn Healthy, nhưng **không chứng minh** mô hình phân biệt bệnh/khỏe tốt hơn. Final Test gần như không đổi, Holdout Accuracy giảm.

## 4. Phân tích lỗi

Bảy ảnh Healthy vẫn bị cả 5 seed EXP-030 nhận sai thành Anthracnose: real_healthy_023, _027, _029, _031, _032, _034, _036. Seed 2026 nhận đúng hơn ở real_healthy_030 và _033, nhưng tăng lỗi bệnh → Healthy trên những ảnh như real_anthracnose_095 và real_leaf_miner_120. Một số ảnh sau xoay/translation có viền màu điền nhân tạo; chỉ ghi là rủi ro cần kiểm tra, chưa chứng minh được là nguyên nhân.

## 5. Giới hạn khoa học

- Nhóm real_healthy trên Validation chỉ có **7 ảnh gốc**, mỗi lần sửa một dự đoán tương ứng 14,29 điểm phần trăm recall của seed đó.
- Tiền tố real_ chưa đủ chứng minh nguồn ảnh độc lập theo cây/vườn/buổi chụp.
- Holdout 05 (120 ảnh) chỉ có 4 lớp ground truth, không có true not_cashew_leaf; Macro F1 chỉ tính các lớp có support.
- Test V11 và Holdout 05 đã được xem trong các vòng nghiên cứu trước: phép đối chiếu **hồi cứu**, không phải fresh blind test; tuyệt đối không dùng chúng để chọn target augmentation mới.
- Kiểm toán gợi ý 51 cặp gần trùng pHash cross-split cần xem provenance, không được coi tất cả là leakage.
- ANALYSIS ZIP không cung cấp đầy đủ binary checkpoint để tự chạy lại inference.

## 6. Kết luận và hành động

**Không promote EXP-030** chỉ vì Healthy Recall tăng. Duy trì EXP-028 làm đối chứng; EXP-029 là thí nghiệm Validation-only; EXP-030 là Target 50 với Test/Holdout hồi cứu. Trước khi huấn luyện lại: rà soát nhãn và Grad-CAM các ảnh lỗi, giảm artifact viền, tách riêng hai đối chứng online targeted augmentation và source-aware sampling, bổ sung lá Healthy nguồn mới đa dạng, rồi dùng một fresh blind field test đủ 5 lớp.

## Hồ sơ liên quan

- [EXP-028](./EXP-RESNET50-TRANSFER-5SEEDS-028/README.md)
- [EXP-029](./EXP-RESNET50-TRANSFER-5SEEDS-029/README.md)
- [EXP-030](./EXP-RESNET50-TRANSFER-5SEEDS-030/README.md)
- [Bảng so sánh 7 thí nghiệm](./RECENT_V11_MODELS_5SEEDS.csv)

**Giới hạn upload:** Các README, metrics CSV và báo cáo tóm lược được ghi vào GitHub; ZIP gốc, tất cả ảnh PNG, dự đoán từng ảnh và binary checkpoints không được mặc định đã upload.