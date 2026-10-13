# Đánh giá DenseNet121 Scratch — IMPROVED002 (Dataset V8 có Real Validation)

**Thí nghiệm:** `EXP-DENSENET121-SCRATCH-IMPROVED-002` — 13/10/2026  
**Mục tiêu:** WP3 — đưa ảnh thực địa từ Train sang Validation, đo khả năng tổng quát hóa trên ảnh thực địa.  
**Nguồn số liệu:** ZIP kết quả do người dùng cung cấp, gồm `metrics.json`, `validation_predictions.csv`, `predictions.csv`, `history.csv`, `dataset_version_integrity_v7_v8.json`. Đồng thời đối chiếu theo ảnh với ZIP IMPROVED001.  
**Phạm vi:** nghiên cứu dataset riêng, không gộp vào benchmark `Cashew_dataV05`.

## 1. Kết luận chính

**WP3 đã đạt mục tiêu:** CashewData_Split (8) có **35 ảnh Real Validation**, chuyển từ 170 ảnh Real Train của V7. Tập Train+Val 6.220 ảnh giữ nguyên nội dung, Test V8 686 ảnh vì đã chủ động loại `not_cashew_leaf_1514.JPG`. 686 ảnh Test còn lại cùng lớp/tên/SHA256 với V7.

Mô hình đạt **Validation Accuracy 93,01%**, nhưng **Real Validation Accuracy chỉ 65,71% (23/35)**, Macro F1 **69,12%**. Non-real Validation đạt **93,69% (1.307/1.395)**. Khi giới hạn Non-real ở bốn lớp lá điều, Accuracy đạt **94,82% (1.025/1.081)**, hơn Real Val khoảng **29,11 điểm phần trăm**. Đây là **domain gap quan sát được**, không phải chứng minh background/resize là nguyên nhân.

**Test V8**: Accuracy **91,11% (625/686)**, Macro F1 **90,60%**; **Real Test 71,43% (20/28)**. So với IMPROVED001 trên cùng 686 ảnh, Accuracy giảm **1,90 điểm phần trăm**, nhưng Real Test cải thiện đúng **một ảnh ròng**. Không thể quy các biến động này cho riêng việc thêm Real Validation vì đồng thời 35 ảnh Real bị chuyển khỏi Train, checkpoint được chọn theo Validation có phân phối mới, và run chỉ sử dụng seed 42.

## 2. Dataset V8, chất lượng và protocol

| Lớp | Train | Validation | Test | Real Train | Real Val | Real Test |
|---|---:|---:|---:|---:|---:|---:|
| Anthracnose | 939 | 308 | 122 | 35 | 9 | 6 |
| Healthy | 803 | 216 | 118 | 11 | 3 | 7 |
| Leaf Miner | 875 | 264 | 132 | 71 | 18 | 9 |
| Red Rust | 1.072 | 328 | 158 | 18 | 5 | 6 |
| Not Cashew Leaf | 1.101 | 314 | 156 | 0 | 0 | 0 |
| **Tổng** | **4.790** | **1.430** | **686** | **135** | **35** | **28** |

**Tổng số ảnh: 6.906.** Real Val chỉ có **2,45%** số ảnh Validation. Số mẫu Healthy Real Val là **3**, nên không được suy khả năng tổng quát hóa từ 3/3 ảnh đúng.

Kiểm định: 0 file ảnh hỏng, 0 trùng SHA256, 0 nhóm trùng byte khác split, 0 exact cross-class label conflict; **136** cặp pHash nghi giống, **38** cặp xuyên split, **15** cặp khác lớp (distance≤4). So sánh danh sách V7/V8: **38/38** cảnh báo xuyên split V8 nằm trong 39 cặp V7 đã được người dùng review, **không phát sinh cặp mới**. Dữ liệu có thể vẫn cần kiểm tra chung lá/cây/phiên chụp vì pHash không đảm bảo phát hiện toàn bộ.

**Integrity:** V8 chỉ khác Test V7 bởi một ảnh `test/not_cashew_leaf/not_cashew_leaf_1514.JPG`. Việc loại ảnh do nghi trùng là quyết định của người xây dựng dataset; ở WP2 trước đó cặp liên quan từng có verdict “không trùng”. Cần lưu cả hai mốc, **không khẳng định trùng thật** nếu chưa có bằng chứng độc lập. V8 là phiên bản Test 686 ảnh riêng.

**Cấu hình:** DenseNet121 Scratch `weights=None`, all trainable, 7.042.629 tham số (6.958.981 trainable), input 224×224, seed 42, batch 16, tối đa 50 epoch, Adam LR=1e-4, Flip/Rotation(0,04)/Zoom(0,05), checkpoint `val_loss` thấp nhất (trên toàn bộ Validation). Dataset SHA256: `75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7`.

## 3. Learning curves

**37 epoch đã chạy**; best checkpoint ở **epoch 27**, `val_loss=0.21242`, Val Accuracy **93,01%**, Train Accuracy **98,77%**. Chênh Train–Val **5,76 điểm phần trăm** ở best epoch. Sau epoch 27, Train Loss vẫn giảm trong khi Val Loss không cải thiện bền vững, cho thấy dấu hiệu gap tổng quát hóa/overfitting nhưng không xác định nguyên nhân cụ thể. Thời gian training **842,25 giây**. Real-Val log loss tính từ xác suất dự đoán khoảng **0,942**, so với Non-real Val **0,194**.

## 4. Real Validation — 35 ảnh

| Subset | Đúng/tổng | Accuracy | Macro F1 |
|---|---:|---:|---:|
| Toàn bộ Validation | 1.330/1.430 | **93,01%** | 92,88% |
| **Real Validation** | **23/35** | **65,71%** | **69,12%** |
| Non-real Validation (5 lớp) | 1.307/1.395 | 93,69% | 93,50% |
| Non-real Validation (4 lớp lá điều) | 1.025/1.081 | **94,82%** | 94,72% |

| Lớp Real Val | Đúng/tổng | Recall | F1 |
|---|---:|---:|---:|
| Anthracnose | **4/9** | **44,44%** | 40,00% |
| Healthy | 3/3 | 100,00% | 100,00% |
| Leaf Miner | 13/18 | 72,22% | 76,47% |
| Red Rust | 3/5 | 60,00% | 60,00% |

**Ma trận nhầm lẫn Real Validation**, hàng là nhãn thật, cột Anthracnose, Healthy, Leaf Miner, Red Rust, Not Cashew:

```text
Anthracnose  [4, 0, 3, 2, 0]
Healthy      [0, 3, 0, 0, 0]
Leaf Miner   [5, 0,13, 0, 0]
Red Rust     [2, 0, 0, 3, 0]
Not Cashew   [0, 0, 0, 0, 0]  # không có Real Not Cashew
```

**8/12 lỗi tập trung ở hai chiều Anthracnose ↔ Leaf Miner**. Có **5/12 lỗi Real Val với confidence ≥90%**, đáng chú ý:
- `real_anthracnose_036.jpg` → Red Rust (98,22%).
- `real_anthracnose_021.jpg` → Leaf Miner (97,40%).
- `real_anthracnose_006.jpg` → Leaf Miner (96,97%).
- `real_anthracnose_014.jpg` → Leaf Miner (93,05%).
- `real_red_rust_013.jpg` → Anthracnose (90,23%).

Các kết quả này là **dự đoán sai theo nhãn hiện tại**, không chứng minh nhãn sai. Real Val n=35 rất nhỏ (Wilson 95% CI tham khảo cho Accuracy: **49,15%–79,17%**, giả định ảnh độc lập). Không có Real Not Cashew để đo khả năng nhận diện vật ngoài miền trong điều kiện chụp thực tế.

## 5. Test V8 — phân tích hậu nghiệm

**Test Accuracy 91,11%; Macro Precision 90,64%; Macro Recall 90,72%; Macro F1 90,60%; 61/686 sai.**

| Lớp Test | Support | Recall | F1 |
|---|---:|---:|---:|
| Anthracnose | 122 | **76,23%** | **78,15%** |
| Healthy | 118 | 94,92% | 94,12% |
| Leaf Miner | 132 | 93,94% | 89,21% |
| Red Rust | 158 | 97,47% | 97,16% |
| Not Cashew Leaf | 156 | **91,03%** | 94,35% |

Lỗi chính: Anthracnose → Leaf Miner **17**, Not Cashew → Anthracnose **12**, Anthracnose → Healthy **7**. Nhóm Not Cashew bị nhầm thành Anthracnose cần phân tích vì liên quan trực tiếp chức năng loại ảnh không phải lá điều.

**Real Test 20/28 = 71,43%**: Anthracnose **5/6**, Healthy **4/7**, Leaf Miner **9/9**, Red Rust **2/6**. Có **2 ảnh Real Red Rust bị dự đoán thành Not Cashew Leaf**. Đây là nhóm ảnh đã được nghiên cứu hậu nghiệm ở WP1, không phải tập để chọn phương án cải tiến.

## 6. V7 vs V8 trên 686 ảnh giống hệt nhau

Dự đoán từ hai ZIP được ghép theo class + filename trên đúng **686 ảnh chung**, kiểm nhãn và integrity SHA256.

| Metric | V7 trên 686 ảnh | V8 | V8 − V7 |
|---|---:|---:|---:|
| Test Accuracy | 638/686 = **93,00%** | 625/686 = **91,11%** | **−1,90 pp** |
| Test Macro F1 | 92,50% | 90,60% | −1,90 pp |
| Real Test Accuracy | 19/28 = 67,86% | 20/28 = 71,43% | +3,57 pp (1 ảnh) |
| Non-real Test Accuracy | 619/658 = 94,07% | 605/658 = 91,95% | −2,13 pp |

**612** ảnh đúng ở cả hai; **35** sai ở cả hai; **26** ảnh đúng V7→sai V8 và **13** sai V7→đúng V8. Ở Real Test, V8 sửa đúng `real_anthracnose_045.jpg` và `real_anthracnose_049.jpg`, nhưng làm sai mới `real_red_rust_025.jpg`. Not Cashew chung 156 ảnh giảm từ **154** xuống **142** ảnh đúng.

**Không quy tác động nhân quả** cho Real Validation: chuyển 35 ảnh khỏi Train đồng thời tạo Validation mới, mô hình chỉ train 1 seed và checkpoint được chọn theo Validation mới.

## 7. Phát hiện, hạn chế, hướng WP4

**Đã xác nhận:** tồn tại Real Validation 35 ảnh; domain gap lớn giữa Real và Non-real; lỗi Real Val tập trung Anthracnose↔Leaf Miner; không phát hiện trùng byte SHA256 khác split.

**Chưa chứng minh:** ảnh nhiều background, tổn thương nhỏ, ảnh nhiều bệnh, direct resize hoặc overfitting là *nguyên nhân trực tiếp* của domain gap. WP1 trước đó chưa cho thấy quan hệ đơn giản giữa background mức “high” và dự đoán sai; nhãn bệnh đa triệu chứng còn cần chuyên gia xác nhận.

**WP4 đề xuất:** Review **12 lỗi Real Validation** bằng kiểm tra nhãn độc lập (nhất là 5 ca confidence≥90%). Trên **cùng Train/Real Val V8**, so A0 Direct Resize, A1 Letterbox, A2 ROI đã annotate chính xác, B1 Background-only perturbation; báo Macro F1 Real Val + Recall lớp + Non-real Val; chạy nhiều seed nếu tài nguyên cho phép. **Không dùng Real Test đã xem để tuning**. Giữ lịch sử Test V7 (687) và Test V8 (686) riêng; dùng holdout mới độc lập khi có thể.

**Kết luận khóa luận:** Bổ sung Real Validation tạo ra phép đánh giá trực tiếp độ khó ảnh ngoài vườn và phát hiện chênh lệch mạnh bị Accuracy chung che khuất, nhưng chưa chứng minh mô hình hay dataset đã tốt hơn. Hướng nghiên cứu phù hợp là xác minh nhãn/ảnh nhiều bệnh, kiểm soát miền ảnh và thí nghiệm preprocessing có đối chứng trên Validation.

**Báo cáo số liệu lấy từ ZIP thí nghiệm; checkpoint và FULL ZIP lớn không được commit lên repo.**
