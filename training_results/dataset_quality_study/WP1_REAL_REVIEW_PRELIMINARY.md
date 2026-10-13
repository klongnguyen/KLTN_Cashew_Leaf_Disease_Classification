# WP1 — Kết quả review ảnh thực địa (sơ bộ 25/28)

**Nguồn:** REAL_DATA_INVESTIGATION_WP1_001_ANALYSIS.zip, ngày 13/10/2026.
**Trạng thái:** WP1 IN PROGRESS. Chưa đánh giá xong 28/28 và chưa có xác minh chuyên gia.
**Báo cáo model để đối chiếu:** [IMPROVED001 — 9 lỗi Real Test](./EXP-DENSENET121-SCRATCH-IMPROVED-001/REAL_TEST_ERRORS.md). ZIP WP1 không chứa predictions; so sánh sau đây ghép theo tên file từ báo cáo IMPROVED001, chưa đối chiếu hash từng ảnh.

## 1. Tiến độ và phát hiện

- Tổng 28 ảnh hợp lệ, 0 ảnh lỗi; đã review 25/28.
- 3 ảnh chưa review: real_red_rust_027.jpg, real_red_rust_028.jpg, real_red_rust_029.jpg.
- Label review: 22 confirmed theo người review, 3 ambiguous (real_healthy_017, real_red_rust_025, real_red_rust_026).
- Background high: 15/25; medium: 9/25; low: 1/25.
- Leaf Miner 9/9 đã review; 7/9 có background high; cả 9 được model lịch sử dự đoán đúng.
- Cả ba nhãn ambiguous cần xác minh độc lập, nhất là dấu hiệu nhiều bệnh trên Red Rust và lá bệnh nằm trong background của Healthy.

## 2. Đối chiếu đúng/sai (chỉ mô tả, cỡ mẫu nhỏ)

| Nhóm đã review | n | Background high | Diện tích lá chính trung bình |
|---|---:|---:|---:|
| Dự đoán sai | 8 | 4 (50,0%) | 53% |
| Dự đoán đúng | 17 | 11 (64,7%) | 48% |

**Không có bằng chứng cho rằng chỉ riêng mức background high gây lỗi.** Ví dụ real_anthracnose_045.jpg có background low, lá chính ~84%, lesion rõ nhưng model đoán thành not_cashew_leaf. Cần phân biệt background nhiều với background chứa lá có triệu chứng.

## 3. Resize và vấn đề chuẩn hóa biểu mẫu

| Độ rõ lesion gốc → 224 | Số ảnh |
|---|---:|
| clear → clear | 17 |
| weak → clear | 4 |
| weak → weak | 3 |
| blank → weak | 1 |

Chưa ghi nhận clear → weak trong form. Tuy nhiên một nhận xét nói rõ “ROI → 224×224”, trong khi cột lesion_visibility_224 được định nghĩa là **direct resize**; vì vậy **không kết luận resize vô hại** trước khi thống nhất tiêu chí.

- Ảnh Healthy không có lesion trên lá mục tiêu: điền `not_applicable`, không điền clear/weak dựa vào lá bệnh phía sau.
- ROI: 24/25 ảnh dùng gần như đúng tọa độ mặc định (x=15%, y=10%, width=70%, height=80%); chưa thể xem đây là ROI hiệu chỉnh theo từng lá.
- Thiếu lighting 19/25, multiple_symptoms 9/25, blur_level 4/25; cần bổ sung nếu quan sát lại.

## 4. Kết luận và bước tiếp

**Đã thấy:** phần lớn ảnh có nền medium/high, nhưng nhóm dự đoán đúng cũng có nhiều ảnh nền phức tạp. Ba nhãn cần review lại; một số ảnh Red Rust được mô tả có thể có nhiều bệnh.

**Chưa chứng minh:** background, resize, ROI là nguyên nhân nhân quả. Không được dựa Test này để chọn preprocessing.

**Bước tiếp:**
1. Hoàn thành review Red Rust 027, 028, 029.
2. Chuẩn hóa ghi nhận Direct 224 vs ROI, Healthy N/A, lighting/blur/multiple_symptoms.
3. Xác minh 3 nhãn ambiguous bằng chuyên môn nếu có thể.
4. Sau WP1, lập Real Validation độc lập (từ Train hoặc dữ liệu mới, group-aware); trên Validation mới kiểm chứng direct resize, letterbox, ROI và background ablation.

**Không thay đổi hay đánh giá lại Test Set gốc từ những nhận xét này.**
