# WP1 — TỔNG KẾT REVIEW 28 ẢNH REAL TEST (HOÀN TẤT PHẦN QUAN SÁT)

**Experiment:** `REAL_DATA_INVESTIGATION_WP1_001`  
**Ngày cập nhật:** 13/10/2026  
**Nguồn:** `REAL_DATA_INVESTIGATION_WP1_001_ANALYSIS (2).zip` do người dùng cung cấp; `test_realdata.zip` gốc, Drive file ID `1WM5d2K9Spz31EFU9Zjtt4KU0chUO3aOr`  
**Review ZIP SHA256:** `18fef01e93070118d67413c95cb785cff8e8fdbb72eca68140e942377261112a`  
**Nguồn ảnh ZIP SHA256 (notebook ghi):** `eba81b9e4cd4748ad0deeaa921a3bce1aed6e2a345c06db6c92217a5b3d56f1e`  
**Trạng thái:** **ĐÃ REVIEW 28/28 ẢNH — CHƯA HOÀN TẤT XÁC MINH NHÃN CHUYÊN MÔN VÀ THÍ NGHIỆM ĐỐI CHỨNG**. Không huấn luyện lại / thay đổi Test.

## 1. Tiến độ và chất lượng biểu mẫu

| Nội dung | Kết quả |
|---|---:|
| Tổng ảnh hợp lệ | 28 |
| Ảnh lỗi | 0 |
| Ảnh `real_` | 28 |
| Đã có phiếu review | **28/28** (bản trước 25/28) |
| Reviewer chọn `confirmed` | **22** |
| Reviewer chọn `ambiguous` | **6** |
| Reviewer confidence `high` / `medium` | 21 / 7 |
| Trường `lighting` còn trống | 20/28 |
| Trường `multiple_symptoms` còn trống | 7/28 |
| Trường `blur_level` còn trống | 4/28 |

**Lưu ý:** `confirmed` là mức xác nhận do reviewer chọn, **không phải nhãn đã được chuyên gia bệnh cây độc lập xác nhận**. Những trường trống không cản trở kết luận **28/28 phiếu đã được lưu**, nhưng hạn chế phân tích sâu yếu tố chiếu sáng/chất lượng ảnh.

### Theo lớp

| Lớp | Đã review | Confirmed | Ambiguous |
|---|---:|---:|---:|
| Anthracnose | 6 | 6 | 0 |
| Healthy | 7 | 6 | 1 |
| Leaf Miner | 9 | 9 | 0 |
| Red Rust | 6 | 1 | **5** |
| **Tổng** | **28** | **22** | **6** |

Sáu ảnh `ambiguous` (theo reviewer): `real_healthy_017.jpg`, `real_red_rust_025.jpg`, `real_red_rust_026.jpg`, `real_red_rust_027.jpg`, `real_red_rust_028.jpg`, `real_red_rust_029.jpg`.

## 2. Background: quan sát và đối chiếu hậu nghiệm

**Đánh giá toàn bộ 28 ảnh:** Background `high` **16/28**, `medium` **9/28**, `low` **3/28**. Tỷ lệ diện tích lá chính do reviewer ước tính trung bình **51,57%**, không phải segmentation pixel chính xác.

ZIP WP1 **không chứa dự đoán model**. Để so nhóm đúng/sai, hậu kiểm này ghép 9 tên ảnh sai từ [danh sách IMPROVED001](./EXP-DENSENET121-SCRATCH-IMPROVED-001/REAL_TEST_ERRORS.md) với 28 `filename` trong WP1; **chỉ ghép theo tên file**, chưa xác thực cùng byte SHA256 từng ảnh giữa hai archive.

| Nhóm từ IMPROVED001 | Số ảnh | Background high | Tỷ lệ | Diện tích lá chính trung bình |
|---|---:|---:|---:|---:|
| Dự đoán sai | 9 | **5** | **55,6%** | 52,9% |
| Dự đoán đúng | 19 | **11** | **57,9%** | 50,9% |

**Nhận định:** Trong 28 ảnh này, chưa thấy mô hình sai thường xuyên hơn khi reviewer chấm background=`high` (hai nhóm có tỷ lệ gần bằng nhau). **Không** khẳng định background vô hại, vì mức độ có lá bệnh khác ở nền, tỷ lệ lá chính, số liệu từng bệnh, ánh sáng và cỡ mẫu đều có thể ảnh hưởng.

Ví dụ: 9/9 Leaf Miner được dự đoán đúng dù **7/9** nền `high`; `real_anthracnose_045.jpg` nền `low`, diện tích lá ~84% vẫn dự đoán sai thành `not_cashew_leaf`. Hướng giả thuyết cần kiểm định kỹ hơn: **lá khác có triệu chứng trong background** và **xác định lá mục tiêu trong ảnh nhiều lá**, không chỉ tổng diện tích nền.

## 3. So sánh khả năng nhìn thấy tổn thương ảnh gốc → 224

Đã điều chỉnh tất cả **7 ảnh Healthy** thành `not_applicable` cho cả hai cột lesion visibility — đây là định nghĩa chính xác hơn nếu đang đánh giá **tổn thương trên lá mục tiêu**, khác việc thấy bệnh trên lá nền.

Đối với **21 ảnh thuộc 3 lớp bệnh**:

| Lesion gốc → 224 | Số ảnh |
|---|---:|
| `clear → clear` | **15** |
| `weak → clear` | **4** |
| `weak → weak` | **2** |
| `clear → weak` | **0** |

Quan sát **không ủng hộ mạnh giả thuyết direct resize làm lesion từ rõ sang mờ**. Tuy nhiên: (a) cần chắc chắn cột 224 dùng đúng *direct resize* của model, vì ghi chú `real_healthy_021.jpg` còn đề cập **ROI→224**; (b) độ rõ bằng mắt **không** đồng nghĩa đặc trưng CNN được bảo toàn; (c) không có thí nghiệm đối chứng hiệu năng nên không suy nguyên nhân.

**ROI:** **27/28** ảnh giữ cùng tọa độ x=15%, y=10%, w=70%, h=80%; ảnh còn lại x=16%, y=10%, w=70%, h=80%. Do đó đây gần như là **crop mặc định chung**, chưa có bằng chứng từng lá đã được khoanh chuẩn; không thể dùng các tọa độ này để khẳng định đã kiểm định chính xác leaf-ROI.

## 4. Trường hợp nhiều bệnh / nhãn mơ hồ là hướng điều tra nổi bật nhất

- Toàn bộ **6/6 ảnh Red Rust** có `multiple_symptoms=yes` trong form.
- **5/6 ảnh Red Rust** có `label_review=ambiguous`; `real_red_rust_024.jpg` là `confirmed` nhưng cũng có `multiple_symptoms=yes`.
- Ghi chú các ảnh `real_red_rust_024…029.jpg` đề cập có thể có **Anthracnose và Red Rust trên cùng lá**. Đây là **quan sát của reviewer**; chưa có xác minh bệnh học độc lập.
- Ảnh `real_healthy_017.jpg` có `label_review=ambiguous` vì reviewer ghi background chứa lá biểu hiện bệnh, trong khi lá mục tiêu được gán Healthy. Cần làm rõ **classification target là lá chính hay toàn ảnh**.

Từ danh sách 9 lỗi model IMPROVED001, **3 ảnh Red Rust sai** (`024`, `026`, `029`) được reviewer đánh dấu `multiple_symptoms=yes`. Nhưng **3 ảnh Red Rust đúng cũng được đánh dấu như vậy** → không kết luận nhiều bệnh tự nó gây sai; mẫu nhỏ. Đối chiếu chỉ để định hình giả thuyết, không sửa Test hay gán nhiều nhãn chính thức từ prediction.

**Giả thuyết được ưu tiên tiếp theo:** ảnh có nhiều loại triệu chứng và/or nhiều lá bệnh không được mô tả đầy đủ bởi nhãn single-label cho toàn ảnh. Nếu được chuyên gia xác nhận, đây là một giới hạn của **định nghĩa bài toán/annotation policy**, cần giải quyết bằng quy tắc nhãn hoặc phát hiện vùng bệnh (YOLO), không đổ lỗi hoàn toàn cho DenseNet121.

## 5. Tồn đọng / giới hạn / việc tiếp theo

1. **WP1 manual review complete**; đóng công việc duyệt 28/28 ảnh. Đưa **6 nhãn ambiguous** cho chuyên gia hoặc reviewer thứ hai kiểm tra độc lập; giữ trạng thái chờ expert review.
2. Chuẩn hóa ghi chú `real_healthy_021.jpg` (Direct 224 khác ROI 224); nếu phân tích lighting/blur/multiple symptoms, cần bổ sung 20/4/7 giá trị còn trống.
3. Khi lập thí nghiệm ROI: cung cấp leaf bounding boxes/masks thật sự từng ảnh từ **Train/Real Validation**, không dùng crop mặc định làm ground-truth ROI.
4. Tiếp tục **WP3 Real Validation** (ảnh từ Train hoặc ảnh mới, group-aware), không chuyển 28 Real Test vào Train/Validation; không dùng kết quả WP1 để chọn hyperparameter.
5. **WP4 controlled ablation** (direct resize / letterbox / ROI / background-only) trên Real Validation, cùng protocol và seed; theo dõi macro F1 + per-class recall.
6. Khi đủ bằng chứng, tiến hành Holdout Real độc lập chung cho bài toán 4 lớp để đánh giá cải tiến dataset; model 5 lớp được báo cáo riêng.

## 6. Kết luận có thể trích trong khóa luận

> Kết quả rà soát thủ công 28 ảnh thực địa thuộc Test Set của IMPROVED001 cho thấy tất cả ảnh đã được đánh giá về nhãn, mức nền và khả năng quan sát triệu chứng. Trong đó 22 ảnh được người review xác nhận nhãn và 6 ảnh có nhãn chưa chắc chắn; 5/6 trường hợp nhãn chưa chắc chắn thuộc nhóm Red Rust. Cả 6 ảnh Red Rust được reviewer nhận xét có nhiều dạng triệu chứng, đặt ra câu hỏi về mức phù hợp của phương pháp gán một nhãn cho một ảnh khi có thể xuất hiện nhiều bệnh. Mức nền phức tạp không cho thấy liên hệ đơn giản với tình trạng dự đoán sai (5/9 ảnh sai và 11/19 ảnh đúng có nền mức cao). Ở 21 ảnh có nhãn bệnh, không có trường hợp được đánh dấu triệu chứng rõ ở ảnh gốc nhưng giảm thành khó thấy sau resize 224×224. Các nhận xét là bằng chứng mô tả của một vòng review, **chưa chứng minh quan hệ nhân quả** và chưa thay thế xác minh nhãn chuyên môn hoặc các thí nghiệm đối chứng trên Validation.

## 7. Nguồn và truy xuất

- [Kế hoạch điều tra WP0–WP7](./INVESTIGATION_PLAN.md).
- [IMPROVED001 — 9 ảnh Real Test dự đoán sai](./EXP-DENSENET121-SCRATCH-IMPROVED-001/REAL_TEST_ERRORS.md).
- [Bản WP1 sơ bộ 25/28 (giữ để truy vết)](./WP1_REAL_REVIEW_PRELIMINARY.md).
- Raw ZIP review **không commit** để tránh lẫn dữ liệu cá nhân/chứa ảnh, người dùng giữ bản gốc tại môi trường làm việc hoặc Drive.
