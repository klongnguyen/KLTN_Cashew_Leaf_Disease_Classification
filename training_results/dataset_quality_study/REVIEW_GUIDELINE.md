# PROTOCOL REVIEW THỦ CÔNG — ẢNH REAL VÀ ẢNH NGHI TRÙNG

> Dùng cùng [INVESTIGATION_PLAN.md](./INVESTIGATION_PLAN.md). Đây là **hướng dẫn thực hiện**, chưa phải kết quả đã review.  
> Không sửa dataset Test đã khóa; kết quả quan sát Test chỉ phục vụ giải thích lỗi, không tuning.

## A. Review 28 ảnh Real Test (9 sai + 19 đúng)

### Chuẩn bị
1. Xuất danh sách **tất cả** `test/**/real_*` và nối `predictions.csv` bằng **path chuẩn hóa**. Bảo đảm **28 ảnh không lặp**; chụp kèm `image_sha256` và phiên bản ZIP.
2. Cung cấp cho reviewer A **ảnh gốc và nhãn bệnh trong manifest**, **ẩn `pred_label` và `confidence`**. Nếu có reviewer B/chuyên gia, review độc lập trước khi tổng hợp.
3. Với ảnh gốc 3024×4032, hiển thị nguyên bản và 224×224 sau **pipeline preprocessing đúng như đã train**; tạo ROI lá chính *chỉ để quan sát*. Không dùng preview ROI từ Test để chọn bộ xử lý ảnh cho lần train mới.

### Chấm từng tiêu chí (không suy từ tên lớp dự đoán)
| Trường | Giá trị hợp lệ / cách ghi |
|---|---|
| `label_review` | `confirmed` / `suspected_wrong` / `ambiguous` / `needs_expert` |
| `reviewed_label` | anthracnose / healthy / leaf_miner / red_rust / multiple / uncertain |
| `multiple_symptoms` | yes / no / uncertain |
| `background_complexity` | low / medium / high (định nghĩa dưới) |
| `main_leaf_area_pct` | ước lượng 0–100; ghi 0 nếu không xác định |
| `lesion_visibility_original` | clear / weak / invisible / not_applicable |
| `lesion_visibility_224` | clear / weak / invisible / not_applicable |
| `blur_level` / `lighting` | none/mild/severe ; even/strong/shadow/backlit/uncertain |
| `possible_factors` | nhãn / nền / resize / bệnh nhẹ / nhiều bệnh / nhận nhầm loài / khác; cho phép nhiều giá trị ngăn cách bằng dấu `|` |
| `reviewer_confidence` | low / medium / high |
| `reviewer_id` / `review_notes` | người review và bằng chứng cụ thể |
| `prediction_correct` | blank khi blind review; chỉ điền sau khi reviewer đã khóa kết quả |

**Quy ước nền ảnh:** `low` = lá chính chiếm phần lớn, ít chi tiết khác; `medium` = thấy thêm lá/cành/nền nhưng vẫn nhận dạng lá chính dễ; `high` = nhiều lá/cành/đất hoặc che khuất làm lá chính/lesion khó tách. Chấm độc lập với việc model đúng hay sai; nếu khó đánh giá ghi uncertain trong note.

**Hai nhóm bắt buộc so sánh sau unblind:** (1) 9 ảnh sai; (2) 19 ảnh đúng. Không chỉ nhìn ảnh sai rồi suy nền là nguyên nhân. Thống kê theo cả lớp và mức nền/lesion; cỡ mẫu thấp, không kết luận nhân quả.

### Chốt nhóm lỗi
- **LABEL**: reviewer/chuyên gia xác minh sai nhãn hoặc có nhiều bệnh khó gán một nhãn; ghi tình trạng quyết định, tránh tự chỉnh Test.
- **PREPROCESSING**: lesion rõ trên ảnh gốc nhưng giảm rõ sau pipeline 224×224; đây là quan sát, cần A1/B1 xác minh tác động đến hiệu năng.
- **BACKGROUND SUSPECTED**: mối liên quan của nền với sai/dúng, cần background-only intervention để xác nhận.
- **MODEL/FEATURE**: triệu chứng nhìn rõ, nhãn đúng nhưng dự đoán sai; dùng Grad-CAM như bằng chứng hỗ trợ, chưa xác định cơ chế.
- **OUT-OF-SCOPE FALSE REJECTION**: lá điều thực địa bị đoán `not_cashew_leaf`.

**Báo cáo cuối WP1:** số lượng nhãn verified/ambiguous; bảng đúng–sai theo lớp, background, lesion; 3–5 hình ví dụ (nếu được phép công bố); danh sách câu hỏi chuyên môn chưa giải quyết.

## B. Review 39 cặp pHash nghi gần trùng xuyên split

### Chuẩn bị
1. Truy xuất `near_duplicates_cross_split_review.csv` từ kết quả IMPROVED001; chuẩn hóa tên split/class/path.
2. Lấy cả ảnh A/B, kích thước, SHA256, pHash, EXIF (nếu còn) và phát hiện có cùng tên gốc/cùng nguồn không.
3. Hiển thị song song ở cùng scale; nếu cần xem vùng lá (vein, lesion, mép lá) và bối cảnh ảnh.

### Phân loại một cặp
| `pair_verdict` | Điều kiện |
|---|---|
| `same_file_variant` | Ảnh gốc giống nhau, chỉ khác nén/định dạng/resize/crop/flip |
| `same_leaf_or_session` | Rõ cùng lá/cây và phiên chụp (không phải cùng byte) |
| `different_subject` | Có bằng chứng là hai mẫu khác nhau |
| `uncertain` | Không đủ chứng cứ; **không** tự suy “không leak” |

Ghi thêm `cross_split` (true/false), `cross_label`, `expert_label_conflict` (confirmed/suspected/no/unknown), `evidence` (vết bệnh/trục gân lá/EXIF/nguồn), `reviewer`.

**Khi nghi leakage:** Dừng tuyên bố Test độc lập cho nhóm đó, trình bày tác động, lập issue và lên phương án version mới. **Không di chuyển ảnh trực tiếp trong dataset đóng băng rồi báo lại metric cũ**.

### Chốt WP2
- Đã xem đủ **39/39** cặp và nếu cần các nhóm similarity liên quan.
- Có số xác nhận `same_file_variant` / `same_leaf_or_session` / `different_subject` / `uncertain`, kèm định danh cặp.
- Có quyết định có cần dataset version/split mới và lý do.

## C. Checklist trước khi kết luận

- [ ] Đã review **28 ảnh Real** không chỉ 9 ảnh sai.
- [ ] Blind review trước khi unblind dự đoán.
- [ ] Hai reviewer hoặc chuyên gia xác minh nhãn bất đồng khi có thể.
- [ ] 39 cặp nghi trùng được xem và phân loại có bằng chứng.
- [ ] Có báo cáo số mẫu theo **lá/cây/session** nếu metadata cho phép.
- [ ] Không lấy Test để chọn siêu tham số, ROI, preprocess hay seed.
- [ ] Bằng chứng đã xác nhận được phân biệt với giả thuyết.
- [ ] Nếu có sửa dữ liệu, phiên bản mới được tạo riêng.
