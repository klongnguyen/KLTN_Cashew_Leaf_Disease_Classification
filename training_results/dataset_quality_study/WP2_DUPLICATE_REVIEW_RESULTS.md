# WP2 — Biên bản review 39 cặp nghi gần trùng xuyên split

**Dataset:** CashewData_Split (7) — EXP-DENSENET121-SCRATCH-IMPROVED-001
**Nguồn kiểm tra:** `near_duplicates_cross_split_review.xlsx` (người dùng cung cấp), sheet `near_duplicates_cross_split_rev`, ô A1:J40.
**Trạng thái:** Hoàn tất review *danh sách 39 cặp được pHash cảnh báo*, theo kết quả tự đánh giá của người dùng.

## Kết quả

| Chỉ tiêu | Số cặp |
|---|---:|
| Tổng cặp đã đánh dấu | 39 |
| `Check duplicate = False` | **39** |
| `Check duplicate = True` | **0** |
| Thiếu đánh giá | 0 |
| Train–Validation | 13 |
| Train–Test | 20 |
| Validation–Test | 6 |

Thống kê bổ sung: 33 cặp cùng lớp, 6 cặp khác lớp; pHash distance 4: 35 cặp, distance 2: 3 cặp, distance 0: 1 cặp. **Tất cả 39 cặp đều có `real_a=False` và `real_b=False`**.

Đặc biệt, `train/not_cashew_leaf/not_cashew_leaf_42.jpg` ↔ `test/not_cashew_leaf/not_cashew_leaf_1514.JPG` có **pHash distance = 0**, nhưng được người dùng đánh dấu `Check duplicate=False`.

## Kết luận

**39/39 cặp nghi gần trùng xuyên split đã được người dùng đánh dấu không trùng.** Trong phạm vi danh sách này, chưa phát hiện cặp trùng được reviewer xác nhận và không có cơ sở loại ảnh chỉ vì pHash.

**Giới hạn:** File Excel chỉ chứa tên ảnh, metadata và verdict, không chứa ảnh nguồn hoặc mô tả bằng chứng thị giác; đây là **kết quả kiểm tra do người dùng xác nhận**, không phải xác minh độc lập bằng mắt của trợ lý. pHash có thể bỏ sót các góc chụp khác nhau của cùng lá/cây/phiên chụp. Kết quả này không chứng minh toàn bộ dataset tuyệt đối không có data leakage.

Do không cặp nào thuộc nhóm `real_`, bước review này **chưa xác nhận tính độc lập ở cấp lá/cây/phiên chụp của dữ liệu thực địa**.

## Tác động đến kế hoạch

- WP2 hoàn thành đối với **39 cảnh báo**; không cần di chuyển ảnh trong split chỉ dựa trên danh sách pHash này.
- Tiếp tục WP1: review **28 ảnh Real Test** gồm 9 sai và 19 đúng, xác minh nhãn và đặc điểm background/resize.
- Khi xây dựng Real Validation và Real Holdout, tiếp tục nhóm ảnh theo leaf/tree/session để tránh leakage không được pHash nhận ra.
- Không huấn luyện lại hay cập nhật Test metrics chỉ vì kết quả review này.

[Kế hoạch tổng thể](./INVESTIGATION_PLAN.md) · [Báo cáo IMPROVED001](./EXP-DENSENET121-SCRATCH-IMPROVED-001/ANALYSIS.md).
