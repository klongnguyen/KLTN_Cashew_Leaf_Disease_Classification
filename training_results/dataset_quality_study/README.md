# Dataset quality study — Kaggle base vs improved dataset

Nhánh nghiên cứu độc lập nhằm kiểm tra dữ liệu Kaggle gốc và dữ liệu cải tiến, **không trộn với benchmark `Cashew_dataV05` hiện tại**.

| Experiment | Dataset | Classes | Test Accuracy | Macro F1 | Ghi chú |
|---|---|---:|---:|---:|---|
| BASE001 | CashewData_base (Kaggle) | 4 | 94,50% | 94,53% | 5.360 ảnh; 1 nhóm file trùng bị xung đột nhãn |
| [IMPROVED001](./EXP-DENSENET121-SCRATCH-IMPROVED-001/) | CashewData_Split (7) | 5 | **93,01%** | **92,50%** | 6.907 ảnh; Real Test 67,86% (19/28) |

**Không so sánh trực tiếp hai Accuracy:** khác số lớp, Test, phân phối ảnh. Mục tiêu tiếp theo: xác thực nguồn/split và cùng Real Holdout 4 lớp, độc lập với cả hai tập Train/Val.

Báo cáo IMPROVED001: [ANALYSIS.md](./EXP-DENSENET121-SCRATCH-IMPROVED-001/ANALYSIS.md).

## Kế hoạch điều tra tiếp theo

**Kế hoạch điều tra tiếp theo (WP0–WP7):** [INVESTIGATION_PLAN.md](./INVESTIGATION_PLAN.md). Kèm [quy trình review ảnh](./REVIEW_GUIDELINE.md) và [mẫu ghi nhận ảnh Real](./templates/real_image_review.csv) / [nghi trùng](./templates/duplicate_pair_review.csv). **Trạng thái:** WP1 đã review thủ công 28/28 ảnh (6 nhãn ambiguous còn cần chuyên môn); WP2 đã review 39/39 cảnh báo pHash (0 cặp trùng theo người dùng). WP3–WP7 chưa triển khai.

## Review nghi trùng — WP2

[Biên bản WP2](./WP2_DUPLICATE_REVIEW_RESULTS.md): cả 39 cặp được đánh dấu không trùng, gồm 13 Train–Val, 20 Train–Test và 6 Val–Test. Không có cặp nào liên quan `real_`; không suy ra toàn bộ dataset không thể có leakage.

## WP1 — Đánh giá ảnh Real Test 28/28

[Báo cáo hoàn thành quan sát WP1](./WP1_REAL_REVIEW_COMPLETE.md): 22 confirmed và 6 ambiguous theo reviewer, nổi bật 5/6 Red Rust ambiguous và 6/6 Red Rust multiple symptoms. Đây là đánh giá hậu nghiệm, **chưa chứng minh nguyên nhân**, cũng chưa thay thế xác minh nhãn chuyên môn.
