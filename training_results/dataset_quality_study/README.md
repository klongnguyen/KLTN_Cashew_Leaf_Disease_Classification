# Dataset quality study — Kaggle base vs improved dataset

Nhánh nghiên cứu độc lập nhằm kiểm tra dữ liệu Kaggle gốc và dữ liệu cải tiến, **không trộn với benchmark `Cashew_dataV05` hiện tại**.

| Experiment | Dataset | Classes | Test Accuracy | Macro F1 | Ghi chú |
|---|---|---:|---:|---:|---|
| BASE001 | CashewData_base (Kaggle) | 4 | 94,50% | 94,53% | 5.360 ảnh; 1 nhóm file trùng bị xung đột nhãn |
| [IMPROVED001](./EXP-DENSENET121-SCRATCH-IMPROVED-001/) | CashewData_Split (7) | 5 | **93,01%** | **92,50%** | 6.907 ảnh; Real Test 67,86% (19/28) |

**Không so sánh trực tiếp hai Accuracy:** khác số lớp, Test, phân phối ảnh. Mục tiêu tiếp theo: xác thực nguồn/split và cùng Real Holdout 4 lớp, độc lập với cả hai tập Train/Val.

Báo cáo IMPROVED001: [ANALYSIS.md](./EXP-DENSENET121-SCRATCH-IMPROVED-001/ANALYSIS.md).

## Kế hoạch điều tra tiếp theo

**Kế hoạch điều tra tiếp theo (WP0–WP7):** [INVESTIGATION_PLAN.md](./INVESTIGATION_PLAN.md). Kèm [quy trình review ảnh](./REVIEW_GUIDELINE.md) và [mẫu ghi nhận ảnh Real](./templates/real_image_review.csv) / [nghi trùng](./templates/duplicate_pair_review.csv). **Trạng thái: Planned, chưa chạy các bước xác minh.**
