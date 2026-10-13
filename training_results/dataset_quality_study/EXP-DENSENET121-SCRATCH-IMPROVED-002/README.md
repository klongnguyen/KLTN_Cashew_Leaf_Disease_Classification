# IMPROVED002 — DenseNet121 Scratch trên CashewData_Split (8)

WP3: **Real Validation 35 ảnh**, chuyển từ V7 Train; Test V8 có 686 ảnh. Không nhập vào official benchmark V05.

**[Báo cáo phân tích đầy đủ](./ANALYSIS.md)**

| Metric | Result |
|---|---:|
| Train / Val / Test | 4.790 / 1.430 / 686 |
| Validation Accuracy | 93,01% |
| **Real Val Accuracy / Macro F1** | **65,71% (23/35) / 69,12%** |
| Non-real Val Accuracy | 93,69% |
| Test Accuracy / Macro F1 | 91,11% / 90,60% |
| Real Test Accuracy | 71,43% (20/28) |
| Best epoch | 27 |

[Metrics CSV](./metrics_summary.csv) · [12 lỗi Real Val để review](./REAL_VALIDATION_12_ERRORS.csv) · [Kế hoạch](../INVESTIGATION_PLAN.md)

**Giới hạn:** 1 seed, Test đã được review từ WP1; chưa được dùng để lựa chọn preprocessing. Group-level leakage chưa được xác minh tuyệt đối.
