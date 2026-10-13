# IMPROVED001 — DenseNet121 Scratch trên CashewData_Split (7)

**Loại:** exploratory dataset-quality audit, **1 seed**, 5 lớp. **Không** thay thế locked-Test benchmark V05.

- [BÁO CÁO ĐÁNH GIÁ CHI TIẾT (ANALYSIS.md)](./ANALYSIS.md)
- [Real Test — 9 ảnh dự đoán sai](./REAL_TEST_ERRORS.md)
- [Bảng chỉ số CSV](./metrics_summary.csv)
- [Biểu đồ Real vs Non-real](./figures/domain_accuracy.svg)
- [Biểu đồ Recall trên Real Test](./figures/real_recall.svg)

| Metric | Result |
|---|---:|
| Train / Val / Test | 4.825 / 1.395 / 687 |
| Test Accuracy | **93,01%** |
| Macro F1 | **92,50%** |
| Real Test Accuracy | **67,86% (19/28)** |
| Non-real Test Accuracy | **94,08% (620/659)** |
| SHA256 trùng byte | **0 nhóm** |
| pHash nghi gần trùng xuyên split | **39 cặp — chưa xác nhận leakage** |

**Lưu trữ:** Mô hình `.keras`, FULL ZIP và file dữ liệu ảnh không commit lên GitHub. Dữ liệu nguồn được định danh bằng SHA256 trong báo cáo.

## Kế hoạch xác minh nguyên nhân

Xem [kế hoạch điều tra WP0–WP7](../INVESTIGATION_PLAN.md) và [quy trình review ảnh Real/near-duplicate](../REVIEW_GUIDELINE.md). Chưa có kết quả ablation; các lý do background/resize hiện là giả thuyết.
