# IMPROVED001 — DenseNet121 Scratch trên CashewData_Split (7)
> **Đồng bộ điều hướng 28/10/2026.** NHÁNH KIỂM ĐỊNH DỮ LIỆU — kết quả khác benchmark chính thức từng V08–V11. [Tổng quan](../../../README.md) · [Danh mục](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


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
| pHash nghi gần trùng xuyên split | **39 cặp, 0 trùng theo review thủ công WP2** |

**Lưu trữ:** Mô hình `.keras`, FULL ZIP và file dữ liệu ảnh không commit lên GitHub. Dữ liệu nguồn được định danh bằng SHA256 trong báo cáo.

## Kế hoạch xác minh nguyên nhân

Xem [kế hoạch điều tra WP0–WP7](../INVESTIGATION_PLAN.md) và [quy trình review ảnh Real/near-duplicate](../REVIEW_GUIDELINE.md). Chưa có kết quả ablation; các lý do background/resize hiện là giả thuyết.

### Review WP2

[39 cặp nghi gần trùng đã được người dùng đánh dấu không trùng](../WP2_DUPLICATE_REVIEW_RESULTS.md); kết quả không loại trừ khả năng leakage ở cấp lá/cây/phiên chụp của ảnh thực địa.

### Quan sát 28 ảnh Real Test (WP1)

[WP1 manual review 28/28 — kết quả và giới hạn](../WP1_REAL_REVIEW_COMPLETE.md): 6 nhãn ambiguous cần xác minh; chưa có ablation hoặc thay đổi kết quả Test.
