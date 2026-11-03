# Dataset audit — DenseNet121 EXP-001

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** Giữ đúng dataset, seed, config và phạm vi đánh giá gốc. Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../../README.md) · [Danh mục 95 README](../../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->


> **Chức năng: audit dữ liệu, không phải training benchmark hay Final Test.** Hồ sơ gồm manifest, phân bố ảnh, bảng nghi trùng gần và ảnh review. Không kết luận tất cả near duplicates là leakage; phải kiểm tra thủ công cùng lá/cây/phiên.

![Phân bố và dịch chuyển dữ liệu](distribution_and_shift.png)

![Ảnh nghi tương tự cần rà soát](near_pair_contact_sheet.png)

[Kiểm tra tổng hợp](audit_summary.json) · [Danh sách near duplicates](near_duplicate_pairs.csv) · [Ảnh review ứng viên](image_review_candidates.csv) · [Exact duplicate pairs](exact_duplicate_pairs.csv) · [Dataset manifest](dataset_manifest.csv)

[Xem mọi experiment](../../../docs/EXPERIMENT_INDEX.md) · [Dataset changelog](../../../DATASET_CHANGELOG.md)
