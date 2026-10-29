# Training Results — Kho dữ liệu thực nghiệm

![V08 Test và Holdout](../docs/assets/benchmark_v08_test_holdout.svg)

Mỗi thư mục thí nghiệm có **ID, dataset snapshot, config và cách đánh giá riêng**. Không gộp tất cả thành bảng xếp hạng; đặc biệt giữ ranh giới V08 với V09–V11, giữa Test và Historical Holdout.

| Khu vực | Nội dung | Xem |
|---|---|---|
| current_dataset | EXP classification Scratch/Transfer/CBAM/ViT-XGBoost, các snapshot V04–V11 | [Current dataset](current_dataset/README.md) |
| dataset_quality_study | Kiểm tra nhãn, real-prefix, ảnh trùng, domain shift | [Dataset quality](dataset_quality_study/README.md) |
| archive_legacy_dataset | Checkpoint / báo cáo gốc lịch sử | [Legacy archive](archive_legacy_dataset/README.md) |
| failure | YOLO26 Take chưa chọn final | [Detection archive](../failure/YOLO26/README.md) |

## Mốc đánh giá quan trọng
- **V08 DenseNet121 Transfer 014 + đánh giá 015/016:** 5-seed Test Accuracy **95,13 ± 0,51%**; Historical Holdout 48 ảnh **80,83 ± 0,93%**.
- **V11 DenseNet121 024+025:** Test **96,09 ± 0,42%**, Holdout05 **80,00 ± 1,56%**.
- **V11 DenseNet121 026:** Test **96,23 ± 0,56%**, Holdout05 **74,83 ± 2,24%**.
- **CBAM V08:** ba ablation mới chỉ Validation; không có Test/Holdout.

![DenseNet121 V11](../docs/assets/v11_retrospective.svg)

Không dùng số Test/Holdout đã tiếp xúc để chọn seed/hyperparameter. Đối với EXP chỉ backfill README/CSV từ ZIP ANALYSIS ngoài Git, không quảng bá là đã có đầy đủ checkpoint/figure/notebook. Không xóa artifact nguồn chỉ vì cũ: cấu trúc phải hỗ trợ tái lập và truy xuất lỗi.

[Danh mục toàn bộ experiment](../docs/EXPERIMENT_INDEX.md) · [Protocol](../docs/RESEARCH_PROTOCOL.md) · [Trang chính](../README.md)
