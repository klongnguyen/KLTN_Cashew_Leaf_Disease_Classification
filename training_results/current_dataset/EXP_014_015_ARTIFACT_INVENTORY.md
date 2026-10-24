# Artifact inventory — EXP-014 / EXP-015 (2026-10-19)

> **Cập nhật 22/10/2026:** Bảng bên dưới là ảnh chụp trạng thái 014/015 tại thời điểm lập hồ sơ. [EXP-016](./EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md) đã hoàn tất Final Test V08 686 ảnh × 5 checkpoint EXP-014, không retrain.

Cơ sở dữ liệu: hai **ANALYSIS ZIP do tác giả gửi**. Repo ưu tiên lưu **báo cáo kết quả và bảng machine-readable** có kiểm chứng để dễ review/diff mà không làm phình history bởi binary checkpoint.

| Nhóm | EXP-014 | EXP-015 |
|---|---|---|
| Report README | Đã lưu | Đã lưu |
| Aggregate 5-seed CSV | Validation + subgroup real_ | Real Holdout + per-class + comparison |
| Config + traceability | experiment_config, validation lock, audit | evaluation_config, execution status, holdout audit |
| Confusion matrix | Có trong original ANALYSIS ZIP, chưa commit PNG gốc | 5 CSV per-seed + hình so sánh SVG tái lập |
| Per-image predictions | Có trong original ANALYSIS ZIP, **chưa commit** | Có trong original ANALYSIS ZIP, **chưa commit** |
| Model binary / checkpoints | **Không có trong ANALYSIS ZIP**; cần FULL ZIP | **Không có binary trong ANALYSIS ZIP**, reused EXP-014 |
| Notebook huấn luyện / chạy 015 | Có notebook 014 trong ZIP, **chưa commit** | Notebook chạy EXP-015 không nằm trong các file được commit tại đây |
| Final Test V08 | **PENDING** | **NOT_RUN** |

**Quan trọng:** các file JSON metadata được xuất sang repo dưới dạng snapshot giữ giá trị từ nguồn; **không nên dùng checksum của file tái-serialize trên repo làm thay thế SHA-256 của bản gốc**. SHA-256 chứa trong metadata chỉ tham chiếu đến **file gốc trong ANALYSIS/FULL archive**, không xác minh được binary checkpoint nếu thiếu FULL ZIP. Cần giữ archive gốc trong kho lưu trữ ngoài GitHub, rồi kiểm tra hash trước khi công bố reproduction.

**Data interpretation:** V08 chứa 51 pHash candidates chưa xác minh thủ công; Holdout 49 → 48 ảnh sau khi loại 1 exact Test duplicate; không có true `not_cashew_leaf`, từng được phân tích trong EXP-013. Đây không phải blind holdout 5-class. Các chỉ số Mean ± SD dùng sample SD trên 5 seeds, không phải confidence interval của ảnh ngoài thực tế.

**Web deployment:** phần `Application/` đang mô tả dùng EXP-009 seed 3407, không tự chuyển sang seed 123.

Liên kết: [014](./EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md) · [015](./EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md) · [Progress](./EXPERIMENT_PROGRESS_2026_10.md).
