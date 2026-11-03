# EXP-DENSENET121-TRANSFER-5SEEDS-027 — DenseNet121 Transfer, 256×256

<!-- docs-sync-2026-11-04:start -->
> **Đồng bộ README 04/11/2026:** Giữ đúng dataset, seed, config và phạm vi đánh giá gốc. Không phát sinh huấn luyện hoặc Test mới; kết quả gốc được giữ nguyên. [Trang chủ](../../../README.md) · [Danh mục 95 README](../../../docs/EXPERIMENT_INDEX.md) · [Tiến độ](../../../docs/RESEARCH_STATUS_2026-11-04.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).
<!-- docs-sync-2026-11-04:end -->

> Hồ sơ được nhập từ ANALYSIS ZIP do tác giả cung cấp; đây là **kết quả đã huấn luyện**, không tái huấn luyện hoặc chọn lại checkpoint. Bản EXP đầy đủ có tên gốc `EXP-DENSENET121-TRANSFER-5SEEDS-027-256px-3e196f0b`.

## Provenance

- Dataset: **Cashew_dataV12 / CashewData_Split (12).zip**; Train **4975**, Validation **1452**, Test **685**.
- Dataset archive SHA256: `e357493562611ae4bbd196bfb6a43659ae9dd6dedaccff231288fa69a8721b9d`
- Dataset manifest SHA256: `086e710127ebf751f5bbbde0b5f7092e8200f46c7c99ef71f8fd64be60e25c2c`
- ANALYSIS ZIP gốc: `EXP-DENSENET121-TRANSFER-5SEEDS-027-256px-3e196f0b_ANALYSIS.zip`, SHA256 `1b9cbe77f35962cbdb5597e54c27e56ef1bb5be195f28d75d4d36ea65aa3e520`.
- Seed [42, 123, 2026, 3407, 7777]; **seed 7777 được chọn bằng Validation Loss**, không bằng Test/Holdout.
- Holdout 06: **125 ảnh**, chỉ có 4 lớp ground-truth; `not_cashew_leaf` không có support.
- **GitHub backfill:** CSV tổng hợp và tài liệu; ZIP gốc, model checkpoint và toàn bộ PNG/predictions gốc chưa được push. Muốn chạy inference cần lấy trọng số từ nơi lưu gốc.

## Kiến trúc và cấu hình

| Thành phần | Giá trị |
|---|---|
| Backbone | DenseNet121, ImageNet pretrained, include_top=False, GAP |
| Input | RGB float32 0–255, resize bilinear 256×256 |
| Preprocess | Trong mô hình, chia 255 rồi chuẩn hóa theo ImageNet mean/std |
| Head | Dropout 0.4 + Dense Softmax 5 classes |
| Augmentation | HorizontalFlip; Rotation 0.08; Zoom 0.10; Contrast 0.10; Brightness 0.08 |
| Optimizer | AdamW, weight decay 1e-4; batch 16 |
| Stage 1 | Frozen backbone, LR 3e-4, max 12 epoch, ES patience 4 |
| Stage 2 | `conv5_*` non-BN only, LR 1e-5, max 30 epoch, ES patience 8 |
| Checkpoint | Validation Loss tối thiểu, tie-break theo Validation macro-F1 |

## Metrics — trung bình 5 seed

| Split | Accuracy | Macro F1 |
|---|---:|---:|
| Validation (1452) | **97.245%** | 97.125% |
| Final Test (685) | **95.912%** | 95.510% |
| Holdout 06 (125) | **76.320%** | 71.045% (chỉ 4 lớp có support) |

| Seed | Val Acc | Test Acc | Holdout 06 Acc | Chính xác /118 ảnh chung |
|---:|---:|---:|---:|---:|
| 42 | 96.8320% | 95.9124% | 72.8000% | 91/118 |
| 123 | 97.2452% | 96.0584% | 76.0000% | 95/118 |
| 2026 | 97.3140% | 95.7664% | 78.4000% | 97/118 |
| 3407 | 97.4518% | 96.0584% | 78.4000% | 96/118 |
| 7777 (**selected**) | 97.3829% | 95.7664% | 76.0000% | 94/118 |

[Xem CSV 5 seed](./metrics_5seeds.csv) · [Báo cáo V12/V13 và 118 ảnh so khớp](../DENSENET121_V12_V13_EXP027_028_COMPARISON.md).

## Ràng buộc kết luận

- Final Test/Real Holdout đã được xem trong nhiều vòng của đồ án; dùng để mô tả hồi cứu, không chọn lại seed hoặc tăng chỉnh hyperparameter.
- **Cần đối chiếu theo ảnh trùng checksum** khi so Holdout 07 vs Holdout 09; số ảnh và một số nội dung khác nhau.
- 51 cặp nghi gần trùng pHash qua split là các trường hợp **cần kiểm tra thủ công**, không mặc định là 51 lỗi leakage.
- Không tự động thay thế mô hình triển khai từ hiệu năng Holdout lịch sử.
