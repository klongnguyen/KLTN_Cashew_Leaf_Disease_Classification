# EXP-DENSENET121-TRANSFER-5SEEDS-028 — DenseNet121 Transfer, 256×256
> Hồ sơ được nhập từ ANALYSIS ZIP do tác giả cung cấp; đây là **kết quả đã huấn luyện**, không tái huấn luyện hoặc chọn lại checkpoint. Bản EXP đầy đủ có tên gốc `EXP-DENSENET121-TRANSFER-5SEEDS-028-256px-09415a54`.

## Provenance

- Dataset: **Cashew_dataV13 / CashewData_Split (13).zip**; Train **4972**, Validation **1452**, Test **685**.
- Dataset archive SHA256: `e1647b72e10af04275dc760545f4ef17c9f44343ae119495b89ba1e3cd954e04`
- Dataset manifest SHA256: `2cee0fb4818f93fba1dda43ab2e83615d33c5f7f3e4e15dc7c647436ff54b607`
- ANALYSIS ZIP gốc: `EXP-DENSENET121-TRANSFER-5SEEDS-028-256px-09415a54_ANALYSIS.zip`, SHA256 `53cb4de990ce310bd4b0512a3c9b8e7ed06637f5f0f0d37c649759937650eb9b`.
- Seed [42, 123, 2026, 3407, 7777]; **seed 42 được chọn bằng Validation Loss**, không bằng Test/Holdout.
- Holdout 09: **126 ảnh**, chỉ có 4 lớp ground-truth; `not_cashew_leaf` không có support.
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
| Validation (1452) | **97.287%** | 97.177% |
| Final Test (685) | **95.912%** | 95.523% |
| Holdout 09 (126) | **79.365%** | 78.549% (chỉ 4 lớp có support) |

| Seed | Val Acc | Test Acc | Holdout 09 Acc | Chính xác /118 ảnh chung |
|---:|---:|---:|---:|---:|
| 42 (**selected**) | 97.5895% | 95.6204% | 81.7460% | 98/118 |
| 123 | 97.0386% | 95.6204% | 78.5714% | 94/118 |
| 2026 | 97.1763% | 95.3285% | 81.7460% | 98/118 |
| 3407 | 97.6584% | 96.3504% | 80.9524% | 96/118 |
| 7777 | 96.9697% | 96.6423% | 73.8095% | 87/118 |

[Xem CSV 5 seed](./metrics_5seeds.csv) · [Báo cáo V12/V13 và 118 ảnh so khớp](../DENSENET121_V12_V13_EXP027_028_COMPARISON.md).

## Ràng buộc kết luận

- Final Test/Real Holdout đã được xem trong nhiều vòng của đồ án; dùng để mô tả hồi cứu, không chọn lại seed hoặc tăng chỉnh hyperparameter.
- **Cần đối chiếu theo ảnh trùng checksum** khi so Holdout 07 vs Holdout 09; số ảnh và một số nội dung khác nhau.
- 51 cặp nghi gần trùng pHash qua split là các trường hợp **cần kiểm tra thủ công**, không mặc định là 51 lỗi leakage.
- Không tự động thay thế mô hình triển khai từ hiệu năng Holdout lịch sử.
