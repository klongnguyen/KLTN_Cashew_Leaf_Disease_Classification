# Training Results

Thư mục này quản lý các kết quả huấn luyện classification theo **phiên bản dataset** để tránh so sánh sai giữa các experiment dùng dữ liệu khác nhau.

## Current Dataset — Cashew_dataV04

Dataset hiện tại có **6,911 ảnh / 5 lớp**, được chia thành **4,822 Train / 1,402 Validation / 687 Test**. Phiên bản V04 được tạo sau khi tiếp tục loại ảnh mờ/chất lượng thấp và xử lý các trường hợp có nguy cơ data leakage.

| Experiment | Model | Seeds | Test Accuracy | Macro F1 | Params | Status |
|---|---|---:|---:|---:|---:|---|
| [`EXP-DENSENET121-SCRATCH-5SEEDS-003`](./current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-003/) | DenseNet121 Scratch | 5 | **91.82 ± 0.86%** | **91.37 ± 0.79%** | 7.57M | ✅ V04 baseline |
| [`EXP-RESNET50-SCRATCH-5SEEDS-003`](./current_dataset/EXP-RESNET50-SCRATCH-5SEEDS-003/) | ResNet50 Scratch | 5 | **90.63 ± 2.11%** | **90.17 ± 2.04%** | 24.64M | ✅ V04 baseline |
| [`EXP-VIT-SCRATCH-5SEEDS-003`](./current_dataset/EXP-VIT-SCRATCH-5SEEDS-003/) | ViT Scratch | 5 | **85.30 ± 1.68%** | **84.41 ± 1.71%** | **0.35M** | ✅ V04 baseline |

### V04 benchmark snapshot

- DenseNet121 currently has the highest mean Test Accuracy and Macro F1 and the lowest variance among the two CNN baselines.
- ResNet50 remains competitive but is more sensitive to seed; seeds `123` and `2026` reduce its 5-seed mean and increase standard deviation.
- ViT is by far the smallest model, but its classification performance remains below both CNN baselines.
- `anthracnose` remains the most difficult class across the V04 experiments.

➡️ Xem dataset, figures và phân tích chi tiết tại [`current_dataset/README.md`](./current_dataset/README.md).

## Dataset versioning

Các experiment `*-002` dùng `Cashew_dataV03` được giữ lại làm lịch sử thực nghiệm. Do Train/Validation/Test đã thay đổi sau quá trình làm sạch, **không so sánh trực tiếp metric V03 với V04 như một controlled model comparison**.

## Protocol classification

- Fixed Train / Validation / Test split trong từng dataset version.
- Seeds: `42, 123, 2026, 3407, 7777`.
- Validation dùng cho checkpoint, EarlyStopping, LR scheduling và deployment-seed selection.
- Test Set không dùng để tuning hoặc chọn seed.
- Báo cáo **Mean ± sample Standard Deviation (`ddof=1`)**.

## YOLO26 Development Archive

Các experiment object detection hiện được quản lý tại [`../failure/YOLO26/README.md`](../failure/YOLO26/README.md) cho đến khi detector cuối được khóa.
