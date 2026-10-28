# Application — AI Cashew Leaf Diagnosis (FastAPI)

![Kiến trúc](../docs/assets/research_pipeline.svg)

Giao diện và API phân loại lá điều **5 lớp**. Theo [config.py](config.py), mô hình đã cấu hình là **DenseNet121 Transfer EXP-014 seed 123**, **không phải** các EXP mới 024–026 trong tài liệu nghiên cứu.

## Thông tin model theo mã nguồn

| Thành phần | Giá trị |
|---|---|
| Model | DenseNet121 Transfer v014, seed 123 |
| Input | RGB 224 × 224, BILINEAR, float32 pixel 0–255 |
| Preprocessing | Đã đóng gói bên trong model, không chia 255 lần nữa |
| Validation seed 123 | Accuracy 97,27%, Macro F1 97,14%, Val Loss 0,0925 |
| Final Test V08 seed 123 | 653/686 = 95,19%, Macro F1 94,84% |
| Lớp | anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf |

**Binary checkpoint không có trong Git.** config.py cần ba file ở thư mục sau, trong đó label_map.json và deployment_metadata.json có trong repository nhưng **deployment_model.keras phải lấy đúng từ source package ngoài Git**:

~~~text
training_results/current_dataset/
└── EXP-DENSENET121-TRANSFER-5SEEDS-014_FULL/
    └── EXP-DENSENET121-TRANSFER-5SEEDS-014/
        └── model_package/
            ├── deployment_model.keras         # KHÔNG có trong Git
            ├── label_map.json
            └── deployment_metadata.json
~~~

Không thay checkpoint bằng tên khác nếu chưa xác nhận version, label order, input preprocess và serialization parity. Khi thiếu binary này, server có thể lỗi ngay khi import model_service.

## Cách khởi chạy không phụ thuộc đường dẫn máy cá nhân

Tại thư mục gốc repo, dùng Python và TensorFlow tương thích checkpoint; cần có **binary đã phục hồi**:

~~~bash
python -m venv .venv
# Kích hoạt môi trường ảo tùy Windows/macOS/Linux
python -m pip install -r Application/requirements.txt
cd Application
python -m uvicorn server:app --host 127.0.0.1 --port 8000
~~~

Truy cập **http://127.0.0.1:8000/**. Cài đặt trong môi trường tách biệt được khuyến nghị vì version Keras/TensorFlow có thể ảnh hưởng load model.

## API và kiểm tra đầu vào

| Route | Vai trò |
|---|---|
| GET /api/health | Kiểm tra dịch vụ |
| GET /api/model-info | Thông tin mô hình |
| GET /api/samples | Danh mục ảnh thử |
| POST /api/predict | Một ảnh upload |
| POST /api/predict-sample/{sample_name} | Dự đoán ảnh mẫu |

Theo [server.py](server.py), server cho phép JPG/JPEG, PNG, WEBP, BMP, TIFF, tối đa **25 MiB**. Giao diện có thể hạn chế định dạng chặt hơn server. Theo [model_service.py](model_service.py), danh sách xác suất được sắp **giảm dần** (xác suất lớn nhất trước).

## Hình ảnh thử nghiệm

| Anthracnose | Leaf Miner | Red Rust |
|:--:|:--:|:--:|
| ![Anthracnose](static/samples/ath_01.png) | ![Leaf Miner](static/samples/mine01.png) | ![Red Rust](static/samples/red01.png) |

Kết quả chỉ hỗ trợ nhận diện từ ảnh, không thay chẩn đoán chuyên môn hoặc khuyến cáo phun thuốc tại thực địa. Nếu thay model WebApp, cần kiểm tra checkpoint SHA, preprocess, nhãn, inference parity, latency và phê duyệt theo quy tắc Validation; không dùng Test/Holdout lịch sử để chọn.

[README chính](../README.md) · [EXP-014](../training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md) · [EXP-016 Test](../training_results/current_dataset/EXP-DENSENET121-TRANSFER-5SEEDS-016/README.md)
