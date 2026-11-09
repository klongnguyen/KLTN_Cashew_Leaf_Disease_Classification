# Application — Nhận diện bệnh lá điều

Ứng dụng FastAPI nhận ảnh lá điều, đánh dấu vùng nghi bệnh và hiển thị kết luận cùng bước chăm sóc tiếp theo. Giao diện tại `http://127.0.0.1:8080/` chỉ hiện thông tin chính trước; chi tiết từng vùng, hướng dẫn đầy đủ và thông số mô hình nằm trong các mục mở rộng.

## Mô hình đang dùng

Theo [config.py](config.py), `MODEL_ENGINE = "exp034"` là tên đường chạy lịch sử; **`ACTIVE_BUNDLE = "035"`** chọn bundle `EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04` hiện dùng. Bundle này gồm DenseNet121 ×5 seed và YOLO26s NEG-V04. Stacker F3 được tính để tham khảo; `FINAL_DECISION = "boxes"` quyết định các bệnh từ những vùng được khoanh và xác minh.

Luồng nhận diện:

1. DenseNet trên toàn ảnh kiểm tra ảnh có phù hợp là lá điều không. Ảnh bị từ chối khi lớp cao nhất là `not_cashew_leaf` hoặc khoảng cách kNN vượt ngưỡng cứng **0,24**. Vùng kNN nghi ngờ chỉ phát cảnh báo; Energy không dùng để từ chối theo cấu hình hiện tại.
2. YOLO26s đề xuất các khung bệnh. DenseNet phân loại từng vùng cắt riêng (cạnh tối thiểu 64 px, ngữ cảnh 1,25 lần cạnh dài khung). Chỉ thay nhãn YOLO khi DenseNet đủ tin cậy theo `LABEL_SWITCH_MIN_CONF = 0.95`.
3. Các bệnh có khung sau xác minh được đưa vào kết luận. Nếu không còn khung, DenseNet toàn lá quyết định lá khỏe hoặc yêu cầu kiểm tra chuyên môn khi vẫn nghi có bệnh.

Điểm tin cậy cạnh tên bệnh lấy từ vùng bệnh rõ nhất khi có khung; khi không có khung, điểm này lấy từ phân loại toàn ảnh. Tỷ lệ diện tích theo khung chỉ là **ước tính vùng nghi bệnh**, không phải diện tích mô lá hoại tử được đo chính xác.

Các chỉ số Test/Holdout trong `EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/aggregate/` mô tả đánh giá offline của Stacker F3; không nên diễn giải trực tiếp thành độ chính xác của luật `FINAL_DECISION = "boxes"` trên web. Ngưỡng OOD đã được lựa chọn sau khi xem ảnh Holdout/thực địa, nên Holdout không còn là benchmark mù cho cổng này.

## Tệp mô hình và cách chạy

Các tệp `.keras` và `.pt` không được lưu trong Git. Trước khi chạy, khôi phục đúng các binary theo `pipeline_bundle/artifact_sha256.csv` của EXP-035:

```text
EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/pipeline_bundle/
├── best.pt
├── seed_42/deployment_model.keras
├── seed_123/deployment_model.keras
├── seed_2026/deployment_model.keras
├── seed_3407/deployment_model.keras
└── seed_7777/deployment_model.keras
```

Ứng dụng còn cần `Application/ood/exp035/ood_reference.npz` và `ood_calibration.json`. Không thay checkpoint bằng tệp khác khi chưa kiểm tra SHA-256, thứ tự nhãn, tiền xử lý và kết quả suy luận tương đương. Theo cấu hình, bundle được kiểm SHA-256 khi khởi tạo.

Từ thư mục gốc repository:

```bash
python -m venv .venv
# Kích hoạt môi trường ảo phù hợp với hệ điều hành.
python -m pip install -r Application/requirements.txt
cd Application
python -m uvicorn server:app --host 127.0.0.1 --port 8080
```

Mở `http://127.0.0.1:8080/`. Nếu muốn dùng lại EXP-034 để đối chiếu, đổi `ACTIVE_BUNDLE` trong [config.py](config.py) và cung cấp đầy đủ bundle cùng bộ OOD tương ứng.

## API

| Route | Vai trò |
|---|---|
| `GET /api/health` | Trạng thái dịch vụ |
| `GET /api/model-info` | Bundle đang dùng |
| `GET /api/samples` | Danh sách ảnh mẫu |
| `POST /api/predict` | Nhận diện một ảnh tải lên |
| `POST /api/predict-sample/{sample_name}` | Nhận diện ảnh mẫu |

Server chấp nhận JPG/JPEG, PNG, WEBP, BMP và TIFF, tối đa 25 MiB; giao diện tải ảnh giới hạn JPG/JPEG, PNG và WEBP. Ảnh có cạnh ngắn dưới 64 px bị từ chối, dưới 224 px có cảnh báo vì có thể bỏ sót vùng bệnh nhỏ.

Kết quả chỉ hỗ trợ nhận diện từ ảnh. Khi kết quả không chắc chắn hoặc ảnh không phù hợp, hãy chụp lại ảnh và tham khảo cán bộ chuyên môn trước khi dùng thuốc bảo vệ thực vật.

[README chính](../README.md) · [Kết quả EXP-035](../EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/aggregate/README.md)
