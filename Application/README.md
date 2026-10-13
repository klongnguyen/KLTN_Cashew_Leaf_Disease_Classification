# Cashew Leaf AI Diagnostic Web Application
### Chuẩn Đoán Bệnh Lá Cây Điều Bằng Học Máy & Thị Giác Máy Tính

Hệ thống web hoàn chỉnh phục vụ chuẩn đoán bệnh trên lá cây điều sử dụng mô hình học sâu **DenseNet121 phiên bản 009** (chọn phiên bản tối ưu nhất theo thực nghiệm là **Seed 3407**).

---

## 1. Thông Tin Mô Hình Triển Khai
- **Mô hình chính**: `DenseNet121 (Scratch v009)`
- **Checkpoint tốt nhất**: `Seed 3407` (Được chọn qua đánh giá trên tập kiểm định độc lập Validation):
  - **Validation Accuracy**: `95.63%`
  - **Validation Loss**: `0.1410` (Thấp nhất trong 5 seeds)
  - **Validation Macro F1**: `95.51%`
- **Kích thước đầu vào**: `224 × 224 × 3` (RGB)
- **Tiền xử lý**: Tích hợp sẵn trực tiếp trong đồ thị tính toán của mô hình (Rescaling & Normalization).
- **Đường dẫn mô hình**: `training_results/current_dataset/EXP-DENSENET121-SCRATCH-5SEEDS-009_FULL/EXP-DENSENET121-SCRATCH-5SEEDS-009/model_package/deployment_model.keras`

---

## 2. Các Tính Năng Đã Thực Hiện Theo Yêu Cầu

1. **Ràng buộc tải ảnh**:
   - Chỉ chấp nhận các định dạng tệp hình ảnh: `.png`, `.jpg`, `.jpeg`, `.webp`.
   - Giới hạn nghiêm ngặt: **Chỉ cho phép tải 1 ảnh duy nhất mỗi lần** (cả trên giao diện chọn file lẫn kéo thả).
   - Kiểm tra định dạng hợp lệ ở cả phía máy khách (Client-side) và máy chủ (FastAPI Server-side).

2. **Cấu trúc giao diện 2 cột chuẩn mực**:
   - **Cột Bên Trái (Kết quả chuẩn đoán)**:
     - **Khung ảnh đã tải lên**: Xem trước ảnh, kích thước ảnh gốc, định dạng và tính năng phóng to (lightbox).
     - **Kết luận chuẩn đoán**: Hiển thị nhãn bệnh có độ tin cậy cao nhất cùng mức độ nguy hiểm và khuyến nghị điều trị.
     - **Độ tin cậy của các loại nhãn bệnh**: Được sắp xếp **lần lượt từ thấp đến cao** (từ hạng #1 thấp nhất đến hạng #5 cao nhất), nhãn cao nhất được gắn huy hiệu **🏆 KẾT LUẬN**.
     - **Cẩm nang nông nghiệp**: Mô tả triệu chứng nhận diện và biện pháp phòng trừ cụ thể.
   - **Cột Bên Phải (Tải ảnh & Công cụ)**:
     - Khu vực kéo thả và duyệt ảnh từ thiết bị.
     - Thanh trạng thái tệp ảnh được chọn.
     - Bộ 3 mẫu thử nhanh (`Bệnh Thán Thư`, `Sâu Vẽ Bùa`, `Bệnh Rỉ Đỏ`) để kiểm thử ngay lập tức chỉ với 1 click chuột.
     - Bảng thông số chi tiết của mô hình DenseNet121 v009.

---

## 3. Hướng Dẫn Khởi Chạy Ứng Dụng

Máy chủ web FastAPI hiện đang được chạy tại địa chỉ:
👉 **`http://127.0.0.1:8000`**

Nếu cần khởi động lại máy chủ trong tương lai:
```powershell
cd d:\Study\KhoaLuanTotNghiep\Resource\Application
& "..\.venv\Scripts\python.exe" -m uvicorn server:app --host 127.0.0.1 --port 8000
```
Sau đó mở trình duyệt web bất kỳ và truy cập vào: **`http://localhost:8000`**.
