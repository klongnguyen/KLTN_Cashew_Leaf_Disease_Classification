# Audit tài liệu repository · 28/10/2026

## Phạm vi kiểm kê
Đã quét cây GitHub nhánh main: **1.479 mục**, bao gồm nội dung lịch sử Scratch/Transfer, notebooks, metrics, hình, ứng dụng FastAPI và YOLO Takes. 940 file ở training_results, 91 file ở Notebook, 53 file ở failure. Đã kiểm tra cấu hình ứng dụng và các tài liệu nguồn then chốt. **Không có nghĩa đã thực thi mọi notebook hay rà từng dòng cả 1.479 mục.**

## Phát hiện chính
- README chính trước đó có nhiều dòng tiến độ chồng nhau, mô tả WebApp đôi khi là Scratch EXP-009 thay vì Transfer EXP-014. Mã config.py xác nhận **EXP-014 seed 123**.
- README Application dùng đường dẫn Windows cá nhân, mô tả MIME/thứ tự xác suất không khớp source và ngầm hiểu chạy ngay sau clone. **Không có deployment_model.keras trong Git**, nên cần artifact ngoài Git.
- Các mốc V08/V09/V10/V11 và Holdout02–05 dùng tập ảnh thay đổi. Chưa có fresh blind 5-class field test. Một số EXP là **backfill README/CSV**, không có full source ZIP/figure/checkpoint.
- Có nhiều artifact legacy và notebook backup. Không xóa chỉ vì tên cũ; ví dụ hai checkpoint DenseNet121 archive legacy mỗi file khoảng **87,7 MiB**. Dọn sạch history/binary cần kế hoạch LFS hoặc kho artifact riêng, nếu xóa tùy tiện sẽ mất provenance.
- Tách phần điều hướng hiện hành khỏi kết quả lịch sử, thêm biểu đồ SVG và chỉ mục theo EXP; các ghi chú root cũ được lưu vào docs/research_notes.

## Quy tắc bảo toàn
Không thay đổi metric CSV, seed checkpoint, notebook huấn luyện, dataset manifest hoặc source archive. Chỉ tài liệu và chỉ mục được tái cấu trúc; EXP lịch sử vẫn xem được. Chỉ chuyển các ghi chú phân tích khỏi root khi đường dẫn nội bộ đã được chỉnh tương ứng. Việc dọn này **không** phải kiểm thử mô hình end-to-end.
