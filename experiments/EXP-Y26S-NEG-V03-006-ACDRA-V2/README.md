# EXP-006 — ACDRA v2 cho vùng bệnh hiển thị

V006 chỉ hậu xử lý các bbox YOLO26s V005. Không đổi checkpoint, detection, GT, split hay nhãn; không chạy DenseNet. Biến thể A gọi nguyên hàm V005 để đối chứng. B dùng khoảng cách theo **kích thước lesion nhỏ hơn** trong cặp (`min(s_i,s_j)`), tránh một box lớn kéo một đốm nhỏ từ xa. C xếp cặp lân cận theo fill ratio; giới hạn 16 láng giềng gần nhất để tránh O(n³). Mỗi box raw thuộc đúng một region cùng class; các class khác nhau không bị gộp.

## Nguồn cố định

| Tệp | SHA-256 |
|---|---|
| `source/EXP-Y26S-NEG-V03-005_FULL.zip` | `990df19b094cc14d6e5706183616245dda83d21fac3b52c20f9a2a3f5488c845` |
| `source/EXP-Y26S-NEG-V03-005_ANALYSIS.zip` | `38ed98c637fd822f0ec534aa28a0954efce301e6d50a8802de3d80ef9ef96859` |
| `source/CASHEW_YOLO_MASTER_FINAL_NEG_V03.zip` | `312870474cfbab8c3fc77db7b19e821c1edf34a0787cb45592f00bde841b4dbe` |
| `best.pt` bên trong FULL ZIP | `12f5717ff7e97c6ffcc3b2ab8c66b869bc8065995507dac3a5dbd95ce7714d19` |

V005 dùng Ultralytics 8.4.198, seed 42. Raw Validation đã được suy luận với `conf=0.2`, NMS `iou=0.7`, `max_det=500`, `imgsz=640`. V006 chỉ dùng raw JSONL này; không thể khảo sát `conf=0.10/0.15` nếu không suy luận lại. Dataset ZIP giữ nguyên, được đọc trực tiếp để lấy kích thước ảnh và GT. Mapping class đọc từ `data_fixed.yaml` trong FULL ZIP.

## Chạy lại từ thư mục gốc repo

```powershell
# Tái hiện V005 và tính lại metric theo GT
& '.venv\Scripts\python.exe' scripts\run_acdra_v2.py baseline

# Quét thô B/C trên Validation, lưu mọi trial, Pareto và ba ứng viên
& '.venv\Scripts\python.exe' -u scripts\tune_acdra_v2_on_val.py

# Sinh ảnh Original → Raw → ACDRA và thống kê confidence/kích thước vùng
& '.venv\Scripts\python.exe' scripts\build_acdra_v2_gallery.py

# Tổng hợp báo cáo Validation (Bảng A–E)
& '.venv\Scripts\python.exe' scripts\write_acdra_v2_report.py

# Unit test
& '.venv\Scripts\python.exe' -m pytest -q tests\test_adaptive_regions_v2.py

# Chỉ sau khi chủ dự án chọn ứng viên từ số liệu và ảnh (ví dụ: balanced):
& '.venv\Scripts\python.exe' scripts\lock_acdra_v2.py --approved-point balanced
# Chạy Test đúng một lần sau khi đã khóa:
& '.venv_yolo_eval\Scripts\python.exe' scripts\evaluate_acdra_v2_locked.py
```

`validation_tuning.csv` ghi từng cấu hình và metric tổng/slice. Ba ứng viên nằm trong `operating_points.yaml` và `reports/EXP-006/operating_points.md`. `config_locked.yaml` được tạo sau khi chủ dự án chọn. Script Test từ chối chạy khi chưa có khóa, và từ chối chạy lại sau khi có `test_evaluation_completed.json`.

## Trạng thái đã chốt

Chủ dự án đã duyệt `balanced` (`B_e6e7d65f`). `config_locked.yaml` có SHA-256 `458fbb0a2a77717be386d0a783a5231bee3314f8c129a12400f879c2fe3efe21`. Test đã chạy đúng một lần trong 35,05 giây trên 412 ảnh, 0 ảnh âm: 19.926 raw box → 13.768 region, giảm 30,90%; ảnh ≥100 raw box giảm 35,73%; `lost_tp=0`. Excess area theo GT tăng từ 28,49% raw lên 37,98% region. Dấu hoàn tất tại `reports/EXP-006/test_evaluation_completed.json`; lệnh Test và lệnh tuning trong khối trên giờ sẽ bị từ chối để tránh chạy lại sau khóa. Bảng Test và các lát cắt nằm trong `reports/EXP-006/EXP006_FINAL_REPORT.md`. `test_summary_complete.json` bổ sung lát cắt class Test từ `per_image.csv` đã lưu, không chạy lại suy luận hay gộp.

## Môi trường hiện tại

Python 3.11.15; NumPy 2.4.6; Pillow 12.3.0; PyYAML 6.0.3; scikit-learn 1.9.1; Matplotlib 3.11.2; pytest 9.2.0 trong `.venv`. Môi trường `.venv_yolo_eval` đã đồng bộ Ultralytics 8.4.198 để khớp V005. Thời gian gộp p50/p95/max và ảnh chậm nhất được ghi trong summary của mỗi trial. Đây là metric vùng hiển thị, không phải AP/mAP detector.
