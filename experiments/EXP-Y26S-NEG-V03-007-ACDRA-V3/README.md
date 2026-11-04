# EXP-007 — ACDRA v3 trên YOLO26s

V007 chỉ xử lý hậu kỳ bbox của checkpoint YOLO26s V005. Biến thể D hợp nhất các region cùng predicted class có IoA đủ cao khi vùng sau gộp vẫn đạt giới hạn diện tích, độ trải và fill ratio. Không đổi nhãn, GT, split, checkpoint hoặc raw detections. Không tạo crop và không huấn luyện DenseNet.

## Nguồn cố định

Các ZIP nguồn nằm trong `experiments/EXP-Y26S-NEG-V03-006-ACDRA-V2/source/` và được `scripts/run_acdra_v2.py` kiểm SHA-256 trước khi đọc. Checkpoint `best.pt`: `12f5717ff7e97c6ffcc3b2ab8c66b869bc8065995507dac3a5dbd95ce7714d19`; dataset ZIP: `312870474cfbab8c3fc77db7b19e821c1edf34a0787cb45592f00bde841b4dbe`; V005 FULL ZIP: `990df19b094cc14d6e5706183616245dda83d21fac3b52c20f9a2a3f5488c845`. Class mapping lấy từ `data_fixed.yaml` trong V005 FULL ZIP. Raw Validation dùng `conf=0,2`, NMS IoU `0,7`, `max_det=500`, `imgsz=640`, seed `42`; không chạy lại YOLO. `validation_raw_detections.jsonl` độc lập có SHA-256 `8a5732eecd9e19f5c1d2be63618ac0a1da1a39ae7efe4b90b61f7522e2ccf6ed`.

## Chạy từ gốc repo

```powershell
# Tái hiện năm trial V006 trước V007; dừng nếu lệch
& '.venv\Scripts\python.exe' scripts\reproduce_acdra_v3_references.py

# Quét Validation; CSV được ghi sau từng trial để có thể tiếp tục
& '.venv\Scripts\python.exe' -u scripts\tune_acdra_v3_on_val.py

# Pareto và ba ứng viên tạm để chủ dự án duyệt
& '.venv\Scripts\python.exe' scripts\select_acdra_v3_points.py

# Chấm mức sẵn sàng crop và confidence trên Validation, không tạo crop
& '.venv\Scripts\python.exe' scripts\evaluate_crop_readiness.py

# Gallery Original → Raw → ACDRA v3, gồm nhóm thưa bị đổi và overlap nặng
& '.venv\Scripts\python.exe' scripts\build_acdra_v3_gallery.py

# Tổng hợp báo cáo
& '.venv\Scripts\python.exe' scripts\write_acdra_v3_report.py

# Unit test V006 + V007
& '.venv\Scripts\python.exe' -m pytest -q tests\test_adaptive_regions_v2.py tests\test_adaptive_regions_v3.py
```

Chỉ khi chủ dự án đã xem số liệu và ảnh rồi chọn một điểm, khóa bằng `scripts/lock_acdra_v3.py --approved-point <conservative|balanced|aggressive>`. Script Test `scripts/evaluate_acdra_v3_locked.py` từ chối chạy khi chưa có khóa và chỉ cho một lần đánh giá. **Không chạy Test trong giai đoạn chọn tham số.** Test đã được dùng ở V006 nên mọi kết quả V007 sau này phải mang nhãn “đánh giá nội bộ, không còn là Test nguyên vẹn”.

`config_locked_v007.yaml` chỉ xuất hiện sau khi chủ dự án phê duyệt; không tạo file khóa từ chính kết quả tuning. Tương tự, không push lên GitHub khi chưa có yêu cầu mới.

Hash SHA-256 cấu hình ứng viên (config JSON sắp xếp khóa): conservative `c2cf6ddd018e3472e29c60968f8adcd0a841fbae638c9d71b80734a8a60a88a6`; balanced `e95d60ebf168140db10545a31369ada3754f8d70bc07718df69698085637b51f`; aggressive `1c8106e16404002bee258a76ec37fceb4a9cb6d76f0d63a562def19ff67469d0`. Bảng đầy đủ cho 18 trial nằm trong `reports/EXP-007/validation_tuning_v007.csv`.

## Môi trường

Python 3.11.15, NumPy 2.4.6, Pillow 12.3.0, PyYAML 6.0.3, scikit-learn 1.9.1, Matplotlib 3.11.2, pytest 9.2.0. Latency p50/p95/max từng trial nằm trong CSV; thời gian chạy lại năm mốc V006 nằm trong `reports/EXP-007/baseline_reproduction.json`. Thời gian thực của toàn bộ lần quét tuning không được ghi tự động (`N/A`), và latency phụ thuộc máy hiện tại. V006 và V005 không bị sửa.
