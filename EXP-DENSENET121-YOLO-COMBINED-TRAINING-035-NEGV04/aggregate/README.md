# DenseNet121 + YOLO26s (NEG_V04) — EXP-035



YOLO: EXP-Y26S-NEG-V04-001 trên CASHEW_YOLO_MASTER_FINAL_NEG_V04.zip (best.pt SHA-256 f9dc3fba1347b321…); DenseNet121: huấn luyện mới, công thức EXP-034.

Báo cáo này chỉ đánh giá pipeline kết hợp. Không chọn hoặc so sánh phương pháp khác.

Test: 97.1% (Wilson 95% 95.5%–98.1%).

Holdout 09: 94.4% (Wilson 95% 89.0%–97.3%).

Khoảng cách Test–Holdout: 2.6 điểm phần trăm.

C=1; nhiệt độ=0.9317; ngưỡng từ chối=0.4262; BOX_CONF=0.25.

Validation là ngoài fold theo nhóm pHash. Holdout là đánh giá mô tả đã được xem trong EXP-032, không phải blind benchmark.

Holdout không có ground truth box; chất lượng box chỉ đo trên YOLO Test.

Các bảng CSV và hình PNG nằm trong aggregate/. Hai ZIP chứa pipeline và báo cáo consistency.
