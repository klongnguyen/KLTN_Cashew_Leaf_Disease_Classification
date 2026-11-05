# DenseNet121 + YOLO26s — EXP-034



Báo cáo này chỉ đánh giá pipeline kết hợp. Không chọn hoặc so sánh phương pháp khác.

Test: 97.1% (Wilson 95% 95.5%–98.1%).

Holdout 09: 96.0% (Wilson 95% 91.0%–98.3%).

Khoảng cách Test–Holdout: 1.0 điểm phần trăm.

C=3; nhiệt độ=1.0135; ngưỡng từ chối=0.5042; BOX_CONF=0.25.

Validation là ngoài fold theo nhóm pHash. Holdout là đánh giá mô tả đã được xem trong EXP-032, không phải blind benchmark.

Holdout không có ground truth box; chất lượng box chỉ đo trên YOLO Test.

Các bảng CSV và hình PNG nằm trong aggregate/. Hai ZIP chứa pipeline và báo cáo consistency.
