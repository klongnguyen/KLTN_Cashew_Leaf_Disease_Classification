# Báo cáo EXP-006 — ACDRA v2 (Validation và Test khóa)

## Phạm vi và nguồn

Chỉ gộp bbox cùng predicted class; detector YOLO26s V005, GT, split và raw detections không đổi. Không có DenseNet/CNN verifier trong V006.
Nguồn FULL V005 và dataset NEG-V03 có SHA256 trong README; checkpoint `best.pt` SHA256 `12f5717ff7e97c6ffcc3b2ab8c66b869bc8065995507dac3a5dbd95ce7714d19`.
Baseline A tái hiện đúng 56.237 raw → 42.602 region, khớp cả 951 ảnh. TP raw: ghép tham lam confidence giảm dần, cùng class, IoU ≥ 0,5, mỗi GT tối đa một lần.

## Bảng A — Baseline V005 tính lại

| Metric | Raw | Region A |
|---|---:|---:|
| GT coverage IoA ≥0,5 | 73.92% | 75.73% |
| GT center coverage | 76.30% | 77.48% |
| Excess area theo GT | 14.71% | 26.72% |
| Display occupancy | 11.01% | 12.61% |
| FP/ảnh | 23.744 | 14.157 |
| Precision | 59.85% | 68.40% |

## Bảng B — So sánh A và ba điểm vận hành

| Điểm | Raw/ảnh | Region/ảnh | Giảm toàn bộ | Giảm ảnh ≥100 | Excess area | Region precision | Lost TP | p95 gộp (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A (V005) | 59.135 | 44.797 | 24.25% | 27.44% | 26.72% | 68.40% | 0 | 47.998 |
| conservative | 59.135 | 45.225 | 23.52% | 26.83% | 25.18% | 68.10% | 0 | 62.365 |
| balanced | 59.135 | 37.309 | 36.91% | 41.53% | 28.97% | 73.95% | 0 | 60.259 |
| aggressive | 59.135 | 30.391 | 48.61% | 52.35% | 46.47% | 69.21% | 0 | 146.783 |

Latency p50 / p95 / max (ms) và ảnh chậm nhất:

| Điểm | p50 | p95 | max | Ảnh chậm nhất |
|---|---:|---:|---:|---|
| A (V005) | 3.241 | 47.998 | 123.080 | red_rust_1427.jpg |
| conservative | 2.723 | 62.365 | 193.525 | red_rust_1427.jpg |
| balanced | 2.692 | 60.259 | 158.465 | red_rust_1427.jpg |
| aggressive | 5.489 | 146.783 | 931.386 | anthracnose_193.jpg |

## Bảng C — Slices theo mật độ và class chính

| Điểm | Slice | Ảnh | Raw/ảnh | Region/ảnh | Giảm | Δ coverage (điểm %) | Excess area |
|---|---|---:|---:|---:|---:|---:|---:|
| A (V005) | density:le5 | 221 | 2.407 | 2.244 | 6.77% | +0.00 | 9.56% |
| A (V005) | density:6_10 | 108 | 7.694 | 7.083 | 7.94% | +0.00 | 10.94% |
| A (V005) | density:11_20 | 96 | 14.448 | 13.062 | 9.59% | +0.14 | 16.72% |
| A (V005) | density:21_40 | 84 | 29.810 | 24.750 | 16.97% | +0.93 | 23.05% |
| A (V005) | density:41_80 | 108 | 62.046 | 50.704 | 18.28% | +1.45 | 27.88% |
| A (V005) | density:81_150 | 247 | 116.688 | 88.486 | 24.17% | +1.96 | 42.94% |
| A (V005) | density:gt150 | 87 | 177.701 | 122.713 | 30.94% | +2.37 | 54.81% |
| A (V005) | class:anthracnose | 300 | 50.617 | 36.047 | 28.78% | +2.76 | 29.14% |
| A (V005) | class:leaf_miner | 267 | 8.569 | 7.783 | 9.18% | +0.08 | 13.09% |
| A (V005) | class:red_rust | 326 | 118.887 | 91.117 | 23.36% | +1.60 | 40.72% |
| A (V005) | class:healthy | 38 | 0.132 | 0.105 | 20.00% | N/A | 100.00% |
| A (V005) | class:not_cashew_leaf | 20 | 0.100 | 0.100 | 0.00% | N/A | 100.00% |
| A (V005) | class:mixed_disease | 503 | 66.048 | 51.155 | 22.55% | +1.67 | 26.28% |
| A (V005) | class:negative | 58 | 0.121 | 0.103 | 14.29% | N/A | 100.00% |
| conservative | density:le5 | 221 | 2.407 | 2.276 | 5.45% | +0.00 | 9.22% |
| conservative | density:6_10 | 108 | 7.694 | 7.083 | 7.94% | +0.00 | 10.77% |
| conservative | density:11_20 | 96 | 14.448 | 13.167 | 8.87% | +0.07 | 16.30% |
| conservative | density:21_40 | 84 | 29.810 | 25.512 | 14.42% | +0.68 | 19.54% |
| conservative | density:41_80 | 108 | 62.046 | 51.287 | 17.34% | +1.39 | 25.34% |
| conservative | density:81_150 | 247 | 116.688 | 88.895 | 23.82% | +1.80 | 41.90% |
| conservative | density:gt150 | 87 | 177.701 | 124.575 | 29.90% | +2.26 | 53.45% |
| conservative | class:anthracnose | 300 | 50.617 | 36.570 | 27.75% | +2.59 | 26.84% |
| conservative | class:leaf_miner | 267 | 8.569 | 7.820 | 8.74% | +0.08 | 12.80% |
| conservative | class:red_rust | 326 | 118.887 | 91.853 | 22.74% | +1.47 | 39.33% |
| conservative | class:healthy | 38 | 0.132 | 0.105 | 20.00% | N/A | 100.00% |
| conservative | class:not_cashew_leaf | 20 | 0.100 | 0.100 | 0.00% | N/A | 100.00% |
| conservative | class:mixed_disease | 503 | 66.048 | 51.545 | 21.96% | +1.53 | 24.81% |
| conservative | class:negative | 58 | 0.121 | 0.103 | 14.29% | N/A | 100.00% |
| balanced | density:le5 | 221 | 2.407 | 2.208 | 8.27% | +0.00 | 9.64% |
| balanced | density:6_10 | 108 | 7.694 | 6.917 | 10.11% | +0.00 | 10.97% |
| balanced | density:11_20 | 96 | 14.448 | 12.125 | 16.08% | +0.34 | 17.38% |
| balanced | density:21_40 | 84 | 29.810 | 22.726 | 23.76% | +1.36 | 22.49% |
| balanced | density:41_80 | 108 | 62.046 | 44.546 | 28.20% | +2.26 | 30.33% |
| balanced | density:81_150 | 247 | 116.688 | 72.789 | 37.62% | +2.62 | 47.37% |
| balanced | density:gt150 | 87 | 177.701 | 96.356 | 45.78% | +3.21 | 58.64% |
| balanced | class:anthracnose | 300 | 50.617 | 30.633 | 39.48% | +3.69 | 30.58% |
| balanced | class:leaf_miner | 267 | 8.569 | 7.476 | 12.76% | +0.16 | 13.07% |
| balanced | class:red_rust | 326 | 118.887 | 74.506 | 37.33% | +2.22 | 45.43% |
| balanced | class:healthy | 38 | 0.132 | 0.105 | 20.00% | N/A | 100.00% |
| balanced | class:not_cashew_leaf | 20 | 0.100 | 0.100 | 0.00% | N/A | 100.00% |
| balanced | class:mixed_disease | 503 | 66.048 | 42.626 | 35.46% | +2.29 | 28.50% |
| balanced | class:negative | 58 | 0.121 | 0.103 | 14.29% | N/A | 100.00% |
| aggressive | density:le5 | 221 | 2.407 | 2.285 | 5.08% | +0.00 | 9.23% |
| aggressive | density:6_10 | 108 | 7.694 | 7.037 | 8.54% | +0.11 | 11.13% |
| aggressive | density:11_20 | 96 | 14.448 | 11.458 | 20.69% | +1.50 | 21.04% |
| aggressive | density:21_40 | 84 | 29.810 | 21.048 | 29.39% | +3.06 | 36.37% |
| aggressive | density:41_80 | 108 | 62.046 | 33.472 | 46.05% | +7.02 | 49.59% |
| aggressive | density:81_150 | 247 | 116.688 | 55.591 | 52.36% | +6.48 | 67.99% |
| aggressive | density:gt150 | 87 | 177.701 | 85.322 | 51.99% | +4.70 | 73.63% |
| aggressive | class:anthracnose | 300 | 50.617 | 21.687 | 57.16% | +7.68 | 47.40% |
| aggressive | class:leaf_miner | 267 | 8.569 | 7.442 | 13.16% | +0.59 | 14.26% |
| aggressive | class:red_rust | 326 | 118.887 | 62.586 | 47.36% | +5.24 | 66.07% |
| aggressive | class:healthy | 38 | 0.132 | 0.105 | 20.00% | N/A | 100.00% |
| aggressive | class:not_cashew_leaf | 20 | 0.100 | 0.100 | 0.00% | N/A | 100.00% |
| aggressive | class:mixed_disease | 503 | 66.048 | 34.590 | 47.63% | +5.47 | 46.67% |
| aggressive | class:negative | 58 | 0.121 | 0.103 | 14.29% | N/A | 100.00% |

Ảnh âm (`healthy`, `not_cashew_leaf`) có trong Validation; bảng class và `negative` cho biết số raw/region còn lại. Coverage của slice không có GT là N/A (không có mẫu số).

## Bảng D — Confidence của region (Validation, n≥2)

Các giá trị max/mean/top-3 là thống kê box thành viên, **không** phải xác suất vùng. File `confidence_statistics.csv` có đủ ROC-AUC và PR-AUC cho hai target (có ≥1 TP; ≥50% thành viên TP).

| Điểm | Target | Thống kê PR-AUC tốt nhất | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|
| conservative | has_tp | mean | 0.743 | 0.940 |
| conservative | tp_member_share_ge_half | max_times_fill | 0.683 | 0.863 |
| balanced | has_tp | mean | 0.741 | 0.947 |
| balanced | tp_member_share_ge_half | mean | 0.660 | 0.835 |
| aggressive | has_tp | mean | 0.798 | 0.979 |
| aggressive | tp_member_share_ge_half | mean | 0.692 | 0.874 |

## Bảng E — Kích thước vùng và chuẩn bị crop

| Điểm | Class | Vùng | Vùng/ảnh Validation | Cạnh ngắn p50/p95 (px) | <32 px | Region class khác chồng lấn |
|---|---|---:|---:|---:|---:|---:|
| conservative | anthracnose | 13079 | 13.753 | 10.819/50.638 | 91.68% | 2.41% |
| conservative | red_rust | 29344 | 30.856 | 8.601/28.677 | 95.80% | 0.48% |
| conservative | leaf_miner | 586 | 0.616 | 131.573/246.901 | 1.71% | 4.74% |
| balanced | anthracnose | 11201 | 11.778 | 11.114/60.577 | 88.10% | 2.81% |
| balanced | red_rust | 23713 | 24.935 | 9.181/40.417 | 92.75% | 0.55% |
| balanced | leaf_miner | 567 | 0.596 | 133.393/247.898 | 1.76% | 4.93% |
| aggressive | anthracnose | 8489 | 8.926 | 10.817/114.812 | 84.63% | 4.82% |
| aggressive | red_rust | 19825 | 20.846 | 8.086/71.868 | 90.50% | 0.50% |
| aggressive | leaf_miner | 588 | 0.618 | 129.965/246.901 | 1.70% | 4.87% |

GT class khác trong vùng là phép đo riêng, không đồng nhất với chồng lấn region khác class:

| Điểm | GT khác class / diện tích region | Cặp region khác class chồng lấn | Ảnh có chồng lấn |
|---|---:|---:|---:|
| A (V005) | 2.42% | 468 | 214 |
| conservative | 2.43% | 468 | 215 |
| balanced | 2.33% | 473 | 221 |
| aggressive | 1.81% | 620 | 248 |

Mức giảm region theo predicted class (so với raw box cùng class):

| Điểm | Anthracnose | Leaf miner | Red rust |
|---|---:|---:|---:|
| A (V005) | 25.92% | 9.86% | 23.72% |
| conservative | 25.03% | 8.29% | 23.09% |
| balanced | 35.79% | 11.27% | 37.85% |
| aggressive | 51.34% | 7.98% | 48.04% |

## Diễn giải và giới hạn

YOLO detector không đổi; mọi thay đổi trong bảng là chất lượng và số lượng **vùng hiển thị**. Coverage có thể tăng chỉ vì bbox lớn hơn; phải xem cùng excess area và occupancy.
Cạnh ngắn <32 px cho biết tỷ lệ crop sau này phải phóng to; chưa tạo crop, chưa xác minh nhãn bằng DenseNet.
Mọi trial, gồm các trial C trước lần sửa chia cụm được gắn `valid_for_selection=false`, vẫn lưu trong `validation_tuning.csv`; chỉ trial hợp lệ dùng chọn ứng viên và Pareto.
Ba gallery (mỗi điểm 20 ảnh ở từng nhóm sparse, moderate, dense, mixed) ở `visual_gallery/`. Chủ dự án đã duyệt balanced trước khi mở Test.

## Test khóa

Đã chạy một lần với `balanced` / `B_e6e7d65f` trên 412 ảnh Test; 0 ảnh âm. Không chọn lại tham số từ Test.

Slice class chính của Test được tính từ `per_image.csv` đã lưu, theo tiền tố tên ảnh; bước này chỉ tổng hợp báo cáo, không chạy lại YOLO hay thuật toán gộp.

| Metric | Raw Test | Region Test |
|---|---:|---:|
| Số box/vùng | 19926 | 13768 |
| Số box/vùng mỗi ảnh | 48.364 | 33.417 |
| GT coverage IoA ≥0,5 | 74.98% | 76.22% |
| GT center coverage | 76.04% | 76.92% |
| Excess area theo GT | 28.49% | 37.98% |
| Display occupancy | 11.73% | 12.89% |
| FP/ảnh | 18.010 | 8.478 |
| Precision | 62.76% | 74.63% |

Giảm toàn bộ: 30.90%; ảnh ≥100 raw box: 35.73% (81 ảnh); ảnh ≤10 raw box: 9.15% (153 ảnh). `lost_tp=0`.
Đã đối chiếu độc lập raw Test JSONL với region JSONL: 19,926 ID raw xuất hiện đúng một lần trong region cùng class.
GT class khác trong region: 2.85%; 200 cặp vùng khác class chồng lấn trên 111 ảnh; 90.46% vùng có cạnh ngắn <32 px.
Latency p50/p95/max: 1.536/39.329/85.718 ms; ảnh chậm nhất: `red_rust_1130.jpg`.

| Slice Test | Ảnh | Raw/ảnh | Region/ảnh | Giảm | Δ coverage (điểm %) | Excess area |
|---|---:|---:|---:|---:|---:|---:|
| density:le5 | 89 | 3.258 | 3.022 | 7.24% | +0.00 | 23.55% |
| density:6_10 | 64 | 7.594 | 6.812 | 10.29% | +0.26 | 24.86% |
| density:11_20 | 38 | 14.553 | 11.868 | 18.44% | +0.00 | 33.99% |
| density:21_40 | 48 | 29.062 | 22.958 | 21.00% | +0.77 | 40.37% |
| density:41_80 | 55 | 59.273 | 41.127 | 30.61% | +1.34 | 51.73% |
| density:81_150 | 102 | 110.510 | 74.804 | 32.31% | +1.39 | 54.46% |
| density:gt150 | 16 | 166.875 | 101.125 | 39.40% | +1.30 | 60.56% |
| class:anthracnose | 123 | 22.325 | 15.455 | 30.77% | +0.80 | 33.49% |
| class:leaf_miner | 131 | 9.206 | 7.901 | 14.18% | +0.72 | 27.54% |
| class:red_rust | 158 | 101.101 | 68.557 | 32.19% | +1.34 | 53.78% |
| class:mixed_disease | 210 | 62.881 | 43.810 | 30.33% | +1.45 | 42.27% |

Confidence vùng trên Test (chẩn đoán sau khóa; không đổi tham số):

| Target | Thống kê PR-AUC cao nhất | ROC-AUC | PR-AUC |
|---|---|---:|---:|
| has_tp | top3_mean | 0.768 | 0.958 |
| tp_member_share_ge_half | mean | 0.660 | 0.857 |

Phân bố kích thước vùng Test theo class nằm trong `test_region_size_statistics.csv`; chi tiết từng vùng nằm trong `results/test_locked/region_statistics.csv`.

Trên ảnh dày, Validation giảm 41.53% và Test giảm 35.73%; đây là độ giảm của vùng hiển thị, không phải thay đổi AP của detector.
Test giảm số vùng nhưng excess area tăng từ 28.49% lên 37.98%; mức tăng coverage không tự nó chứng minh crop tốt hơn. Chưa đánh giá DenseNet hoặc chất lượng nhãn vùng cuối pipeline.
