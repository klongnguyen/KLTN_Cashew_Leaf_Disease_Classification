# EXP-007 — ACDRA v3, Pareto và mức sẵn sàng cho crop

## Nguồn và phạm vi

Validation gồm 951 ảnh, 56.237 raw box; checkpoint V005 SHA-256 `12f5717ff7e97c6ffcc3b2ab8c66b869bc8065995507dac3a5dbd95ce7714d19`, dataset ZIP SHA-256 `312870474cfbab8c3fc77db7b19e821c1edf34a0787cb45592f00bde841b4dbe`, V005 FULL ZIP SHA-256 `990df19b094cc14d6e5706183616245dda83d21fac3b52c20f9a2a3f5488c845`. Class mapping đọc từ `data_fixed.yaml`. Không huấn luyện lại detector, không tạo crop và không chạy Test.
Năm trial V006 đã được chạy lại trước khi sửa mã V007 và khớp chính xác ba chỉ số tham chiếu; xem `baseline_reproduction.md`.

## Bảng A — So sánh toàn Validation

`region_precision_strict` là tỷ lệ region có ≥50% raw member TP. `region_precision` kiểu V006 (chỉ cần ≥1 TP) vẫn nằm trong CSV, nhưng không dùng làm chỉ số chính.

| Điểm | Vùng/ảnh | Giảm | Excess | Strict precision | FP raw bị hấp thụ | Chồng lấn cùng class | Lost TP | p50/p95/max ms |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Raw | 59.13 | 0.00% | 14.71% | 59.85% | 0 | 14.41% | 0 | N/A |
| A | 44.80 | 24.25% | 26.72% | 67.22% | 7,808 (34.58%) | 8.21% | 0 | 3.0/44.0/110.6 |
| V006 balanced | 37.31 | 36.91% | 28.97% | 70.42% | 11,289 (49.99%) | 8.06% | 0 | 2.6/54.3/131.6 |
| conservative | 32.93 | 44.31% | 30.81% | 73.93% | 13,001 (57.57%) | 7.83% | 0 | 2.6/57.2/142.6 |
| balanced | 25.48 | 56.91% | 43.67% | 74.53% | 15,095 (66.85%) | 6.97% | 0 | 2.7/59.8/159.9 |
| aggressive | 21.30 | 63.98% | 48.19% | 74.60% | 16,394 (72.60%) | 6.46% | 0 | 2.7/64.0/157.4 |

### Chỉ ảnh dày (≥100 raw box)

| Điểm | Ảnh | Vùng/ảnh | Giảm | Excess | Strict precision | Chồng lấn cùng class |
|---|---:|---:|---:|---:|---:|---:|
| Raw | 275 | 141.66 | 0.00% | 21.79% | 58.52% | 11.04% |
| A | 275 | 102.79 | 27.44% | 47.76% | 66.58% | 4.17% |
| V006 balanced | 275 | 82.82 | 41.53% | 51.98% | 70.58% | 2.02% |
| conservative | 275 | 72.35 | 48.93% | 54.00% | 74.74% | 1.90% |
| balanced | 275 | 55.04 | 61.15% | 67.27% | 75.33% | 2.62% |
| aggressive | 275 | 45.07 | 68.18% | 71.01% | 75.54% | 2.40% |

Diện tích chồng lấn cùng class tính đúng một lần ở chỗ được ≥2 bbox cùng class che; mẫu số là tổng diện tích bbox (vì vậy raw và region có cùng quy tắc). Số cặp với IoU>0,1 và tỷ lệ ảnh có cặp nằm trong `validation_tuning_v007.csv`.

## Pareto và ba ứng viên

Có 18 trial V007 ghi đầy đủ config hash; ba điểm dưới đây là **ứng viên chờ chủ dự án duyệt**, chưa khóa.

| Điểm | Trial | Variant | Giảm ảnh dày | Excess | Strict precision | Chồng lấn cùng class |
|---|---|---|---:|---:|---:|---:|
| conservative | `B_08ff744d` | B | 48.93% | 30.81% | 73.93% | 7.83% |
| balanced | `V7_e95d60ebf1` | B | 61.15% | 43.67% | 74.53% | 6.97% |
| aggressive | `V7_1c8106e164` | B | 68.18% | 48.19% | 74.60% | 6.46% |

Điểm giữa được chọn bằng quy tắc: trong các điểm Pareto đạt ít nhất 60% giảm trên ảnh dày, lấy điểm có excess thấp nhất. Đây là cách trình bày ba mức đánh đổi để duyệt ảnh, không phải ngưỡng loại trial.

Mục tiêu thăm dò đạt được ở balanced: giảm 61.15% trên ảnh dày, excess 43.67% thấp hơn mốc C2 V006 (46,47%), strict precision 74.53% cao hơn raw 59.85%. Đây là kết quả Validation; gallery vẫn cần được duyệt trước khi khóa.

- [Pareto giảm–excess](pareto_excess_area.png)
- [Pareto giảm–strict precision](pareto_strict_precision.png)
- [Pareto giảm–chồng lấn cùng class](pareto_same_class_overlap.png)
- [Toàn bộ trial](validation_tuning_v007.csv) và [các điểm không bị trội](pareto_frontier_v007.csv)

## Bảng B — Ảnh thưa (≤10 raw box)

Cờ duyệt mềm bật khi >15% ảnh thưa có ít nhất một box được gộp; không loại tự động.

| Điểm | Ảnh thưa bị đổi / 329 | Box được gộp | Giảm trên nhóm | Cờ duyệt |
|---|---:|---:|---:|---|
| A | 77/329 (23.40%) | 102 | 7.48% | Cần xem ảnh |
| V006 balanced | 85/329 (25.84%) | 128 | 9.39% | Cần xem ảnh |
| conservative | 109/329 (33.13%) | 162 | 11.89% | Cần xem ảnh |
| balanced | 128/329 (38.91%) | 196 | 14.38% | Cần xem ảnh |
| aggressive | 136/329 (41.34%) | 218 | 15.99% | Cần xem ảnh |

## Bảng C — Mức sẵn sàng cho crop

Vùng đủ điều kiện xác minh: `n_detections≥2` và cạnh ngắn ≥`s_min`. GT lesion được tính nếu ≥50% diện tích GT nằm trong ít nhất một vùng đủ điều kiện cùng class. Đây là phép đo hình học, chưa đánh giá DenseNet.

| Điểm | s_min | Vùng đủ điều kiện | Raw box trong vùng | Raw TP trong vùng | GT lesion được phủ |
|---|---:|---:|---:|---:|---:|
| conservative | 32 | 3,091/31,317 (9.87%) | 14,865/56,237 (26.43%) | 8,344/33,656 (24.79%) | 10,232/47,258 (21.65%) |
| conservative | 48 | 1,620/31,317 (5.17%) | 9,726/56,237 (17.29%) | 5,302/33,656 (15.75%) | 6,863/47,258 (14.52%) |
| conservative | 64 | 1,005/31,317 (3.21%) | 6,974/56,237 (12.40%) | 3,723/33,656 (11.06%) | 5,046/47,258 (10.68%) |
| balanced | 32 | 3,827/24,230 (15.79%) | 24,226/56,237 (43.08%) | 14,497/33,656 (43.07%) | 17,687/47,258 (37.43%) |
| balanced | 48 | 2,463/24,230 (10.17%) | 19,191/56,237 (34.13%) | 11,289/33,656 (33.54%) | 14,219/47,258 (30.09%) |
| balanced | 64 | 1,815/24,230 (7.49%) | 16,042/56,237 (28.53%) | 9,319/33,656 (27.69%) | 12,031/47,258 (25.46%) |
| aggressive | 32 | 4,425/20,256 (21.85%) | 29,363/56,237 (52.21%) | 17,627/33,656 (52.37%) | 21,245/47,258 (44.96%) |
| aggressive | 48 | 2,860/20,256 (14.12%) | 23,242/56,237 (41.33%) | 13,740/33,656 (40.82%) | 17,005/47,258 (35.98%) |
| aggressive | 64 | 2,067/20,256 (10.20%) | 19,308/56,237 (34.33%) | 11,238/33,656 (33.39%) | 14,212/47,258 (30.07%) |

**Kết luận crop:** ở điểm balanced và `s_min=32`, có 3,827/24,230 region (15.79%) đủ điều kiện hình học; các vùng này chứa 14,497/33,656 raw TP (43.07%) và phủ 17,687/47,258 GT lesion (37.43%). Có vùng đủ lớn để thử crop, nhưng phần lớn region vẫn quá nhỏ hoặc chỉ có một detection; chất lượng phân loại crop chưa được kiểm chứng.

Bảng theo từng class, tỷ lệ cạnh ngắn <32/<64 và các ngưỡng 32/48/64 px nằm trong [crop_readiness.csv](crop_readiness.csv).

Cửa sổ giả định là hình vuông quanh tâm region, cạnh bằng `max(cạnh dài region, W_min)`; đo cắt biên trước khi kẹp vào kích thước ảnh. `other_class_region_inside` nghĩa là crop giao với region khác class, không phải GT khác class.

| Điểm | W_min | Crop bị cắt biên | Crop gặp region khác class | Cặp crop cùng class IoU>0,5 |
|---|---:|---:|---:|---:|
| conservative | 64 | 2,914/31,317 (9.30%) | 2,367/31,317 (7.56%) | 83 |
| conservative | 96 | 4,625/31,317 (14.77%) | 3,462/31,317 (11.05%) | 1,631 |
| conservative | 128 | 6,556/31,317 (20.93%) | 4,644/31,317 (14.83%) | 6,764 |
| balanced | 64 | 2,490/24,230 (10.28%) | 2,156/24,230 (8.90%) | 67 |
| balanced | 96 | 3,769/24,230 (15.56%) | 2,958/24,230 (12.21%) | 726 |
| balanced | 128 | 5,240/24,230 (21.63%) | 3,844/24,230 (15.86%) | 3,302 |
| aggressive | 64 | 2,219/20,256 (10.95%) | 2,030/20,256 (10.02%) | 54 |
| aggressive | 96 | 3,305/20,256 (16.32%) | 2,693/20,256 (13.29%) | 212 |
| aggressive | 128 | 4,508/20,256 (22.26%) | 3,369/20,256 (16.63%) | 1,254 |

[Bảng crop theo class](crop_window_readiness.csv). Ảnh Validation trong dataset ZIP có kích thước:

| W×H | Số ảnh |
|---|---:|
| 194×259 | 1 |
| 256×256 | 15 |
| 293×172 | 1 |
| 400×400 | 38 |
| 427×341 | 1 |
| 512×384 | 2 |
| 512×512 | 52 |
| 640×640 | 841 |

Ảnh gốc độ phân giải cao hơn: **N/A** — ZIP nguồn YOLO chỉ có `images/train`, `images/val`, `images/test`; chưa xác định được cặp ảnh gốc có ánh xạ tên/GT để dùng crop. Không thể kết luận ảnh YOLO còn nguyên độ phân giải gốc chỉ từ kích thước hiện có.

## Bảng D — Confidence region n≥2

PR-AUC cần đọc cùng base rate của target; score mean/top3 là thống kê thành viên, không phải xác suất vùng.

| Điểm | Target | Score | Base rate | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|---:|
| conservative | has_tp | mean | 86.77% | 0.731 | 0.940 |
| conservative | has_tp | top3_mean | 86.77% | 0.732 | 0.940 |
| conservative | tp_member_share_ge_half | mean | 76.77% | 0.686 | 0.877 |
| conservative | tp_member_share_ge_half | top3_mean | 76.77% | 0.649 | 0.850 |
| balanced | has_tp | mean | 88.25% | 0.756 | 0.954 |
| balanced | has_tp | top3_mean | 88.25% | 0.757 | 0.954 |
| balanced | tp_member_share_ge_half | mean | 78.14% | 0.701 | 0.892 |
| balanced | tp_member_share_ge_half | top3_mean | 78.14% | 0.654 | 0.862 |
| aggressive | has_tp | mean | 89.92% | 0.761 | 0.960 |
| aggressive | has_tp | top3_mean | 89.92% | 0.761 | 0.962 |
| aggressive | tp_member_share_ge_half | mean | 78.00% | 0.703 | 0.891 |
| aggressive | tp_member_share_ge_half | top3_mean | 78.00% | 0.646 | 0.857 |

[Toàn bộ score và target](confidence_statistics_v007.csv).

## Bảng E — Slice theo mật độ và class chính

| Điểm | Slice | Ảnh | Giảm | Excess | Strict precision |
|---|---|---:|---:|---:|---:|
| conservative | density:le5 | 221 | 9.59% | 9.91% | 75.47% |
| conservative | density:6_10 | 108 | 13.36% | 12.10% | 65.42% |
| conservative | density:11_20 | 96 | 23.22% | 18.24% | 70.99% |
| conservative | density:21_40 | 84 | 30.47% | 23.96% | 69.16% |
| conservative | density:41_80 | 108 | 36.11% | 32.36% | 70.68% |
| conservative | density:81_150 | 247 | 45.42% | 49.70% | 76.94% |
| conservative | density:gt150 | 87 | 52.79% | 60.39% | 71.65% |
| conservative | class:anthracnose | 300 | 46.34% | 32.22% | 68.32% |
| conservative | class:leaf_miner | 267 | 17.96% | 13.69% | 68.62% |
| conservative | class:red_rust | 326 | 45.08% | 47.87% | 76.57% |
| conservative | class:mixed_disease | 503 | 43.09% | 30.40% | 75.23% |
| balanced | density:le5 | 221 | 11.84% | 10.43% | 75.69% |
| balanced | density:6_10 | 108 | 16.00% | 13.08% | 66.05% |
| balanced | density:11_20 | 96 | 31.87% | 22.17% | 70.69% |
| balanced | density:21_40 | 84 | 40.81% | 35.20% | 68.62% |
| balanced | density:41_80 | 108 | 52.68% | 46.89% | 72.53% |
| balanced | density:81_150 | 247 | 59.45% | 64.62% | 78.03% |
| balanced | density:gt150 | 87 | 62.62% | 71.46% | 71.60% |
| balanced | class:anthracnose | 300 | 62.11% | 45.37% | 70.02% |
| balanced | class:leaf_miner | 267 | 22.60% | 15.05% | 68.77% |
| balanced | class:red_rust | 326 | 56.91% | 62.69% | 76.71% |
| balanced | class:mixed_disease | 503 | 55.58% | 43.53% | 75.17% |
| aggressive | density:le5 | 221 | 12.22% | 10.61% | 75.59% |
| aggressive | density:6_10 | 108 | 18.41% | 13.92% | 65.93% |
| aggressive | density:11_20 | 96 | 36.70% | 23.90% | 71.30% |
| aggressive | density:21_40 | 84 | 47.68% | 39.87% | 68.55% |
| aggressive | density:41_80 | 108 | 60.33% | 51.90% | 72.27% |
| aggressive | density:81_150 | 247 | 66.88% | 68.60% | 78.61% |
| aggressive | density:gt150 | 87 | 69.48% | 74.97% | 71.24% |
| aggressive | class:anthracnose | 300 | 68.27% | 49.21% | 70.57% |
| aggressive | class:leaf_miner | 267 | 26.44% | 15.90% | 68.75% |
| aggressive | class:red_rust | 326 | 64.52% | 67.28% | 76.76% |
| aggressive | class:mixed_disease | 503 | 62.90% | 48.19% | 75.35% |

## Ảnh trước/sau và giới hạn suy luận

Gallery có 330 ảnh. Mỗi ứng viên có 20 ảnh cho từng nhóm sparse, moderate, dense, mixed; thêm ảnh sparse bị thay đổi và 10 ảnh có chồng lấn cùng class nặng nhất.

- [Danh sách ảnh](visual_gallery_manifest.csv)
- [Cấu hình ba ứng viên](operating_points_v007.yaml)

Test V006 đã được dùng một lần cho `B_e6e7d65f`. Vì vậy EXP-007 chưa chạy Test; nếu chủ dự án khóa một ứng viên và đánh giá tiếp, kết quả đó phải ghi là **đánh giá nội bộ, không còn là Test nguyên vẹn**. Không chọn lại tham số từ Test. Một Real Holdout độc lập mới sẽ phù hợp hơn cho báo cáo công bố.

Môi trường: Python 3.11.15; seed gallery 42; checkpoint/dataset không đổi. Hash CSV trial: `f66107befd6b7eb4ef7a8fbc8abdbdeed20c3f565b52dcefd8ab2ad74d31c59c`.
