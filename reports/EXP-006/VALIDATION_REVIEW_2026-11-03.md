# Hồ sơ đánh giá EXP-006 trên Validation

Các kết quả dưới đây lấy từ artifact Validation đã lưu. Việc bổ sung chỉ số không mở lại Test, không đổi trial, cấu hình khóa hoặc `validation_tuning.csv`.

## Cấu hình đã khóa

Điểm **balanced**, trial `B_e6e7d65f`. SHA-256 `config_locked.yaml`: `458fbb0a2a77717be386d0a783a5231bee3314f8c129a12400f879c2fe3efe21`.

| Tham số đang dùng trong B | Giá trị |
|---|---:|
| `k` | 0.5 |
| `cluster_gap_ratio` | 0.02 |
| `overlap_iou` | 0.6 |
| `containment_ratio` | 0.8 |
| `min_fill_ratio` | 0.15 |
| `min_group_members` | 3 |
| `max_region_area_ratio` | 0.1 |
| `max_region_span_ratio` | 0.35 |

`τ_fill` không tham gia biến thể B; biến thể C mạnh tay dùng `τ_fill=0.15`. Khoảng cách B dùng `max(0.02 × cạnh ngắn ảnh, 0.5 × kích thước lesion nhỏ hơn trong cặp)`.

## Các trial và ba điểm vận hành

`validation_tuning.csv` chứa **26 trial**: A=1, B=11, C=14. 19 trial hợp lệ để chọn; 7 trial C từ phiên bản chia cụm cũ được giữ lại nhưng đánh dấu không hợp lệ.

| Điểm | Trial | Giảm ảnh ≥100 box | Excess area | Precision vùng | Nhiễm GT khác class | Giảm ảnh ≤10 box |
|---|---|---:|---:|---:|---:|---:|
| baseline_A | `A_v005` | 27.44% | 26.72% | 68.40% | 2.42% | 7.48% |
| conservative | `B_608386fc` | 26.83% | 25.18% | 68.10% | 2.43% | 6.97% |
| balanced | `B_e6e7d65f` | 41.53% | 28.97% | 73.95% | 2.33% | 9.39% |
| aggressive | `C2_6451497a` | 52.35% | 46.47% | 69.21% | 1.81% | 7.19% |

Điểm mạnh tay đạt mức giảm 50% trên ảnh dày ở Validation, nhưng excess area là 46,47%, so với 28,97% của balanced. Ba đồ thị Pareto dùng 19 trial hợp lệ:

- [Ba mặt cắt Pareto, đánh dấu ba điểm vận hành](pareto_review_2026-11-03.png)
- [Giảm vùng và excess area](pareto_excess_area.png)
- [Giảm vùng và precision](pareto_region_precision.png)
- [Giảm vùng và display occupancy](pareto_occupancy.png)

## Hai chỉ số bổ sung

TP raw được ghép một-một theo confidence giảm dần, cùng class và IoU ≥0,5. Tỷ lệ TP đa số = số region có `2 × TP members ≥ n_detections` chia tổng region. FP bị hấp thụ = tổng số thành viên FP nằm trong region có ít nhất một TP. Tỷ lệ FP bị hấp thụ chia cho tổng raw FP. Region đơn lẻ được tính trong tỷ lệ toàn bộ; cột `n≥2` tách riêng các region thực sự gộp.

| Điểm | Region TP đa số | TP đa số trong region n≥2 | Raw FP bị hấp thụ | Tỷ lệ raw FP | Region trộn TP+FP |
|---|---:|---:|---:|---:|---:|
| baseline_A | 28,636/42,602 (67.22%) | 3,945/5,184 (76.10%) | 7,808 | 34.58% | 4,371 |
| conservative | 28,787/43,009 (66.93%) | 3,960/5,197 (76.20%) | 7,514 | 33.28% | 4,381 |
| balanced | 24,984/35,481 (70.42%) | 6,099/8,395 (72.65%) | 11,289 | 49.99% | 6,756 |
| aggressive | 19,172/28,902 (66.33%) | 3,362/4,517 (74.43%) | 12,383 | 54.84% | 3,996 |

FP bị hấp thụ không biến thành TP của detector; chỉ không còn khung riêng trên màn hình. Metric này phản ánh khả năng che lỗi khi hiển thị và cần đọc cùng precision, excess area và ảnh.

### Ảnh dày và ảnh nhiều bệnh

| Điểm | Nhóm | Region TP đa số | Raw FP bị hấp thụ / tổng raw FP |
|---|---|---:|---:|
| baseline_A | ≥100 raw box | 66.58% | 6,119/16,158 (37.87%) |
| baseline_A | GT nhiều bệnh | 68.57% | 4,224/12,679 (33.31%) |
| conservative | ≥100 raw box | 66.42% | 5,935/16,158 (36.73%) |
| conservative | GT nhiều bệnh | 68.37% | 4,107/12,679 (32.39%) |
| balanced | ≥100 raw box | 70.58% | 8,946/16,158 (55.37%) |
| balanced | GT nhiều bệnh | 71.70% | 6,351/12,679 (50.09%) |
| aggressive | ≥100 raw box | 65.17% | 9,496/16,158 (58.77%) |
| aggressive | GT nhiều bệnh | 67.10% | 6,833/12,679 (53.89%) |

## Theo predicted class và ảnh nhiều bệnh

| Điểm | Class | Giảm số vùng | TP đa số | FP bị hấp thụ / raw FP |
|---|---|---:|---:|---:|
| baseline_A | anthracnose | 25.92% | 63.18% | 2,439/7,552 (32.30%) |
| baseline_A | leaf_miner | 9.86% | 72.74% | 52/214 (24.30%) |
| baseline_A | red_rust | 23.72% | 68.90% | 5,317/14,815 (35.89%) |
| conservative | anthracnose | 25.03% | 62.80% | 2,323/7,552 (30.76%) |
| conservative | leaf_miner | 8.29% | 71.67% | 43/214 (20.09%) |
| conservative | red_rust | 23.09% | 68.68% | 5,148/14,815 (34.75%) |
| balanced | anthracnose | 35.79% | 65.16% | 3,171/7,552 (41.99%) |
| balanced | leaf_miner | 11.27% | 72.49% | 59/214 (27.57%) |
| balanced | red_rust | 37.85% | 72.85% | 8,059/14,815 (54.40%) |
| aggressive | anthracnose | 51.34% | 64.97% | 4,266/7,552 (56.49%) |
| aggressive | leaf_miner | 7.98% | 71.43% | 41/214 (19.16%) |
| aggressive | red_rust | 48.04% | 66.77% | 8,076/14,815 (54.51%) |

Nhóm GT nhiều bệnh có 503 ảnh. Số liệu reduction và excess area theo nhóm/class chính nằm trong `EXP006_FINAL_REPORT.md`; các cột class trên đây phân theo **predicted class của region**.

## Thống kê confidence trên region n≥2

Các score là thống kê confidence thành viên, không phải xác suất region. Bảng lấy thống kê PR-AUC tốt nhất cho từng target trong `confidence_statistics.csv`.

| Điểm | Target | Score | ROC-AUC | PR-AUC |
|---|---|---|---:|---:|
| conservative | has_tp | mean | 0.743 | 0.940 |
| conservative | tp_member_share_ge_half | max_times_fill | 0.683 | 0.863 |
| balanced | has_tp | mean | 0.741 | 0.947 |
| balanced | tp_member_share_ge_half | mean | 0.660 | 0.835 |
| aggressive | has_tp | mean | 0.798 | 0.979 |
| aggressive | tp_member_share_ge_half | mean | 0.692 | 0.874 |

## Ảnh Original → Raw → ACDRA

Gallery có 20 ảnh cố định cho mỗi tổ hợp của 4 nhóm (sparse, moderate, dense, mixed) và 3 điểm vận hành, tổng cộng 240 ảnh. Mỗi JPG đã ghép ba khung để đối chiếu trực tiếp.

- sparse / conservative: [ảnh mẫu](visual_gallery/sparse/conservative/anthracnose_1256.jpg), 20 ảnh trong thư mục cùng tên.
- sparse / balanced: [ảnh mẫu](visual_gallery/sparse/balanced/anthracnose_1256.jpg), 20 ảnh trong thư mục cùng tên.
- sparse / aggressive: [ảnh mẫu](visual_gallery/sparse/aggressive/anthracnose_1256.jpg), 20 ảnh trong thư mục cùng tên.
- moderate / conservative: [ảnh mẫu](visual_gallery/moderate/conservative/anthracnose_1047.jpg), 20 ảnh trong thư mục cùng tên.
- moderate / balanced: [ảnh mẫu](visual_gallery/moderate/balanced/anthracnose_1047.jpg), 20 ảnh trong thư mục cùng tên.
- moderate / aggressive: [ảnh mẫu](visual_gallery/moderate/aggressive/anthracnose_1047.jpg), 20 ảnh trong thư mục cùng tên.
- dense / conservative: [ảnh mẫu](visual_gallery/dense/conservative/anthracnose_115.jpg), 20 ảnh trong thư mục cùng tên.
- dense / balanced: [ảnh mẫu](visual_gallery/dense/balanced/anthracnose_115.jpg), 20 ảnh trong thư mục cùng tên.
- dense / aggressive: [ảnh mẫu](visual_gallery/dense/aggressive/anthracnose_115.jpg), 20 ảnh trong thư mục cùng tên.
- mixed / conservative: [ảnh mẫu](visual_gallery/mixed/conservative/anthracnose_112.jpg), 20 ảnh trong thư mục cùng tên.
- mixed / balanced: [ảnh mẫu](visual_gallery/mixed/balanced/anthracnose_112.jpg), 20 ảnh trong thư mục cùng tên.
- mixed / aggressive: [ảnh mẫu](visual_gallery/mixed/aggressive/anthracnose_112.jpg), 20 ảnh trong thư mục cùng tên.

Các ảnh cho thấy một số vùng của điểm mạnh tay bao phủ cả khoảng nền giữa các đốm. Vì vậy mức giảm khung 52,35% cần được đánh giá cùng excess area và ảnh từng trường hợp. Balanced đã được khóa và Test đã đánh giá một lần; việc đổi điểm sau khi xem Test sẽ là một thí nghiệm mới, không phải điều chỉnh EXP-006 hiện tại.
