# Ba điểm vận hành trên Validation

Điều kiện: lost_tp=0, precision vùng ≥ raw precision − 2 điểm %, giảm ảnh thưa ≤10%.

| Điểm | Trial | Giảm ảnh ≥100 box | Excess area | Region precision | Giảm ảnh ≤10 box | Vùng cạnh ngắn <32 px |
|---|---|---:|---:|---:|---:|---:|
| conservative | B_608386fc | 0.268 | 0.252 | 0.681 | 0.070 | 0.933 |
| balanced | B_e6e7d65f | 0.415 | 0.290 | 0.740 | 0.094 | 0.898 |
| aggressive | C2_6451497a | 0.524 | 0.465 | 0.692 | 0.072 | 0.870 |

Các điểm Pareto được đánh dấu trên ba biểu đồ trong thư mục báo cáo.
Chủ dự án đã chốt `balanced` (`B_e6e7d65f`). Cấu hình đã khóa trong `config_locked.yaml`; Test đã chạy đúng một lần. Kết quả Test nằm trong `EXP006_FINAL_REPORT.md`.
