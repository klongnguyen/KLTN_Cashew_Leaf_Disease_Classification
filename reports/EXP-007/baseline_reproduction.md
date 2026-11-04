# Tái hiện năm trial V006 trước khi làm EXP-007

Chạy lại trực tiếp trên 951 ảnh Validation từ ZIP V005 và dataset NEG-V03, trước khi tạo mã V007. Đối chiếu với `reports/EXP-006/validation_tuning.csv`; sai lệch lớn nhất ở ba chỉ số dưới đây bằng 0 cho cả năm trial. `lost_tp=0`.

| Trial | Giảm ảnh ≥100 raw box | Excess area | Giảm ảnh ≤10 raw box | Sai lệch lớn nhất |
|---|---:|---:|---:|---:|
| `A_v005` | 27,4412% | 26,7168% | 7,4835% | 0 |
| `B_e6e7d65f` | 41,5340% | 28,9747% | 9,3910% | 0 |
| `B_08ff744d` | 48,9270% | 30,8131% | 11,8855% | 0 |
| `C2_6451497a` | 52,3514% | 46,4696% | 7,1900% | 0 |
| `C2_fc918698` | 52,8468% | 41,8102% | 16,7278% | 0 |

Phép tính gọi `scripts/run_acdra_v2.py::load_split` và `evaluate` với config của từng trial trong CSV, có `baseline_check` trên từng ảnh cho A. Đã chạy lại qua CLI `scripts/reproduce_acdra_v3_references.py`; thời gian từng trial và các giá trị đầy đủ ở [baseline_reproduction.json](baseline_reproduction.json). Không tạo kết quả Test hoặc sửa artifact V006.
