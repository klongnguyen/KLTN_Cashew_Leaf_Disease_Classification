# Current Dataset Experiments

Thư mục này dành riêng cho các kết quả huấn luyện sử dụng **Master Dataset hiện tại** sau khi dataset được cập nhật. Chỉ các experiment trong thư mục này mới được xem là ứng viên cho bảng so sánh chính thức của khóa luận, trừ khi có ghi chú khác.

## Dataset hiện tại — Cashew_dataV03

Dataset classification hiện tại có **7,213 ảnh thuộc 5 lớp** và sử dụng một split cố định cho toàn bộ baseline.

| Class | Train | Validation | Test | Total |
|---|---:|---:|---:|---:|
| `anthracnose` | 1,096 | 313 | 156 | 1,565 |
| `healthy` | 818 | 225 | 128 | 1,171 |
| `leaf_miner` | 919 | 262 | 131 | 1,312 |
| `not_cashew_leaf` | 1,101 | 314 | 157 | 1,572 |
| `red_rust` | 1,115 | 319 | 159 | 1,593 |
| **TOTAL** | **5,049** | **1,433** | **731** | **7,213** |

- Tỉ lệ split xấp xỉ **70% / 20% / 10%**.
- **Test Set đã khóa** và giữ nguyên giữa các model / seed.
- File Excel mô tả phân bố dataset: [`dataset_cashew.xlsx`](../../dataset_cashew.xlsx).

## Baseline hiện tại

| Experiment | Model | Mode | Seeds | Validation Accuracy | Test Accuracy | Macro F1 | Status |
|---|---|---|---:|---:|---:|---:|---|
| [`EXP-RESNET50-SCRATCH-5SEEDS-002`](./EXP-RESNET50-SCRATCH-5SEEDS-002/) | ResNet50 | Scratch | 5 | **86.76 ± 0.74%** | **90.10 ± 1.82%** | **90.12 ± 1.81%** | ✅ Baseline hợp lệ |

## Kết quả trực quan

Seed 42 được dùng làm **representative seed** cho hình minh họa vì đây là seed cố định đầu tiên trong protocol. Kết luận chính thức vẫn dựa trên **Mean ± Std của cả 5 seeds**.

<table>
<tr>
<th>Training / Validation Accuracy — Seed 42</th>
<th>Normalized Test Confusion Matrix — Seed 42</th>
</tr>
<tr>
<td width="50%"><img src="./EXP-RESNET50-SCRATCH-5SEEDS-002/5_seed_resnet50_v02/5_seed_resnet50_v02/42/accuracy_curve.png" width="100%" alt="Seed 42 Accuracy Curve"></td>
<td width="50%"><img src="./EXP-RESNET50-SCRATCH-5SEEDS-002/5_seed_resnet50_v02/5_seed_resnet50_v02/42/confusion_matrix_normalized.png" width="100%" alt="Seed 42 Normalized Test Confusion Matrix"></td>
</tr>
</table>

### Insight chính từ baseline ResNet50

- `red_rust` là lớp ổn định nhất với **F1 = 94.86 ± 1.03%**.
- `not_cashew_leaf` đạt **F1 = 93.27 ± 3.11%**.
- `leaf_miner` đạt **F1 = 90.36 ± 1.30%**.
- `healthy` đạt **F1 = 89.85 ± 4.75%**.
- `anthracnose` là lớp khó nhất với **F1 = 82.26 ± 2.55%**, cần ưu tiên phân tích nhầm lẫn trong các experiment cải tiến.

Xem đầy đủ 5 seed, learning curves, confusion matrices và các file CSV tại README của experiment: [`EXP-RESNET50-SCRATCH-5SEEDS-002`](./EXP-RESNET50-SCRATCH-5SEEDS-002/README.md).

## Protocol đánh giá

- Cùng một Train / Validation / Test split cho tất cả seed.
- Mỗi configuration chạy trên 5 seed: `42, 123, 2026, 3407, 7777`.
- Validation dùng cho checkpoint, EarlyStopping và learning-rate scheduling.
- Test Set không dùng để tuning hoặc chọn seed tốt nhất.
- Báo cáo kết quả theo **Mean ± sample Standard Deviation (ddof=1)**.

## Quy tắc lưu experiment

Mỗi experiment nên có cấu trúc tối thiểu:

```text
EXP-{MODEL}-{MODE}-{NO}/
├── README.md
├── experiment_config.json
├── aggregate/
└── figures / per-seed artifacts
```

Không sử dụng kết quả trong `../archive_legacy_dataset/` để so sánh trực tiếp với các experiment mới vì chúng được huấn luyện trên dataset cũ.

## Lưu ý dung lượng

Checkpoint `.weights.h5` dung lượng lớn không commit trực tiếp vào GitHub. Repo ưu tiên lưu config, metrics, CSV và figures để bảo đảm khả năng kiểm tra experiment mà không làm repository quá nặng.
