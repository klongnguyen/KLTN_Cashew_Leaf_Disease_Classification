# Audit & changelog — ResNet50 Scratch EXP-006/007/008

## Source archives and SHA256

| Run | Archive | SHA256 |
|---|---|---|
| 006 | `EXP-RESNET50-SCRATCH-5SEEDS-006_ANALYSIS (1).zip` | `d1654999d1003f131c25153b705d49798c94c4e2737a15f156f3c7cbb32555e4` |
| 007 | `EXP-RESNET50-SCRATCH-5SEEDS-007_ANALYSIS.zip` | `8d4d8d2c02877c9b98481de1c28ad285cf980710c57bfe917aec7e5b7e38824f` |
| 008 | `EXP-RESNET50-SCRATCH-5SEEDS-008_ANALYSIS.zip` | `5bdcd4382a13afb7f805030e478a8f7bd63e0ea95355a3b35f97e1a43f43a6a9` |

## V05 006 → V07 007 → V07 008

| Experiment | Dataset | Input | Params | LR/batch/head | Val Acc | Val Macro F1 | Val/Test status |
|---|---|---|---:|---|---:|---:|---|
| 006 | V05 6.911 | /255 scratch | 24.64 M | 3e-4 / 32 / 512+drop .4 | 86.46 ± 2.27% | 86.12 ± 2.56% | Val done / Test pending |
| 007 | V07 6.907 | Caffe scratch | 23.60 M | 1e-4 / 16 / drop .3 | 89.20 ± 2.04% | 88.61 ± 2.19% | Val done / Test pending |
| 008 | V07 6.907 | Caffe scratch | 23.60 M | same as 007 | **89.20 ± 2.04%** | **88.61 ± 2.19%** | Source Val same as 007 |

**006 → 007:** nhiều thay đổi đồng thời (dataset, head, learning rate, batch, preprocessing, augment/callback), nên không xem chênh lệch +2.74 điểm Accuracy là tác động riêng của một kỹ thuật.

**007 → 008:** cả 5 seed predictions/history và 2 aggregate validation CSV SHA256 trùng; chưa thể coi là hai lần chạy độc lập. Binary checkpoints ở FULL ZIP chưa xác minh.

**Validation seed-selection:** 006 seed 42 (`val_loss=0.354891`); 007/008 seed 123 (`val_loss=0.332464`). 007/008 seed 2026 có Val Accuracy cao nhất 91.83% nhưng **không được đổi deployment seed theo Accuracy** sau khi đã ấn định rule min Val Loss.

**Dataset audit V07:** 0 exact cross-split duplicates, 52 pHash candidates + 1 thumbnail candidate chưa thủ công; không có `real_` trong Val. **Không suy ra độ chính xác trên ảnh thực địa/Real Holdout** từ Val V07.

## Phạm vi commit

Các tệp trên GitHub là báo cáo README/aggregate CSV/compact config + audit, confusion matrix seed đại diện và bảng truy xuất SHA của source archive. **Chưa upload original 3 ANALYSIS ZIP, binary FULL ZIP, per-image predictions của toàn bộ seeds, notebook, log epoch-level và PNG.** Chúng vẫn ở source archive được user cung cấp; **không ghi đường dẫn chưa tồn tại làm link GitHub**.

Đọc: [006](./EXP-RESNET50-SCRATCH-5SEEDS-006/README.md) · [007](./EXP-RESNET50-SCRATCH-5SEEDS-007/README.md) · [008](./EXP-RESNET50-SCRATCH-5SEEDS-008/README.md).
