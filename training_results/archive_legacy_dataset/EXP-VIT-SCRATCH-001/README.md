# EXP-VIT-SCRATCH-001
> **Cập nhật điều hướng 28/10/2026.** LỊCH SỬ / SCRATCH — Không phải model WebApp Transfer EXP-014. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


## Model
Vision Transformer - Scratch

## Config
- Input: 224x224
- Patch size: 16
- Projection dim: 64
- Heads: 4
- Transformer layers: 8
- MLP dim: 128
- Batch: 32
- Max epochs: 60
- AdamW LR: 0.0003
- Weight decay: 0.0001

## Training
- Actual epochs: 43
- Best epoch: 33
- Best val accuracy: 0.898521
- Best val loss: 0.313227
- Training time: 15.94 min

## Dataset
- Valid: 7450
- Invalid skipped: 0
- Train: 5215
- Val: 1488
- Test: 747
- Test locked: True

## Manual Demo
Upload Image -> Predict included.

## Version trace / tiến độ (đồng bộ 19/10/2026)

- **Experiment 001 – Compact ViT Scratch**, dataset đầu tiên 7,450 ảnh: Train 5,215, Val 1,488, Test 747.
- **Config:** Patch 16, dim 64, 4 heads, 8 transformer layers, MLP 128, batch 32, 60 max epochs, AdamW LR 3e-4, weight decay 1e-4 (theo mục Config phía trên).
- **Kết quả có nguồn trong README:** best Val Accuracy `0.898521`, best val_loss `0.313227`; **README chưa cung cấp Final Test metrics** nên không suy đoán chúng.
- **Tiến độ:** archived (1 seed), không so trực tiếp với V04/V05 năm seeds hoặc V08; xem [bảng tiến độ](../../current_dataset/EXPERIMENT_PROGRESS_2026_10.md).
