# EXP-RESNET50-SCRATCH-5SEEDS-002

## Dataset
- Version: Cashew_dataV03
- Train: 5049
- Validation: 1433
- Test: 731
- Total: 7213
- Test locked: True

## Model
- Architecture: ResNet50
- Training mode: Scratch
- Pretrained weights: None
- Input: 224x224
- Head: Dense(512) + BatchNorm + Dropout(0.4)
- Backbone call: x = backbone(x)
- Forced training=True: False

## Training
- Seeds: [42, 123, 2026, 3407, 7777]
- Batch size: 32
- Max epochs: 50
- Optimizer: Adam
- Initial learning rate: 0.001
- Loss: SparseCategoricalCrossentropy
- EarlyStopping patience: 8
- ReduceLROnPlateau patience: 3
- ReduceLROnPlateau factor: 0.3

## Checkpoint
- Selection metric: minimum val_loss
- Storage: best weights only (.weights.h5)

## Training Summary
- Best epoch Mean ± Std: 18.00 ± 4.47
- Training time / seed Mean ± Std: 27.66 ± 4.65 min

## Validation 5-Seed Summary
- val_accuracy: 86.76 ± 0.74%
- val_macro_precision: 87.25 ± 0.81%
- val_macro_recall: 86.60 ± 0.43%
- val_macro_f1: 86.59 ± 0.63%
- val_balanced_accuracy: 86.60 ± 0.43%

## Final Test 5-Seed Summary
- accuracy: 90.10 ± 1.82%
- macro_precision: 90.42 ± 1.82%
- macro_recall: 90.09 ± 1.79%
- macro_f1: 90.12 ± 1.81%
- balanced_accuracy: 90.09 ± 1.79%
- weighted_f1: 90.14 ± 1.75%

## Scientific Rule
- Same Train/Val/Test split for all seeds.
- Do not select the best seed using Test performance.
- Report Mean ± sample Standard Deviation (ddof=1).

## GitHub Artifacts

Repo chỉ lưu các artifact nhẹ phục vụ tái lập và phân tích. Checkpoint `.weights.h5` (~282 MB/seed) không được commit để tránh làm repo quá nặng. Repo lưu config và các bảng tổng hợp chính để theo dõi experiment.

### Included
- `experiment_config.json`
- `aggregate/` với kết quả 5 seeds, Mean ± Std và per-class summary
- `aggregate/seed_details.json` tổng hợp training summary + test metrics từng seed

### Main result
- Test Accuracy: **90.10 ± 1.82%**
- Test Macro F1: **90.12 ± 1.81%**
- Test Balanced Accuracy: **90.09 ± 1.79%**
