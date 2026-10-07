# KLCN Classification Training Archive Format v1

- Experiment: EXP-DENSENET121-SCRATCH-5SEEDS-006
- Dataset: Cashew_dataV05
- Model: DenseNet121
- Seeds: [42, 123, 2026, 3407, 7777]
- Results directory: /content/drive/MyDrive/Cashew_Leaf_model_result/training_results/current_dataset

## Required archive content
- all five best checkpoints;
- model_package for deployment / YOLO integration;
- code package for rebuilding and continued training;
- histories, metrics, reports, predictions and error cases;
- paper-style seed comparison panels;
- environment and requirements;
- README, manifest and SHA256 checksums.

## Figure rule
Any figure whose purpose is to compare seeds must present all 5 seeds side-by-side in one panel.

## Scientific rule
Same split + same hyperparameters for all seeds. Test is locked for tuning. Deployment seed is selected from Validation only.
