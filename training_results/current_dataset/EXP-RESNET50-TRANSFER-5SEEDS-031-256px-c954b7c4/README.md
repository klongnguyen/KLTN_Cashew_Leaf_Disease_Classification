# EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4

ResNet50 ImageNet Transfer, two-stage, 256x256, five seeds, trained under the DenseNet121 recipe EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05 (same dataset, augmentation, schedule and selection rule).

Dataset: CashewData_Split (13).zip (SHA256 e1647b72e10af04275dc760545f4ef17c9f44343ae119495b89ba1e3cd954e04, manifest 2cee0fb4818f93fba1dda43ab2e83615d33c5f7f3e4e15dc7c647436ff54b607).
Stage 2 unfreezes conv5_block1-3 non-BatchNorm layers (14,953,472 of 23,587,712 backbone parameters; DenseNet121 reference: 2,129,920).
Deployment seed selected by Validation loss: 123.
Test: FINAL_TEST_EVALUATED; configured real holdout: REAL_HOLDOUT_EVALUATED. Neither evaluation selects the model or seed.
The real holdout has 4 present classes (no not_cashew_leaf images) and is a descriptive audit, not a blind benchmark.
predictions_standard.csv: dataset_manifest_sha256 is the Split manifest hash for Validation/Test and the Holdout archive SHA256 for the Holdout.
Seed intervals describe variation across seeds on one fixed split, not a field-population CI.
No automatic promotion or WebApp replacement.
