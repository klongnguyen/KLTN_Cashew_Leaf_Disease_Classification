# EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-256px-1192fd08

DeiT-Small (distilled) ImageNet Transfer, two-stage, 256x256, five seeds. EXP-016 changes ONE factor versus the baseline EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015-256px-d4379a48: the Stage 2 learning-rate scheme (layer-wise LR decay) and unfreeze depth. Everything else follows the DenseNet121 recipe EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05 (dataset, augmentation, Stage 1, schedule and selection rule).

Stage 2 configuration: n08_lr5e-05 (last 8 of 12 encoder blocks + final LayerNorm (embeddings and earlier blocks frozen); LR = 5e-05 x 0.75 ** (11 - block index)); source: screening lock /content/training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-LLRD-SCREEN-016-256px-1192fd08/screening_selection_lock.json.
Screening experiment: EXP-VIT-DEITSMALL-TRANSFER-LLRD-SCREEN-016-256px-1192fd08. Several configurations were compared on Validation, so the Validation figures can be slightly optimistic.
The archived notebook copy carries the default PHASE; this run used PHASE = 'CONFIRM' (see experiment_config.json).

Dataset: CashewData_Split (13).zip (SHA256 e1647b72e10af04275dc760545f4ef17c9f44343ae119495b89ba1e3cd954e04, manifest 2cee0fb4818f93fba1dda43ab2e83615d33c5f7f3e4e15dc7c647436ff54b607).
Stage 2 rule: last 8 of 12 encoder blocks + final LayerNorm (embeddings and earlier blocks frozen); LR = 5e-05 x 0.75 ** (11 - block index) (14,196,480 of 21,689,472 backbone parameters; DenseNet121 reference: 2,129,920).
Pretrained weights: deit_small_distilled_patch16_224_imagenet at 224 px, position embedding resampled to 256 px (weights SHA256 9f41529cb0b85ef1a04a236180b72d6aad91fd1974e665644a3c00c5a67262de).
Deployment seed selected by Validation loss: 7777.
Test: FINAL_TEST_EVALUATED; configured real holdout: REAL_HOLDOUT_EVALUATED. Neither evaluation selects the model or seed.
The real holdout has 4 present classes (no not_cashew_leaf images) and is a descriptive audit, not a blind benchmark.
predictions_standard.csv: dataset_manifest_sha256 is the Split manifest hash for Validation/Test and the Holdout archive SHA256 for the Holdout.
Seed intervals describe variation across seeds on one fixed split, not a field-population CI.
No automatic promotion or WebApp replacement.
