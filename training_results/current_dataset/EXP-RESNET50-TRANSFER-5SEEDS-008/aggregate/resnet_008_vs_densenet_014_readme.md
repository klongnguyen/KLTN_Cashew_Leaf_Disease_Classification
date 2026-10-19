# Controlled dataset, different training methods

Both EXP-RESNET50-TRANSFER-5SEEDS-008 and EXP-DENSENET121-TRANSFER-5SEEDS-014 use V08 with the same manifest and 5 fixed seeds. The comparison remains **method-level**, because DenseNet121-014 unfreezes the last Dense Block while ResNet50-008 freezes the backbone throughout.

| Split | ResNet50-008 | DenseNet121-014 (evaluated in 015 where noted) |
|---|---|---|
| Validation Accuracy | 95.09 ± 0.22% | 97.10 ± 0.20% |
| Validation Macro F1 | 94.91 ± 0.23% | 96.96 ± 0.21% |
| 35 real_ Validation subset Accuracy | 71.43% | 79.43% |
| 48 historical previously-exposed Holdout Accuracy | 77.08 ± 2.55% | 80.83 ± 0.93% |
| Locked Final Test V08 Accuracy | **93.24 ± 0.24% (done)** | **pending for EXP-014** |

Do not compare this V08 ResNet50 to ResNet50 Scratch-008 V07 and attribute the difference to pretrained weights alone. Real Holdout is **not a fresh blind sample**, and SD over seeds is not a generalization confidence interval.

[ResNet50-008](../README.md) · [DenseNet121-014](../../EXP-DENSENET121-TRANSFER-5SEEDS-014/README.md) · [DenseNet121 Holdout 015](../../EXP-DENSENET121-TRANSFER-5SEEDS-015/README.md).
