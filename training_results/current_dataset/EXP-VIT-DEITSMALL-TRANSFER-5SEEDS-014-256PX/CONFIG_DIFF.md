# EXP-011 → EXP-014 protocol differences

| Item | EXP-011 | EXP-014 |
|---|---|---|
| Dataset | V08 Split (8) | V11 Split (11), SHA256 `5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49` |
| Input | 224×224 | 256×256; 16×16 patch grid, positional interpolation required |
| Model source | KerasHub Kaggle preset, CLS head | Facebook DeiT-Small distilled through Transformers, CLS and DIST heads averaged |
| Framework | TensorFlow/Keras 3 | PyTorch/Transformers |
| Augmentation | flip, rotation .04, zoom .05, translation .03, contrast .08 | flip, rotation .08, zoom .10, contrast .10, brightness .08, no translation |
| Stages | 30/30 epochs, frozen then last two blocks | Same caps, patience 7, AdamW 1e-4/1e-5, 3-epoch warmup + cosine in Stage 2 |
| Selection | Prior V08 rules | V11 Validation minimum loss, tie Macro F1, Stage 1, then seed number |

EXP-011 and EXP-014 differ in dataset, size, augmentation, framework, checkpoint representation, and head behavior. Their difference is not a one-factor ablation. The switch to Transformers follows [KerasHub's CLS-only DeiTImageClassifier documentation](https://keras.io/keras_hub/api/models/deit/deit_image_classifier/) and [Transformers' distilled two-head and interpolation API](https://huggingface.co/docs/transformers/en/model_doc/deit). Exact source revision and artifact SHA256 are captured when the notebook runs.
