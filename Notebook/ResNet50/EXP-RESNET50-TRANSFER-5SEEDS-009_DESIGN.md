# ResNet50 transfer 009: rationale and protocol

Source reviewed: `EXP-RESNET50-TRANSFER-5SEEDS-008_ANALYSIS.zip` supplied with the request (SHA256 `a408af9b540208045f2214c71cb6072b75e8b1442efacd12d88ae510107bc24f`). Its CSV/JSON records, rather than a rendered learning curve alone, inform this design. 009 has **not** been trained.

## What 008 shows

| Measurement | Five-seed 008 result |
|---|---:|
| Pre-fit Validation accuracy | 9.23–27.06% |
| First Train batch accuracy | 6.25–18.75% |
| End of Stage 1 epoch 1 Train accuracy | 68.58–72.30% |
| End of Stage 1 epoch 1 Validation accuracy | 88.95–92.17% |
| Selected clean Train probe / Validation accuracy | 97.08% / 95.09% (five-seed means) |
| Final Test accuracy | 93.24% mean |
| Historical 48-image holdout accuracy | 77.08% mean |

The first Validation score follows about 300 minibatch updates to the head, and the model uses pretrained ImageNet features. It is not an epoch-zero or first-batch result. Online Train uses augmentation/dropout while Validation uses inference mode; its early Train/Val gap is not a paired comparison. The later clean Train/Validation gap is about 1.99 percentage points, which does not show severe within-V08 overfitting. The much larger V08/holdout gap is a domain-generalization concern. The holdout has already been exposed and is only a historical comparison, not a blind tuning target.

## Fixed 009 changes

- Preserve the exact V08 archive, class order, ResNet50 Caffe preprocessing, five seeds, Final Test and historical holdout gates.
- Stage 1 keeps the ImageNet backbone frozen, lowers head LR from `3e-4` to `1e-4`, allows up to 18 epochs, and uses Validation-loss early stopping. This slows optimizer updates; it does not guarantee a chosen curve shape.
- Stage 2 reopens only `conv5_block3_*` non-BatchNorm weights (4,459,520 backbone trainable parameters in local architecture smoke) at LR `3e-6`, up to 8 epochs. Other backbone layers and BatchNorm statistics stay frozen. This is a cautious partial fine-tune, weaker than the 007 Stage 2 learning rate of `1e-5`.
- A fixed balanced probe of 20 images per class per split is evaluated in **the same inference mode** before fitting, every 50 batches within the first Stage 1 epoch, and at epoch ends. The measured Train/Validation curves are saved as CSV and a five-seed panel. They do not select a checkpoint or seed. The ordinary full Validation loss still controls selection.
- The notebook exports selected and stage candidate checkpoints, model package, training/inference times, Validation/Test/holdout diagnostics and two ZIPs. Both ZIPs include `notebook/resnet50_transfer_5seed_009.ipynb` under the experiment folder.

The curves may still rise quickly or separate. A visually gradual curve is not a performance criterion. The next useful independent check is a fresh field dataset grouped by plant/capture session and covering all five classes, including `not_cashew_leaf`.

## Verification before Colab training

The output-free 009 notebook has 26 cells. Local smoke with randomly initialized backbone weights verified Caffe preprocessing parity, Stage 1 freeze, Stage 2 trainability, BatchNorm invariance and clean-process `.keras` reload. A separate synthetic fit verified the matched-probe callback records step 0, a point within epoch 1 and both epoch ends without changing the model. Temporary FULL/ANALYSIS ZIPs both contained the exact reconstructed 009 notebook. The mandatory ImageNet/V08 pipeline smoke and five-seed run remain to be executed in Colab.
