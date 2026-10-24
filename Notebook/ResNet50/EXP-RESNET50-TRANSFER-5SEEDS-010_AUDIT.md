# ResNet50 transfer 010: why early Validation looks high

This review uses the supplied `README (4).md` and the original `EXP-RESNET50-TRANSFER-5SEEDS-009_ANALYSIS.zip` (SHA256 `65b77db413d3b9e0c559686116f023c4d299a666e6533fe905d8d2246ec03762`). `README (3).md` is project context; `README (2).md` describes a different DenseNet scratch run. The 009 ZIP CSV/JSON records are the numerical source below. 010 has **not** been trained.

## Diagnosis

- Before training, full Validation accuracy was **9.23–27.06%** over five seeds. The first Train batch was **6.25–18.75%**. The model did not start with a 78–86% Validation score.
- Each first epoch processed `ceil(4790 / 16) = 300` minibatches. The fixed 100-image balanced Validation inference probe progressed from **7–26% at step 0**, through **26–35% at step 50**, **43–54% at step 100**, **55–70% at step 150**, and **75–86% at step 300**. Its rise is visible *within* the epoch. Full Validation measured at epoch end was **78.32–86.08%**.
- ResNet50 starts with ImageNet features. Only the random five-class head learned in Stage 1, and 300 updates are enough for rapid separation of these V08 classes. The online Train metric averages training-mode predictions while augmentation/dropout are active; Validation is measured at the end in inference mode. Plotting them as a same-time pair at epoch 1 is misleading.
- Across selected 009 checkpoints, clean Train probe mean was **97.16%** and full Validation mean **95.23%**: about **1.93 percentage points** apart. The more important gap is full Validation **95.23%** versus the 35 `real_` Validation images **73.71%** and the previously exposed 48-image holdout **75.83%**. The 35-image filename group is a noisy proxy, and the holdout is not a new blind test.
- Exact cross-split duplicates were **0** in the 009 audit. The **51 pHash candidate pairs** have not all been manually checked for the same leaf, plant, session or garden. Ten candidate Validation paths were correct in all five seeds, yet removing those ten still left high Validation accuracy. This does not prove leakage absent or present.

The high first **epoch-end** score is explained by pretrained features and the timing of the measurement. Current evidence does not identify a software bug that can be repaired by forcing a lower accuracy. Reducing the score for appearance would not establish better generalization.

## What 010 changes

The V08 ZIP, class order, five seeds, Stage 1/2 learning rates, partial fine-tuning scope, checkpoint rule, Final Test and historical holdout gates are preserved from 009. The training recipe adds only train-time brightness (`0.12` of the 0–255 range) and translation (`0.05` of image size) to the existing flip/rotation/zoom/contrast stack. This is a fixed domain-robustness experiment for lighting and framing variation; its outcome must be measured.

The primary training panel starts at **optimizer step 0** and plots fixed, balanced Train and Validation probes under the **same inference conditions**, with accuracy and cross-entropy loss. It displays the 35-image `real_` Validation subgroup as a separate dotted accuracy curve. The original online Train/full Validation epoch panel remains available under a different filename. No diagnostic probe or real subgroup chooses a checkpoint, seed or hyperparameter.

Both learning-curve panels now mark the **measured full Validation score before fitting** at step 0. The full Validation epoch panel connects that point to the measured end of epoch 1; it does not replace or lower any result. Epoch 1 represents approximately 300 optimizer updates. The primary probe panel labels the full-split point separately because its sample size differs from the fixed balanced probes.

Every per-seed and five-seed aggregate confusion matrix for Validation, Final Test and historical real holdout uses the same white-to-blue scale, a white background, and legible dark or white annotations according to cell intensity. These are display changes only; the confusion-matrix CSV values and model predictions are unchanged.

010 exports contact sheets for all pHash candidate pairs to support manual leakage review. It does not change V08 or automatically remove images. A fresh independent field set with source grouping and all five labels remains necessary for a credible real-world estimate.

## Verification

The initial 010 audit cell had a forward reference to the JSON helper defined in cell 12. The contact-sheet manifest now writes JSON directly in cell 8. The corrected block was executed locally with zero and one candidate pair; both wrote the expected manifest, and the one-pair case wrote its image. The embedded notebook snapshot was checked to include this fix.

The output-free 010 notebook has 26 cells. Local smoke with `weights=None` verified Keras Caffe preprocessing parity, train-time augmentation forward pass, Stage 1 freeze, Stage 2's 4,459,520 trainable backbone parameters, fixed BatchNorm statistics, checkpoint reload and clean-process `.keras` parity. A synthetic fit verified step-zero/intra-epoch matched-probe logging; synthetic images verified contact-sheet generation. Temporary FULL and ANALYSIS ZIPs each contained the exact reconstructed 010 notebook. ImageNet/V08 smoke and five-seed training must run in Colab before reporting 010 metrics.
