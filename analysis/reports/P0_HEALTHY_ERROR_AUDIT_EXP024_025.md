# P0 Healthy error audit — DenseNet121 EXP-024/025

## Scope and sources

This is a read-only error audit. It does not train, change labels, select a new seed, or rerun Test/Holdout evaluation. The selected model is EXP-024 256 px, seed 42. EXP-025 provides saved Holdout predictions. The complete local output, including image sheets and CSV tables, is under `analysis_results/ANALYSIS-DENSENET121-EXP025-P0-GRADCAM-005/` and is intentionally excluded from Git because it contains source image pixels.

The following source hashes were verified locally:

| Source | SHA-256 |
|---|---|
| EXP-024 FULL ZIP | `ff2efb4fcd6a20d7b176a7aedf794a938a1a3dc39c73556ea432f79e9997e4d1` |
| Split (11) ZIP | `5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49` |
| Holdout (5) ZIP | `8d5c3758b5869c3ae19238fa8d78e53ad15bb40544f4c70f632602896d80045f` |
| Seed-42 `best.weights.h5` | `fddcde3988af72b4446468a813a0d516f45663459fd7c761bec4aed58ff662c5` |

The EXP-024 config and seed-42 weight hashes match its Validation selection lock. Validation predictions came directly from EXP-024 FULL ZIP. Holdout predictions came from a cached inventory created from EXP-025 ANALYSIS ZIP during an earlier run; that ZIP was no longer available at its former Downloads path for this final run.

## Prediction findings

There are 7,835 saved prediction rows for 1,567 unique images: five seeds on 1,447 Validation images and 120 Holdout images. Seed 42 misclassifies 4 of 222 Healthy Validation images and 5 of 13 Healthy Holdout images. It predicts Healthy for 10 diseased Validation images and 3 diseased Holdout images.

All five seeds predict the five priority Healthy Holdout images `027`, `029`, `033`, `034`, `036` as Anthracnose. All five predict Anthracnose `078` as Healthy. Three of five predict Leaf Miner `131` as Healthy, while all five predictions on that image are wrong. Healthy controls `022`–`026` are correct for all five seeds.

## Image review and quality

The audit read 27 priority images plus 3 seed-dependent Healthy Holdout images from the verified original ZIPs. A blinded contact sheet and empty human review form are available locally. No expert review decision has been entered and no label was changed.

Whole-image mean HSV V is 126.4 for five consistently wrong Healthy images and 131.6 for five consistently correct controls. Mean HSV S is 106.7 versus 125.1. These groups are small and unmatched by tree, visit, or camera; no verified leaf ROI was available. Visible darker regions and small spots in some wrong images are observations, not a disease diagnosis or a causal explanation.

## Grad-CAM gate

No Grad-CAM heatmaps were exported. The saved `.keras` classifier and the model rebuilt from the seed-42 checkpoint agree locally; the unpooled 4D feature-map branch differs from the local classifier by only about `2.7e-7` on the first smoke image. Across five smoke images, predicted labels match the EXP-025 logs, but confidence differences range from `0.000115` to `0.000726`. Every image exceeds the plan's `1e-4` parity threshold. The original run used TensorFlow 2.21.0 / Keras 3.13.2 on Linux/GPU; the local check used TensorFlow 2.21.0 / Keras 3.15.1 on Windows/CPU. Environment differences are a plausible explanation, not a confirmed cause.

Use `analysis/notebooks/densenet121_exp025_p0_parity_colab.ipynb` in a matching Colab runtime to recheck parity before producing CAMs. Holdout (5) is historically exposed and has only 13 Healthy images; it is diagnostic, not a new blind benchmark.
