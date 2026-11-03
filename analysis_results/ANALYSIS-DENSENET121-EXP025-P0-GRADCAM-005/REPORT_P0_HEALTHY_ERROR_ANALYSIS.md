# P0 Error Analysis — DenseNet121 EXP-024/025

## Executive summary
Loaded 7,835 prediction rows for 1,567 distinct images (1,447 Validation and 120 Holdout, five seeds each). Read and measured 30 original images. Seed 42: 4/222 Healthy Validation and 5/13 Healthy Holdout errors; 10 Validation and 3 Holdout diseased images predicted Healthy. Grad-CAM is blocked by the strict probability parity gate, not by missing sources.

## Source and checkpoint integrity
{
  "split11": {
    "path": "D:\\Study\\KhoaLuanTotNghiep\\Resource\\dataset\\CashewData_Split (11).zip",
    "sha256": "5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49",
    "expected": "5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49",
    "verified": true
  },
  "holdout05": {
    "path": "D:\\Study\\KhoaLuanTotNghiep\\Resource\\dataset\\RL_Cashew_holdout (5).zip",
    "sha256": "8d5c3758b5869c3ae19238fa8d78e53ad15bb40544f4c70f632602896d80045f",
    "expected": "8d5c3758b5869c3ae19238fa8d78e53ad15bb40544f4c70f632602896d80045f",
    "verified": true
  },
  "seed42": {
    "path": "D:\\Study\\KhoaLuanTotNghiep\\Resource\\analysis\\source_024\\seed_42\\best.weights.h5",
    "sha256": "fddcde3988af72b4446468a813a0d516f45663459fd7c761bec4aed58ff662c5",
    "expected": "fddcde3988af72b4446468a813a0d516f45663459fd7c761bec4aed58ff662c5",
    "verified": true
  },
  "exp024_full": {
    "path": "D:\\Study\\KhoaLuanTotNghiep\\Resource\\training_results\\current_dataset\\EXP-DENSENET121-TRANSFER-5SEEDS-024-256px-f9fd5d5f_FULL.zip",
    "sha256": "ff2efb4fcd6a20d7b176a7aedf794a938a1a3dc39c73556ea432f79e9997e4d1"
  },
  "holdout_cached_inventory": {
    "path": "D:\\Study\\KhoaLuanTotNghiep\\Resource\\analysis_results\\ANALYSIS-DENSENET121-EXP025-P0-GRADCAM-001\\tables\\error_inventory_all_seeds.csv",
    "sha256": "c4d69a54afe7f99795bbff66ab5ecae9d369a1d47be8be16b3ec64043d05a032",
    "note": "Derived from EXP-025 ANALYSIS ZIP in earlier run; original ZIP unavailable now"
  }
}

## Prediction error inventory
See `tables/error_inventory_all_seeds.csv` and `tables/p0_priority_cases.csv`. Validation predictions came directly from the EXP-024 FULL ZIP. Holdout predictions came from the cached inventory derived from EXP-025 ANALYSIS during the earlier run. The source EXP-025 ZIP is no longer at its old Downloads path. All five seeds misclassified the five priority Healthy Holdout images (`027`, `029`, `033`, `034`, `036`) as Anthracnose. Anthracnose `078` was predicted Healthy by all five seeds. Leaf Miner `131` was predicted Healthy by three seeds, while all five predictions were wrong. Healthy `022`–`026` were correct for all five seeds.

## Image and label audit
Original images and review sheets are in `figures/`. The blinded sheet uses IDs mapped by `tables/review_blinded_template.csv`; no model predictions or confidence are shown in that sheet. No human label decision has been entered and no label was changed. In the unblinded comparison, several misclassified Healthy images visibly have darker regions or small spots; this is an observation, not a diagnosis of Anthracnose or a causal explanation. Anthracnose `078` has visible small dark spots; Leaf Miner `131` has a discolored area near the lower edge. Expert review is required to classify symptoms or question labels.

## Image quality comparison

`tables/image_quality_features.csv` and `tables/group_comparison_summary.csv` contain whole-image brightness, saturation, gray-level contrast and Laplacian blur proxy. The five consistently wrong Healthy images have mean HSV V of 126.4 versus 131.6 for five consistently correct controls; mean S is 106.7 versus 125.1. Three seed-dependent Healthy cases are kept separate. These tiny groups are unmatched by tree, visit or camera, and there is no verified leaf ROI; the differences are descriptive only.

## Grad-CAM and limitations
The seed-42 checkpoint and both image ZIP hashes are verified. The saved `.keras` model and the model rebuilt from the seed-42 checkpoint agree on the local CPU. The unpooled feature-map branch agrees with the local classifier to 2.7e-7 on the first smoke image. Five smoke images across four classes all retain the logged predicted label, but confidence differences range from 0.000115 to 0.000726, exceeding the required 0.0001 threshold on every image. The original run used Keras 3.13.2 on Linux/GPU; the local runtime has Keras 3.15.1 on Windows/CPU. This is a plausible source of numerical drift, not a proven cause. No CAM is exported until parity is resolved or the review protocol explicitly changes the threshold. See `tables/parity_smoke.csv`.

## Next controlled experiment
Complete blinded human label review, then reproduce inference in the original Keras/Linux/GPU environment to test parity. If parity passes, generate Grad-CAM from the verified pre-pooling 4D feature map. Holdout 05 has only 13 Healthy images and is historically exposed; use a new independent five-class field test for final assessment. Do not tune a new model against this Holdout.


## Visual audit and parity gate

Verified ZIP hashes and measured 30 original images. The first classifier parity check failed on `anthracnose/real_anthracnose_051.jpg` (absolute confidence difference 0.00048798); a further four-image diagnostic also failed the 0.0001 confidence threshold. The first unpooled branch difference was 2.682209e-07. No CAM was exported. See `tables/parity_smoke.csv`, `figures/healthy_originals_blinded.png`, `figures/healthy_correct_vs_wrong.png`, and `figures/quality_feature_comparison.png`.
