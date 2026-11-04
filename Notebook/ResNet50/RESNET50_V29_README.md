# ResNet50 V29 — Real Healthy Train target 30

Notebook: `resnet50-v29.ipynb`. Base: `resnet50_transfer_exp009_recipe_split11_028_256px.ipynb` (V28, unexecuted).

## Run on Colab

Run the notebook top to bottom on a Colab GPU with access to the same frozen `CashewData_Split (11).zip` and Google Drive result folder as V28. The notebook audits the source ZIP, then selects **only** `train/healthy/` images whose filename begins `real_`. It counts those originals and creates enough offline variants to bring their total to 30. If 30 or more originals already exist, it adds none. The generated files are saved separately under `EXP-RESNET50-TRANSFER-5SEEDS-029/augmentation/train/healthy` and are appended to the Train loader once for all five seeds. The source ZIP, Validation and Test lists are unchanged.

The existing dataset audit checks frozen ZIP hash, split counts, image integrity, exact duplicates and the expected `real_` filename counts. V29 also checks that a `real_` filename stem does not occur across splits and saves `source_rule_manifest.csv` with the exact filenames and image hashes used. This **filename convention is the selected source rule**; it is not independent proof of capture source or `leaf_id`/`tree_id`. Without those IDs, tree-level cross-split leakage cannot be certified. The notebook states this limitation instead of claiming the group check passed.

Set `PREVIOUS_EXPERIMENT_DIR` to a completed V28 result folder only if one exists. If V28 metrics are unavailable, V29 marks the comparison unavailable. No V28 result directory is present in the workspace.

`RUN_FINAL_TEST` and `RUN_REAL_HOLDOUT` default to `False`. Compare augmentation targets using Validation only. After choosing a target and completing the Validation lock, run Final Test and Holdout separately if needed.

## Changes from V28

| Item | V28 | V29 |
|---|---|---|
| Real Healthy Train | Original frozen images | Originals plus fixed offline images, total target 30 |
| Offline augmentation | None | Seed 2029; mild flips, rotation, zoom, translation, brightness and contrast |
| Source rule | `real_` filename proxy in subgroup reports | Same prefix selects only Healthy Train sources; exact list and hashes saved |
| Target tuning evaluation | Final Test and Holdout enabled in config | Disabled until target choice is frozen |
| Training recipe | ImageNet ResNet50 transfer, five seeds, 256 px, stage 1 max 18, stage 2 max 16 | Same |

The model, online augmentation, optimizer, learning rates, callbacks, loss, batch size, seeds, preprocessing, class order and original Train/Validation/Test split are unchanged. The evaluation flag change prevents Test and Holdout from influencing target choice; it does not change training. Generated images are dependent variants of originals, not independent field samples.

The output keeps the existing seed directories, checkpoints, Validation reports and archive exports. V29 adds `augmentation/`, `comparison/` and `analysis.md`. The augmentation folder contains `augmentation_manifest.csv`, `source_rule_manifest.csv`, `real_healthy_augmentation_summary.json`, count chart, paired sample sheet and offline-plus-online sample sheet.

The notebook has passed JSON and code-cell syntax checks. The augmentation step has been smoke tested on synthetic files; full training has not been run in this workspace.
