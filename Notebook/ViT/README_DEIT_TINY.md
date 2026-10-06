# DeiT-tiny fine-tune on Cashew_dataV05

## Files

- `VIT_5seed.ipynb`: active DeiT-tiny fine-tuning notebook, experiment `006`.
- `vit_scratch_5seed.ipynb`: compact ViT scratch experiment `006`, with training-time summaries and blue/white matrices with black text.
- `VIT_5seed_scratch_backup_2026-10-04.ipynb`: original scratch ViT backup, SHA-256 `B1E63A2E3733BC9F205381ECC4712B5A6233522E3DB21E186DC78D543D480DFB`.
- `deit_tiny_finetune_5seed.py`: matching Python source.

## Run in Colab

1. Open the updated `VIT_5seed.ipynb` in Colab and select a GPU runtime. Upload/open the updated file, restart its runtime, and clear historical outputs; re-running an old tab still runs its old code. Retained notebook outputs are not results for `006`.
2. Run the package-install cell, then run all cells in order.
3. The notebook reads `MyDrive/Cashew_Leaf_model_result/Dataset/CashewData_Split (5).zip`. Its required SHA-256 is `37E76938B930025B88154D53EA56084E0828C57A0B0FB725387462C835EA8CDF`. A mismatch stops the run before training.
4. Results go to `MyDrive/Cashew_Leaf_model_result/training_results/cashew_datav05/EXP-DEIT-TINY-FINETUNE-5SEEDS-006/`. Earlier experiments are not overwritten.

The old scratch run `EXP-VIT-SCRATCH-5SEEDS-004` is preserved on Drive. Its existing `seed_42/model_summary.txt` correctly triggers the scratch notebook's overwrite guard; the DeiT notebook uses a separate experiment ID and directory.

The experiment uses version 1 of the KerasHub `deit_tiny_distilled_patch16_224_imagenet` preset. It trains the new classification head for 5 epochs with the backbone frozen, then fine-tunes the full model for up to 30 epochs. Five seeds use the same V05 archive, class mapping, and hyperparameters. The preset preprocessor is part of the saved `.keras` deployment model; inference inputs must be RGB `[0,255]`, with no additional division by 255.

`RUN_FINAL_TEST=False` by default. After all five validation runs are complete and the checkpoints/configuration are frozen, set `RUN_TRAINING=False` and `RUN_FINAL_TEST=True` to evaluate the locked V05 Test split once and, with `RUN_REAL_HOLDOUT=True`, the separate real-world holdout. If enabling Test after training, rerun the data-only holdout preflight before the Test drivers. Do not use either Test set to choose hyperparameters or the deployment seed. Resolve the overlap described below before unlocking.

## Verification status

The V05 archive was checked for counts and cross-split byte duplicates. Python syntax and notebook JSON were checked. KerasHub preset loading, a forward pass, one synthetic batch for each training stage, checkpoint loading, and `.keras` save/reload were tested with TensorFlow 2.21.0 and KerasHub 0.32.0 on CPU. Full five-seed fine-tuning has not run here: TensorFlow 2.11+ does not use the local NVIDIA GPU on native Windows. Validation/Test metrics must be produced by the Colab run.

The new experiment is a **pretrained** model. Compare it with scratch baselines on the same V05 split, and report the different training mode and parameter count.

## Training time and confusion matrices

`training_summary_5seeds.csv` records `head_training_seconds`, `finetune_training_seconds`, and their sum in `training_seconds`. The timers surround the two `model.fit` calls, including their epoch validation and callbacks/checkpoint writes. Post-training evaluation, prediction, and deployment-model saving are excluded. `training_time_mean_std_seconds.csv` reports the mean and sample standard deviation across the five seeds.

When loading older DeiT results without stage timers, the notebook labels their timing scope as legacy and warns that post-fit work was included. Their fit-only time cannot be recovered from the old summary.

Confusion-matrix figures use a white-to-light-blue scale (`#ffffff`, `#dbeafe`, `#93c5fd`) with black annotations. Run the figure cells again to regenerate existing plots with this style.

## External real-world holdout in 006

The embedded evaluator reads `MyDrive/Cashew_Leaf_model_result/Dataset/RL_Cashew_holdout.zip` on Colab. Set `REAL_HOLDOUT_ARCHIVE_OVERRIDE` for a different mounted/local ZIP location. No additional helper-file upload is needed. The archive is pinned by SHA-256 and checked against cached extraction and all V05 image bytes. No holdout images enter training or Validation. When Final Test is enabled from the start, data preflight runs before costly training.

The uploaded ZIP has 49 images: 13 anthracnose, 5 healthy, 24 leaf_miner, 7 red_rust, and **zero not_cashew_leaf**. Models keep five outputs. Headline holdout macro metrics use four supported true classes; absent-class per-class metrics are N/A, and predicting that absent class still counts as an error.

**Pending data decision:** `red_rust/IMG_3244.jpg` is byte-identical to V05 Test `red_rust/real_red_rust_029.jpg`. Preflight currently stops before training/Test when the Final Test gate is open. The recommended solution is to keep the ZIP unchanged but exclude that image from holdout evaluation, leaving 48 images; this exclusion has **not** been applied. Do not bypass the audit or change a frozen dataset after seeing metrics.

Results are written separately under `real_holdout/`: archive/manifest audit, immutable checkpoint/Validation-selection lock, per-seed predictions/reports/matrices, five-seed metrics, mean/sample SD and the Validation-selected deployment-seed result. Completed external predictions are read back, not rerun. Existing V05 `test_*` files are not replaced or merged.

See [comparison proposal and the 006 protocol](../../PROPOSAL_COMPARISON_SCRATCH_FINETUNE_AND_006.md). Training modes and hyperparameters remain unchanged; no fairness-design proposal has been applied.

CPU-only tests: `.venv/Scripts/python.exe Notebook/common/test_real_holdout.py`. Full 006 training and real-model Test/holdout inference have not run locally.
