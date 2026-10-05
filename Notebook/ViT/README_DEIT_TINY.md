# DeiT-tiny fine-tune on Cashew_dataV05

## Files

- `VIT_5seed.ipynb`: active DeiT-tiny fine-tuning notebook.
- `VIT_5seed_scratch_backup_2026-10-04.ipynb`: original scratch ViT backup, SHA-256 `B1E63A2E3733BC9F205381ECC4712B5A6233522E3DB21E186DC78D543D480DFB`.
- `deit_tiny_finetune_5seed.py`: matching Python source.

## Run in Colab

1. Open the updated `VIT_5seed.ipynb` in Colab and select a GPU runtime. If a Colab tab is already open with the old scratch notebook, upload/open the updated file and restart its runtime; re-running the old tab will still run the scratch code.
2. Run the package-install cell, then run all cells in order.
3. The notebook reads `MyDrive/Cashew_Leaf_model_result/Dataset/CashewData_Split (5).zip`. Its required SHA-256 is `37E76938B930025B88154D53EA56084E0828C57A0B0FB725387462C835EA8CDF`. A mismatch stops the run before training.
4. Results go to `MyDrive/Cashew_Leaf_model_result/training_results/cashew_datav05/EXP-DEIT-TINY-FINETUNE-5SEEDS-001/`.

The old scratch run `EXP-VIT-SCRATCH-5SEEDS-004` is preserved on Drive. Its existing `seed_42/model_summary.txt` correctly triggers the scratch notebook's overwrite guard; the DeiT notebook uses a separate experiment ID and directory.

The experiment uses version 1 of the KerasHub `deit_tiny_distilled_patch16_224_imagenet` preset. It trains the new classification head for 5 epochs with the backbone frozen, then fine-tunes the full model for up to 30 epochs. Five seeds use the same V05 archive, class mapping, and hyperparameters. The preset preprocessor is part of the saved `.keras` deployment model; inference inputs must be RGB `[0,255]`, with no additional division by 255.

`RUN_FINAL_TEST=False` by default. After all five validation runs are complete and the checkpoints/configuration are frozen, set `RUN_TRAINING=False` and `RUN_FINAL_TEST=True` to evaluate the locked Test split once. Do not use Test to choose hyperparameters or the deployment seed.

## Verification status

The V05 archive was checked for counts and cross-split byte duplicates. Python syntax and notebook JSON were checked. KerasHub preset loading, a forward pass, one synthetic batch for each training stage, checkpoint loading, and `.keras` save/reload were tested with TensorFlow 2.21.0 and KerasHub 0.32.0 on CPU. Full five-seed fine-tuning has not run here: TensorFlow 2.11+ does not use the local NVIDIA GPU on native Windows. Validation/Test metrics must be produced by the Colab run.

The new experiment is a **pretrained** model. Compare it with scratch baselines on the same V05 split, and report the different training mode and parameter count.
