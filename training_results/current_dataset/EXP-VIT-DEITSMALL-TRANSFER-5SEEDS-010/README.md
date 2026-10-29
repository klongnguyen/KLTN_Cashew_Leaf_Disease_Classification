# EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010 — Stage A
> **Cập nhật điều hướng 28/10/2026.** VALIDATION-ONLY DEIT-SMALL — không tự suy ra Test/triển khai từ Val. [README chính](../../../README.md) · [Danh mục EXP](../../../docs/EXPERIMENT_INDEX.md) · [Protocol](../../../docs/RESEARCH_PROTOCOL.md).


**Status (2026-10-22): Gate 0 and Gate 1 technical checks passed locally; Gate 2 five-seed training has not run.** The local TensorFlow 2.21.0 runtime on native Windows reports no GPU. Consequently there are no new training checkpoints, Validation metrics, confusion matrices, predictions, scientific outcome, or PROMOTED/NOT PROMOTED decision yet. `gate_status.json` is the machine-readable state. The Colab notebook is ready to execute the frozen protocol on a GPU; it stops before Gate 2 if no GPU is visible.

## Protocol

- Dataset: `CashewData_Split (8).zip`, SHA256 `75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7`; Train 4,790, Validation 1,430, Test 686. Class order: `anthracnose`, `healthy`, `leaf_miner`, `red_rust`, `not_cashew_leaf`.
- Pretrained model: KerasHub `kaggle://keras/deit/keras/deit_small_distilled_patch16_224_imagenet/1`, 21,668,357 parameters after replacing the head with 5 logits; `task.weights.h5` SHA256 `186714690069C17876BA8D6479F27E5C345AAC8A3B37574EE04B8B1D949AB72F`. The input is RGB float32 0–255, bilinear 224×224, then the preset preprocessor exactly once.
- Seeds: `42, 123, 2026, 3407, 7777`, each reset from the same ImageNet preset. Stage 1 updates only the 1,925-parameter head. Stage 2 reloads the same seed's best Stage 1 checkpoint, updates only encoder blocks 11 and 12 plus head (3,550,853 parameters).
- Batch 16; AdamW weight decay `1e-4`; Stage 1 LR `1e-4`; Stage 2 common LR `1e-5` with 3-epoch warmup then cosine decay to `1e-6`; maximum 15 epochs per stage, early stopping patience 5. Both stages save the minimum Validation loss checkpoint. Stage 2 does **not** claim differential LR.
- Train augmentation matches the EXP-007 recipe: horizontal flip, rotation `0.04`, zoom `±0.05`, translation `0.03`, contrast `0.08`. Validation has no augmentation. Head emits logits; cross entropy uses `from_logits=True`; softmax is reporting only.
- `test_enabled=false`, `historical_holdout_enabled=false`. Test members are read only for ZIP integrity audit. Test/Holdout never enter model training, inference, selection, or this notebook's figures.

The frozen values and pre-registered guardrails are in `experiment_config.json`. The selection rule picks an eligible stage per seed by minimum Validation loss, then an eligible seed by minimum Validation loss. Promotion additionally requires selected-stage 5-seed mean Macro F1 above Tiny A, seed 123 D2H below 18/900, and seed 123 guardrails. A missing improvement is reported as **NOT PROMOTED** after results exist. EXP-008's deployment seed 123/A stays unchanged.

## Gate checks performed

| Gate | Evidence | State |
|---|---|---|
| 0 | ZIP SHA/CRC, per-image SHA, 6,906-image manifest, class counts, decode, exact duplicate and pHash audit | Pass with baseline parity caveat |
| 1 | Real Train-image gradient steps, 5 logits, 384-D head input, frozen/open layer inventory, preprocessing and save/reload parity; a compiled `fit` step per stage | Pass on local CPU |
| 2 | Full 5 seeds and both stages | Blocked by unavailable TensorFlow GPU runtime |
| 3 | Validation comparison, 5-seed mean ± sample SD, selection lock | Not started |
| 4 | Validation error analysis, staged inference package and report | Not started |

Gate 0 found no broken images or exact duplicates. It found **226 cross-split pHash pairs with Hamming distance ≤6**. These are candidates for image/source-group review, not proof of leakage; group independence remains unverified. The Tiny A baseline is copied from EXP-008 Validation artifacts. The EXP-007 checkpoint archive was found on Drive, but full Tiny inference parity was not rerun locally. EXP-009 source notebook was reviewed; retrospective Test/Holdout results were not available locally and are not used here.

## Run on Colab

Open [`Notebook/ViT/vit_deit_small_transfer_5seeds_010.ipynb`](../../../Notebook/ViT/vit_deit_small_transfer_5seeds_010.ipynb) from the `exp/vit-deitsmall-transfer-010` branch and enable a GPU. Place the unchanged V08 ZIP at:

`MyDrive/Cashew_Leaf_model_result/Dataset/CashewData_Split (8).zip`

The notebook clones the branch, installs `requirements.lock.txt`, mounts Drive, reruns Gate 0 and Gate 1 in that runtime, then runs all five seeds. It writes to:

`MyDrive/Cashew_Leaf_model_result/training_results/current_dataset/EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010`

The notebook records its source Git commit. Completed seed outputs are reused only after checkpoint SHA checks. It refuses to overwrite an incomplete seed folder, preserving evidence for review. Large `.keras` checkpoints and the preset cache remain on Drive, not GitHub. Checkpoint SHA256 values are generated in each seed's `metrics_stage*.json`, then the chosen one is recorded in `selection/validation_selection_lock.json` and `deployment_package/metadata.json` after Gates 2–4 finish.

## Expected outputs after GPU run

Each seed yields Stage 1/2 history, trained checkpoint, metric JSON, raw/normalized confusion matrices, per-class report, per-image Validation probabilities/predictions, errors and the 35 `real_` Validation cases. `aggregate/` yields the 5-seed table, mean ± sample SD, Stage 1/2 ablation, Tiny/Small comparison, per-class and D2H tables. `figures/` yields learning curves, confusion matrices, class recall, Tiny/Small comparison and error examples. `analysis/SCIENTIFIC_REPORT.md` and `selection/validation_selection_lock.json` are generated only after all five seeds complete. The staging package is research-only; it does not modify the WebApp.

See `analysis/ARTIFACT_INVENTORY.md` and `analysis/METHOD_AND_LEAKAGE_REVIEW.md` for provenance and current limitations.
