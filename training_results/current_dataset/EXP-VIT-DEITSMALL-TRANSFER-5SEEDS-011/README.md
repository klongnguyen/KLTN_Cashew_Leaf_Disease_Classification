# EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011

Source notebook: Notebook/ViT/vit_deit_small_transfer_5seeds_011.ipynb.
Source branch: exp/vit-deitsmall-extended-epochs-011.

This is a new, untrained experiment specification. It extends the EXP-010
DeiT-Small two-stage recipe to at most 30 epochs per stage and EarlyStopping
patience 7. Five fixed seeds: 42, 123, 2026, 3407, 7777. Batch size is 16.
Stage 1 trains the head only. Stage 2 starts from the best same-seed Stage 1
checkpoint and trains only the last two encoder blocks plus head.

Dataset: CashewData_Split (8).zip, SHA256
75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7.
Only Train V08 updates weights. Only Validation V08 is used for development
metrics and selection. Test V08 is read only by the ZIP integrity audit.
Historical Holdout is not used.

Run notebook cells in order on a Colab GPU. The last VIEW RESULTS ONLY cell
can be run independently in a fresh runtime after results exist. Each seed
is a separate notebook cell. Completed stages and seeds are SHA256-checked
on resume; partial stages are preserved under _incomplete_backups before
their stage restarts. This is a new run from pretrained weights, never a
continuation from EXP-010 fine-tuned checkpoints.

No EXP-011 metrics, figures, scientific claims, or promotion status exist
until the notebook actually runs and verifies them.
