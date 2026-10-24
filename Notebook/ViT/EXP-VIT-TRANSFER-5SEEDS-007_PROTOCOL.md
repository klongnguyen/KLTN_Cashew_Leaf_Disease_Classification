# ViT transfer 007: implementation and run status

**Status:** Notebook generated but not trained. No 007 accuracy, checkpoint or
real-holdout result is claimed.

## Fixed experiment

| Setting | Value |
| --- | --- |
| Notebook | vit_transfer_5seed_007.ipynb |
| Output ID | EXP-VIT-TRANSFER-5SEEDS-007 |
| Dataset | CashewData_Split (8).zip, Drive file ID `1vaaOwAi72byaxFb7w_ixd_Y2HboKn2-G`, 1,058,695,554 bytes, SHA-256 75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7 |
| Architecture | KerasHub DeiT-tiny distilled patch16/224 ImageNet preset, new five-class logits head |
| Class order | anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf |
| Seeds | 42, 123, 2026, 3407, 7777 |
| Input | RGB float32 [0,255], 224×224; preset preprocessor embedded once |
| Stage 1 | backbone frozen, head LR 1e-4, up to 8 epochs |
| Stage 2 | full fine-tuning, LR peak 5e-6 with warm-up/cosine decay, up to 12 epochs |
| Optimizer | AdamW, weight decay 1e-4, batch 16 |
| Checkpoint | best Validation loss per stage; lower-loss stage selected |
| Deployment seed | best Validation macro F1, then Validation loss |
| Evaluation | Final Test and historical real holdout enabled after immutable Validation lock |

V08 has 4,790 Train, 1,430 Validation and 686 Test images. The pinned historical
holdout archive has 49 images; exactly one image is byte-identical to V08 Test.
The notebook records and excludes only that image, then scores 48 images.
Four true holdout classes have support. This set was previously inspected and
is diagnostic; it is not a fresh blind field test.

The notebook records pre-fit Validation accuracy, augmented online Train and
full Validation curves, stage training time, image decode/resize delivery time,
Validation and Test inference time, per-class reports, and white-blue
confusion matrices. Both FULL and ANALYSIS ZIPs embed the exact 007 notebook.
The ANALYSIS ZIP omits model weights and saved model files.

## Local verification

- The V08 ZIP hash and all split/class counts match the frozen configuration.
- All 6,906 V08 image hashes were checked against the pinned holdout archive.
  The preflight returned 48 scored images and precisely the known Test overlap.
- All 21 notebook cells compile and have no execution outputs. A temporary ZIP
  packaging smoke verified the exact notebook bytes in FULL and ANALYSIS.
- The archive resolver was checked with nested Drive-style paths, an explicit
  override, a missing override and a same-name ZIP with a wrong SHA-256. Both
  real local archives pass the pinned-hash lookup.
- A mocked Drive API download checked metadata, streamed bytes, final SHA-256
  validation and cleanup of a rejected download. An actual Colab download has
  not been run here.
- An offline tiny DeiT structural smoke checks one-pass preprocessing,
  Stage 1/2 trainability and saved-model reload.
- The full ImageNet preset cannot be downloaded in this local sandbox because
  outbound Kaggle access is blocked. The original 006 preset was tested with
  KerasHub 0.32.0; the 007 notebook's mandatory smoke must pass in Colab before
  five-seed training starts.

The user supplied [this Dataset folder](https://drive.google.com/drive/folders/1-Mrqs8Y9dwLexSKIVKBvyYvvQ8X8fXti).
It contains the V08 ZIP and `RL_Cashew_holdout.zip` (Drive file ID
`1PBry_y2bMsvd5_YdmEUy9erSFarS3O52`, 150,455,582 bytes). Run 007 in a GPU
Colab runtime. The default mounted locations are under
`/content/drive/MyDrive/Cashew_Leaf_model_result/Dataset/`. If the files are
elsewhere in mounted My Drive or Shared drives, the notebook searches for the
exact filenames. If that shared folder is not mounted, it authenticates the
Colab user and streams both pinned file IDs through the Drive API into
`/content/verified_drive_archives/`. In all cases, it checks SHA-256 before
extraction. The user must authenticate with an account that can open the
shared folder. For a mounted ZIP, `DATASET_ARCHIVE_OVERRIDE` and
`REAL_HOLDOUT_ARCHIVE_OVERRIDE` accept exact paths. The notebook writes only
to its new experiment ID.
