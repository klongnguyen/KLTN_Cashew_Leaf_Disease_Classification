# EXP-032 protocol — DenseNet121 + YOLO26s

**R1 implementation correction:** The first notebook scored classifier images in batches but exported a one-image deployment API. A real Holdout consistency check found different YOLO box counts and a small DenseNet numeric drift. R1 scores YOLO and DenseNet one image at a time, uses fixed YOLO `rect=False`, and writes a new frozen-config/result namespace. Verified model weights from the earlier run may be reused; its fusion tables and thresholds are not reused. Holdout remains a descriptive audit.

## Goal and fixed inputs

Build one importable pipeline returning a five-class label, Validation-calibrated confidence, top two labels, an abstain flag, and lesion boxes. Part A studies frozen EXP-029 probabilities; Part B retrains the EXP-029 recipe and is the only basis for deployment claims. DenseNet classes are anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf. YOLO classes are anthracnose, leaf_miner, red_rust. Source archives, Drive IDs, bytes/hashes, detector commit and checkpoint hash are pinned in the notebook. No split or Holdout membership changes.

Part B uses five seeds (42, 123, 2026, 3407, 7777), two stages, 256 px, batch 16, AdamW weight decay 1e-4, dropout 0.40, EXP-029 photometric augmentation, no real-image oversampling, and Validation loss for checkpoint selection. Stage 1 trains the head at 3e-4 for up to 12 epochs. Stage 2 trains non-BatchNorm `conv5_*` and the head at 1e-5 for up to 30 epochs. Every completed seed is saved and uploaded separately, and may be restored after disconnect.

Each seed ZIP is a recovery checkpoint. After all evaluation and pipeline consistency gates pass, the final cell uploads `ANALYSIS.zip` (reports, tables, per-seed predictions; no weights) and `FULL.zip` (the complete result folder, including models and pipeline). Existing Drive files are verified and never overwritten with different content.

A separate manual cell can upload an `ANALYSIS_SNAPSHOT` immediately from existing tables, even when the pipeline consistency gate has not passed. Its manifest explicitly marks it as a snapshot rather than a certified final result.

The figure cell immediately before the snapshot upload reads the saved aggregate tables and writes `aggregate/figures/`: accuracy with Wilson intervals, paired bootstrap differences from F0, confusion matrices, class recall, risk–coverage, ECE, and YOLO Test localization F1. It does not refit or retune a rule. Run it before uploading a snapshot so that the figures are included. Holdout plots are labeled descriptive audits.

## Validation lock and fusion

Use mean softmax probabilities across all five seeds. Compare F0 (DenseNet), F1 (EXP-031 anthracnose gate, tau 0.60), F2 (zero-evidence anthracnose gate), F3 (logistic stacker on five probabilities plus nine detector features), F3a (probabilities only), and F3b (detector only). F2 is **post hoc**. Detector features come from boxes at confidence 0.05, IoU 0.7, image size 640. Remove Validation images with pixel-exact or pHash Hamming ≤4 overlap with YOLO Train. Group 5-fold CV by pHash cluster; choose C from 0.1, 0.3, 1, 3, 10 using log loss. Fit temperature and the smallest threshold reaching 98% selective accuracy on clean Validation. Write and print the `frozen_config.json` SHA256 before reading Test or Holdout probabilities. Temperature fitted on stacker training predictions may be optimistic; ECE is reported.

The feature set was chosen after an earlier exploratory Test/Holdout glance. That run showed roughly 96.9% Test and 97.62% Holdout (three anthracnose→red_rust errors). This exposed Holdout result is neither a target nor a blind claim. Holdout 09 is a **descriptive audit, not a blind benchmark**.

## Evaluation and decision

Report image accuracy with Wilson 95% interval, macro-F1, recall by class, confusion matrices, ECE, top-2 accuracy, error taxonomy, abstention, and paired 10,000-resample intervals for each rule versus F0 on Test, Holdout, and their gap. Report Holdout n=124 after excluding `real_healthy_40.jpg` and `real_healthy_42.jpg` as a secondary metric; retain their healthy ground truth. Exact Test/Holdout pixel overlap with YOLO Train aborts; near matches are listed and Test is reported with and without them.

Choose `BOX_CONF` on YOLO Validation by macro F1 at IoU 0.5, then apply class-wise NMS 0.5, final-label consistency and a cap of 20 boxes. Healthy, not_cashew_leaf and abstain show no boxes. Score detector and fused-display localization on YOLO Test (412 images), including `real_` and near-excluded subsets; mAP uses boxes retained from confidence 0.001. YOLO Validation is slightly optimistic because the checkpoint was selected there. Holdout has no box labels; its first six sorted images per class are qualitative only.

F3 is the Part B default only if Test is within 1 point of F0, its Holdout improvement over F0 has bootstrap lower bound above zero, and it beats F3a on Holdout. Otherwise F1 is used unless it regresses by over 1 Test point, in which case F2 is used with its post-hoc warning. The gap below 10 points is descriptive, not a success criterion. No second configuration is tried on Holdout. Optional `blind10` is off by default and runs only after the final Holdout report.
