# EXP-Y26S-NEG-V02-001-CLASS-WEIGHTED

- Architecture: **YOLO26s**; Dataset: **Cashew YOLO Master Final NEG V02**; Train/Validation/Test: **3152/951/412**.
- Negative images: Train **276**, Validation **58**; Test **unchanged, locked**. Healthy and not-cashew are 0-box background images, not fourth/fifth classes.
- Model config: `cls_pw=0.25` (Leaf Miner weight 1.5752), seed 42, image 640px, AdamW, batch 8, up to 80 epochs.
- **Validation:** Precision 68.95%, Recall 66.16%, mAP50 71.31%, mAP50-95 40.08%.
- **Leaf Miner:** Precision 75.43%, Recall 69.19%, AP50 78.42%, AP50-95 52.85%.
- Trained 61 epochs, about 16.0 minutes.
- Original file supplied in conversation: `EXP-Y26S-NEG-V02-001_ANALYSIS.zip` — SHA256 `2028def8b43046a3c4532252b02714291a750bab97d3cdfefd34cd9ee4e035fa`.
- [Negative image false-positive evaluation](../../negative_image_evaluation/README.md).

**Scope / provenance:** these are extracted summary files. The original ANALYSIS.zip contains learning curves, PR/confidence curves, confusion matrices, per-image failure CSV and prediction JPGs; those original binary artifacts have **not yet been committed to GitHub**. Use the archive synchronization script in this folder's parent directory to import them.

**Limitations:** only one seed and Validation evaluation; classification split unchanged, but NEG V02 has one fewer Anthracnose and one fewer Red Rust training annotation than V003. Avoid attributing all differences exclusively to negative images.
