# EXP-Y8S-NEG-V02-001-CLASS-WEIGHTED

- Architecture: **YOLOv8s**; Dataset: **Cashew YOLO Master Final NEG V02**; Train/Validation/Test: **3152/951/412**.
- Negative images: Train **276**, Validation **58**; Test **unchanged, locked**. Healthy and not-cashew are 0-box background images, not fourth/fifth classes.
- Model config: `cls_pw=0.25` (Leaf Miner weight 1.5752), seed 42, image 640px, AdamW, batch 8, up to 80 epochs.
- **Validation:** Precision 69.00%, Recall 64.74%, mAP50 69.90%, mAP50-95 38.67%.
- **Leaf Miner:** Precision 72.23%, Recall 68.66%, AP50 74.68%, AP50-95 49.40%.
- Trained 48 epochs, about 8.1 minutes.
- Original file supplied in conversation: `EXP-Y8S-NEG-V02-001_ANALYSIS.zip` — SHA256 `b8af8ab5a38b346c942250df6307910583f6f4ba74e8c0c4e6c679ce5c60ed6c`.
- [Negative image false-positive evaluation](../../negative_image_evaluation/README.md).

**Scope / provenance:** these are extracted summary files. The original ANALYSIS.zip contains learning curves, PR/confidence curves, confusion matrices, per-image failure CSV and prediction JPGs; those original binary artifacts have **not yet been committed to GitHub**. Use the archive synchronization script in this folder's parent directory to import them.

**Limitations:** only one seed and Validation evaluation; classification split unchanged, but NEG V02 has one fewer Anthracnose and one fewer Red Rust training annotation than V003. Avoid attributing all differences exclusively to negative images.
