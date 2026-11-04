# Model result inventory (2026-11-05)

This branch starts from the latest `main` results and adds the locally available outputs for the trained and evaluated models.

| Model family | Result locations |
| --- | --- |
| DenseNet121 | [`current_dataset`](current_dataset/), [`archive_legacy_dataset`](archive_legacy_dataset/), [`analysis_results`](../analysis_results/) |
| ResNet50 | [`current_dataset`](current_dataset/), including EXP-032 regularized analysis and EXP-033 YOLO pipeline analysis |
| ViT / DeiT | [`current_dataset`](current_dataset/), with notebooks in [`Notebook/ViT`](../Notebook/ViT/) |
| YOLOv8 / YOLO26 | [`negative_evaluation`](negative_evaluation/), [`ACDRA experiments`](../experiments/), [`reports`](../reports/), and notebooks in [`Notebook/Yolov8`](../Notebook/Yolov8/) and [`Notebook/Yolov26`](../Notebook/Yolov26/) |

Training checkpoints, full result archives, and large JSONL evaluations added in this branch use Git LFS. The `dataset/` source directory and temporary downloaded model caches remain outside Git.
