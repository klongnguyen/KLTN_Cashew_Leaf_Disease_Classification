# DenseNet121 EXP-029–032: notebooks and result artifacts

This index records which artifacts are available on the `results/densenet-029-032-complete` branch. The ZIPs below are in `training_results/current_dataset/` and are recorded in `training_results/DENSENET_RESULTS_MANIFEST_2026-11-04.csv` with SHA-256 hashes.

| EXP | Notebook | Result artifacts on this branch | Status |
| --- | --- | --- | --- |
| 029 | `densenet121_transfer_photometric_029.ipynb`; `densenet121_transfer_photometric_029_fixed.ipynb` | `EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05_ANALYSIS.zip`; eight `exp029_full_drive_parts/*_FULL_part*.zip` | ANALYSIS and split FULL available. See `EXP-029_FULL_DRIVE_PARTS.md` and `restore_exp029_full.py`. |
| 030 | `densenet121_transfer_oversample_030.ipynb` | `EXP-DENSENET121-TRANSFER-5SEEDS-030-256px-9c23ac05_ANALYSIS.zip` | ANALYSIS available. No FULL archive was found in the local workspace, Downloads, or connected Drive search. |
| 031 | `densenet121_yolo_evidence_gate_031.ipynb` | No separate training archive found | EXP-031 is an inference/evidence gate using EXP-029 predictions; its protocol states that it does not train a new model. See `EXP-031_PROTOCOL.md`. |
| 032 | `cashew_densenet_yolo_pipeline_032.ipynb` | No FULL or ANALYSIS archive found | The protocol defines the intended exports, but their completed ZIPs were not found in the local workspace, Downloads, or connected Drive search. See `EXP-032_PROTOCOL.md`. |

Notebook files are included to preserve the experiment definitions and any saved cell outputs. A notebook with saved outputs is not a substitute for a verified FULL or ANALYSIS export.
