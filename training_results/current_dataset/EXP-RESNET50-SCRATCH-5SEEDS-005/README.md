# EXP-RESNET50-SCRATCH-5SEEDS-005

**Model:** ResNet50 Scratch  
**Dataset:** `Cashew_dataV05`  
**Seeds:** `42, 123, 2026, 3407, 7777`  
**Runtime:** Google Colab / NVIDIA L4 / OneDeviceStrategy  
**Status:** ✅ **Final locked Test complete**

## Validation results

| Seed | Val Accuracy | Val Macro F1 | Val Loss |
|---:|---:|---:|---:|
| 42 | 94.22% | 94.20% | 0.1736 |
| 123 | 92.30% | 92.19% | 0.2471 |
| 2026 | 89.87% | 89.73% | 0.2894 |
| 3407 | **94.44%** | **94.29%** | 0.2064 |
| 7777 | 91.30% | 91.11% | 0.2665 |

Aggregate:
- Validation Accuracy: **92.43 ± 1.94%**
- Validation Macro F1: **92.30 ± 1.98%**
- Candidate checkpoint by Validation Macro-F1: **seed 3407**
- Mean training time: **25.45 min/seed**
- Parameters: **24,641,413**

## Final locked Test — 5 seeds

| Metric | Mean ± sample SD |
|---|---:|
| Test Accuracy | **89.78 ± 2.50%** |
| Macro Precision | **89.72 ± 2.08%** |
| Macro Recall | **89.37 ± 2.56%** |
| Macro F1 | **89.36 ± 2.47%** |
| Balanced Accuracy | **89.37 ± 2.56%** |

ResNet50 shows the largest seed variability of the three V05 baselines. Test Accuracy ranges from **86.61%** to **93.16%**. `anthracnose` remains the weakest class (**79.32 ± 4.40% F1**).

## Per-class Validation F1

| Class | F1 ± SD |
|---|---:|
| `anthracnose` | **86.99 ± 3.19%** |
| `healthy` | **90.87 ± 4.28%** |
| `leaf_miner` | **92.79 ± 1.39%** |
| `not_cashew_leaf` | **94.61 ± 1.25%** |
| `red_rust` | **96.24 ± 1.62%** |

Main recurring errors: `healthy → anthracnose` (9.24%), `leaf_miner → anthracnose` (6.43%), and `anthracnose → red_rust` (4.15%).

Seed 2026 is notably weaker and increases the between-seed variance. Compared with the latest V04 validation rerun, mean Validation Accuracy increases **+2.20 pp** and Macro F1 **+2.27 pp**, but standard deviation increases.

**The final locked Test results are now included in this repository experiment.**
