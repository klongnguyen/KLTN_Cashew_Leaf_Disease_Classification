# EXP-DENSENET121-SCRATCH-5SEEDS-005

**Model:** DenseNet121 Scratch  
**Dataset:** `Cashew_dataV05`  
**Seeds:** `42, 123, 2026, 3407, 7777`  
**Status:** ✅ **Final locked Test complete**

## Validation results

| Seed | Val Accuracy | Val Macro F1 | Val Loss |
|---:|---:|---:|---:|
| 42 | 94.44% | 94.28% | 0.1923 |
| 123 | 95.01% | 94.81% | 0.1566 |
| 2026 | **95.44%** | **95.25%** | **0.1489** |
| 3407 | 95.01% | 94.94% | 0.1771 |
| 7777 | 93.01% | 92.88% | 0.2246 |

Aggregate:
- Validation Accuracy: **94.58 ± 0.95%**
- Validation Macro F1: **94.43 ± 0.93%**
- Candidate checkpoint by Validation Macro-F1: **seed 2026**
- Mean training time: **33.37 min/seed**
- Parameters: **7,566,917**

## Final locked Test — 5 seeds

| Metric | Mean ± sample SD |
|---|---:|
| Test Accuracy | **91.97 ± 1.11%** |
| Macro Precision | **91.48 ± 1.13%** |
| Macro Recall | **91.42 ± 1.15%** |
| Macro F1 | **91.41 ± 1.13%** |
| Balanced Accuracy | **91.42 ± 1.15%** |

Per-seed Test Accuracy ranges from **90.10%** to **93.01%**. On Test, `anthracnose` remains the weakest class (**81.17 ± 1.66% F1**), while `red_rust` reaches **96.40 ± 1.41% F1**.

## Per-class Validation F1

| Class | F1 ± SD |
|---|---:|
| `anthracnose` | **90.75 ± 2.49%** |
| `healthy` | **93.73 ± 0.81%** |
| `leaf_miner` | **93.81 ± 0.83%** |
| `not_cashew_leaf` | **96.32 ± 1.02%** |
| `red_rust` | **97.55 ± 0.48%** |

Main recurring errors: `healthy → anthracnose` (5.60%), `leaf_miner → anthracnose` (4.90%), and `anthracnose → red_rust` (3.67%).

Compared with the latest V04 validation rerun, mean Validation Accuracy increases **+2.45 pp** and Macro F1 **+2.53 pp**. Because the dataset version changed, this is a V05 result rather than proof of an architecture improvement.

**The final locked Test results are now included in this repository experiment.**
