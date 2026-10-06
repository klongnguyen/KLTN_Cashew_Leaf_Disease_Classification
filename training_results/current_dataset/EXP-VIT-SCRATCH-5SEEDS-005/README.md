# EXP-VIT-SCRATCH-5SEEDS-005

**Model:** Compact Vision Transformer Scratch  
**Dataset:** `Cashew_dataV05`  
**Seeds:** `42, 123, 2026, 3407, 7777`  
**Status:** ✅ **Final locked Test complete**

> Source archive self-reports `EXP-VIT-SCRATCH-5SEEDS-004`. That repository ID already belongs to the V04 ViT rerun, so this V05 result is normalized to **EXP-005** to preserve history.

## Validation results

| Seed | Val Accuracy | Val Macro F1 | Val Loss |
|---:|---:|---:|---:|
| 42 | 88.94% | 88.35% | 0.3215 |
| 123 | 86.80% | 86.19% | 0.3898 |
| 2026 | 87.80% | 87.31% | 0.3508 |
| 3407 | **90.51%** | **90.16%** | **0.3037** |
| 7777 | 87.59% | 87.05% | 0.3755 |

Aggregate:
- Validation Accuracy: **88.33 ± 1.44%**
- Validation Macro F1: **87.81 ± 1.52%**
- Candidate checkpoint by Validation Macro-F1: **seed 3407**
- Mean training time: **6.20 min/seed**
- Parameters: **347,717**

## Final locked Test — 5 seeds

| Metric | Mean ± sample SD |
|---|---:|
| Test Accuracy | **85.04 ± 0.44%** |
| Macro Precision | **84.35 ± 0.30%** |
| Macro Recall | **84.12 ± 0.35%** |
| Macro F1 | **83.93 ± 0.37%** |
| Balanced Accuracy | **84.12 ± 0.35%** |

Compact ViT has the lowest overall Test performance but the lowest seed variability in Test Accuracy. `not_cashew_leaf` remains its strongest class (**96.95 ± 1.18% F1**), while `anthracnose` is the main weakness (**68.33 ± 2.22% F1**).

## Per-class Validation F1

| Class | F1 ± SD |
|---|---:|
| `anthracnose` | **81.50 ± 3.03%** |
| `healthy` | **81.92 ± 3.70%** |
| `leaf_miner` | **89.04 ± 0.70%** |
| `not_cashew_leaf` | **93.84 ± 1.87%** |
| `red_rust` | **92.76 ± 1.32%** |

Main recurring errors: `healthy → anthracnose` (12.98%), `leaf_miner → anthracnose` (8.35%), and `anthracnose → red_rust` (7.96%).

Compared with the V04 ViT rerun, mean Validation Accuracy increases **+1.20 pp** and Macro F1 **+1.13 pp**, but seed variability increases. ViT remains much lighter than both CNN baselines.

**The final locked Test results are now included in this repository experiment.**
