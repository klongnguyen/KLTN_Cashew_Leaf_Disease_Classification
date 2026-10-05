# Cashew_dataV05 — Latest 5-seed Validation Benchmark

**Status:** Training + Validation complete for all three scratch baselines. **Locked Test is still pending.**

## Dataset consistency

All three uploaded V05 analysis archives report the same dataset size and the same normalized `split/class/filename` manifest:

- Train: **4,822**
- Validation: **1,402**
- Test: **687**
- Total: **6,911**
- Classes: `anthracnose`, `healthy`, `leaf_miner`, `not_cashew_leaf`, `red_rust`
- Invalid images: **0**
- Unknown files: **0**
- Normalized manifest SHA-256: `0712a1b61789ab7d55d025589174c59eabb9d01532cd09ff5f673f5bb39f939b`

For the ResNet50 runs, the notebook-recorded split fingerprints show:

| Split | V04 vs V05 |
|---|---|
| Train | **Different** |
| Validation | **Different** |
| Test | **Same** |

The locked Test fingerprint remains `ee49ef59c791a7519c172316afd48554f92fbb1b4f47d11b57c82d6ae1624dc4`. Train and Validation fingerprints changed, so V05 is treated as a new dataset snapshot even though the counts are unchanged.

## Latest V05 validation results

| Model | Repository experiment | Val Accuracy | Val Macro F1 | Candidate seed | Params |
|---|---|---:|---:|---:|---:|
| **DenseNet121** | `EXP-DENSENET121-SCRATCH-5SEEDS-005` | **94.58 ± 0.95%** | **94.43 ± 0.93%** | **2026** | 7.57M |
| **ResNet50** | `EXP-RESNET50-SCRATCH-5SEEDS-005` | **92.43 ± 1.94%** | **92.30 ± 1.98%** | **3407** | 24.64M |
| **Compact ViT** | `EXP-VIT-SCRATCH-5SEEDS-005`* | **88.33 ± 1.44%** | **87.81 ± 1.52%** | **3407** | **0.35M** |

\* Source archive self-reports `EXP-VIT-SCRATCH-5SEEDS-004`. The repository stores this V05 run as `...-005` because `EXP-VIT-SCRATCH-5SEEDS-004` already exists for the V04 run.

DenseNet121 currently leads ResNet50 by **2.15 percentage points Validation Accuracy** and **2.13 pp Macro F1**. It leads Compact ViT by **6.25 pp Accuracy** and **6.62 pp Macro F1**. This ranking is provisional until the locked Test is evaluated.

## Per-class Validation F1

| Class | DenseNet121 | ResNet50 | Compact ViT |
|---|---:|---:|---:|
| `anthracnose` | **90.75 ± 2.49%** | 86.99 ± 3.19% | 81.50 ± 3.03% |
| `healthy` | **93.73 ± 0.81%** | 90.87 ± 4.28% | 81.92 ± 3.70% |
| `leaf_miner` | **93.81 ± 0.83%** | 92.79 ± 1.39% | 89.04 ± 0.70% |
| `not_cashew_leaf` | **96.32 ± 1.02%** | 94.61 ± 1.25% | 93.84 ± 1.87% |
| `red_rust` | **97.55 ± 0.48%** | 96.24 ± 1.62% | 92.76 ± 1.32% |

`anthracnose` remains the weakest class in all three models. `red_rust` remains the strongest. Recurring errors are still centered on Anthracnose, especially `healthy → anthracnose`, `leaf_miner → anthracnose`, and `anthracnose → red_rust`.

## V04 rerun → V05 comparison

| Model | V04 Val Accuracy | V05 Val Accuracy | Δ mean | V04 Macro F1 | V05 Macro F1 | Δ mean |
|---|---:|---:|---:|---:|---:|---:|
| DenseNet121 | 92.13 ± 1.31% | **94.58 ± 0.95%** | **+2.45 pp** | 91.90 ± 1.37% | **94.43 ± 0.93%** | **+2.53 pp** |
| ResNet50 | 90.23 ± 1.45% | **92.43 ± 1.94%** | **+2.20 pp** | 90.03 ± 1.44% | **92.30 ± 1.98%** | **+2.27 pp** |
| Compact ViT | 87.13 ± 0.62% | **88.33 ± 1.44%** | **+1.20 pp** | 86.68 ± 0.70% | **87.81 ± 1.52%** | **+1.13 pp** |

These gains are **not architecture-improvement claims** because the dataset version changed. DenseNet becomes more stable by seed; ResNet and ViT improve in mean score but show larger seed variability than their latest V04 reruns.

## Training behavior

- **DenseNet121:** 33.37 min/seed on average; best checkpoint around epoch 22.2; Macro-F1 SD **0.93 pp**.
- **ResNet50:** 25.45 min/seed; best checkpoint around epoch 20.6; Macro-F1 SD **1.98 pp**.
- **Compact ViT:** 6.20 min/seed; best checkpoint around epoch 26.0; Macro-F1 SD **1.52 pp**.

The CNN curves show large transient Validation-loss spikes in early epochs but recover before the selected checkpoints. This does not indicate a failed run, but it shows scratch CNN optimization remains sensitive early in training. ViT converges more smoothly.

## Conclusion

At the **Validation stage**, the latest V05 ranking is:

1. **DenseNet121** — strongest mean performance and best stability.
2. **ResNet50** — strong performance but more seed-sensitive.
3. **Compact ViT** — lower Accuracy/F1 but dramatically smaller and faster to train.

This is **not yet the final benchmark**. Freeze the current five checkpoints for each model and evaluate the same locked Test Set once per seed, without any additional tuning, before promoting V05 to the official Test benchmark.
