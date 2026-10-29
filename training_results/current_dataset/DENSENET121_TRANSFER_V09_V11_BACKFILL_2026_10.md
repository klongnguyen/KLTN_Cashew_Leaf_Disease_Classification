# DenseNet121 Transfer EXP-018–026 — Experiment archive backfill (28/10/2026)

> This index records **curated results** from ANALYSIS archives provided by the project owner, and additionally links the ViT/XGBoost EXP-009 inference results. It does **not** imply that the full ZIP, image figures, per-image predictions, notebooks or binary checkpoints have been uploaded. EXP-019 was already in GitHub and has not been overwritten.

## Status and headline results

| EXP | Dataset/recipe | Protocol | Validation accuracy / Macro F1 (mean, 5 seeds) | Test accuracy / Macro F1 | Holdout accuracy / Macro F1 | Validation-selected seed |
|---|---|---|---|---|---|---:|
| [018](./EXP-DENSENET121-TRANSFER-5SEEDS-018/README.md) | V09, 224px, ImageNet fresh | Train + Validation | 96.76% / 96.62% | — | — | 123 |
| [019](./EXP-DENSENET121-TRANSFER-5SEEDS-019/README.md) | V09, checkpoints 018 | Evaluation only | inherited from 018 | 95.13% / 94.79% | Holdout02 (127): 67.72% / 63.72% | 123 |
| [020](./EXP-DENSENET121-TRANSFER-5SEEDS-020/README.md) | V09, checkpoints 018 | Holdout-only, no training | inherited from 018 | NOT RUN | Holdout03 (126): 68.25% / 64.60% | 123 |
| [021](./EXP-DENSENET121-TRANSFER-5SEEDS-021/README.md) | V09 checkpoints, reduced Holdout | Holdout-only, no training | inherited from 018 | NOT RUN | Holdout04 (120): 71.33% / 68.31% | 123 |
| [022](./EXP-DENSENET121-TRANSFER-5SEEDS-022/README.md) | V10, 224px, no brightness | Fresh training + Val/Test/Holdout | 96.72% / 96.58% | 94.84% / 94.47% | Holdout04 (120): 72.17% / 71.94% | 123 |
| [023](./EXP-DENSENET121-TRANSFER-5SEEDS-023/README.md) | V10, 224px + brightness 0.08 | Train+Validation only, research-only | 96.58% / 96.44% | PENDING | PENDING | 123 |
| [024](./EXP-DENSENET121-TRANSFER-5SEEDS-024/README.md) | V11, 256px + brightness 0.08 | Fresh training + Validation | 97.26% / 97.14% | in 025 | in 025 | 42 |
| [025](./EXP-DENSENET121-TRANSFER-5SEEDS-025/README.md) | V11, checkpoints 024 | Evaluation only | inherited from 024 | 96.09% / 95.71% | Holdout05 (120): 80.00% / 76.56% | 42 |
| [026](./EXP-DENSENET121-TRANSFER-5SEEDS-026/README.md) | V11, 224px, EXP-014 recipe, fresh | Train+Val+historical Test/Holdout | 97.00% / 96.88% | 96.23% / 95.88% | Holdout05 (120): 74.83% / 73.21% | 2026 |

All percentages in the table are **mean over 5 seeds** and rounded. Test/Validation Macro F1 uses the five true classes; Holdout has **no not_cashew_leaf** and uses only **four supported classes**. Consult linked per-experiment CSVs for per-seed metrics.

## Scope, research interpretation, and leakage risk

1. **EXP-018 vs 019:** EXP-019 evaluated, but did not retrain, EXP-018 weights.
2. **EXP-020 and 021:** apparent metric improvements arose after modifying/reducing the same Holdout data; source checkpoints never changed, so this is **not improved model learning**.
3. **EXP-022:** V10 retraining adds six previously held-out Healthy examples to Train. Holdout04 already had its errors inspected; treat comparisons as retrospective diagnostics.
4. **EXP-023:** the isolated RandomBrightness(0.08) V10 Validation ablation did **not** improve Macro F1. Do not silently promote.
5. **EXP-024/025:** V11 256px with brightness; EXP-025 is only Test/Holdout inference from EXP-024's seed-42 selection lock.
6. **EXP-026:** original EXP-014 recipe retrained independently on V11/224px; direct retrospective comparison to 024/025 uses the **same V11 + Holdout05**, but simultaneously changes **resolution and brightness**, so it cannot attribute effects to either alone.
7. **No blind test claim:** Historical Test and Holdout versions have been repeatedly examined or revised (including label corrections). No current Holdout version should be used to tune and then reported as an untouched prospective field test.
8. **No deployment changes:** WebApp checkpoint remains unchanged. Each EXP's deployment-candidate seed was selected only by its own Validation.
9. **Image provenance:** investigate near-duplicate/same leaf/capture session groups; zero exact byte duplicates is not proof of independence.

## Archival coverage

| Source archive | GitHub coverage |
|---|---|
| DenseNet EXP-018, 020–026 ANALYSIS ZIPs | Newly added `README.md` + `metrics_5seeds.csv` per EXP |
| DenseNet EXP-019 ANALYSIS ZIP | Previously uploaded curated README, Test/Holdout aggregates, selected-seed confusion/error tables |
| [ViT/XGBoost EXP-009](./EXP-VIT-TRANSFER-XGBOOST-009/README.md) RESULTS ZIP | Newly added README + **full-precision 30-row** A/B/C × five seeds × Test/Holdout summary |
| Other older ResNet50, ViT, DenseNet, CBAM and YOLO experiments | Existing reports remain untouched; this backfill does not claim to have reuploaded or audited every older archive |

**Original full ANALYSIS ZIPs, binary weights, figures, notebook and per-image predictions** are not contained in this GitHub backfill. They are stored in the user-uploaded conversation artifacts; this repo contains key indexed **text and CSV summaries**, not a full reproducibility package.

## Dataset lineages

- **V09 (Split 9)**: Train 4,914 / Val 1,445 / Test 686; source EXP-018, eval EXP-019–021.
- **V10 (Split 10)**: Train 4,920 / Val 1,445 / Test 686; train EXP-022 and brightness ablation EXP-023.
- **V11 (Split 11)**: Train 4,916 / Val 1,447 / Test 685; train EXP-024, eval EXP-025, train-from-scratch-via-ImageNet EXP-026.
- **Historic Holdouts**: Holdout02=127, Holdout03=126, Holdout04=120, Holdout05=120; memberships and, in some cases, file contents were revised. Compare on identical image hashes only when claiming paired image-level effects.

[Main README](../../README.md) · [Dataset changelog](../../DATASET_CHANGELOG.md)
