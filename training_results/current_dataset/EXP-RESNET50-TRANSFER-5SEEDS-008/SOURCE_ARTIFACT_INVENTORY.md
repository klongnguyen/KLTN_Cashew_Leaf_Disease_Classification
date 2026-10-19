# Inventory of imported EXP-008 artifacts (20/10/2026)

Source archive: `EXP-RESNET50-TRANSFER-5SEEDS-008_ANALYSIS.zip`. SHA256 ZIP **`a408af9b540208045f2214c71cb6072b75e8b1442efacd12d88ae510107bc24f`**. Archive contains **234 entries**, including notebook, per-epoch histories, extensive per-image predictions and PNG panels; this GitHub snapshot is a **curated subset** only.

| Artifact | Imported to GitHub | Source details |
|---|---|---|
| README, method and data-leakage review | ✅ | README and independent notes in this directory |
| 5-seed Validation results and SD | ✅ | `aggregate/validation_results_5seeds.csv` and `validation_mean_std_summary.csv` |
| 5-seed Final Test results and SD | ✅ | `aggregate/test_results_5seeds.csv` and `test_mean_std_summary.csv` |
| 5-seed Historical Holdout and SD | ✅ | `real_holdout/aggregate/` |
| 15 Confusion matrix CSVs, each original split and seed | ✅ | `seed_{s}/validation`, `seed_{s}/test`, `real_holdout/seed_{s}` |
| Split provenance, config, smoke, selection and holdout audit | ✅ summaries | `experiment_config_snapshot.json`, `validation_selection_lock_summary.json`, etc. |
| Visual comparison | ✅ reconstructed SVG | `figures/*.svg` redrawn from aggregate numbers, **not original PNG panels** |
| Original Python notebook `resnet50_transfer_5seed_008.ipynb` | ❌ not uploaded | Also supplied separately in conversation; source ZIP embeds notebook, **193,240 bytes** |
| Original per-image predictions (all images, all seeds) | ❌ not uploaded | Only confusion matrices, class-level and aggregate summaries on GitHub |
| Original training logs/history per epoch | ❌ not uploaded | See ZIP (can be added in separate data import) |
| Original chart PNG panels | ❌ not uploaded | See ZIP; SVG here is a newly rendered summary |
| Original `EXP-008 ANALYSIS.zip` and `EXP-008 FULL.zip` | ❌ not uploaded | Retain and verify checksum in external canonical storage |
| Binary checkpoints (90 MB each), deployment `.keras`, feature models | ❌ not uploaded | Need separate artifact storage/Git LFS/Drive link, not ordinary Git history |

**Integrity rules:** The SHA-256 hashes in [audit](./ANALYSIS_AND_PROTOCOL_AUDIT.md) are from original source artifacts; reconstructed/summarized GitHub files can have different bytes and should NOT be presented as byte-for-byte copies. Original `validation_selection_lock.json` is larger and includes stage candidates, whereas GitHub stores an explicitly **curated summary**. Model-binary verification can't be reproduced from this repo alone.

**No changes** made to `Application/`, training notebooks already in repo, dataset splits, or deployment weights. To run inference one must obtain the original FULL ZIP/checkpoints, validate provenance and preprocessing, and re-evaluate.

Read [experiment README](./README.md) and [protocol audit](./ANALYSIS_AND_PROTOCOL_AUDIT.md).
