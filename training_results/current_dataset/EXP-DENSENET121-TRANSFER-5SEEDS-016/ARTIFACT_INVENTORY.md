# EXP-016 artifact inventory and raw-data provenance

**Original archive**: `EXP-DENSENET121-TRANSFER-5SEEDS-016_ANALYSIS.zip`, SHA256 `02c1c0dcd6c8a7c06d8dffab5fe6d8514871590c7e071def6c4f2fdec5a32ea5`; **27 ZIP members**, ~294 KiB compressed, ~757,499 bytes uncompressed.

| Artifact | Github state | Notes |
|---|---|---|
| README, QUALITY_AND_PROTOCOL_REVIEW, ARTIFACT_INVENTORY | **Imported** | Curated user-facing documentation |
| Source evaluation config and execution status | **Imported** | Original logical configuration, reformatted JSON |
| Aggregate test results, mean±sample SD, per-class by seed, sum confusion | **Imported** | Derived from source CSV, same metrics |
| 5 seed `confusion_matrix.csv` | **Imported** | All five checked against supports and accuracy |
| 5 seed `classification_report.csv` | **Imported/reconstructed values** | Same metrics; serialization/float formatting can differ from source |
| 5 seed `metrics.json` | **Imported/reformatted** | Same metric values, source bytes may differ |
| 5 seed `predictions.csv` (686 rows each) | **NOT IMPORTED** | Present in source ZIP, ~748 KB across 5 CSVs; required for full per-image audit |
| Source full ZIP | **NOT IMPORTED** | Available from original author upload, SHA provided |
| EXP-014 5 binary checkpoint weights `.keras` | **NOT IMPORTED** | Binary checkpoints not included in EXP-016 source ZIP |
| Error audit & real-prefix test subgroup | **Imported derived subset** | Built from original `predictions.csv`, not full per-image logs |
| SVG charts | **Imported derived charts** | New vectors based on source metrics; not original images |
| Original notebook | **NONE in source ZIP** | EXP-016 source archive has config/results only |

**Meaning of an original checksum:** hashes listed for EXP-014 checkpoint files and original ZIP do not authenticate a reserialized report; verify raw binaries/checkpoint hashes separately. The experiment remains **inference-only on locked EXP-014 checkpoints**, deployment seed **123**.

**No direct production deployment:** `Application/` remains the previous DenseNet121 Scratch EXP-009 seed3407. EXP-014 model package, preprocessing parity and fresh 5-class external validation are outstanding.

[Results](./README.md) · [Protocol review](./QUALITY_AND_PROTOCOL_REVIEW.md).
