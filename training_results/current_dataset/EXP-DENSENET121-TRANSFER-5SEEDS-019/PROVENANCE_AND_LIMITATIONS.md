# EXP-019 audit / provenance (excerpt)

Source: original user-supplied `EXP-DENSENET121-TRANSFER-5SEEDS-019_ANALYSIS.zip`. SHA256 `596c4fa817cb3054de98abe4798584b041a909aac04f91f6b31b538b1ac512d0`. Archive has **169 members** and expands to about **6.48 MB**. `CHECKSUMS.sha256` lists 167 entries; independent rehash gave **167/167 PASS**; ZIP CRC test PASS.

`execution_status.json`: training `REUSED_EXP018`; final_test `COMPLETE`; real_holdout `COMPLETE`.

`evaluation_lock.json`: source EXP-018, selection source `EXP-018 Validation only`, selected seed 123, five model checkpoint SHA256s:
- 42: `12e5be0b043467362536a1f357ec486d3a762e322eaeed02c2395393205b99d7`
- 123: `376cca9f44b08c1543fe15115dd4cea9374a0d740ddd34d25db20bbc3b7e31f7`
- 2026: `c5b5f21c955c2c2b9f56ca39d49eac58a964990af8e27c8f0dc5b46a180466e9`
- 3407: `1e1c0d3af10cfe7015dbb4af6d25e06cb80c144eda586c8b24f6cf67fd6b67cd`
- 7777: `21b356913f74d0984396c37cb5bee227e2abe06ed673598d9a9bef8d8fe5f8f3`

Source config SHA256 `370778fb64853b31f0b6e32a7dfbfc0038b273ed4bd8a6bc7a00c6b4cfc08396`; source Validation lock SHA256 `dc9f56b70d20d692947d4d0927840b6cba52167908ca1937e8f97e0846f405d3`; source manifest SHA256 `65f3f0dfb3f50e110d39ee67fb9970c13e4e0468894d03278e8c4fbc28e761c0`.

V09 data ZIP SHA256 `1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4`; Holdout02 ZIP SHA256 `47a5722660e2c8dc7cc04c9ff720e1f889c2481ec99112b41bc0606958a7fdd7`. Evaluation lock SHA256 `3b43f6ce32d1e5dda6bc851fb22cada96a25d27c3ad812b49e92b2132c300a64`.

Original package includes 5-seed complete per-image prediction CSVs, raw/normalized confusion matrices and PNG, calibration, ROC/PR, original notebook, metadata, figures and manifests. **The complete raw result artifacts and original ZIP remain only in the supplied archive; the GitHub commit intentionally includes a reproducible, curated text/CSV report without checkpoint binaries.** SHA metadata alone cannot verify absent checkpoint binary bytes.

Holdout metadata asserts 0 exact overlap with train/validation/test, but near-duplicate/shared leaf or orchard provenance needs manual field audit. No independent newly gathered not-cashew Holdout examples. No training or threshold choice from Test/Holdout should be performed.

**Macro reporting caveat:** official Holdout aggregate `metrics.json` excludes the class with zero true support for macro averages, giving seed123 F1=0.6219711538461539 on four classes. Original `classification_report.csv` prints a five-label macro average including unsupported `not_cashew_leaf` (F1=0.4975769230769231); do not mix them.
