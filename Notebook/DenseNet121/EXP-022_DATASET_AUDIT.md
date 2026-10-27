# EXP-022 dataset preflight — 27/10/2026

New five-seed DenseNet121 training uses these pinned archives:

| Archive | SHA-256 | Images |
|---|---|---:|
| `CashewData_Split (10).zip` | `b24f0faa5e918fb43d7d16500c6b8234df1cd771cc4a27eb2354051bce84edac` | 7,051 |
| `RL_Cashew_holdout (4).zip` | `3b340ea830418638fa610fb2176dcc171739cc7fe0bd8e8430c8ed2a614717cc` | 120 |

| Class | Train | Val | Test | Holdout |
|---|---:|---:|---:|---:|
| anthracnose | 982 | 316 | 122 | 46 |
| healthy | 834 | 216 | 118 | 13 |
| leaf_miner | 875 | 264 | 132 | 36 |
| red_rust | 1,127 | 335 | 158 | 25 |
| not_cashew_leaf | 1,102 | 314 | 156 | 0 |
| **Total** | **4,920** | **1,445** | **686** | **120** |

The Split (10) ZIP adds exactly six `train/healthy` images to Split (9); all
existing image paths retain identical bytes. The six additions were removed
from Holdout (3) when Holdout (4) was prepared. There are no exact byte
duplicates within Split (10), within Holdout (4), or between them. This byte
check does not establish independence of photos from the same leaf, tree or
capture session. The notebook writes pHash candidates for manual review.

EXP-022 starts with new ImageNet-initialized training. Its Final Test and
holdout evaluations run only after all five Validation checkpoints and the
selection lock have been written. No training metrics are prefilled here.
