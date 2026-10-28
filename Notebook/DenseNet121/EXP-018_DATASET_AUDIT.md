# EXP-018 dataset preflight (27/10/2026)

The local copies of the two revised ZIP archives were inspected before writing
the notebook. The notebook repeats these checks in its own runtime before
training and freezes the archive hashes in its configuration.

| Archive | SHA-256 | Images |
|---|---|---:|
| `CashewData_Split (9).zip` | `1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4` | 7,045 |
| `RL_Cashew_holdout (2).zip` | `47a5722660e2c8dc7cc04c9ff720e1f889c2481ec99112b41bc0606958a7fdd7` | 127 |

| Class | Train | Val | Test | Holdout |
|---|---:|---:|---:|---:|
| anthracnose | 982 | 316 | 122 | 46 |
| healthy | 828 | 216 | 118 | 20 |
| leaf_miner | 875 | 264 | 132 | 36 |
| red_rust | 1,127 | 335 | 158 | 25 |
| not_cashew_leaf | 1,102 | 314 | 156 | 0 |
| **Total** | **4,914** | **1,445** | **686** | **127** |

`real_` filename counts in Split 9: Train 236, Val 50, Test 28.
SHA-256 was also computed for each of the 7,172 archived images: **no exact
byte duplicate** was found within or between the two archives. This does not
establish independence of different photos from the same leaf, tree, or capture
session. The notebook writes pHash candidates for manual review.

The notebook has no executed training output. Run it in a TensorFlow/GPU
environment, with the archive paths in its configuration pointing to the
corresponding copies, to obtain metrics and model exports.
