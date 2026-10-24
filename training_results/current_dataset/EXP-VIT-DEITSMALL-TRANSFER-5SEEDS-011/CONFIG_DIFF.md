# EXP-011 configuration lock versus EXP-010

This file records every intentional value change in `experiment_config.json`.
All other values, including V08 fingerprint, class order, five seeds, model preset,
batch size, augmentation, optimizer settings, and Validation guardrails, are inherited.

| JSON key | EXP-010 | EXP-011 |
|---|---|---|
| `experiment_id` | `EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010` | `EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011` |
| `source_experiment` | `EXP-VIT-TRANSFER-5SEEDS-007` | `EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010` |
| `stage1.max_epochs` | 15 | 30 |
| `stage1.early_stopping_patience` | 5 | 7 |
| `stage2.max_epochs` | 15 | 30 |
| `stage2.early_stopping_patience` | 5 | 7 |
| `stage2.schedule` | 3-epoch warmup, cosine decay within 15 epochs | 3-epoch warmup, cosine decay within 30 epochs |

EXP-011 tests the combined extended recipe. It cannot isolate the effect of
`max_epochs` from the patience and schedule changes. Each seed starts from the
same pretrained DeiT-Small preset with a newly initialized head, then Stage 2
starts from that seed's best EXP-011 Stage 1 checkpoint. EXP-010 fine-tuned
checkpoints are comparison inputs only.

Before training, the notebook checks that the actual frozen config differs
from EXP-010 only in the keys above. Test V08 is limited to ZIP integrity
audit; Historical Holdout is never loaded into a model.
