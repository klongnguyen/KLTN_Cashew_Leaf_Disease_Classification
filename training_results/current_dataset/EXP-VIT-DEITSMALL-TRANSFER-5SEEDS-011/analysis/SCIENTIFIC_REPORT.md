# EXP-011 — Validation V08 scientific report

## Method

DeiT-Small distilled ImageNet pretrained, 5 seeds. Stage 1 trains only the head; Stage 2 fine-tunes the last two encoder blocks and head from the same-seed EXP-011 best Stage 1 checkpoint. Epoch caps are 30/30 and patience is 7.
Only Train updates weights. Only Validation selects epochs, stages and seeds. Test V08 is read solely for ZIP integrity. Historical Holdout is not used.

The epoch cap, patience and Stage 2 cosine horizon change together. This paired comparison does not isolate the effect of epoch cap.

## Five-seed results

| Model | Accuracy mean +/- sample SD | Macro F1 mean +/- sample SD | D2H / 900 mean +/- sample SD |
|---|---:|---:|---:|
| Tiny A EXP-008 | 0.9441 +/- 0.0073 | 0.9429 +/- 0.0070 | 20.8000 +/- 4.4385 |
| Small EXP-010 S1 | 0.8627 +/- 0.0134 | 0.8595 +/- 0.0143 | 17.8000 +/- 5.4498 |
| Small EXP-010 S2 | 0.9281 +/- 0.0100 | 0.9274 +/- 0.0096 | 21.2000 +/- 6.7231 |
| Small EXP-011 S1 | 0.8890 +/- 0.0048 | 0.8876 +/- 0.0048 | 20.2000 +/- 5.3572 |
| Small EXP-011 S2 | 0.9345 +/- 0.0128 | 0.9340 +/- 0.0130 | 19.2000 +/- 5.2631 |

## Paired seed deltas, EXP-011 minus EXP-010

| Stage | Accuracy delta | Macro F1 delta | D2H delta |
|---|---:|---:|---:|
| stage1 | +0.0263 +/- 0.0105 | +0.0281 +/- 0.0117 | +2.4000 +/- 2.3022 |
| stage2 | +0.0064 +/- 0.0090 | +0.0066 +/- 0.0092 | -2.0000 +/- 5.7879 |

## Validation selection

Status: NOT PROMOTED. Selected research checkpoint: none met guardrails.
The EXP-010 guardrails and tie rules were frozen before this run.

## Limits

Five seeds share one Validation split; sample SD is seed variation rather than independent dataset uncertainty. The V08 audit has 226 cross-split pHash candidates requiring source/group review. Only 35 Validation images have the real_ prefix. Tiny and Small differ in capacity. No Test/Holdout inference was run.
