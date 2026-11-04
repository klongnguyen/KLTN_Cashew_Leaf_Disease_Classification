# EXP-033 result

See classifier_comparison.csv for ResNet50 versus F0 and DenseNet121; DenseNet cells are NA unless its result ZIP is supplied.

Default rule: **F3**

Holdout 09: descriptive audit, not a blind benchmark. It has been inspected previously.
Two known duplicate Holdout images retain their healthy labels; see n=124 metric.
ResNet50 032 was selected over 031 for Stage 2 stability after Test/Holdout inspection; 031 has lower val_loss. F2 is post hoc.
Confidence uses Validation-fitted temperature scaling. In-sample stacker calibration may be optimistic.
YOLO best.pt was chosen on YOLO Val; BOX_CONF chosen there is slightly optimistic. YOLO Test is the clean localization split.

## Part A

| Rule | Clean Val | Test | Holdout | Holdout n=124 | Gap |
|---|---:|---:|---:|---:|---:|
| F0 | 0.9618 | 0.9504 | 0.7698 | 0.7742 | 0.1805 |
| F1 | 0.9581 | 0.9299 | 0.8254 | 0.8226 | 0.1045 |
| F2 | 0.9633 | 0.9533 | 0.8254 | 0.8226 | 0.1279 |
| F3 | 0.9846 | 0.9810 | 0.9127 | 0.9113 | 0.0683 |
| F3a | 0.9633 | 0.9504 | 0.7857 | 0.7903 | 0.1647 |
| F3b | 0.7950 | 0.7723 | 0.8413 | 0.8468 | -0.0690 |

## Part B

| Rule | Clean Val | Test | Holdout | Holdout n=124 | Gap |
|---|---:|---:|---:|---:|---:|
| F0 | 0.9611 | 0.9474 | 0.7698 | 0.7742 | 0.1776 |
| F1 | 0.9589 | 0.9285 | 0.8175 | 0.8145 | 0.1110 |
| F2 | 0.9633 | 0.9504 | 0.8254 | 0.8226 | 0.1250 |
| F3 | 0.9838 | 0.9810 | 0.9048 | 0.9032 | 0.0763 |
| F3a | 0.9611 | 0.9489 | 0.7857 | 0.7903 | 0.1632 |
| F3b | 0.7950 | 0.7723 | 0.8413 | 0.8468 | -0.0690 |

## Localization

See `localization_test.csv` for per-class IoU 0.5 precision/recall/F1, mAP50, mAP50-95, `real_` and near-duplicate exclusions.
BOX_CONF=0.250; detector boxes are filtered by final class, class-wise NMS 0.5, and capped at 20 for display.
Holdout panels are qualitative only: Holdout has no box ground truth.

## Decision and limits

The predeclared Part B rule decides the deployable default. F2 is explicitly post hoc.
Abstained images mean expert review and do not count as correct.
The 126-image Holdout has a wide confidence interval; a gap below 10 points alone does not establish success.
A future blind real-field set is needed for a confirmatory claim.
Completed runs export both ANALYSIS (tables/predictions) and FULL (models + tables) ZIPs. Per-seed ZIPs are disconnect recovery checkpoints.
