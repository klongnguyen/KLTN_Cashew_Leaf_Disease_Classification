# ResNet50 EXP-032 and EXP-033 results

The directories below contain the extracted ANALYSIS artifacts. Model weights and the original ZIP files remain outside Git because the repository ignores those file types.

| Experiment | Results | Notebook | Source ANALYSIS ZIP SHA256 | Files |
|---|---|---|---|---:|
| EXP-032, five-seed ResNet50 | [Results](EXP-RESNET50-REGULARIZED-5SEEDS-032-B3-LR5e6-256px-5411b7ad/README.md) | [Notebook](../../Notebook/ResNet50/resnet50_transfer_regularized_5seed_032.ipynb) | `0d3d339438726cf03867b5f02ea781dc64946919238ed28a1e8307883be6d6bd` | 413 |
| EXP-033, ResNet50 + YOLO pipeline | [Results](EXP-RESNET50-YOLO-PIPELINE-033/aggregate/README.md) | [Notebook](../../Notebook/ResNet50/cashew_resnet50_yolo_pipeline_033.ipynb) | `1cd29a14593b3748f6e5b5716f85fdc0e1e5280e77f40365f74da3bf2ff8dad6` | 312 |

EXP-033's default F3 rule reached 98.10% on Test in both Part A and Part B. Holdout 09 is a descriptive audit because it was inspected during development; see the [result notes](EXP-RESNET50-YOLO-PIPELINE-033/aggregate/README.md) for the complete tables and limitations. The [notebook guide](../../Notebook/ResNet50/EXP-033_README.md) documents inputs and execution.
