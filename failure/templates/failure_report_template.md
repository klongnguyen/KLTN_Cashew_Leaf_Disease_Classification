# FAILURE REPORT TEMPLATE

## 1. Experiment Information

| Field | Value |
|---|---|
| Experiment ID | |
| Failure Take | |
| Date | |
| Model | |
| Dataset Version | |
| Failure Type | |
| Status | FAILED / BASELINE ONLY |

## 2. Objective

Mục tiêu của experiment:

- ...

## 3. Configuration

| Parameter | Value |
|---|---:|
| Image Size | |
| Epochs | |
| Batch Size | |
| Optimizer | |
| Learning Rate | |
| Seed | |
| Augmentation | |

## 4. Dataset Summary

| Split/Class | Value |
|---|---:|
| Train images | |
| Validation images | |
| Test images | |
| Class 1 boxes | |
| Class 2 boxes | |
| Class 3 boxes | |

## 5. Main Metrics

| Metric | Result |
|---|---:|
| Precision | |
| Recall | |
| F1 | |
| mAP@0.50 | |
| mAP@0.75 | |
| mAP@0.50:0.95 | |
| Mean IoU | |

## 6. Per-Class Performance

| Class | Precision | Recall | F1 | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|
| | | | | | |

## 7. Observed Failure Symptoms

- ...
- ...

## 8. Root Cause Analysis

### Data / Annotation

- ...

### Model / Training

- ...

### Evaluation

- ...

## 9. What Worked

- ...

## 10. Why This Experiment Is Not Selected

- ...

## 11. Corrective Actions

1. ...
2. ...
3. ...

## 12. Next Experiment

```text
Experiment ID:
Main change:
Variables kept fixed:
Expected improvement:
```

## 13. Final Decision

- [ ] Keep as final candidate
- [ ] Keep as baseline only
- [ ] Re-run after data fix
- [ ] Discard configuration

## 14. Artifacts

```text
config/
metrics/
figures/
predictions/
checkpoints/ (optional)
```
