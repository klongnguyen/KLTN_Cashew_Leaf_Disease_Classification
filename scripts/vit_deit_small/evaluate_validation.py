"""Validation-only metrics and per-image exports for EXP-010."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support)

from modeling import CLASSES


def score(frame, logits, seed, stage):
    if len(frame) != 1430 or set(frame["split"]) != {"val"}:
        raise ValueError("Only 1,430 V08 Validation rows are accepted")
    logits = np.asarray(logits, dtype=np.float32)
    if logits.shape != (1430, 5) or not np.isfinite(logits).all():
        raise ValueError(f"Invalid Validation logits: {logits.shape}")
    y = frame.label.to_numpy(dtype=int)
    p = softmax(logits, axis=1)
    pred = p.argmax(axis=1)
    cm = confusion_matrix(y, pred, labels=np.arange(5))
    if cm.sum() != 1430 or int(np.isin(y, [0, 2, 3]).sum()) != 900:
        raise ValueError("Validation count or D2H denominator changed")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=np.arange(5), zero_division=0)
    d2h = int(sum(cm[i, 1] for i in (0, 2, 3)))
    result = {"seed": seed, "stage": stage, "n": 1430,
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(np.mean(precision)), "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)), "log_loss": float(log_loss(y, p, labels=np.arange(5))),
        "disease_to_healthy_count": d2h, "disease_to_healthy_rate": d2h / 900,
        "healthy_recall": float(recall[1]), "healthy_precision": float(precision[1]),
        "not_cashew_leaf_recall": float(recall[4]),
        "anthracnose_to_leaf_miner": int(cm[0, 2]),
        "leaf_miner_to_anthracnose": int(cm[2, 0]),
        "confusion_matrix": cm.tolist(),
        "per_class": [{"class": klass, "support": int(support[i]),
            "precision": float(precision[i]), "recall": float(recall[i]), "f1": float(f1[i]),
            "to_healthy": int(cm[i, 1]) if i in (0, 2, 3) else None}
            for i, klass in enumerate(CLASSES)]}
    predictions = frame[["relative_path", "class", "label", "sha256"]].copy()
    predictions = predictions.rename(columns={"class": "true_class", "label": "true_label",
                                               "sha256": "image_sha256"})
    predictions["predicted_label"] = pred
    predictions["predicted_class"] = [CLASSES[i] for i in pred]
    predictions["confidence"] = p.max(axis=1)
    predictions["correct"] = pred == y
    predictions["seed"] = seed
    predictions["stage"] = stage
    for i, klass in enumerate(CLASSES):
        predictions[f"p_{klass}"] = p[:, i]
    return result, predictions


def export_validation(frame, logits, seed, stage, seed_dir):
    result, predictions = score(frame, logits, seed, stage)
    seed_dir = Path(seed_dir)
    seed_dir.mkdir(parents=True, exist_ok=True)
    (seed_dir / f"metrics_{stage}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    cm = np.asarray(result["confusion_matrix"], dtype=int)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(seed_dir / f"confusion_matrix_val_{stage}.csv")
    pd.DataFrame(cm / cm.sum(axis=1, keepdims=True), index=CLASSES, columns=CLASSES).to_csv(
        seed_dir / f"confusion_matrix_val_normalized_{stage}.csv")
    pd.DataFrame(result["per_class"]).to_csv(seed_dir / f"classification_report_val_{stage}.csv", index=False)
    predictions.to_csv(seed_dir / f"predictions_val_{stage}.csv", index=False)
    errors = predictions.loc[~predictions.correct].copy()
    errors["disease_to_healthy"] = errors.true_label.isin([0, 2, 3]) & (errors.predicted_label == 1)
    errors["anthracnose_leaf_miner"] = ((errors.true_label == 0) & (errors.predicted_label == 2)) | (
        (errors.true_label == 2) & (errors.predicted_label == 0))
    errors.sort_values("confidence", ascending=False).to_csv(seed_dir / f"errors_val_{stage}.csv", index=False)
    real = predictions.loc[predictions.relative_path.str.contains(r"/(?:real_)", regex=True)].copy()
    real.to_csv(seed_dir / f"real_prefix_val_{stage}.csv", index=False)
    return result, predictions
