# %% [markdown]
# # EXP-DEIT-TINY-FINETUNE-5SEEDS-006
#
# DeiT-tiny ImageNet pretrained, fine-tuned on the fixed Cashew_dataV05 split.
# Run this file in Colab, or run the generated notebook of the same name.
# The original scratch ViT is preserved in VIT_5seed_scratch_backup_2026-10-04.ipynb.
# Test stays locked until the configuration and all five validation runs are frozen.
#
# KerasHub preset: https://keras.io/keras_hub/api/models/deit/deit_image_classifier/

# %%
# In Colab, run once if KerasHub is missing:
# %pip install -q "keras-hub==0.32.0" scikit-learn pandas matplotlib pillow

# %%
import gc
import hashlib
import json
import math
import os
import platform
import random
import shutil
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import keras
import keras_hub
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

MATRIX_CMAP = LinearSegmentedColormap.from_list(
    "readable_blues", ["#ffffff", "#dbeafe", "#93c5fd"]
)

# %% [markdown]
# ## 1. Frozen configuration
# V05 archive SHA-256 comes from the local CashewData_Split (5).zip. Counts
# alone cannot identify a split: V04 and V05 have identical counts but differ
# in 78 train/validation assignments.

# %%
NOTEBOOK_VERSION = "006"
EXPERIMENT_ID = "EXP-DEIT-TINY-FINETUNE-5SEEDS-006"
DATASET_VERSION = "Cashew_dataV05"
DATASET_ARCHIVE_SHA256 = "37E76938B930025B88154D53EA56084E0828C57A0B0FB725387462C835EA8CDF"
DATASET_ARCHIVE_NAME = "CashewData_Split (5).zip"
PRESET = "kaggle://keras/deit/keras/deit_tiny_distilled_patch16_224_imagenet/1"
CLASS_NAMES = ["anthracnose", "healthy", "leaf_miner", "not_cashew_leaf", "red_rust"]
EXPECTED_COUNTS = {
    "train": [945, 806, 893, 1101, 1077],
    "val": [294, 225, 249, 314, 320],
    "test": [122, 118, 132, 157, 158],
}
SEEDS = [42, 123, 2026, 3407, 7777]
IMG_SIZE = 224
BATCH_SIZE = 16  # Reduce to 8 and use a new experiment ID if GPU memory is insufficient.
HEAD_EPOCHS = 5
FINETUNE_EPOCHS = 30
HEAD_LR = 3e-4
BACKBONE_LR = 2e-5
MIN_LR = 1e-6
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7
RUN_TRAINING = True
RUN_FINAL_TEST = False

# External images are never used by model.fit or checkpoint/seed selection.
RUN_REAL_HOLDOUT = True  # Effective only when RUN_FINAL_TEST=True.
REAL_HOLDOUT_ARCHIVE_NAME = "RL_Cashew_holdout.zip"
REAL_HOLDOUT_ARCHIVE_SHA256 = "BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165"
REAL_HOLDOUT_EXPECTED_COUNTS = {"anthracnose": 13, "healthy": 5, "leaf_miner": 24, "red_rust": 7}
REAL_HOLDOUT_ARCHIVE_OVERRIDE = None  # Optional mounted/local ZIP path.

# %% [markdown]
# ## 2. Runtime, archive integrity, and dataset audit
# The archive is read from the same Drive location as the V05 scratch notebook.
# Extraction uses a separate local directory and never deletes existing data.

# %%
if Path("/content").is_dir():
    from google.colab import drive

    drive.mount("/content/drive")
    project_root = Path("/content/drive/MyDrive/Cashew_Leaf_model_result")
    archive_path = project_root / "Dataset" / DATASET_ARCHIVE_NAME
    extraction_root = Path("/content/deit_v05_dataset")
    results_root = project_root / "training_results" / "cashew_datav05"
elif os.name == "nt":
    project_root = Path(r"D:\Study\KhoaLuanTotNghiep\Resource")
    archive_path = Path(r"D:\Study\KhoaLuanTotNghiep\Dataset") / DATASET_ARCHIVE_NAME
    extraction_root = project_root / ".deit_v05_dataset"
    results_root = project_root / "training_results" / "cashew_datav05"
else:
    raise RuntimeError("Use Google Colab or configure local dataset/results paths.")

if not archive_path.is_file():
    raise FileNotFoundError(archive_path)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest().upper()


actual_archive_hash = sha256_file(archive_path)
if actual_archive_hash != DATASET_ARCHIVE_SHA256:
    raise ValueError(
        f"Wrong V05 archive: {actual_archive_hash}. Expected {DATASET_ARCHIVE_SHA256}."
    )

marker = extraction_root / ".archive_sha256"
if extraction_root.exists():
    if not marker.is_file() or marker.read_text().strip() != DATASET_ARCHIVE_SHA256:
        raise RuntimeError(f"Unverified extraction at {extraction_root}; inspect it before reuse.")
else:
    extraction_root.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        root_resolved = extraction_root.resolve()
        for member in archive.infolist():
            target = (extraction_root / member.filename).resolve()
            if target != root_resolved and root_resolved not in target.parents:
                raise ValueError(f"Unsafe ZIP path: {member.filename}")
        archive.extractall(extraction_root)
    marker.write_text(DATASET_ARCHIVE_SHA256, encoding="utf-8")

dataset_root = extraction_root / "CashewData_Split"
for split in ("train", "val", "test"):
    if not (dataset_root / split).is_dir():
        raise FileNotFoundError(dataset_root / split)

out_dir = results_root / EXPERIMENT_ID
aggregate_dir = out_dir / "aggregate"
package_dir = out_dir / "model_package"
if RUN_TRAINING and out_dir.exists() and any(out_dir.rglob("*.weights.h5")):
    raise FileExistsError(f"Existing checkpoints at {out_dir}; choose a new ID or RUN_TRAINING=False.")
if not RUN_TRAINING and not (out_dir / "experiment_config.json").is_file():
    raise FileNotFoundError("Evaluation-only mode needs a completed saved experiment.")
for folder in (out_dir, aggregate_dir, package_dir):
    folder.mkdir(parents=True, exist_ok=True)

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
rows = []
for split in ("train", "val", "test"):
    found = sorted(p.name for p in (dataset_root / split).iterdir() if p.is_dir())
    if found != CLASS_NAMES:
        raise ValueError(f"Class directories mismatch in {split}: {found}")
    for label, class_name in enumerate(CLASS_NAMES):
        files = sorted(p for p in (dataset_root / split / class_name).rglob("*") if p.is_file())
        if len(files) != EXPECTED_COUNTS[split][label]:
            raise ValueError(f"Unexpected {split}/{class_name} count: {len(files)}")
        for path in files:
            if path.suffix.lower() not in valid_ext:
                raise ValueError(f"Unknown image extension: {path}")
            rows.append({
                "split": split,
                "class": class_name,
                "label": label,
                "relative_path": path.relative_to(dataset_root).as_posix(),
                "sha256": sha256_file(path),
            })

manifest = pd.DataFrame(rows)
manifest.to_csv(out_dir / "dataset_manifest.csv", index=False)
split_overlap = manifest.groupby("sha256")["split"].nunique()
if (split_overlap > 1).any():
    raise ValueError("Exact duplicate image bytes occur across train/val/test splits.")
manifest_hash = hashlib.sha256(
    manifest[["split", "class", "relative_path", "sha256"]]
    .to_csv(index=False, lineterminator="\n")
    .encode()
).hexdigest()
print("Dataset:", DATASET_VERSION, "images:", len(manifest), "manifest SHA-256:", manifest_hash)

gpus = tf.config.list_physical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if not gpus:
    raise RuntimeError("Fine-tuning requires a GPU runtime. Select GPU in Colab.")
print("GPU:", gpus, "TensorFlow:", tf.__version__, "KerasHub:", keras_hub.__version__)

config = {
    "experiment_id": EXPERIMENT_ID,
    "notebook_version": NOTEBOOK_VERSION,
    "external_holdout_protocol": {
        "archive_name": REAL_HOLDOUT_ARCHIVE_NAME,
        "archive_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,
        "expected_counts": REAL_HOLDOUT_EXPECTED_COUNTS,
        "gate": "RUN_FINAL_TEST and RUN_REAL_HOLDOUT",
        "separate_from_v05_test": True,
        "macro_scope": "true classes with holdout support; currently four",
        "selection_source": "validation only",
    },
    "dataset_version": DATASET_VERSION,
    "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
    "dataset_manifest_sha256": manifest_hash,
    "classes": CLASS_NAMES,
    "counts": EXPECTED_COUNTS,
    "seeds": SEEDS,
    "preset": PRESET,
    "training_mode": "ImageNet pretrained, head then full fine-tune",
    "image_size": IMG_SIZE,
    "batch_size": BATCH_SIZE,
    "head_epochs": HEAD_EPOCHS,
    "finetune_epochs": FINETUNE_EPOCHS,
    "head_lr": HEAD_LR,
    "backbone_lr": BACKBONE_LR,
    "min_lr": MIN_LR,
    "weight_decay": WEIGHT_DECAY,
    "checkpoint_monitor": "val_loss",
    "deployment_seed_metric": "val_macro_f1",
    "test_locked_by_default": True,
}
config_path = out_dir / "experiment_config.json"
if config_path.exists():
    saved_config = json.loads(config_path.read_text(encoding="utf-8"))
    if saved_config != config:
        raise ValueError("Saved experiment config differs from this run. Use a new experiment ID.")
else:
    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
environment_path = out_dir / "environment.json"
if not environment_path.exists():
    environment_path.write_text(json.dumps({
        "tensorflow": tf.__version__, "keras": keras.__version__,
        "keras_hub": keras_hub.__version__, "python": platform.python_version(),
        "gpu": [gpu.name for gpu in gpus],
    }, indent=2), encoding="utf-8")

# %% [markdown]
# ## External holdout preflight

# %%
"""Locked external classification evaluation; embedded in standalone notebooks.

This file has no training code. Holdout labels never select checkpoints or seeds.
"""

import hashlib
import json
import os
import stat
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import f1_score, precision_score, recall_score


REAL_HOLDOUT_HELPER_VERSION = "006.1"
REAL_HOLDOUT_CMAP = LinearSegmentedColormap.from_list(
    "real_holdout_readable_blues", ["#ffffff", "#dbeafe", "#93c5fd"]
)


def holdout_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def resolve_holdout_archive(archive_name, override=None):
    if override is not None:
        candidate = Path(override)
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate
    if Path("/content/drive/MyDrive").is_dir():
        candidate = Path("/content/drive/MyDrive/Cashew_Leaf_model_result/Dataset") / archive_name
        if candidate.is_file():
            return candidate
    if os.name == "nt":
        candidate = Path(r"D:\Study\KhoaLuanTotNghiep\Dataset") / archive_name
        if candidate.is_file():
            return candidate
    if Path("/kaggle/input").is_dir():
        candidates = sorted(Path("/kaggle/input").rglob(archive_name))
        if len(candidates) == 1:
            return candidates[0]
    raise FileNotFoundError(
        f"Cannot uniquely locate {archive_name}. Set REAL_HOLDOUT_ARCHIVE_OVERRIDE "
        "to the mounted ZIP path; on Colab add the supplied Dataset folder to My Drive."
    )


def prepare_real_holdout(archive_path, extraction_dir, class_names, expected_sha256,
                         expected_counts, reference_rows, audit_dir):
    """Verify the frozen archive, labels, decoding and byte overlap with V05.

    reference_rows: dictionaries with path/split and optional existing sha256.
    Exact-byte checking does not establish independence of near-duplicate photos.
    """
    archive_path, extraction_dir, audit_dir = map(Path, (archive_path, extraction_dir, audit_dir))
    actual_hash = holdout_sha256(archive_path)
    if actual_hash != expected_sha256.upper():
        raise ValueError(f"Holdout archive changed: {actual_hash}; expected {expected_sha256}.")
    marker = extraction_dir / ".archive_sha256"
    if extraction_dir.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != actual_hash:
            raise FileExistsError(f"Unverified extraction: {extraction_dir}; choose a new directory.")
    else:
        with zipfile.ZipFile(archive_path) as archive:
            resolved_root = extraction_dir.resolve()
            for member in archive.infolist():
                target = (extraction_dir / member.filename.replace("\\", "/")).resolve()
                is_symlink = stat.S_ISLNK(member.external_attr >> 16)
                if is_symlink or (target != resolved_root and resolved_root not in target.parents):
                    raise ValueError(f"Unsafe ZIP member: {member.filename}")
            extraction_dir.mkdir(parents=True)
            archive.extractall(extraction_dir)
        marker.write_text(actual_hash, encoding="utf-8")

    dataset_root = extraction_dir / "RL_Cashew_holdout"
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    found_classes = sorted(p.name for p in dataset_root.iterdir() if p.is_dir())
    if found_classes != sorted(expected_counts) or not set(found_classes).issubset(class_names):
        raise ValueError(f"Unexpected holdout class folders: {found_classes}")
    rows = []
    for label, name in enumerate(class_names):
        class_dir = dataset_root / name
        files = sorted(p for p in class_dir.rglob("*") if p.is_file()) if class_dir.is_dir() else []
        if len(files) != expected_counts.get(name, 0):
            raise ValueError(f"Unexpected holdout count for {name}: {len(files)}")
        for image_path in files:
            if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                raise ValueError(f"Unsupported holdout image: {image_path}")
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                image.convert("RGB").load()
            rows.append({"path": str(image_path), "relative_path": image_path.relative_to(dataset_root).as_posix(),
                         "class": name, "label": label, "sha256": holdout_sha256(image_path)})
    manifest = pd.DataFrame(rows)
    if manifest.empty or manifest["sha256"].duplicated().any():
        raise ValueError("Holdout is empty or contains duplicate image bytes.")
    # A marker alone is insufficient: verify cached extraction bytes against the ZIP.
    with zipfile.ZipFile(archive_path) as frozen_archive:
        for row in rows:
            digest = hashlib.sha256()
            with frozen_archive.open("RL_Cashew_holdout/" + row["relative_path"]) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            if digest.hexdigest().upper() != row["sha256"]:
                raise ValueError(f"Cached holdout extraction changed: {row['relative_path']}")
    audit_dir.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(audit_dir / "dataset_manifest.csv", index=False)

    references = list(reference_rows)
    if not references or not {"train", "val", "test"}.issubset({r["split"] for r in references}):
        raise ValueError("Overlap audit needs the full train/val/test V05 reference manifest.")
    holdout_hashes = set(manifest["sha256"])
    overlaps = []
    reference_digest = hashlib.sha256()
    for row in references:
        digest = row.get("sha256") or holdout_sha256(row["path"])
        digest = digest.upper()
        identity = row.get("relative_path", str(row["path"]))
        reference_digest.update(f"{row['split']}\0{identity}\0{digest}\n".encode("utf-8"))
        if digest in holdout_hashes:
            overlaps.append({"reference_path": str(row["path"]), "split": row["split"], "sha256": digest})
    present = [name for name in class_names if expected_counts.get(name, 0) > 0]
    audit = {"archive_sha256": actual_hash, "archive_name": archive_path.name,
             "manifest_sha256": hashlib.sha256(manifest[["relative_path", "class", "label", "sha256"]]
                 .to_csv(index=False).encode()).hexdigest(),
             "counts": {name: expected_counts.get(name, 0) for name in class_names},
             "images": len(manifest), "present_classes": present,
             "unsupported_classes": [name for name in class_names if name not in present],
             "reference_images_checked": len(references), "reference_fingerprint": reference_digest.hexdigest(),
             "byte_overlap_count": len(overlaps), "byte_overlaps": overlaps,
             "limitation": "Byte audit only; near-duplicate photos and shared trees/sessions need a separate audit."}
    (audit_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if overlaps:
        raise ValueError("Real holdout overlaps V05 by exact bytes; inspect dataset_audit.json before evaluation.")
    print("Real holdout:", len(manifest), "images; present classes:", present)
    print("Not evaluated (no true examples):", audit["unsupported_classes"])
    return manifest, audit


def save_real_holdout_seed(manifest, probabilities, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if probabilities.shape != (len(manifest), len(class_names)) or not np.isfinite(probabilities).all():
        raise ValueError("Holdout predictions have invalid shape or non-finite values.")
    if (probabilities < 0).any() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Holdout evaluator expects probabilities, not logits.")
    y_true = manifest["label"].to_numpy(dtype=np.int32)
    y_pred = probabilities.argmax(axis=1)
    present_ids = sorted(set(y_true.tolist()))
    all_ids = list(range(len(class_names)))
    result = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision_present_classes": float(precision_score(y_true, y_pred, labels=present_ids, average="macro", zero_division=0)),
        "macro_recall_present_classes": float(recall_score(y_true, y_pred, labels=present_ids, average="macro", zero_division=0)),
        "macro_f1_present_classes": float(f1_score(y_true, y_pred, labels=present_ids, average="macro", zero_division=0)),
        "balanced_accuracy_present_classes": float(recall_score(y_true, y_pred, labels=present_ids, average="macro", zero_division=0)),
        "images": len(manifest), "macro_class_count": len(present_ids),
    }
    report = pd.DataFrame(classification_report(y_true, y_pred, labels=all_ids,
        target_names=class_names, output_dict=True, zero_division=0)).transpose()
    per_class = report.loc[class_names, ["precision", "recall", "f1-score", "support"]].copy()
    per_class["evaluated"] = per_class["support"] > 0
    # No evidence for unsupported true classes: don't present placeholder zeros as performance.
    per_class.loc[~per_class["evaluated"], ["precision", "recall", "f1-score"]] = np.nan
    per_class.to_csv(output_dir / "per_class_metrics.csv", index_label="class")
    readable_report = per_class.copy()
    readable_report["scope"] = np.where(readable_report["evaluated"], "true class present", "not evaluated: zero true support")
    readable_report.loc["macro avg (present classes)"] = [
        result["macro_precision_present_classes"], result["macro_recall_present_classes"],
        result["macro_f1_present_classes"], len(manifest), True,
        f"{len(present_ids)} supported true classes only",
    ]
    readable_report.loc["weighted avg"] = [
        report.loc["weighted avg", "precision"], report.loc["weighted avg", "recall"],
        report.loc["weighted avg", "f1-score"], len(manifest), True, "weighted by true support",
    ]
    readable_report.to_csv(output_dir / "classification_report.csv", index_label="class")
    predictions = manifest[["relative_path", "class", "label"]].copy()
    predictions["predicted_label"] = y_pred
    predictions["predicted_class"] = [class_names[i] for i in y_pred]
    predictions["confidence"] = probabilities.max(axis=1)
    predictions["correct"] = y_true == y_pred
    for label, name in enumerate(class_names):
        predictions[f"probability_{name}"] = probabilities[:, label]
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    matrix = confusion_matrix(y_true, y_pred, labels=all_ids)
    for normalized, suffix in ((False, ""), (True, "_normalized")):
        values = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1) if normalized else matrix
        pd.DataFrame(values, index=class_names, columns=class_names).to_csv(output_dir / f"confusion_matrix{suffix}.csv")
        fig, ax = plt.subplots(figsize=(9, 8), facecolor="white")
        ax.set_facecolor("white")
        ax.imshow(values, cmap=REAL_HOLDOUT_CMAP, vmin=0,
                  vmax=1 if normalized else max(int(matrix.max()), 1))
        ax.set_xticks(all_ids, class_names, rotation=40, ha="right", color="black")
        ax.set_yticks(all_ids, class_names, color="black")
        ax.set_xlabel("Predicted class", color="black")
        ax.set_ylabel("True class", color="black")
        ax.set_title("Real holdout" + (" (row normalized)" if normalized else ""), color="black")
        for row in all_ids:
            for col in all_ids:
                text = f"{values[row, col]:.2f}" if normalized else str(int(values[row, col]))
                if normalized and row not in present_ids:
                    text = "N/A"
                ax.text(col, row, text, ha="center", va="center", color="black")
        fig.tight_layout()
        fig.savefig(output_dir / f"confusion_matrix{suffix}.png", dpi=180)
        plt.close(fig)
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def evaluate_real_holdout(manifest, audit, class_names, seeds, validation_results,
                          checkpoints, load_model, make_dataset, clear_session,
                          output_dir, outputs_are_logits=False):
    """Freeze all checkpoints and validation selection BEFORE any prediction.

    Completed evaluations are read back, not repeated. Interrupted evaluations
    may resume only with the same checkpoint, validation and archive fingerprints.
    """
    output_dir = Path(output_dir)
    seeds = [int(seed) for seed in seeds]
    validation_results = validation_results.copy()
    if validation_results["seed"].tolist() != seeds:
        raise ValueError("Holdout evaluation requires completed, ordered validation for every seed.")
    if not np.isfinite(validation_results[["val_macro_f1", "val_loss"]].to_numpy()).all():
        raise ValueError("Validation selection metrics are incomplete.")
    ranked = validation_results.sort_values(["val_macro_f1", "val_loss", "seed"], ascending=[False, True, True])
    deployment_seed = int(ranked.iloc[0]["seed"])
    checkpoint_hashes = {str(seed): holdout_sha256(checkpoints[seed]) for seed in seeds}
    lock = {"helper_version": REAL_HOLDOUT_HELPER_VERSION, "archive_sha256": audit["archive_sha256"],
            "manifest_sha256": audit["manifest_sha256"],
            "reference_fingerprint": audit["reference_fingerprint"], "seeds": seeds,
            "checkpoint_sha256": checkpoint_hashes, "deployment_seed": deployment_seed,
            # Canonical float precision avoids false lock changes on CSV reload.
            "validation_sha256": hashlib.sha256(validation_results.to_csv(index=False, float_format="%.12g").encode()).hexdigest(),
            "model_classes": class_names, "macro_classes": audit["present_classes"],
            "selection_source": "validation only; macro F1, val_loss, seed tie-break",
            "outputs_are_logits": outputs_are_logits}
    lock_path = output_dir / "evaluation_lock.json"
    if lock_path.exists():
        if json.loads(lock_path.read_text(encoding="utf-8")) != lock:
            raise ValueError("Holdout lock changed. Do not retune using holdout or overwrite its results.")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(json.dumps(lock, indent=2), encoding="utf-8")
    aggregate_dir = output_dir / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    results_path = aggregate_dir / "real_holdout_results_5seeds.csv"
    if (output_dir / "evaluation_completed.json").exists():
        print("Real holdout already completed; reading frozen results, not running predictions again.")
        return pd.read_csv(results_path)
    result_rows = []
    for seed in seeds:
        seed_dir = output_dir / f"seed_{seed}"
        completed_path = seed_dir / "evaluation_completed.json"
        if completed_path.exists():
            row = json.loads(completed_path.read_text(encoding="utf-8"))
        else:
            clear_session()
            model = load_model(seed)
            dataset = make_dataset(manifest)
            start = time.perf_counter()
            scores = np.asarray(model.predict(dataset, verbose=0))
            prediction_seconds = time.perf_counter() - start
            if outputs_are_logits:
                exponentials = np.exp(scores - scores.max(axis=1, keepdims=True))
                scores = exponentials / exponentials.sum(axis=1, keepdims=True)
            row = {"seed": seed, **save_real_holdout_seed(manifest, scores, class_names, seed_dir),
                   "prediction_seconds_including_input_pipeline": prediction_seconds}
            completed_path.write_text(json.dumps(row, indent=2), encoding="utf-8")
            del model, scores, dataset
            clear_session()
        result_rows.append(row)
    results = pd.DataFrame(result_rows)
    results.to_csv(results_path, index=False)
    metric_columns = [c for c in results if c not in {"seed", "images", "macro_class_count"}]
    results[metric_columns].agg(["mean", "std"]).transpose().to_csv(aggregate_dir / "real_holdout_mean_std.csv")
    per_class = pd.concat([pd.read_csv(output_dir / f"seed_{seed}" / "per_class_metrics.csv")
                          .assign(seed=seed) for seed in seeds], ignore_index=True)
    per_class.groupby("class")[["precision", "recall", "f1-score"]].agg(["mean", "std"]).to_csv(
        aggregate_dir / "per_class_real_holdout_mean_std.csv")
    deployment_result = next(row for row in result_rows if row["seed"] == deployment_seed)
    (aggregate_dir / "deployment_seed_result.json").write_text(json.dumps({
        "selection_source": "validation only, fixed before holdout", "result": deployment_result,
    }, indent=2), encoding="utf-8")
    (output_dir / "evaluation_completed.json").write_text(json.dumps({
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": "No model or seed selection on holdout. SD measures training-seed variability, not sampling uncertainty.",
    }, indent=2), encoding="utf-8")
    print("Real holdout results (macro metrics over supported true classes only):")
    print(results)
    return results

# --- Notebook-specific real holdout preflight (no prediction/training here) ---
if RUN_FINAL_TEST and RUN_REAL_HOLDOUT:
    holdout_archive = resolve_holdout_archive(REAL_HOLDOUT_ARCHIVE_NAME, REAL_HOLDOUT_ARCHIVE_OVERRIDE)
    holdout_reference_rows = manifest.assign(
        path=manifest["relative_path"].map(lambda relative: str(dataset_root / relative))
    ).to_dict("records")
    holdout_out_dir = out_dir / "real_holdout"
    # Use runtime-local storage, not the Drive result directory, for images.
    if Path("/content").is_dir():
        holdout_extract_dir = Path("/content/real_holdout_006") / REAL_HOLDOUT_ARCHIVE_SHA256[:12]
    elif Path("/kaggle/working").is_dir():
        holdout_extract_dir = Path("/kaggle/working/real_holdout_006") / REAL_HOLDOUT_ARCHIVE_SHA256[:12]
    else:
        import tempfile
        holdout_extract_dir = Path(tempfile.gettempdir()) / "cashew_real_holdout_006" / REAL_HOLDOUT_ARCHIVE_SHA256[:12]
    real_holdout_manifest, real_holdout_audit = prepare_real_holdout(
        holdout_archive, holdout_extract_dir, CLASS_NAMES,
        REAL_HOLDOUT_ARCHIVE_SHA256, REAL_HOLDOUT_EXPECTED_COUNTS,
        holdout_reference_rows, holdout_out_dir,
    )
else:
    print("Real-world holdout remains locked; no holdout extraction/evaluation runs.")

# %% [markdown]
# ## 3. Input pipeline and pretrained model
# The KerasHub preset includes its own DeiT image preprocessor. Dataset tensors
# remain RGB float32 in [0, 255]; no external 1/255 or ImageNet normalization.
# Augmentation runs only during training. Validation and Test are deterministic.

# %%
paths = {}
labels = {}
for split in ("train", "val", "test"):
    part = manifest.loc[manifest["split"] == split]
    paths[split] = [str(dataset_root / p) for p in part["relative_path"]]
    labels[split] = part["label"].to_numpy(dtype=np.int32)


def decode_image(path, label):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE), method="bilinear")
    return tf.cast(image, tf.float32), label


def make_dataset(split, seed, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((paths[split], labels[split]))
    if shuffle:
        ds = ds.shuffle(len(paths[split]), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


def build_model(seed):
    set_seed(seed)
    for attempt in range(3):
        try:
            classifier = keras_hub.models.DeiTImageClassifier.from_preset(
                PRESET,
                num_classes=len(CLASS_NAMES),
                activation=None,
                head_dtype="float32",
            )
            break
        except ValueError as exc:
            if attempt == 2 or not any(code in str(exc) for code in ("500", "502", "503")):
                raise
            time.sleep(3 * (attempt + 1))
    if classifier.preprocessor is None:
        raise RuntimeError("DeiT preset lacks preprocessing; stop before training.")
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed + 1),
        keras.layers.RandomRotation(0.04, seed=seed + 2),
        keras.layers.RandomZoom((-0.05, 0.05), (-0.05, 0.05), seed=seed + 3),
        keras.layers.RandomTranslation(0.03, 0.03, seed=seed + 4),
        keras.layers.RandomContrast(0.08, seed=seed + 5),
    ], name="augmentation")
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), dtype="float32", name="image_0_255")
    # Task.preprocessor is normally invoked by Task.fit/predict/evaluate.
    # This classifier is nested in a keras.Model, so call it explicitly.
    processed = classifier.preprocessor(augmentation(inputs))
    outputs = classifier(processed)
    model = keras.Model(inputs, outputs, name="deit_tiny_finetune")
    if model.output_shape != (None, len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected classifier shape: {model.output_shape}")
    return model, classifier


def compile_model(model, learning_rate):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=WEIGHT_DECAY),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def finetune_lr(epoch, unused_lr):
    if epoch < 3:
        return BACKBONE_LR * (epoch + 1) / 3
    progress = (epoch - 3) / max(1, FINETUNE_EPOCHS - 4)
    return MIN_LR + 0.5 * (BACKBONE_LR - MIN_LR) * (1 + math.cos(math.pi * progress))


def prediction_metrics(y_true, logits):
    y_pred = np.argmax(logits, axis=1)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }, y_pred

# %% [markdown]
# ## 4. Five-seed training and validation
# Stage 1 trains the new head with the backbone frozen. Stage 2 unfreezes the
# backbone and uses a small learning rate with warm-up and cosine decay.
# Both stages save their own best val_loss weights, then the better checkpoint
# is chosen. Test is never used for checkpoint or seed selection.

# %%
validation_rows = []
training_rows = []
all_histories = []

if RUN_TRAINING:
    for seed in SEEDS:
        print(f"\n===== SEED {seed} =====")
        keras.backend.clear_session()
        gc.collect()
        set_seed(seed)
        seed_dir = out_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        head_path = seed_dir / "head_best.weights.h5"
        ft_path = seed_dir / "finetune_best.weights.h5"
        model_path = seed_dir / "best_model.keras"
        if any(p.exists() for p in (head_path, ft_path, model_path)):
            raise FileExistsError(f"Refusing to overwrite seed {seed} outputs.")

        train_ds = make_dataset("train", seed, shuffle=True)
        train_clean_ds = make_dataset("train", seed, shuffle=False)
        val_ds = make_dataset("val", seed, shuffle=False)
        model, classifier = build_model(seed)
        total_params = int(model.count_params())
        classifier.backbone.trainable = False
        compile_model(model, HEAD_LR)
        head_start = time.perf_counter()
        h1 = model.fit(
            train_ds, validation_data=val_ds, epochs=HEAD_EPOCHS, verbose=1,
            callbacks=[keras.callbacks.ModelCheckpoint(
                str(head_path), monitor="val_loss", mode="min",
                save_best_only=True, save_weights_only=True,
            )],
        )
        head_training_seconds = time.perf_counter() - head_start
        model.load_weights(head_path)
        head_best_loss = float(min(h1.history["val_loss"]))

        classifier.backbone.trainable = True
        compile_model(model, BACKBONE_LR)
        finetune_start = time.perf_counter()
        h2 = model.fit(
            train_ds, validation_data=val_ds, epochs=FINETUNE_EPOCHS, verbose=1,
            callbacks=[
                keras.callbacks.LearningRateScheduler(finetune_lr, verbose=0),
                keras.callbacks.ModelCheckpoint(
                    str(ft_path), monitor="val_loss", mode="min",
                    save_best_only=True, save_weights_only=True,
                ),
                keras.callbacks.EarlyStopping(
                    monitor="val_loss", mode="min",
                    patience=EARLY_STOPPING_PATIENCE, restore_best_weights=True,
                ),
            ],
        )
        finetune_training_seconds = time.perf_counter() - finetune_start
        ft_best_loss = float(min(h2.history["val_loss"]))
        selected_stage = "finetune" if ft_best_loss < head_best_loss else "head"
        model.load_weights(ft_path if selected_stage == "finetune" else head_path)
        train_clean_loss, train_clean_acc = model.evaluate(train_clean_ds, verbose=0)
        val_loss, _ = model.evaluate(val_ds, verbose=0)
        val_logits = model.predict(val_ds, verbose=0)
        val_metrics, val_pred = prediction_metrics(labels["val"], val_logits)
        model.save(model_path)

        history_rows = []
        for stage, history, offset in (("head", h1, 0), ("finetune", h2, HEAD_EPOCHS)):
            for i in range(len(history.history["loss"])):
                history_rows.append({
                    "seed": seed, "stage": stage, "epoch": offset + i + 1,
                    "train_loss": float(history.history["loss"][i]),
                    "train_aug_accuracy": float(history.history["accuracy"][i]),
                    "val_loss": float(history.history["val_loss"][i]),
                    "val_accuracy": float(history.history["val_accuracy"][i]),
                })
        pd.DataFrame(history_rows).to_csv(seed_dir / "history.csv", index=False)
        all_histories.extend(history_rows)

        cm = confusion_matrix(labels["val"], val_pred, labels=range(len(CLASS_NAMES)))
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(seed_dir / "validation_confusion_matrix.csv")
        report = classification_report(
            labels["val"], val_pred, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )
        pd.DataFrame(report).transpose().to_csv(seed_dir / "validation_classification_report.csv")
        pd.DataFrame({
            "relative_path": manifest.loc[manifest["split"] == "val", "relative_path"].to_numpy(),
            "true_label": labels["val"], "predicted_label": val_pred,
        }).to_csv(seed_dir / "validation_predictions.csv", index=False)

        validation_rows.append({"seed": seed, "val_loss": float(val_loss), **{
            f"val_{k}": v for k, v in val_metrics.items()
        }})
        training_rows.append({
            "seed": seed, "selected_stage": selected_stage,
            "head_best_epoch": int(np.argmin(h1.history["val_loss"]) + 1),
            "finetune_best_epoch": int(np.argmin(h2.history["val_loss"]) + 1),
            "head_best_val_loss": head_best_loss,
            "finetune_best_val_loss": ft_best_loss,
            "actual_epochs": len(h1.history["loss"]) + len(h2.history["loss"]),
            "train_clean_loss": float(train_clean_loss),
            "train_clean_accuracy": float(train_clean_acc),
            "head_training_seconds": float(head_training_seconds),
            "finetune_training_seconds": float(finetune_training_seconds),
            "training_seconds": float(head_training_seconds + finetune_training_seconds),
            "training_time_scope": "model.fit including epoch validation and callbacks",
            "total_parameters": total_params,
            "checkpoint": str(model_path),
        })
        print("Val:", validation_rows[-1], "Clean train accuracy:", train_clean_acc)
        del model, classifier, train_ds, train_clean_ds, val_ds, h1, h2
        keras.backend.clear_session()
        gc.collect()

    pd.DataFrame(validation_rows).to_csv(aggregate_dir / "validation_results_5seeds.csv", index=False)
    pd.DataFrame(training_rows).to_csv(aggregate_dir / "training_summary_5seeds.csv", index=False)
    pd.DataFrame(all_histories).to_csv(aggregate_dir / "history_5seeds.csv", index=False)
else:
    validation_rows = pd.read_csv(aggregate_dir / "validation_results_5seeds.csv").to_dict("records")
    training_rows = pd.read_csv(aggregate_dir / "training_summary_5seeds.csv").to_dict("records")
    all_histories = pd.read_csv(aggregate_dir / "history_5seeds.csv").to_dict("records")

validation_df = pd.DataFrame(validation_rows)
training_df = pd.DataFrame(training_rows)
if validation_df["seed"].tolist() != SEEDS or training_df["seed"].tolist() != SEEDS:
    raise ValueError("Five-seed results are incomplete or out of order.")
time_columns = [
    c for c in ("head_training_seconds", "finetune_training_seconds", "training_seconds")
    if c in training_df.columns
]
timing_scope = "model.fit including epoch validation and callbacks"
if "head_training_seconds" not in training_df.columns:
    timing_scope = "legacy: fit plus post-fit evaluation and artifact saving"
    print("WARNING: Loaded legacy timings include post-fit work; fit-only time cannot be recovered.")
time_summary = training_df[time_columns].agg(["mean", "std"]).transpose()
time_summary["measurement_scope"] = timing_scope
time_summary.to_csv(aggregate_dir / "training_time_mean_std_seconds.csv")
print("\nTraining time mean and sample SD (seconds):\n", time_summary)
metric_columns = [c for c in validation_df if c.startswith("val_") and c != "val_loss"]
summary = validation_df[metric_columns].agg(["mean", "std"]).transpose()
summary.to_csv(aggregate_dir / "validation_mean_std.csv")
print("\nValidation mean ± sample SD:\n", summary)
per_class_rows = []
summed_val_cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
for seed in SEEDS:
    seed_dir = out_dir / f"seed_{seed}"
    report = pd.read_csv(seed_dir / "validation_classification_report.csv", index_col=0)
    for class_name in CLASS_NAMES:
        per_class_rows.append({
            "seed": seed, "class": class_name,
            "precision": float(report.loc[class_name, "precision"]),
            "recall": float(report.loc[class_name, "recall"]),
            "f1": float(report.loc[class_name, "f1-score"]),
        })
    summed_val_cm += pd.read_csv(
        seed_dir / "validation_confusion_matrix.csv", index_col=0
    ).to_numpy(dtype=np.int64)
pd.DataFrame(per_class_rows).groupby("class")[["precision", "recall", "f1"]].agg(
    ["mean", "std"]
).to_csv(aggregate_dir / "per_class_validation_mean_std.csv")
pd.DataFrame(summed_val_cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
    aggregate_dir / "validation_confusion_matrix_5seeds_sum.csv"
)

ranked = validation_df.sort_values(["val_macro_f1", "val_loss"], ascending=[False, True])
deployment_seed = int(ranked.iloc[0]["seed"])
source_model = out_dir / f"seed_{deployment_seed}" / "best_model.keras"
deployment_model = package_dir / "deployment_model.keras"
if not deployment_model.exists():
    shutil.copy2(source_model, deployment_model)
(package_dir / "label_map.json").write_text(
    json.dumps({str(i): name for i, name in enumerate(CLASS_NAMES)}, indent=2), encoding="utf-8"
)
(package_dir / "deployment_metadata.json").write_text(json.dumps({
    "experiment_id": EXPERIMENT_ID, "dataset_version": DATASET_VERSION,
    "seed": deployment_seed, "selection": "highest validation macro F1, val_loss tie-break",
    "input": "RGB float32 [0,255], shape [batch,224,224,3]",
    "preprocessing": "included in deployment_model.keras; do not divide by 255 externally",
    "output": "5 logits in CLASS_NAMES order",
    "class_names": CLASS_NAMES,
}, indent=2), encoding="utf-8")
print("Deployment seed selected on validation:", deployment_seed)

# %% [markdown]
# ## 5. Validation figures

# %%
history_df = pd.DataFrame(all_histories)
fig, axes = plt.subplots(2, len(SEEDS), figsize=(25, 8), sharey="row")
for col, seed in enumerate(SEEDS):
    hist = history_df.loc[history_df["seed"] == seed]
    axes[0, col].plot(hist["epoch"], hist["train_aug_accuracy"], label="train augmented")
    axes[0, col].plot(hist["epoch"], hist["val_accuracy"], label="validation")
    axes[1, col].plot(hist["epoch"], hist["train_loss"], label="train")
    axes[1, col].plot(hist["epoch"], hist["val_loss"], label="validation")
    for row in (0, 1):
        axes[row, col].axvline(HEAD_EPOCHS + 0.5, color="gray", linestyle="--", alpha=0.6)
        axes[row, col].set_title(f"Seed {seed}")
        axes[row, col].set_xlabel("Epoch")
        axes[row, col].grid(alpha=0.2)
axes[0, 0].set_ylabel("Accuracy")
axes[1, 0].set_ylabel("Loss")
axes[0, 0].legend()
fig.tight_layout()
fig.savefig(aggregate_dir / "training_curves_5seeds.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, len(SEEDS), figsize=(26, 5.5))
for col, seed in enumerate(SEEDS):
    cm = pd.read_csv(
        out_dir / f"seed_{seed}" / "validation_confusion_matrix.csv", index_col=0
    ).to_numpy(dtype=np.float64)
    normalized = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    axes[col].imshow(normalized, cmap=MATRIX_CMAP, vmin=0, vmax=1)
    axes[col].set_title(f"Seed {seed}")
    axes[col].set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right", fontsize=8)
    axes[col].set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES, fontsize=8)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            axes[col].text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")
fig.tight_layout()
fig.savefig(aggregate_dir / "validation_confusion_5seeds.png", dpi=180)
plt.close(fig)

# %% [markdown]
# ## 6. Locked final Test
# Set RUN_TRAINING=False and RUN_FINAL_TEST=True only after freezing the five
# checkpoints and validation selection. No tuning is permitted after Test.

# %%
if RUN_FINAL_TEST:
    test_rows = []
    per_class_test_rows = []
    test_ds = make_dataset("test", SEEDS[0], shuffle=False)
    for seed in SEEDS:
        model = keras.models.load_model(out_dir / f"seed_{seed}" / "best_model.keras", compile=False)
        logits = model.predict(test_ds, verbose=0)
        metrics, predictions = prediction_metrics(labels["test"], logits)
        test_rows.append({"seed": seed, **{f"test_{k}": v for k, v in metrics.items()}})
        seed_dir = out_dir / f"seed_{seed}"
        pd.DataFrame(confusion_matrix(
            labels["test"], predictions, labels=range(len(CLASS_NAMES))
        ), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(seed_dir / "test_confusion_matrix.csv")
        report = pd.DataFrame(classification_report(
            labels["test"], predictions, target_names=CLASS_NAMES,
            output_dict=True, zero_division=0,
        )).transpose()
        report.to_csv(seed_dir / "test_classification_report.csv")
        for class_name in CLASS_NAMES:
            per_class_test_rows.append({
                "seed": seed, "class": class_name,
                "precision": float(report.loc[class_name, "precision"]),
                "recall": float(report.loc[class_name, "recall"]),
                "f1": float(report.loc[class_name, "f1-score"]),
            })
        pd.DataFrame({
            "relative_path": manifest.loc[manifest["split"] == "test", "relative_path"].to_numpy(),
            "true_label": labels["test"], "predicted_label": predictions,
        }).to_csv(seed_dir / "test_predictions.csv", index=False)
        del model
        keras.backend.clear_session()
        gc.collect()
    test_df = pd.DataFrame(test_rows)
    test_df.to_csv(aggregate_dir / "test_results_5seeds.csv", index=False)
    test_df.drop(columns="seed").agg(["mean", "std"]).transpose().to_csv(
        aggregate_dir / "test_mean_std.csv"
    )
    pd.DataFrame(per_class_test_rows).groupby("class")[["precision", "recall", "f1"]].agg(
        ["mean", "std"]
    ).to_csv(aggregate_dir / "per_class_test_mean_std.csv")
    print(test_df.drop(columns="seed").agg(["mean", "std"]).transpose())
else:
    print("Final Test remains locked. RUN_FINAL_TEST=False.")

# %% [markdown]
# ## 7. Locked external holdout results

# %%
# External evaluation is separate from the internal V05 Test driver above.
def make_real_holdout_dataset(holdout_manifest):
    dataset = tf.data.Dataset.from_tensor_slices((
        holdout_manifest["path"].tolist(),
        holdout_manifest["label"].to_numpy(dtype=np.int32),
    ))
    return dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(
        BATCH_SIZE, drop_remainder=False
    ).prefetch(tf.data.AUTOTUNE)  # No shuffle or augmentation; same input decoding.


def load_real_holdout_model(seed):
    return keras.models.load_model(holdout_checkpoints[seed], compile=False)


def clear_real_holdout_session():
    tf.keras.backend.clear_session()
    gc.collect()


if RUN_FINAL_TEST and RUN_REAL_HOLDOUT:
    holdout_checkpoints = {seed: out_dir / f"seed_{seed}" / "best_model.keras" for seed in SEEDS}
    real_holdout_results_df = evaluate_real_holdout(
        real_holdout_manifest, real_holdout_audit, CLASS_NAMES, SEEDS, validation_df,
        holdout_checkpoints, load_real_holdout_model, make_real_holdout_dataset,
        clear_real_holdout_session, out_dir / "real_holdout", outputs_are_logits=True,
    )
    print(real_holdout_results_df)
else:
    print("External holdout evaluation skipped. Enable RUN_FINAL_TEST after validation is frozen.")
