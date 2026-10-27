"""Build the standalone, evaluation-only EXP-019 notebook from EXP-018's model recipe."""

import base64
import json
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "densenet121_transfer_5seed_018.ipynb"
TARGET = HERE / "densenet121_transfer_5seed_019.ipynb"
source = json.loads(SOURCE.read_text(encoding="utf-8"))


def md(value):
    return {"cell_type": "markdown", "metadata": {}, "source": value.splitlines(keepends=True)}


def py(value):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": value.splitlines(keepends=True)}


cells = [md("""# EXP-DENSENET121-TRANSFER-5SEEDS-019 — Final Test + real holdout

**Evaluation only.** Load the five Validation-locked checkpoints from EXP-018.
There is no training call or optimizer update in this notebook. Results are
written to a separate EXP-019 directory. The internal Test has 686 images and
five true classes; the external holdout has 127 images and four true classes.
Both are scored once, with the same five checkpoints and no threshold tuning.

Run all cells in order on Colab. The configuration searches the project
shortcut in the active account's My Drive. If this runtime still has the old
account mounted, set `FORCE_REMOUNT_DRIVE=True`. If the shared project is
read-only, set `DRIVE_RESULTS_ROOT_OVERRIDE` to a writable folder. If EXP-018
is stored elsewhere, set `SOURCE_RESULTS_ROOT_OVERRIDE` to its parent
`current_dataset` folder; the notebook appends the EXP-018 directory name.
""")]

cells.append(py('''# 01. Imports and fixed evaluation protocol
import base64
import csv
import gc
import hashlib
import json
import os
import platform
import random
import shutil
import sys
import time
import zipfile
import zlib
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageOps
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score, roc_curve)
from tensorflow import keras
from tensorflow.keras import layers

SOURCE_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-018"
EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-019"
SEEDS = [42, 123, 2026, 3407, 7777]
CLASSES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
HOLDOUT_CLASSES = ["anthracnose", "healthy", "leaf_miner", "red_rust"]
SPLIT_ZIP = "CashewData_Split (9).zip"
HOLDOUT_ZIP = "RL_Cashew_holdout (2).zip"
SPLIT_SHA256 = "1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4"
HOLDOUT_SHA256 = "47a5722660e2c8dc7cc04c9ff720e1f889c2481ec99112b41bc0606958a7fdd7"
EXPECTED_TEST_COUNTS = {"anthracnose": 122, "healthy": 118, "leaf_miner": 132,
                        "red_rust": 158, "not_cashew_leaf": 156}
EXPECTED_HOLDOUT_COUNTS = {"anthracnose": 46, "healthy": 20,
                           "leaf_miner": 36, "red_rust": 25}
IMG_SIZE, BATCH_SIZE = 224, 16
NUM_CLASSES = len(CLASSES)
HEAD_DROPOUT = 0.40
WEIGHT_DECAY = 1e-4
FORCE_REMOUNT_DRIVE = False
DRIVE_PROJECT_DIR_OVERRIDE = None
SOURCE_RESULTS_ROOT_OVERRIDE = None
DRIVE_RESULTS_ROOT_OVERRIDE = None
DATASET_ZIP_OVERRIDE = None
HOLDOUT_ZIP_OVERRIDE = None

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")

print("TensorFlow", tf.__version__, "GPUs", tf.config.list_physical_devices("GPU"))
'''))

cells.append(py('''# 02. Mount the current account and locate source/output directories
try:
    from google.colab import drive
except ImportError:
    drive = None
if drive is not None:
    drive.mount("/content/drive", force_remount=FORCE_REMOUNT_DRIVE)
    drive_root = Path("/content/drive/MyDrive")
    if DRIVE_PROJECT_DIR_OVERRIDE is not None:
        project_dir = Path(DRIVE_PROJECT_DIR_OVERRIDE).expanduser()
    else:
        direct = drive_root / "Cashew_Leaf_model_result"
        candidates = [direct] if (direct / "Dataset" / SPLIT_ZIP).is_file() else []
        if not candidates:
            candidates = [p for p in drive_root.iterdir() if p.is_dir() and
                          (p / "Dataset" / SPLIT_ZIP).is_file()]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Found {len(candidates)} project shortcuts. Set DRIVE_PROJECT_DIR_OVERRIDE.")
        project_dir = candidates[0]
    default_results = project_dir / "training_results" / "current_dataset"
    source_root = (Path(SOURCE_RESULTS_ROOT_OVERRIDE).expanduser() if
                   SOURCE_RESULTS_ROOT_OVERRIDE else default_results)
    results_root = (Path(DRIVE_RESULTS_ROOT_OVERRIDE).expanduser() if
                    DRIVE_RESULTS_ROOT_OVERRIDE else default_results)
    split_archive = Path(DATASET_ZIP_OVERRIDE) if DATASET_ZIP_OVERRIDE else project_dir / "Dataset" / SPLIT_ZIP
    holdout_archive = Path(HOLDOUT_ZIP_OVERRIDE) if HOLDOUT_ZIP_OVERRIDE else project_dir / "Dataset" / HOLDOUT_ZIP
    cache_root = Path("/content/dataset_cache/exp019")
elif os.name == "nt":
    project_dir = Path(r"D:\\Study\\KhoaLuanTotNghiep\\Resource")
    default_results = project_dir / "training_results" / "current_dataset"
    source_root = Path(SOURCE_RESULTS_ROOT_OVERRIDE) if SOURCE_RESULTS_ROOT_OVERRIDE else default_results
    results_root = Path(DRIVE_RESULTS_ROOT_OVERRIDE) if DRIVE_RESULTS_ROOT_OVERRIDE else default_results
    dataset_dir = Path(r"D:\\Study\\KhoaLuanTotNghiep\\Dataset")
    split_archive = Path(DATASET_ZIP_OVERRIDE) if DATASET_ZIP_OVERRIDE else dataset_dir / SPLIT_ZIP
    holdout_archive = Path(HOLDOUT_ZIP_OVERRIDE) if HOLDOUT_ZIP_OVERRIDE else dataset_dir / HOLDOUT_ZIP
    cache_root = project_dir / "dataset_cache" / "exp019"
else:
    raise RuntimeError("EXP-019 supports Colab or Windows; set explicit paths for another runtime.")

source_dir = source_root / SOURCE_ID
out_dir = results_root / EXPERIMENT_ID
aggregate_dir = out_dir / "aggregate"
figures_dir = out_dir / "figures"
for directory in (out_dir, aggregate_dir, figures_dir):
    directory.mkdir(parents=True, exist_ok=True)
probe = out_dir / ".write_probe"
try:
    probe.write_text("ok", encoding="utf-8")
finally:
    probe.unlink(missing_ok=True)
if not source_dir.is_dir():
    raise FileNotFoundError(f"EXP-018 result directory not found: {source_dir}")
for path in (split_archive, holdout_archive):
    if not path.is_file():
        raise FileNotFoundError(path)
print("EXP-018:", source_dir)
print("EXP-019:", out_dir)
print("Split ZIP:", split_archive)
print("Holdout ZIP:", holdout_archive)
'''))

cells.append(md("""## Verify the EXP-018 lock and the two revised archives

Any SHA-256 or checkpoint mismatch stops the evaluation before inference.
"""))

cells.append(py('''# 03. Verify recipe, selection lock, five checkpoints and archive identities
config_path = source_dir / "experiment_config.json"
lock_path = source_dir / "validation_selection_lock.json"
config = json.loads(config_path.read_text(encoding="utf-8"))
lock = json.loads(lock_path.read_text(encoding="utf-8"))
if config.get("experiment_id") != SOURCE_ID or lock.get("experiment_id") != SOURCE_ID:
    raise ValueError("Source experiment identity mismatch")
if config.get("dataset", {}).get("archive_sha256", "").lower() != SPLIT_SHA256:
    raise ValueError("Source recipe uses a different dataset ZIP")
if lock.get("dataset_sha256", "").lower() != SPLIT_SHA256:
    raise ValueError("Validation lock uses a different dataset ZIP")
if lock.get("config_sha256", "").lower() != sha256_file(config_path):
    raise ValueError("Source config changed after Validation lock")
if config.get("dataset", {}).get("class_names") != CLASSES or lock.get("class_names") != CLASSES:
    raise ValueError("Class order changed")
if config.get("seeds") != SEEDS or lock.get("seeds") != SEEDS:
    raise ValueError("Five-seed protocol changed")
if lock.get("dataset_manifest_sha256") != config["dataset"]["manifest_sha256"]:
    raise ValueError("Source manifest fingerprint mismatch")
if sha256_file(split_archive) != SPLIT_SHA256 or sha256_file(holdout_archive) != HOLDOUT_SHA256:
    raise ValueError("Split 9 or Holdout 2 archive SHA-256 mismatch")
checkpoint_paths = {seed: source_dir / f"seed_{seed}" / "best.weights.h5" for seed in SEEDS}
checkpoint_hashes = {}
for seed, path in checkpoint_paths.items():
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual.lower() != lock["selected_checkpoint_sha256"][str(seed)].lower():
        raise ValueError(f"Checkpoint changed: seed {seed}")
    checkpoint_hashes[str(seed)] = actual
deployment_seed = int(lock["deployment_seed"])
if deployment_seed not in SEEDS:
    raise ValueError("Invalid deployment seed in source lock")
validation_csv = source_dir / "aggregate" / "validation_results_5seeds.csv"
if sha256_file(validation_csv) != lock["validation_sha256"]:
    raise ValueError("Validation results changed after lock")
validation_rows = pd.read_csv(validation_csv)
if validation_rows["seed"].tolist() != SEEDS:
    raise ValueError("Source Validation results are incomplete")
print("Source lock PASS; deployment seed:", deployment_seed)
'''))

cells.append(py('''# 04. Safe extraction and exact manifest verification
def extract_frozen_zip(archive_path, target_parent, root_name, archive_sha):
    target = target_parent / root_name
    marker = target_parent / ".archive_sha256"
    if target_parent.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != archive_sha:
            raise RuntimeError(f"Unverified extraction cache: {target_parent}")
    else:
        staging = target_parent.with_name(target_parent.name + "_staging")
        if staging.exists():
            raise RuntimeError(f"Partial extraction requires review: {staging}")
        staging.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                name = member.filename.replace("\\\\", "/")
                parts = Path(name).parts
                resolved = (staging / name).resolve()
                if not parts or parts[0] != root_name or staging.resolve() not in resolved.parents:
                    raise ValueError(f"Unsafe or unexpected ZIP member: {member.filename}")
            archive.extractall(staging)
        (staging / ".archive_sha256").write_text(archive_sha, encoding="utf-8")
        staging.rename(target_parent)
    if not target.is_dir():
        raise FileNotFoundError(target)
    return target

split_root = extract_frozen_zip(split_archive, cache_root / "split09", "CashewData_Split", SPLIT_SHA256)
holdout_root = extract_frozen_zip(holdout_archive, cache_root / "holdout02", "RL_Cashew_holdout", HOLDOUT_SHA256)
source_manifest_path = source_dir / "dataset_manifest.csv"
source_manifest = pd.read_csv(source_manifest_path, dtype={"sha256": str, "relative_path": str})
needed_columns = {"relative_path", "label", "sha256", "split", "class"}
if not needed_columns.issubset(source_manifest.columns) or len(source_manifest) != 7045:
    raise ValueError("EXP-018 source manifest is missing or incomplete")
manifest_records = [{"relative_path": str(row.relative_path), "label": int(row.label),
                     "sha256": str(row.sha256)}
                    for row in source_manifest.itertuples(index=False)]
manifest_digest = hashlib.sha256(json.dumps(
    manifest_records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if manifest_digest != lock["dataset_manifest_sha256"]:
    raise ValueError("EXP-018 source manifest content changed")
reference_hashes = set()
for row in source_manifest.itertuples(index=False):
    relative = Path(row.relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe source manifest path: {row.relative_path}")
    image = split_root / relative
    if not image.is_file() or sha256_file(image) != row.sha256:
        raise ValueError(f"Split 9 extracted image changed: {row.relative_path}")
    reference_hashes.add(row.sha256)
test_manifest = source_manifest[source_manifest["split"] == "test"].copy().reset_index(drop=True)
if test_manifest["class"].value_counts().to_dict() != EXPECTED_TEST_COUNTS:
    raise ValueError("Split 9 Test class counts changed")
test_manifest["image_path"] = test_manifest["relative_path"].map(lambda x: str(split_root / x))
val_manifest = source_manifest[source_manifest["split"] == "val"].head(2).copy()
val_manifest["image_path"] = val_manifest["relative_path"].map(lambda x: str(split_root / x))
if not test_manifest["image_path"].map(lambda x: Path(x).is_file()).all():
    raise FileNotFoundError("Some Test images are missing")

holdout_rows = []
for name in HOLDOUT_CLASSES:
    folder = holdout_root / name
    files = sorted(p for p in folder.iterdir() if p.is_file())
    if len(files) != EXPECTED_HOLDOUT_COUNTS[name]:
        raise ValueError(f"Holdout class count changed: {name}")
    for image in files:
        with Image.open(image) as opened:
            opened.verify()
        digest = sha256_file(image)
        if digest in reference_hashes:
            raise ValueError(f"Holdout image overlaps Split 9: {image}")
        holdout_rows.append({"relative_path": image.relative_to(holdout_root).as_posix(),
                             "image_path": str(image), "class": name,
                             "label": CLASSES.index(name), "sha256": digest})
holdout_manifest = pd.DataFrame(holdout_rows)
if len(holdout_manifest) != 127 or holdout_manifest["sha256"].duplicated().any():
    raise ValueError("Holdout 2 contains missing or duplicate images")
test_manifest.to_csv(out_dir / "test_dataset_manifest.csv", index=False)
holdout_manifest.to_csv(out_dir / "holdout_dataset_manifest.csv", index=False)
save_json(out_dir / "dataset_audit.json", {
    "split_zip_sha256": SPLIT_SHA256, "source_manifest_sha256": manifest_digest,
    "test_counts": EXPECTED_TEST_COUNTS, "holdout_zip_sha256": HOLDOUT_SHA256,
    "holdout_counts": EXPECTED_HOLDOUT_COUNTS, "holdout_exact_overlap": 0,
    "limitation": "Near duplicates and shared leaf/tree/session IDs need separate review.",
})
print("Dataset audit PASS: Test", len(test_manifest), "Holdout", len(holdout_manifest))
'''))

# Reuse the exact architecture source from EXP-018, without its smoke training block.
model_cell = "".join(source["cells"][14]["source"])
model_functions = model_cell.split("if RUN_PIPELINE_SMOKE:", 1)[0]
if "def build_transfer_model" not in model_functions:
    raise RuntimeError("EXP-018 model source changed")
seed_cell = "".join(source["cells"][5]["source"])
cells.append(md("""## Rebuild the EXP-018 model graph and load frozen weights

The model definition is copied from EXP-018. `weights=None` only avoids
redownloading ImageNet weights; the trained checkpoint is loaded immediately.
"""))
cells.append(py(seed_cell + "\n" + model_functions + '''
def load_locked_model(seed):
    keras.backend.clear_session()
    gc.collect()
    model, backbone = build_transfer_model(seed, weights=None)
    selection = json.loads((source_dir / f"seed_{seed}" / "stage_selection.json").read_text(encoding="utf-8"))
    stage = selection["selected_stage"]
    if stage not in {"stage1", "stage2"}:
        raise ValueError(f"Unexpected selected stage for seed {seed}: {stage}")
    if selection.get("checkpoint_sha256", "").lower() != checkpoint_hashes[str(seed)]:
        raise ValueError(f"Stage selection/checkpoint fingerprint mismatch: seed {seed}")
    set_backbone_stage(backbone, stage)
    model.load_weights(checkpoint_paths[seed])
    return model

def make_eval_dataset(frame):
    paths = frame["image_path"].astype(str).to_numpy()
    labels = frame["label"].to_numpy(dtype=np.int32)
    def decode(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
        return tf.cast(image, tf.float32), label
    return tf.data.Dataset.from_tensor_slices((paths, labels)).map(
        decode, num_parallel_calls=tf.data.AUTOTUNE).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Verify the selected seed model against the source .keras deployment package.
source_deployment = source_dir / "model_package" / "deployment_model.keras"
if not source_deployment.is_file():
    raise FileNotFoundError(source_deployment)
selected_model = load_locked_model(deployment_seed)
reference_model = keras.models.load_model(source_deployment, compile=False)
sample = next(iter(make_eval_dataset(val_manifest)))[0]
if not np.allclose(selected_model(sample, training=False).numpy(),
                   reference_model(sample, training=False).numpy(), atol=1e-5):
    raise RuntimeError("Rebuilt checkpoint disagrees with EXP-018 deployment model")
del selected_model, reference_model
keras.backend.clear_session(); gc.collect()
print("Checkpoint/model parity PASS")
'''))

cells.append(py('''# 06. Freeze evaluation inputs before any Test or holdout inference
evaluation_lock = {
    "experiment_id": EXPERIMENT_ID, "source_experiment_id": SOURCE_ID,
    "source_config_sha256": sha256_file(config_path),
    "source_validation_lock_sha256": sha256_file(lock_path),
    "source_validation_results_sha256": sha256_file(validation_csv),
    "source_manifest_sha256": manifest_digest,
    "split_zip_sha256": SPLIT_SHA256, "holdout_zip_sha256": HOLDOUT_SHA256,
    "selected_checkpoint_sha256": checkpoint_hashes,
    "seeds": SEEDS, "deployment_seed": deployment_seed,
    "class_order": CLASSES, "selection_source": "EXP-018 Validation only",
    "test_images": 686, "holdout_images": 127,
}
evaluation_lock_path = out_dir / "evaluation_lock.json"
if evaluation_lock_path.exists():
    if json.loads(evaluation_lock_path.read_text(encoding="utf-8")) != evaluation_lock:
        raise RuntimeError("EXP-019 evaluation lock changed; do not mix datasets or checkpoints")
else:
    save_json(evaluation_lock_path, evaluation_lock)
print("Evaluation lock PASS")
'''))

cells.append(md("""## Final Test and external holdout

The five checkpoints are fixed. Holdout macro metrics use only its four
supported true classes. Every prediction retains all five class probabilities.
"""))

cells.append(py('''# 07. Evaluate every frozen checkpoint once and export standard atomic outputs
PROB_COLUMNS = [f"prob_{name}" for name in CLASSES]

def write_matrix(matrix, output, normalized):
    suffix = "normalized" if normalized else "raw"
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(
        output / f"confusion_matrix_{suffix}.csv")
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else max(1, matrix.max()))
    ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=40, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            ax.text(c, r, f"{matrix[r,c]:.2f}" if normalized else str(int(matrix[r,c])),
                    ha="center", va="center", fontsize=9)
    ax.set(xlabel="Predicted", ylabel="True", title=f"{output.parent.name} / {output.name}")
    fig.tight_layout(); fig.savefig(output / f"confusion_matrix_{suffix}.png", dpi=180); plt.close(fig)

def write_calibration(predictions, output):
    confidence = predictions["confidence"].to_numpy(dtype=float)
    correct = predictions["correct"].to_numpy(dtype=bool)
    probs = predictions[PROB_COLUMNS].to_numpy(dtype=float)
    truth = predictions["true_class_idx"].to_numpy(dtype=int)
    rows = []
    edges = np.linspace(0, 1, 11)
    for i in range(10):
        mask = (confidence >= edges[i]) & (confidence < edges[i+1] if i < 9 else confidence <= 1)
        n = int(mask.sum())
        rows.append({"bin": i, "n": n, "lower": edges[i], "upper": edges[i+1],
                     "accuracy": float(correct[mask].mean()) if n else np.nan,
                     "mean_confidence": float(confidence[mask].mean()) if n else np.nan})
    bins = pd.DataFrame(rows)
    bins.to_csv(output / "calibration_bins.csv", index=False)
    ece = float(sum(r["n"] / len(predictions) * abs(r["accuracy"] - r["mean_confidence"])
                    for r in rows if r["n"]))
    brier = float(np.mean(np.sum((probs - np.eye(NUM_CLASSES)[truth]) ** 2, axis=1)))
    save_json(output / "calibration_metrics.json", {"ece_10_bins": ece,
              "multiclass_brier": brier, "n": len(predictions),
              "note": "Diagnostic only; no threshold or calibration fit."})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    used = bins[bins["n"] > 0]
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].plot(used["mean_confidence"], used["accuracy"], "o-")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Confidence", ylabel="Accuracy",
                title=f"Reliability; ECE={ece:.3f}")
    axes[1].hist(confidence[correct], bins=edges, alpha=.7, label="Correct")
    axes[1].hist(confidence[~correct], bins=edges, alpha=.7, label="Incorrect")
    axes[1].set(xlabel="Confidence", ylabel="Images", title="Confidence distribution")
    axes[1].legend(); fig.tight_layout()
    fig.savefig(output / "reliability_confidence.png", dpi=180); plt.close(fig)
    return ece, brier

def write_roc_pr(predictions, output):
    truth = predictions["true_class_idx"].to_numpy(dtype=int)
    probs = predictions[PROB_COLUMNS].to_numpy(dtype=float)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for index, name in enumerate(CLASSES):
        binary = (truth == index).astype(int)
        if binary.sum() in (0, len(binary)):
            rows.append({"class": name, "support": int(binary.sum()), "roc_auc": np.nan,
                         "average_precision": np.nan, "status": "UNDEFINED"})
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, index])
        precision, recall, _ = precision_recall_curve(binary, probs[:, index])
        auc_score = float(roc_auc_score(binary, probs[:, index]))
        ap = float(average_precision_score(binary, probs[:, index]))
        rows.append({"class": name, "support": int(binary.sum()),
                     "roc_auc": auc_score, "average_precision": ap, "status": "OK"})
        axes[0].plot(fpr, tpr, label=f"{name} {auc_score:.3f}")
        axes[1].plot(recall, precision, label=f"{name} {ap:.3f}")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC OVR")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR OVR")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output / "roc_pr_ovr.png", dpi=180); plt.close(fig)
    pd.DataFrame(rows).to_csv(output / "roc_pr_ovr_metrics.csv", index=False)

def score_split(split_name, frame, seed, model, output):
    dataset = make_eval_dataset(frame)
    _ = model.predict(dataset.take(1), verbose=0)
    started = time.perf_counter()
    probabilities = np.asarray(model.predict(dataset, verbose=0))
    elapsed = time.perf_counter() - started
    y_true = frame["label"].to_numpy(dtype=int)
    if probabilities.shape != (len(frame), NUM_CLASSES) or not np.isfinite(probabilities).all():
        raise RuntimeError("Prediction matrix has wrong shape or non-finite values")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise RuntimeError("Prediction probabilities do not sum to one")
    y_pred = probabilities.argmax(axis=1)
    supported = [CLASSES.index(x) for x in (CLASSES if split_name == "test" else HOLDOUT_CLASSES)]
    report = classification_report(y_true, y_pred, labels=np.arange(NUM_CLASSES),
                                   target_names=CLASSES, output_dict=True, zero_division=0)
    report_frame = pd.DataFrame(report).T
    if split_name == "real_holdout":
        report_frame.loc["not_cashew_leaf", ["precision", "recall", "f1-score"]] = np.nan
    report_frame.to_csv(output / "classification_report.csv")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    cm_norm = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES), normalize="true")
    write_matrix(cm, output, False)
    write_matrix(cm_norm, output, True)
    relative = frame["relative_path"].astype(str).to_numpy()
    predictions = pd.DataFrame({
        "experiment_id": EXPERIMENT_ID, "source_experiment_id": SOURCE_ID,
        "architecture": "DenseNet121", "training_mode": "ImageNet Transfer, two-stage",
        "dataset_version": ("Cashew_dataV09_more_real" if split_name == "test" else
                            "RL_Cashew_holdout_02"),
        "dataset_sha256": SPLIT_SHA256 if split_name == "test" else HOLDOUT_SHA256,
        "split": split_name, "seed": seed, "image_id": relative, "relative_path": relative,
        "true_label": [CLASSES[i] for i in y_true],
        "predicted_label": [CLASSES[i] for i in y_pred],
        "true_class_idx": y_true, "predicted_class_idx": y_pred,
        "correct": y_true == y_pred, "confidence": probabilities.max(axis=1),
    })
    for index, name in enumerate(CLASSES):
        predictions[f"prob_{name}"] = probabilities[:, index]
    predictions.to_csv(output / "predictions.csv", index=False)
    predictions.loc[~predictions["correct"]].to_csv(output / "error_cases.csv", index=False)
    ece, brier = write_calibration(predictions, output)
    write_roc_pr(predictions, output)
    disease = np.isin(y_true, [CLASSES.index(x) for x in ("anthracnose", "leaf_miner", "red_rust")])
    false_healthy = int(np.sum(disease & (y_pred == CLASSES.index("healthy"))))
    metrics = {
        "experiment_id": EXPERIMENT_ID, "source_experiment_id": SOURCE_ID,
        "split": split_name, "seed": seed, "n": len(frame),
        "macro_class_count": len(supported), "supported_true_classes": [CLASSES[i] for i in supported],
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, labels=supported,
                                                    average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, labels=supported,
                                              average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=supported,
                                      average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "loss": float(-np.log(np.clip(probabilities[np.arange(len(y_true)), y_true], 1e-7, 1)).mean()),
        "false_healthy_count": false_healthy, "disease_images": int(disease.sum()),
        "false_healthy_rate": float(false_healthy / disease.sum()),
        "ece_10_bins": ece, "multiclass_brier": brier,
        "prediction_seconds_including_input_pipeline": elapsed,
        "inference_ms_per_image_batch16": 1000 * elapsed / len(frame),
        "checkpoint_sha256": checkpoint_hashes[str(seed)],
    }
    save_json(output / "metrics.json", metrics)
    return metrics

all_metrics = {}
for split_name, frame in (("test", test_manifest), ("real_holdout", holdout_manifest)):
    rows = []
    for seed in SEEDS:
        output = out_dir / f"seed_{seed}" / split_name
        marker = output / "evaluation_completed.json"
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("evaluation_lock_sha256") != sha256_file(evaluation_lock_path):
                raise RuntimeError(f"Completed {split_name} seed {seed} has a different lock")
            metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
            if not (output / "predictions.csv").is_file():
                raise RuntimeError(f"Completed {split_name} seed {seed} is missing predictions")
            print("Reusing completed", split_name, seed)
        else:
            if output.exists() and any(output.iterdir()):
                raise RuntimeError(f"Partial evaluation requires inspection: {output}")
            output.mkdir(parents=True, exist_ok=True)
            model = load_locked_model(seed)
            metrics = score_split(split_name, frame, seed, model, output)
            del model
            keras.backend.clear_session(); gc.collect()
            save_json(marker, {"evaluation_lock_sha256": sha256_file(evaluation_lock_path),
                               "checkpoint_sha256": checkpoint_hashes[str(seed)],
                               "split": split_name, "seed": seed,
                               "completed_at_utc": datetime.now(timezone.utc).isoformat()})
        rows.append(metrics)
    if [row["seed"] for row in rows] != SEEDS:
        raise RuntimeError("Five-seed evaluation incomplete")
    all_metrics[split_name] = pd.DataFrame(rows)
    all_metrics[split_name].to_csv(aggregate_dir / f"{split_name}_results_5seeds.csv", index=False)
print("Final Test and holdout inference complete")
'''))

cells.append(py('''# 08. Aggregate summaries and visual panels
for split_name in ("test", "real_holdout"):
    results = all_metrics[split_name]
    metric_names = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                    "balanced_accuracy", "loss", "false_healthy_rate",
                    "ece_10_bins", "multiclass_brier", "inference_ms_per_image_batch16"]
    summary = pd.DataFrame([{"metric": name, "mean": float(results[name].mean()),
                             "sample_sd": float(results[name].std(ddof=1)),
                             "n_seeds": len(SEEDS), "images_per_seed": int(results.iloc[0]["n"])}
                            for name in metric_names])
    summary.to_csv(aggregate_dir / f"{split_name}_mean_std.csv", index=False)
    per_class = []
    for seed in SEEDS:
        report = pd.read_csv(out_dir / f"seed_{seed}" / split_name /
                             "classification_report.csv", index_col=0)
        for name in CLASSES:
            per_class.append({"split": split_name, "seed": seed, "class": name,
                              "support": int(report.loc[name, "support"]),
                              "precision": report.loc[name, "precision"],
                              "recall": report.loc[name, "recall"],
                              "f1": report.loc[name, "f1-score"]})
    per_class = pd.DataFrame(per_class)
    per_class.to_csv(aggregate_dir / f"per_class_{split_name}_by_seed.csv", index=False)
    per_class.groupby("class")[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(
        aggregate_dir / f"per_class_{split_name}_mean_std.csv")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    primary = summary[summary["metric"].isin(["accuracy", "macro_recall", "macro_f1", "balanced_accuracy"])]
    ax.bar(primary["metric"], primary["mean"], yerr=primary["sample_sd"], capsize=5)
    ax.set(ylim=(0, 1.05), ylabel="Score", title=f"{split_name}: mean ± sample SD across five seeds")
    fig.tight_layout(); fig.savefig(figures_dir / f"{split_name}_metrics_mean_seed_sd.png", dpi=180); plt.close(fig)
    heat = per_class.pivot(index="class", columns="seed", values="f1").reindex(CLASSES)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    image = ax.imshow(heat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(SEEDS)), SEEDS)
    ax.set_yticks(range(NUM_CLASSES), CLASSES)
    ax.set(xlabel="Seed", title=f"{split_name}: per-class F1")
    for r in range(NUM_CLASSES):
        for c in range(len(SEEDS)):
            value = heat.iloc[r, c]
            ax.text(c, r, f"{value:.2f}" if np.isfinite(value) else "N/A",
                    ha="center", va="center")
    fig.colorbar(image, ax=ax, label="F1")
    fig.tight_layout(); fig.savefig(figures_dir / f"{split_name}_per_class_f1_heatmap.png", dpi=180); plt.close(fig)
    for normalized in (False, True):
        suffix = "normalized" if normalized else "raw"
        fig, axes = plt.subplots(1, len(SEEDS), figsize=(27, 5.5))
        for ax, seed in zip(axes, SEEDS):
            matrix = pd.read_csv(out_dir / f"seed_{seed}" / split_name /
                                 f"confusion_matrix_{suffix}.csv", index_col=0).to_numpy()
            ax.imshow(matrix, cmap="Blues", vmin=0, vmax=1 if normalized else max(1, matrix.max()))
            ax.set_xticks(range(NUM_CLASSES), CLASSES, rotation=50, ha="right")
            ax.set_yticks(range(NUM_CLASSES), CLASSES)
            ax.set_title(f"seed {seed}")
            for r in range(NUM_CLASSES):
                for c in range(NUM_CLASSES):
                    ax.text(c, r, f"{matrix[r,c]:.2f}" if normalized else str(int(matrix[r,c])),
                            ha="center", va="center", fontsize=8)
        fig.tight_layout()
        fig.savefig(figures_dir / f"{split_name}_confusion_{suffix}_5seeds.png", dpi=180)
        plt.close(fig)

gap = pd.DataFrame({"seed": SEEDS,
                    "test_5class_macro_f1": all_metrics["test"]["macro_f1"].to_numpy(),
                    "holdout_4class_macro_f1": all_metrics["real_holdout"]["macro_f1"].to_numpy()})
gap.to_csv(aggregate_dir / "domain_gap_descriptive.csv", index=False)
ax = gap.set_index("seed").plot.bar(figsize=(9, 4.5), ylim=(0, 1), rot=0,
    title="Internal Test vs external holdout (different class support)")
ax.set_ylabel("Macro F1"); ax.figure.tight_layout()
ax.figure.savefig(figures_dir / "domain_gap_descriptive.png", dpi=180); plt.close(ax.figure)
print("Aggregate tables and figures complete")
'''))

archive_code = '''# 09. Report, manifest and archives; source checkpoints included in FULL ZIP
import base64
import zlib

readme = f"""# {EXPERIMENT_ID}

Evaluation-only continuation of {SOURCE_ID}. No training or selection on Test/holdout.
Source Validation lock: `{sha256_file(lock_path)}`. Deployment seed {deployment_seed}.
Split 9 SHA-256 `{SPLIT_SHA256}`; holdout 2 SHA-256 `{HOLDOUT_SHA256}`.
Test: 686 images, five true classes. Holdout: 127 images, four true classes.
The two macro metrics use different class support. `not_cashew_leaf` holdout
F1 is undefined because it has zero true support.

Per-seed outputs: `seed_*/test|real_holdout/` contains metrics, standard
predictions with five probabilities, errors, reports, confusion CSV/PNG,
calibration ECE/Brier and ROC/PR one-vs-rest. Aggregate summaries and panels
are in `aggregate/` and `figures/`. SD is sample SD across five seeds, not an
image-sampling confidence interval. Prediction timing is batch {BATCH_SIZE}
including data loading after one-batch warm-up, and should not be compared
across different devices as a deployment latency benchmark.

`FULL.zip` adds the five EXP-018 source checkpoints and deployment model;
`ANALYSIS.zip` contains reports, notebook and figures without model binaries.
The source EXP-018 result directory remains unmodified.
"""
(out_dir / "README.md").write_text(readme, encoding="utf-8")
save_json(out_dir / "execution_status.json", {
    "experiment_id": EXPERIMENT_ID, "source_experiment_id": SOURCE_ID,
    "training": "REUSED_EXP018", "final_test": "COMPLETE", "real_holdout": "COMPLETE",
    "source_validation_lock_sha256": sha256_file(lock_path),
    "evaluation_lock_sha256": sha256_file(evaluation_lock_path),
})
required = ["README.md", "execution_status.json", "evaluation_lock.json", "dataset_audit.json",
            "test_dataset_manifest.csv", "holdout_dataset_manifest.csv",
            "aggregate/test_results_5seeds.csv", "aggregate/test_mean_std.csv",
            "aggregate/real_holdout_results_5seeds.csv", "aggregate/real_holdout_mean_std.csv"]
for seed in SEEDS:
    for split_name in ("test", "real_holdout"):
        required += [f"seed_{seed}/{split_name}/{name}" for name in
                     ("metrics.json", "predictions.csv", "classification_report.csv",
                      "confusion_matrix_raw.csv", "confusion_matrix_raw.png",
                      "confusion_matrix_normalized.csv", "confusion_matrix_normalized.png",
                      "calibration_metrics.json", "roc_pr_ovr_metrics.csv",
                      "evaluation_completed.json")]
missing = [name for name in required if not (out_dir / name).is_file() or
           (out_dir / name).stat().st_size == 0]
save_json(out_dir / "audit_required_artifacts.json", {
    "experiment_id": EXPERIMENT_ID, "required_count": len(required),
    "missing": missing, "status": "PASS" if not missing else "FAIL"})
if missing:
    raise RuntimeError(f"Required evaluation artifacts missing: {missing}")

NOTEBOOK_SOURCE_ZLIB_B64 = ""
notebook_dir = out_dir / "notebook"
notebook_dir.mkdir(exist_ok=True)
notebook_copy = notebook_dir / "densenet121_transfer_5seed_019.ipynb"
source_book = json.loads(zlib.decompress(base64.b64decode(NOTEBOOK_SOURCE_ZLIB_B64)))
for cell in source_book["cells"]:
    if cell["cell_type"] == "code":
        cell["source"] = [line.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                                       f'NOTEBOOK_SOURCE_ZLIB_B64 = "{NOTEBOOK_SOURCE_ZLIB_B64}"', 1)
                          if line.startswith('NOTEBOOK_SOURCE_ZLIB_B64 = ""') else line
                          for line in cell["source"]]
notebook_copy.write_text(json.dumps(source_book, ensure_ascii=False, indent=1) + "\\n", encoding="utf-8")

save_json(out_dir / "analysis_manifest.json", {
    "archive_version": "v1", "experiment_id": EXPERIMENT_ID,
    "source_experiment_id": SOURCE_ID, "source_config_sha256": sha256_file(config_path),
    "source_validation_lock_sha256": sha256_file(lock_path),
    "split_zip_sha256": SPLIT_SHA256, "holdout_zip_sha256": HOLDOUT_SHA256,
    "class_order": CLASSES, "seeds": SEEDS, "checkpoint_sha256": checkpoint_hashes,
})
manifest = []
for path in sorted(out_dir.rglob("*")):
    if path.is_file() and path.name not in {"FILE_MANIFEST.csv", "CHECKSUMS.sha256"}:
        manifest.append({"path": path.relative_to(out_dir).as_posix(),
                         "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
pd.DataFrame(manifest).to_csv(out_dir / "FILE_MANIFEST.csv", index=False)
(out_dir / "CHECKSUMS.sha256").write_text(
    "\\n".join(f"{row['sha256']}  {row['path']}" for row in manifest) + "\\n", encoding="utf-8")

allowed_analysis = {".csv", ".json", ".md", ".png", ".txt", ".sha256", ".ipynb"}
for target, include_models in ((results_root / f"{EXPERIMENT_ID}_ANALYSIS.zip", False),
                               (results_root / f"{EXPERIMENT_ID}_FULL.zip", True)):
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out_dir.rglob("*")):
            if path.is_file() and (include_models or path.suffix.lower() in allowed_analysis):
                archive.write(path, arcname=(Path(EXPERIMENT_ID) /
                                               path.relative_to(out_dir)).as_posix())
        if include_models:
            for seed in SEEDS:
                archive.write(checkpoint_paths[seed], arcname=(Path(EXPERIMENT_ID) /
                    "source_checkpoints" / f"seed_{seed}" / "best.weights.h5").as_posix())
            archive.write(source_deployment, arcname=(Path(EXPERIMENT_ID) /
                "source_checkpoints" / "deployment_model.keras").as_posix())
            archive.write(config_path, arcname=(Path(EXPERIMENT_ID) /
                "source_checkpoints" / "experiment_config.json").as_posix())
            archive.write(lock_path, arcname=(Path(EXPERIMENT_ID) /
                "source_checkpoints" / "validation_selection_lock.json").as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if (archive.testzip() is not None or
                f"{EXPERIMENT_ID}/notebook/{notebook_copy.name}" not in archive.namelist()):
            raise RuntimeError(f"Archive verification failed: {temporary}")
    temporary.replace(target)
    print("Verified archive:", target, f"{target.stat().st_size / 2**20:.1f} MiB")
'''
cells.append(py(archive_code))
cells.append(md("""## Done

EXP-019 reads EXP-018 and leaves its checkpoints/results intact. It writes
complete Test and holdout artifacts plus FULL and ANALYSIS ZIPs. Check the
near-duplicate audit and class support before interpreting the domain gap.
"""))

book = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3",
          "language": "python", "name": "python3"},
          "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
archive_index = len(cells) - 2
embedded = base64.b64encode(zlib.compress(json.dumps(
    book, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
archive_source = "".join(book["cells"][archive_index]["source"])
book["cells"][archive_index]["source"] = archive_source.replace(
    'NOTEBOOK_SOURCE_ZLIB_B64 = ""', f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"', 1
).splitlines(keepends=True)
TARGET.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(cells)} cells, {TARGET.stat().st_size} bytes")
