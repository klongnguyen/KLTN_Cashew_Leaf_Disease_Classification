"""Final V08 Test evaluation of the five Validation-locked EXP-014 checkpoints.

Inference only. The deployment seed remains the one frozen in EXP-014.
"""

import argparse
import gc
import hashlib
import io
import json
import os
import random
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from tensorflow import keras
from tensorflow.keras import layers


SOURCE_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-014"
EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-016"
SEEDS = [42, 123, 2026, 3407, 7777]
CLASSES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
DATASET_SHA256 = "75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7"
SOURCE_FULL_ZIP_SHA256 = "78643d522a56437f5c278c19564e226bd67a8304c46223187dfa511788f59336"
MANIFEST_SHA256 = "1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b"
IMG_SIZE = 224
BATCH_SIZE = 16


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def source_bytes(archive, relative):
    return archive.read(f"{SOURCE_ID}/{relative}")


def set_global_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def densenet_preprocessing(inputs):
    x = layers.Rescaling(1.0 / 255.0, name="densenet_rescale")(inputs)
    return layers.Normalization(mean=[0.485, 0.456, 0.406],
                                variance=[0.229**2, 0.224**2, 0.225**2],
                                axis=-1, name="densenet_preprocess")(x)


def build_transfer_model(seed):
    # Identical graph and layer names to the EXP-014 training notebook.
    set_global_seed(seed)
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="raw_rgb")
    augmented = keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.08, seed=seed),
        layers.RandomZoom(0.10, seed=seed),
        layers.RandomContrast(0.10, seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")(inputs)
    x = densenet_preprocessing(augmented)
    backbone = keras.applications.DenseNet121(
        include_top=False, weights=None, input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    backbone.trainable = False
    features = backbone(x, training=False)
    x = layers.Dropout(0.40, seed=seed)(features)
    outputs = layers.Dense(len(CLASSES), activation="softmax", dtype="float32",
                           name="disease_output")(x)
    return keras.Model(inputs, outputs, name=f"DenseNet121_Transfer_seed_{seed}"), backbone


def set_backbone_stage(backbone, stage):
    if stage == "stage1":
        backbone.trainable = False
    elif stage == "stage2":
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = (layer.name.startswith("conv5_") and
                               not isinstance(layer, layers.BatchNormalization))
    else:
        raise ValueError(f"Unknown selected stage: {stage}")


def decode_image(path, label):
    image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
    return tf.cast(image, tf.float32), label


def make_test_dataset(paths, labels):
    return (tf.data.Dataset.from_tensor_slices((paths, labels))
            .map(decode_image, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE))


def verify_source(archive):
    config_bytes = source_bytes(archive, "experiment_config.json")
    lock_bytes = source_bytes(archive, "validation_selection_lock.json")
    validation_bytes = source_bytes(archive, "aggregate/validation_results_5seeds.csv")
    config, lock = json.loads(config_bytes), json.loads(lock_bytes)
    status = json.loads(source_bytes(archive, "execution_status.json"))
    if config["experiment_id"] != SOURCE_ID or lock["experiment_id"] != SOURCE_ID:
        raise ValueError("EXP-014 identity mismatch")
    if status.get("five_seed_training") != "COMPLETED" or status.get("pipeline_smoke") != "PASS":
        raise ValueError("EXP-014 training/smoke status is incomplete")
    if (sha256_bytes(config_bytes) != lock["config_sha256"] or
            sha256_bytes(validation_bytes) != lock["validation_sha256"]):
        raise ValueError("EXP-014 config or Validation table differs from the lock")
    dataset = config["dataset"]
    if (dataset["name"] != "Cashew_dataV08_real_val" or
            dataset["archive"] != "CashewData_Split (8).zip" or
            dataset["archive_sha256"] != DATASET_SHA256 or
            dataset["manifest_sha256"] != MANIFEST_SHA256 or
            dataset["counts"] != {"train": 4790, "val": 1430, "test": 686, "total": 6906} or
            dataset["class_names"] != CLASSES or config["seeds"] != SEEDS or
            lock["dataset_sha256"] != DATASET_SHA256 or
            lock["dataset_manifest_sha256"] != MANIFEST_SHA256 or
            lock["class_names"] != CLASSES or lock["seeds"] != SEEDS):
        raise ValueError("Dataset, class order, or seeds differ from the Validation lock")
    model = config["model"]
    if (model["backbone"] != "DenseNet121" or model["weights"] != "imagenet" or
            model["include_top"] is not False or model["pooling"] != "avg" or
            model["head_dropout"] != 0.4 or
            model["preprocessing"] != "inside model: /255 then ImageNet mean/std" or
            model["backbone_call"] != "backbone(x, training=False)" or
            config["input"] != "raw RGB float32 0..255, bilinear resize 224x224" or
            config["batch_size"] != BATCH_SIZE):
        raise ValueError("Inference graph differs from EXP-014")
    val = pd.read_csv(io.BytesIO(validation_bytes))
    if val["seed"].tolist() != SEEDS:
        raise ValueError("EXP-014 Validation seed list is incomplete")
    ranked = val.sort_values(["val_loss", "val_macro_f1", "seed"], ascending=[True, False, True])
    if int(ranked.iloc[0]["seed"]) != lock["deployment_seed"]:
        raise ValueError("Validation deployment seed lock mismatch")
    manifest_bytes = source_bytes(archive, "dataset_manifest.csv")
    manifest = pd.read_csv(io.BytesIO(manifest_bytes))
    records = manifest[["relative_path", "label", "sha256"]].to_dict("records")
    manifest_digest = sha256_bytes(json.dumps(records, sort_keys=True, separators=(",", ":")).encode())
    if (manifest_digest != MANIFEST_SHA256 or len(manifest) != 6906 or
            manifest["split"].value_counts().to_dict() != {"train": 4790, "val": 1430, "test": 686}):
        raise ValueError("EXP-014 manifest does not match the frozen dataset")
    selected = {}
    for seed in SEEDS:
        selection = json.loads(source_bytes(archive, f"seed_{seed}/stage_selection.json"))
        if (selection["checkpoint_sha256"] != lock["selected_checkpoint_sha256"][str(seed)] or
                selection["selected_stage"] not in ("stage1", "stage2")):
            raise ValueError(f"Checkpoint selection mismatch for seed {seed}")
        selected[seed] = selection
    return config, lock, val, manifest, selected, sha256_bytes(lock_bytes)


def extract_test_images(dataset_archive, manifest, cache_dir):
    test = manifest.loc[manifest["split"] == "test"].copy().sort_values("relative_path")
    if len(test) != 686 or test["relative_path"].duplicated().any():
        raise ValueError("Expected 686 unique Test image paths")
    expected = {f"CashewData_Split/{relative}" for relative in test["relative_path"]}
    actual = {name for name in dataset_archive.namelist()
              if name.startswith("CashewData_Split/test/") and not name.endswith("/")}
    if actual != expected:
        raise ValueError(f"Dataset ZIP Test membership differs: missing={len(expected-actual)}, extra={len(actual-expected)}")
    paths = []
    for row in test.itertuples(index=False):
        relative = row.relative_path
        if Path(relative).is_absolute() or ".." in Path(relative).parts or not relative.startswith("test/"):
            raise ValueError(f"Unsafe Test path: {relative}")
        data = dataset_archive.read(f"CashewData_Split/{relative}")
        if sha256_bytes(data) != row.sha256:
            raise ValueError(f"Image differs from EXP-014 manifest: {relative}")
        target = cache_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        paths.append(str(target))
    test["local_path"] = paths
    if set(test["label"].astype(int)) != set(range(len(CLASSES))):
        raise ValueError("Test does not cover all five classes")
    if not test["class"].eq(test["label"].map(dict(enumerate(CLASSES)))).all():
        raise ValueError("Test class labels disagree with class order")
    return test


def write_seed_results(output_dir, seed, test, probabilities, lock, selected_stage):
    if probabilities.shape != (686, len(CLASSES)) or not np.isfinite(probabilities).all():
        raise ValueError(f"Invalid prediction shape/values for seed {seed}")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4):
        raise ValueError(f"Invalid softmax probabilities for seed {seed}")
    truth = test["label"].to_numpy(dtype=int)
    predicted = probabilities.argmax(axis=1)
    matrix = confusion_matrix(truth, predicted, labels=list(range(len(CLASSES))))
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=list(range(len(CLASSES))), zero_division=0)
    seed_dir = output_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({
        "relative_path": test["relative_path"].to_numpy(),
        "image_sha256": test["sha256"].to_numpy(),
        "true_label": [CLASSES[i] for i in truth],
        "predicted_label": [CLASSES[i] for i in predicted],
        "confidence": probabilities.max(axis=1),
        "correct": truth == predicted,
    })
    for i, name in enumerate(CLASSES):
        predictions[f"prob_{name}"] = probabilities[:, i]
    predictions.to_csv(seed_dir / "predictions.csv", index=False, float_format="%.10g")
    report = pd.DataFrame({"class": CLASSES, "precision": precision, "recall": recall,
                           "f1": f1, "support": support})
    report.to_csv(seed_dir / "classification_report.csv", index=False)
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).rename_axis("true\\predicted").to_csv(
        seed_dir / "confusion_matrix.csv")
    result = {"seed": seed, "selected_stage": selected_stage,
              "checkpoint_sha256": lock["selected_checkpoint_sha256"][str(seed)],
              "n_test": len(test), "correct": int((truth == predicted).sum()),
              "accuracy": float(accuracy_score(truth, predicted)),
              "macro_f1": float(f1_score(truth, predicted, labels=list(range(len(CLASSES))),
                                          average="macro", zero_division=0))}
    for name, value in zip(CLASSES, recall):
        result[f"recall_{name}"] = float(value)
    write_json(seed_dir / "metrics.json", result)
    return result, report, matrix


def default_paths():
    if Path("/content/drive/MyDrive/Cashew_Leaf_model_result").is_dir():
        root = Path("/content/drive/MyDrive/Cashew_Leaf_model_result")
        return (root / "training_results/current_dataset" / f"{SOURCE_ID}_FULL.zip",
                root / "Dataset/CashewData_Split (8).zip",
                root / "training_results/current_dataset" / EXPERIMENT_ID)
    repo = Path(__file__).resolve().parents[2] if "__file__" in globals() else Path.cwd()
    return (repo / "training_results/current_dataset" / f"{SOURCE_ID}_FULL.zip",
            repo.parent / "Dataset/CashewData_Split (8).zip",
            repo / "training_results/current_dataset" / EXPERIMENT_ID)


def assemble_zip_from_parts(parts_dir, cache_dir, archive_name, expected_sha256):
    manifest_path = parts_dir / "manifest_v2.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "exp016_zip_parts_v2":
        raise ValueError(f"Invalid EXP-016 parts manifest: {manifest_path}")
    entries = [entry for entry in manifest["files"] if entry.get("name") == archive_name]
    if len(entries) != 1 or entries[0].get("sha256") != expected_sha256:
        raise ValueError(f"Parts manifest does not identify the expected {archive_name}")
    entry = entries[0]
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / archive_name
    if target.is_file() and target.stat().st_size == entry["size"]:
        if sha256_file(target) == expected_sha256:
            print("Using previously verified assembled ZIP:", target, flush=True)
            return target
    temporary = cache_dir / f"{archive_name}.assembling"
    total = 0
    combined = hashlib.sha256()
    with temporary.open("wb") as destination:
        for number, part in enumerate(entry["parts"], start=1):
            name = f"{archive_name}.part{number:03d}"
            if part.get("name") != name or not 0 < int(part.get("size", 0)) <= 100 * 1024 * 1024:
                raise ValueError(f"Invalid part metadata for {archive_name}, part {number}")
            path = parts_dir / name
            if not path.is_file() or path.stat().st_size != part["size"]:
                raise FileNotFoundError(f"Missing or incomplete Drive part: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
                    combined.update(block)
                    destination.write(block)
                    total += len(block)
            if digest.hexdigest() != part["sha256"]:
                raise ValueError(f"SHA256 mismatch in Drive part: {path}")
            print(f"Verified {archive_name} part {number}/{len(entry['parts'])}", flush=True)
    if total != entry["size"] or combined.hexdigest() != expected_sha256:
        raise ValueError(f"Assembled {archive_name} differs from the original ZIP")
    temporary.replace(target)
    print("Assembled and verified:", target, flush=True)
    return target


def resolve_zip(override, expected_path, archive_name, expected_sha256, override_name,
                parts_dir=None):
    if override is not None:
        selected = Path(override).expanduser()
        if not selected.is_file():
            raise FileNotFoundError(f"{override_name} points to a missing file: {selected}")
        return selected
    if expected_path.is_file():
        return expected_path
    if parts_dir is not None:
        cache_dir = Path("/content/exp016_inputs") if Path("/content").is_dir() else Path(tempfile.gettempdir()) / "exp016_inputs"
        assembled = assemble_zip_from_parts(Path(parts_dir), cache_dir, archive_name,
                                            expected_sha256)
        if assembled is not None:
            return assembled
    drive_root = Path("/content/drive")
    matches = []
    if drive_root.is_dir():
        visited = 0
        for parent, dirs, files in os.walk(drive_root):
            visited += 1
            depth = len(Path(parent).relative_to(drive_root).parts)
            if depth >= 8:
                dirs.clear()
            if archive_name in files:
                matches.append(Path(parent) / archive_name)
            if visited >= 2000:
                break
    if len(matches) == 1:
        print(f"Found {archive_name} in Drive: {matches[0]}", flush=True)
        return matches[0]
    if matches:
        raise FileNotFoundError(
            f"Multiple {archive_name} files in Drive. Set {override_name} to one exact path: "
            + ", ".join(str(path) for path in matches))
    raise FileNotFoundError(
        f"Missing {archive_name}. Expected {expected_path}. Upload the ZIP to mounted Drive "
        f"or set {override_name} to its exact Colab path. "
        "The notebook cannot read a ZIP that is only on your Windows computer.")


def run(source_zip=None, dataset_zip=None, output_dir=None, parts_dir=None):
    default_source, default_dataset, default_output = default_paths()
    if parts_dir is None and Path("/content/drive/MyDrive/Cashew_Leaf_model_result").is_dir():
        parts_dir = Path("/content/drive/MyDrive/Cashew_Leaf_model_result/EXP-016_INPUT_PARTS")
    source_zip = resolve_zip(source_zip, default_source, f"{SOURCE_ID}_FULL.zip",
                             SOURCE_FULL_ZIP_SHA256, "SOURCE_ZIP_OVERRIDE", parts_dir)
    dataset_zip = resolve_zip(dataset_zip, default_dataset, "CashewData_Split (8).zip",
                              DATASET_SHA256, "DATASET_ZIP_OVERRIDE", parts_dir)
    output_dir = Path(output_dir or default_output)
    print("Source ZIP:", source_zip, flush=True)
    print("Dataset ZIP:", dataset_zip, flush=True)
    if sha256_file(dataset_zip) != DATASET_SHA256:
        raise ValueError("V08 dataset ZIP SHA256 does not match EXP-014")
    with zipfile.ZipFile(source_zip) as source_archive, zipfile.ZipFile(dataset_zip) as dataset_archive:
        config, lock, val, manifest, selected, lock_sha = verify_source(source_archive)
        if output_dir.exists() and any(output_dir.iterdir()):
            status_path = output_dir / "execution_status.json"
            config_path = output_dir / "evaluation_config.json"
            results_path = output_dir / "aggregate/test_results_5seeds.csv"
            if status_path.is_file() and config_path.is_file() and results_path.is_file():
                status = json.loads(status_path.read_text(encoding="utf-8"))
                previous = json.loads(config_path.read_text(encoding="utf-8"))
                existing = pd.read_csv(results_path)
                if (status.get("final_test") == "COMPLETED" and
                        status.get("deployment_seed") == lock["deployment_seed"] and
                        previous.get("source_validation_lock_sha256") == lock_sha and
                        previous.get("dataset_zip_sha256") == DATASET_SHA256 and
                        existing["seed"].tolist() == SEEDS and
                        all((output_dir / f"seed_{seed}/predictions.csv").is_file()
                            for seed in SEEDS)):
                    print("EXP-016 is already complete; returning saved results:", output_dir,
                          flush=True)
                    return existing
            raise FileExistsError(
                f"Output directory has incomplete or different results: {output_dir}. "
                "Set OUTPUT_DIR_OVERRIDE to a new empty folder; existing files were preserved.")
        output_dir.mkdir(parents=True, exist_ok=True)
        print("Validated EXP-014 lock and V08 archive; deployment seed:", lock["deployment_seed"], flush=True)
        with tempfile.TemporaryDirectory(prefix="exp016_") as temporary:
            cache_dir = Path(temporary)
            test = extract_test_images(dataset_archive, manifest, cache_dir)
            print("Verified 686 Test images against frozen manifest", flush=True)
            paths = test["local_path"].tolist()
            labels = test["label"].to_numpy(dtype=np.int32)
            results, reports, matrices = [], [], []
            for seed in SEEDS:
                checkpoint_bytes = source_bytes(source_archive, f"seed_{seed}/best.weights.h5")
                if sha256_bytes(checkpoint_bytes) != lock["selected_checkpoint_sha256"][str(seed)]:
                    raise ValueError(f"Checkpoint SHA256 mismatch for seed {seed}")
                checkpoint_path = cache_dir / f"seed_{seed}.weights.h5"
                checkpoint_path.write_bytes(checkpoint_bytes)
                keras.backend.clear_session()
                model, backbone = build_transfer_model(seed)
                set_backbone_stage(backbone, selected[seed]["selected_stage"])
                model.load_weights(checkpoint_path)
                if model.count_params() != 7042629:
                    raise ValueError(f"Unexpected parameter count for seed {seed}")
                data = make_test_dataset(paths, labels)
                probabilities = model.predict(data, verbose=0)
                result, report, matrix = write_seed_results(
                    output_dir, seed, test, probabilities, lock, selected[seed]["selected_stage"])
                results.append(result)
                reports.append(report.assign(seed=seed))
                matrices.append(matrix)
                print(f"Seed {seed}: {result['correct']}/686, Accuracy={result['accuracy']:.4%}, "
                      f"Macro F1={result['macro_f1']:.4%}", flush=True)
                del data, model, backbone, probabilities
                checkpoint_path.unlink()
                keras.backend.clear_session()
                gc.collect()
    aggregate = output_dir / "aggregate"
    aggregate.mkdir(exist_ok=True)
    result_df = pd.DataFrame(results)
    result_df.to_csv(aggregate / "test_results_5seeds.csv", index=False)
    pd.concat(reports, ignore_index=True)[["seed", "class", "precision", "recall", "f1", "support"]].to_csv(
        aggregate / "per_class_test_by_seed.csv", index=False)
    metric_columns = ["accuracy", "macro_f1"] + [f"recall_{name}" for name in CLASSES]
    summary = pd.DataFrame([{"metric": name, "mean": result_df[name].mean(),
                             "sample_sd": result_df[name].std(ddof=1)} for name in metric_columns])
    summary.to_csv(aggregate / "test_mean_std_summary.csv", index=False)
    pd.DataFrame(sum(matrices), index=CLASSES, columns=CLASSES).rename_axis("true\\predicted").to_csv(
        aggregate / "confusion_matrix_5seeds_sum.csv")
    evaluation_config = {
        "experiment_id": EXPERIMENT_ID, "source_experiment_id": SOURCE_ID,
        "evaluation_type": "inference_only_five_locked_checkpoints",
        "source_full_zip_sha256": sha256_file(source_zip),
        "source_config_sha256": lock["config_sha256"],
        "source_validation_lock_sha256": lock_sha,
        "dataset_zip_sha256": DATASET_SHA256, "dataset_manifest_sha256": MANIFEST_SHA256,
        "checkpoint_sha256": lock["selected_checkpoint_sha256"],
        "seeds": SEEDS, "deployment_seed": lock["deployment_seed"],
        "deployment_selection": "EXP-014 Validation lock; unchanged after Test",
        "test_images": 686, "class_order": CLASSES,
        "runtime_tensorflow": tf.__version__, "runtime_gpu": [str(x) for x in tf.config.list_physical_devices("GPU")],
        "run_full_training": False, "run_final_test": True,
    }
    write_json(output_dir / "evaluation_config.json", evaluation_config)
    write_json(output_dir / "execution_status.json", {
        "experiment_id": EXPERIMENT_ID, "source_training_run": SOURCE_ID,
        "training_in_016": "NOT_RUN_REUSED_014_CHECKPOINTS", "final_test": "COMPLETED",
        "test_images_per_seed": 686, "seeds_evaluated": SEEDS,
        "deployment_seed": lock["deployment_seed"], "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    })
    deployment = result_df.loc[result_df["seed"] == lock["deployment_seed"]].iloc[0]
    lines = [
        f"# {EXPERIMENT_ID} — Final Test V08", "",
        "Inference only on the five EXP-014 Validation-locked checkpoints. No retraining, "
        "checkpoint changes, or seed selection using Test results.", "",
        f"Dataset: 686 Test images from `CashewData_Split (8).zip` (SHA256 `{DATASET_SHA256}`). "
        f"The 686 individual image SHA256 values match the frozen EXP-014 manifest.", "",
        f"Deployment seed: **{lock['deployment_seed']}**, frozen by EXP-014 Validation loss before Test.", "",
        "| Seed | Correct | Accuracy | Macro F1 | " + " | ".join(f"Recall {name}" for name in CLASSES) + " |",
        "|---:|---:|---:|---:|" + "---:|" * len(CLASSES),
    ]
    for row in results:
        lines.append(f"| {row['seed']} | {row['correct']}/686 | {row['accuracy']:.2%} | "
                     f"{row['macro_f1']:.2%} | " + " | ".join(
                         f"{row[f'recall_{name}']:.2%}" for name in CLASSES) + " |")
    lines += ["", f"Five-seed Accuracy: **{summary.iloc[0]['mean']:.2%} ± {summary.iloc[0]['sample_sd']:.2%}** "
              "(sample SD across seeds).",
              f"Five-seed Macro F1: **{summary.iloc[1]['mean']:.2%} ± {summary.iloc[1]['sample_sd']:.2%}**.",
              f"Deployment seed {lock['deployment_seed']}: **{int(deployment['correct'])}/686**, "
              f"Accuracy **{deployment['accuracy']:.2%}**, Macro F1 **{deployment['macro_f1']:.2%}**.", "",
              "Confusion matrices use true class as rows and predicted class as columns. "
              "Each seed folder contains `predictions.csv`, `metrics.json`, "
              "`classification_report.csv`, and `confusion_matrix.csv`. The aggregate folder "
              "contains five-seed metrics, per-class recall, sample mean/SD, and a sum of five "
              "confusion matrices (3,430 predictions, not 3,430 distinct images).", "",
              f"Portable results archive: `{EXPERIMENT_ID}_ANALYSIS.zip` beside this folder.", "",
              "The Test set is used once for this final report; the results are descriptive "
              "for the frozen V08 split. Seed SD measures training-run variation, not "
              "uncertainty for future field photos.", ""]
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")
    analysis_zip = output_dir.parent / f"{EXPERIMENT_ID}_ANALYSIS.zip"
    temporary_zip = analysis_zip.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file():
                archive.write(path, arcname=f"{EXPERIMENT_ID}/{path.relative_to(output_dir).as_posix()}")
    with zipfile.ZipFile(temporary_zip) as archive:
        if (archive.testzip() is not None or
                f"{EXPERIMENT_ID}/README.md" not in archive.namelist()):
            raise RuntimeError("EXP-016 ANALYSIS ZIP integrity check failed")
    temporary_zip.replace(analysis_zip)
    print("Completed:", output_dir, flush=True)
    return result_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path)
    parser.add_argument("--dataset-zip", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    run(args.source_zip, args.dataset_zip, args.output_dir)
