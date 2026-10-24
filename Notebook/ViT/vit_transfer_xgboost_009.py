"""EXP-009: inference-only retrospective Test V08 and historical Holdout."""

import gc
import hashlib
import json
import platform
import shutil
import zipfile
from pathlib import Path

import joblib
import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from xgboost import XGBClassifier

EXPERIMENT = "EXP-VIT-TRANSFER-XGBOOST-009"
RUNNER_REVISION = "009.1-validation-replay"
SOURCE_008 = "EXP-VIT-TRANSFER-XGBOOST-008"
SOURCE_007 = "EXP-VIT-TRANSFER-5SEEDS-007"
SEEDS = (42, 123, 2026, 3407, 7777)
TYPES = ("A", "B", "C")
RUN_V08_TEST = True
RUN_HISTORICAL_HOLDOUT = True
TRAINING_MODE = "inference_only"
CLASSES = ("anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf")
DATASET_SHA = "75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7"
HOLDOUT_SHA = "BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165"
HOLDOUT_MANIFEST_SHA = "BD885CF682292FA6474C915114CEDD4C64B4B986CD90BF2AF47C0119E01CEFC5"
LOCK_008_SHA = "DE4697F76DFA5FD87E6F6C02E09A7935DFA15210473FD4654CD79A1BF450F305"
ARTIFACT_MANIFEST_008_SHA = "7E012D0F9A32CC1EAA7D2FDFDD2856775ADD5D0823C6F8EBBCBC98005E7A8B2B"
DRIVE_IDS = {
    "source_008": "1CdCJ5DIf6klkdcTlcpwytcjHIItyxT8c",
    "source_007": "1WgYA9ZWzwNvGiOfBGV9jWSx4mdER4-px",
    "dataset": "1vaaOwAi72byaxFb7w_ixd_Y2HboKn2-G",
    "holdout": "1PBry_y2bMsvd5_YdmEUy9erSFarS3O52",
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def find_file(name, override, roots):
    if override:
        path = Path(override)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for root in roots:
        path = root / name
        if path.is_file():
            return path
    return None


def drive_download(file_id, target):
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    auth.authenticate_user()
    credentials, _ = google.auth.default()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    info = service.files().get(fileId=file_id, fields="id,name,size,capabilities(canDownload)",
                               supportsAllDrives=True).execute()
    require(info.get("capabilities", {}).get("canDownload") is not False, f"Drive download denied: {info}")
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == int(info["size"]):
        return target
    partial = target.with_name(target.name + ".partial")
    with partial.open("wb") as stream:
        request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
        downloader = MediaIoBaseDownload(stream, request, chunksize=16 * 1024 * 1024)
        done = False
        while not done:
            _, done = downloader.next_chunk(num_retries=3)
    require(partial.stat().st_size == int(info["size"]), "Incomplete Drive download")
    partial.replace(target)
    return target


def resolve_input(name, override, roots, key, cache):
    path = find_file(name, override, roots)
    if path is None:
        if not Path("/content").exists():
            raise FileNotFoundError(f"Set {key.upper()}_OVERRIDE to {name}")
        path = drive_download(DRIVE_IDS[key], cache / name)
    return path


def zip_members(path):
    with zipfile.ZipFile(path) as archive:
        require(archive.testzip() is None, f"ZIP CRC failed: {path}")
        names = archive.namelist()
        for name in names:
            rel = Path(name)
            require(not rel.is_absolute() and ".." not in rel.parts, f"Unsafe ZIP member: {name}")
    return names


def extract_selected(path, destination, predicate):
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for member in archive.infolist():
            rel = Path(member.filename)
            require(not rel.is_absolute() and ".." not in rel.parts, f"Unsafe ZIP member: {member.filename}")
            if member.is_dir() or not predicate(member.filename):
                continue
            target = destination / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


def decode_resize(path):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    return tf.cast(tf.image.resize(image, (224, 224), method="bilinear"), tf.float32)


def make_extractor(checkpoint):
    model = keras.models.load_model(checkpoint, compile=False)
    heads = [layer for layer in model.layers if isinstance(layer, keras_hub.models.DeiTImageClassifier)]
    require(len(heads) == 1, "Expected exactly one saved DeiTImageClassifier")
    head = heads[0]
    augmentations = [layer for layer in model.layers if layer.name == "augmentation"]
    require(len(augmentations) == 1, "Saved EXP-007 augmentation layer missing")
    prehead = keras.Model(head.input, head.output_dense.input, name="verified_prehead")
    x = augmentations[0](model.input, training=False)
    x = head.preprocessor(x)
    extractor = keras.Model(model.input, prehead(x, training=False), name="exp007_prehead_extractor")
    model.trainable = False
    extractor.trainable = False
    require(int(extractor.output_shape[-1]) == 192, "EXP-008 pre-head feature dimension changed")
    return model, head, extractor


def load_classifier(seed, typ, source_008):
    path = source_008 / f"seed_{seed}" / ("logistic_regression.joblib" if typ == "B" else "xgboost.json")
    if typ == "B":
        classifier = joblib.load(path)
    elif typ == "C":
        selected = read_json(source_008 / f"seed_{seed}/C_cv_selection.json")["chosen"]
        xgb_keys = ("n_estimators", "max_depth", "learning_rate", "subsample",
                    "colsample_bytree", "min_child_weight", "reg_lambda")
        classifier = XGBClassifier(objective="multi:softprob", num_class=5, tree_method="hist",
                                   eval_metric="mlogloss", random_state=seed, n_jobs=4,
                                   **{key: selected[key] for key in xgb_keys})
        classifier.load_model(path)
    else:
        raise ValueError(f"No separate saved classifier for {typ}")
    require(np.array_equal(classifier.classes_, np.arange(5)), f"Seed {seed}/{typ}: class order changed")
    return classifier


def probabilities(seed, frame, root, checkpoint, source_008, return_features=False):
    model, head, extractor = make_extractor(checkpoint)
    # Validate the saved feature endpoint before touching either evaluation split.
    probe = tf.stack([decode_resize(str(root / p)) for p in frame.relative_path.iloc[:2]])
    features = extractor(probe, training=False).numpy()
    logits = model(probe, training=False).numpy()
    rebuilt = head.output_dense(features).numpy()
    require(np.array_equal(logits.argmax(1), rebuilt.argmax(1)), f"Seed {seed}: pre-head label parity failed")
    require(np.linalg.norm(logits - rebuilt) / max(np.linalg.norm(logits), 1.0) <= .001,
            f"Seed {seed}: pre-head logit parity failed")
    ds = tf.data.Dataset.from_tensor_slices([str(root / p) for p in frame.relative_path])
    ds = ds.map(decode_resize, num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)
    all_features, all_softmax = [], []
    for batch in ds:
        f = extractor(batch, training=False).numpy().astype("float32")
        p = tf.nn.softmax(head.output_dense(f), axis=1).numpy()
        all_features.append(f)
        all_softmax.append(p)
    features = np.concatenate(all_features)
    outputs = {"A": np.concatenate(all_softmax)}
    require(features.shape == (len(frame), 192), "Feature shape changed")
    for typ in ("B", "C"):
        classifier = load_classifier(seed, typ, source_008)
        outputs[typ] = classifier.predict_proba(features)
    del model, head, extractor
    gc.collect()
    keras.backend.clear_session()
    for typ, p in outputs.items():
        require(p.shape == (len(frame), 5) and np.isfinite(p).all(), f"Seed {seed}/{typ}: invalid probabilities")
        require(np.allclose(p.sum(1), 1, atol=1e-5), f"Seed {seed}/{typ}: probabilities do not sum to one")
    return (outputs, features) if return_features else outputs


def score(frame, p, seed, typ, split, target):
    y = frame.label.to_numpy(dtype=int)
    pred = p.argmax(1)
    cm = confusion_matrix(y, pred, labels=np.arange(5))
    require(cm.sum() == len(frame), "Confusion matrix count mismatch")
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=np.arange(5), zero_division=0)
    present = np.flatnonzero(cm.sum(1))
    metrics = {
        "split": split, "seed": seed, "classifier": typ, "n": len(frame),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1_present_classes": float(f1_score(y, pred, labels=present, average="macro", zero_division=0)),
        "macro_f1_all_5_classes": float(f1_score(y, pred, labels=np.arange(5), average="macro", zero_division=0)),
        "present_true_classes": [CLASSES[i] for i in present],
        "disease_to_healthy_count": int(np.sum(np.isin(y, [0, 2, 3]) & (pred == 1))),
        "confusion_matrix": cm.tolist(),
        "per_class": [{"class": CLASSES[i], "support": int(support[i]), "precision": float(precision[i]),
                       "recall": float(recall[i]), "f1": float(f1[i])} for i in range(5)],
    }
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / "metrics.json", metrics)
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(target / "confusion_matrix.csv")
    pd.DataFrame(metrics["per_class"]).to_csv(target / "per_class.csv", index=False)
    result = frame[["relative_path", "class", "label", "sha256"]].copy()
    result = result.rename(columns={"class": "true_class", "label": "true_label", "sha256": "image_sha256"})
    result["predicted_label"] = pred
    result["predicted_class"] = [CLASSES[i] for i in pred]
    for i, klass in enumerate(CLASSES):
        result[f"p_{klass}"] = p[:, i]
    result.to_csv(target / "predictions.csv", index=False)
    return metrics


def run(dataset_override=None, holdout_override=None, source_008_override=None,
        source_007_override=None, result_root_override=None):
    print(f"{EXPERIMENT} revision {RUNNER_REVISION}", flush=True)
    require(RUN_V08_TEST and RUN_HISTORICAL_HOLDOUT and TRAINING_MODE == "inference_only",
            "EXP-009 must evaluate both frozen splits without training")
    project = Path("/content/drive/MyDrive/Cashew_Leaf_model_result") if Path("/content").exists() else Path.cwd()
    if Path("/content").exists():
        from google.colab import drive
        drive.mount("/content/drive")
    roots = [project / "training_results/current_dataset", project / "Dataset", project,
             project.parent / "Dataset", Path.cwd(), Path("/content")]
    result_root = Path(result_root_override) if result_root_override else project / "training_results/current_dataset"
    out = result_root / EXPERIMENT
    if (out / "execution_status.json").is_file() and read_json(out / "execution_status.json").get("status") == "COMPLETED":
        raise FileExistsError(f"EXP-009 already completed at {out}; outputs are immutable")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "run_config.json", {"experiment_id": EXPERIMENT, "runner_revision": RUNNER_REVISION,
        "source_008": SOURCE_008, "source_007": SOURCE_007,
        "RUN_V08_TEST": RUN_V08_TEST, "RUN_HISTORICAL_HOLDOUT": RUN_HISTORICAL_HOLDOUT,
        "TRAINING_MODE": TRAINING_MODE, "seeds": SEEDS, "classifiers": TYPES,
        "deployment_seed": 123, "selected_classifier": "A"})
    cache = Path("/content/vit009_work") if Path("/content").exists() else out / "_work"
    cache.mkdir(parents=True, exist_ok=True)
    paths = {
        "dataset": resolve_input("CashewData_Split (8).zip", dataset_override, roots, "dataset", cache),
        "holdout": resolve_input("RL_Cashew_holdout.zip", holdout_override, roots, "holdout", cache),
        "source_008": resolve_input(f"{SOURCE_008}_FULL.zip", source_008_override, roots, "source_008", cache),
    }
    require(digest(paths["dataset"]) == DATASET_SHA, "V08 ZIP SHA256 mismatch")
    require(digest(paths["holdout"]) == HOLDOUT_SHA, "Historical Holdout ZIP SHA256 mismatch")
    for path in paths.values():
        zip_members(path)
    # EXP-008 archive stores the trained B/C classifiers, validation lock, and Test image manifest.
    extract_selected(paths["source_008"], cache, lambda name: name.endswith((
        "validation_selection_lock.json", "experiment_config.json", "dataset_manifest.csv",
        "artifact_sha256_manifest.csv", "deployment_manifest.json", "preprocess_contract.json",
        "feature_extractor_config.json", "logistic_regression.joblib", "xgboost.json",
        "C_cv_selection.json", "_val.npy", "_val_metadata.csv")) or
        "_validation_predictions.csv" in name)
    source_008 = cache / SOURCE_008
    lock_path = source_008 / "selection/validation_selection_lock.json"
    require(digest(lock_path) == LOCK_008_SHA, "EXP-008 Validation lock SHA256 mismatch")
    lock = read_json(lock_path)
    require(lock["experiment_id"] == SOURCE_008 and lock["deployment_seed"] == 123 and
            lock["selected_classifier"] == "A" and lock["threshold"] is None,
            "EXP-008 locked deployment changed")
    source_lock = read_json(source_008 / "source/exp007_validation_selection_lock.json")
    require(source_lock["deployment_seed"] == 123 and source_lock["checkpoint_sha256"] == lock["checkpoint_sha256"],
            "EXP-007 and EXP-008 checkpoint locks differ")
    require(lock["dataset_archive_sha256"].upper() == DATASET_SHA, "EXP-008 dataset lock mismatch")
    require(digest(source_008 / "experiment_config.json") == lock["config_sha256"].upper(),
            "EXP-008 config SHA256 mismatch")
    deployment = read_json(source_008 / "model_package/deployment_manifest.json")
    require(deployment["seed"] == 123 and deployment["model_type"] == "A" and
            deployment["source_checkpoint_sha256"].upper() == lock["checkpoint_sha256"]["123"].upper(),
            "EXP-008 deployment manifest differs from Validation lock")
    contract = read_json(source_008 / "model_package/preprocess_contract.json")
    require(contract == {"image": "RGB", "resize": "bilinear 224x224", "range": "float32 0..255",
                         "saved_model_preprocessor": "exactly once", "augmentation": "disabled during extraction"},
            "EXP-008 preprocessing contract changed")
    manifest = pd.read_csv(source_008 / "dataset_manifest.csv")
    require(len(manifest) == 6906 and not manifest.relative_path.duplicated().any(), "EXP-008 dataset manifest changed")
    manifest_csv = manifest[["split", "class", "relative_path", "sha256"]].to_csv(index=False, lineterminator="\n")
    require(hashlib.sha256(manifest_csv.encode()).hexdigest().upper() == lock["manifest_sha256"].upper(),
            "EXP-008 dataset manifest SHA256 mismatch")
    test = manifest.loc[manifest.split == "test", ["class", "label", "relative_path", "sha256"]].reset_index(drop=True)
    val_probe = manifest.loc[manifest.split == "val", ["class", "label", "relative_path", "sha256"]].head(16).reset_index(drop=True)
    require(len(test) == 686 and [int((test["class"] == c).sum()) for c in CLASSES] == [122, 118, 132, 158, 156],
            "Test V08 class counts changed")
    dataset_root = cache / "dataset" / "CashewData_Split"
    with zipfile.ZipFile(paths["dataset"]) as archive:
        for row in pd.concat([val_probe, test], ignore_index=True).itertuples(index=False):
            require(CLASSES[int(row.label)] == row[0], "Test label map changed")
            name = "CashewData_Split/" + row.relative_path
            raw = archive.read(name)
            require(hashlib.sha256(raw).hexdigest().upper() == row.sha256.upper(), f"Test image changed: {name}")
            target = dataset_root / row.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    # EXP-007 stores the immutable DeiT checkpoints and the historical Holdout image manifest.
    source_007 = Path(source_007_override) if source_007_override and Path(source_007_override).is_dir() else roots[0] / SOURCE_007
    if not all((source_007 / f"seed_{seed}/best_model.keras").is_file() for seed in SEEDS) or not (source_007 / "real_holdout/dataset_manifest.csv").is_file():
        source_zip = resolve_input(f"{SOURCE_007}_FULL.zip", source_007_override if source_007_override and Path(source_007_override).is_file() else None,
                                   roots, "source_007", cache)
        zip_members(source_zip)
        extract_selected(source_zip, cache, lambda name: name.endswith("best_model.keras") or
                         name.endswith("real_holdout/dataset_manifest.csv"))
        source_007 = cache / SOURCE_007
    checkpoints = {}
    for seed in SEEDS:
        checkpoint = source_007 / f"seed_{seed}/best_model.keras"
        require(checkpoint.is_file(), f"Missing EXP-007 checkpoint for seed {seed}")
        require(digest(checkpoint) == lock["checkpoint_sha256"][str(seed)].upper(),
                f"EXP-007 checkpoint SHA256 mismatch for seed {seed}")
        checkpoints[seed] = checkpoint
    require(digest(source_008 / "artifact_sha256_manifest.csv") == ARTIFACT_MANIFEST_008_SHA,
            "EXP-008 artifact manifest SHA256 mismatch")
    artifact_hashes = pd.read_csv(source_008 / "artifact_sha256_manifest.csv").set_index("relative_path")
    protected = ["dataset_manifest.csv", "experiment_config.json", "selection/validation_selection_lock.json",
                 "model_package/deployment_manifest.json", "model_package/preprocess_contract.json"]
    for seed in SEEDS:
        for name in ("logistic_regression.joblib", "xgboost.json"):
            protected.append(f"seed_{seed}/{name}")
        protected.append(f"seed_{seed}/C_cv_selection.json")
        protected.extend((f"embeddings/seed_{seed}_val.npy",
                          f"embeddings/seed_{seed}_val_metadata.csv"))
        for typ in TYPES:
            protected.append(f"seed_{seed}/{typ}_validation_predictions.csv")
    for rel in protected:
        require(digest(source_008 / rel) == artifact_hashes.loc[rel, "sha256"].upper(),
                f"EXP-008 artifact SHA256 mismatch: {rel}")
    require(digest(source_007 / "real_holdout/dataset_manifest.csv") == HOLDOUT_MANIFEST_SHA,
            "EXP-007 Historical Holdout manifest SHA256 mismatch")
    holdout_source = pd.read_csv(source_007 / "real_holdout/dataset_manifest.csv")
    holdout_source = holdout_source[["relative_path", "class", "label", "sha256"]]
    require(len(holdout_source) == 49, "Historical Holdout manifest count changed")
    holdout_root = cache / "holdout" / "RL_Cashew_holdout"
    with zipfile.ZipFile(paths["holdout"]) as archive:
        for row in holdout_source.itertuples(index=False):
            require(CLASSES[int(row.label)] == row[1], "Holdout label map changed")
            raw = archive.read("RL_Cashew_holdout/" + row.relative_path)
            require(hashlib.sha256(raw).hexdigest().upper() == row.sha256.upper(),
                    f"Historical Holdout image changed: {row.relative_path}")
            target = holdout_root / row.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    duplicate_test = set(test.sha256.str.upper()) & set(holdout_source.sha256.str.upper())
    require(len(duplicate_test) == 1, "Expected exactly one known Test/Holdout duplicate")
    require(not (set(manifest.loc[manifest.split.isin(["train", "val"]), "sha256"].str.upper()) &
                 set(holdout_source.sha256.str.upper())), "Historical Holdout overlaps Train/Val")
    holdout = holdout_source.loc[~holdout_source.sha256.str.upper().isin(duplicate_test)].reset_index(drop=True)
    require(len(holdout) == 48 and [int((holdout["class"] == c).sum()) for c in CLASSES] == [13, 5, 24, 6, 0],
            "Historical Holdout 48 class counts changed")
    holdout.to_csv(out / "historical_holdout_48_verified_manifest.csv", index=False)
    write_json(out / "source_provenance.json", {"source_008_lock_sha256": LOCK_008_SHA,
        "source_008_config_sha256": lock["config_sha256"], "dataset_zip_sha256": DATASET_SHA,
        "holdout_zip_sha256": HOLDOUT_SHA, "checkpoint_sha256": lock["checkpoint_sha256"],
        "selected_classifier": lock["selected_classifier"], "deployment_seed": lock["deployment_seed"],
        "excluded_holdout_test_sha256": sorted(duplicate_test), "source_paths": {k: str(v) for k, v in paths.items()},
        "tensorflow": tf.__version__, "keras": keras.__version__, "keras_hub": keras_hub.__version__,
        "xgboost": xgboost.__version__, "python": platform.python_version()})
    write_json(out / "execution_status.json", {"status": "RUNNING", "training": "NEVER", "test": "PENDING", "holdout": "PENDING"})
    summaries = []
    for seed in SEEDS:
        probe_outputs, fresh_features = probabilities(seed, val_probe, dataset_root, checkpoints[seed],
                                                       source_008, return_features=True)
        stored_features = np.load(source_008 / f"embeddings/seed_{seed}_val.npy", mmap_mode="r")[:16]
        stored_rows = pd.read_csv(source_008 / f"embeddings/seed_{seed}_val_metadata.csv").head(16)
        require(stored_rows.relative_path.tolist() == val_probe.relative_path.tolist(),
                f"Seed {seed}: EXP-008 cached feature row order changed")
        require(stored_features.shape == fresh_features.shape == (16, 192),
                f"Seed {seed}: EXP-008 cached feature shape changed")
        feature_relative_l2 = float(np.linalg.norm(fresh_features - stored_features) /
                                    max(np.linalg.norm(stored_features), 1.0))
        require(np.isfinite(feature_relative_l2) and feature_relative_l2 <= .01,
                f"Seed {seed}: pre-head features drift beyond 1% from EXP-008 cache: {feature_relative_l2:.6g}")
        replay_rows = []
        for typ in TYPES:
            reference = pd.read_csv(source_008 / f"seed_{seed}/{typ}_validation_predictions.csv").head(16)
            require(reference.relative_path.tolist() == val_probe.relative_path.tolist(),
                    f"Seed {seed}/{typ}: EXP-008 Validation probe row order changed")
            reference_p = reference[[f"p_{klass}" for klass in CLASSES]].to_numpy()
            observed_p = probe_outputs[typ]
            # First prove that the serialized EXP-008 classifier reproduces its own saved
            # Validation probabilities on the *saved* embeddings. Fresh DeiT embeddings
            # can differ slightly by runtime and XGBoost trees can cross a split boundary.
            if typ in ("B", "C"):
                saved_p = load_classifier(seed, typ, source_008).predict_proba(stored_features)
                require(np.array_equal(saved_p.argmax(1), reference.predicted_label.to_numpy()) and
                        np.max(np.abs(saved_p - reference_p)) <= 1e-4,
                        f"Seed {seed}/{typ}: saved classifier does not replay EXP-008 Validation cache")
            differences = np.max(np.abs(reference_p - observed_p), axis=1)
            for i, relative in enumerate(val_probe.relative_path):
                replay_rows.append({"seed": seed, "classifier": typ, "relative_path": relative,
                    "reference_label": int(reference.predicted_label.iloc[i]),
                    "fresh_label": int(observed_p[i].argmax()),
                    "max_probability_difference": float(differences[i]),
                    "feature_relative_l2_16_images": feature_relative_l2})
            mismatches = int(np.sum(reference.predicted_label.to_numpy() != observed_p.argmax(1)))
            if mismatches or float(differences.max()) >= .01:
                print(f"Validation replay drift seed={seed}/{typ}: {mismatches}/16 labels, "
                      f"max probability difference={differences.max():.5f}, "
                      f"feature relative L2={feature_relative_l2:.6g}. "
                      "Saved classifier/cache parity passed; inference continues.", flush=True)
        replay_dir = out / "validation_replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(replay_rows).to_csv(replay_dir / f"seed_{seed}.csv", index=False)
    # Finish all five source-replay audits before opening Test or Holdout for inference.
    for seed in SEEDS:
        for split, frame, root in (("v08_test_686", test, dataset_root), ("historical_holdout_48", holdout, holdout_root)):
            print(f"Inference: seed={seed}, split={split}, n={len(frame)}", flush=True)
            outputs = probabilities(seed, frame, root, checkpoints[seed], source_008)
            for typ in TYPES:
                metrics = score(frame, outputs[typ], seed, typ, split,
                                out / split / f"seed_{seed}" / typ)
                summaries.append({key: value for key, value in metrics.items()
                                  if key not in ("confusion_matrix", "per_class", "present_true_classes")})
    summary = pd.DataFrame(summaries)
    summary.to_csv(out / "summary_all_5seeds_ABC.csv", index=False)
    selected = summary[(summary.seed == 123) & (summary.classifier == "A")]
    selected.to_csv(out / "locked_deployment_seed_123_A.csv", index=False)
    write_json(out / "execution_status.json", {"status": "COMPLETED", "training": "NEVER",
        "test": "RETROSPECTIVE_COMPLETED", "holdout": "RETROSPECTIVE_COMPLETED",
        "deployment_seed": 123, "selected_classifier": "A", "evaluated_seed_count": 5,
        "evaluated_classifier_count_per_seed": 3, "test_count": 686, "holdout_count": 48,
        "fresh_blind_field_test": "NOT_RUN"})
    (out / "README.md").write_text(
        "# EXP-VIT-TRANSFER-XGBOOST-009\n\nInference only. EXP-008 Validation lock fixes deployment "
        "to seed 123 / A. A, B and C were evaluated for all five frozen seeds on historical V08 Test "
        "and 48 Holdout images. One exact Test/Holdout duplicate was excluded. Test and Holdout are "
        "previously exposed; these results are retrospective, not a new blind field test. "
        "`macro_f1_present_classes` uses five classes on Test and four present classes on Holdout; "
        "`macro_f1_all_5_classes` is also reported. No classifier or seed is chosen from these scores. "
        "Validation replay diagnostics compare fresh features with the frozen EXP-008 cache. "
        "Classifier replay on the frozen cache must pass; small runtime feature drift is recorded.\n",
        encoding="utf-8")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    for key in ("dataset", "holdout", "source_008", "source_007", "result_root"):
        parser.add_argument(f"--{key.replace('_', '-')}", dest=f"{key}_override")
    args = parser.parse_args()
    print(run(**vars(args)).to_string(index=False))
