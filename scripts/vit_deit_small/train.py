"""Five-seed, Validation-only DeiT-Small experiment EXP-010.

The Test members of V08 are never extracted or passed to a model here.
"""

import argparse
import hashlib
import json
import math
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import keras
import keras_hub
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset_audit import MANIFEST_SHA, ZIP_SHA, sha_file
from evaluate_validation import export_validation, score
from modeling import CLASSES, PRESET, build_model, set_stage1, set_stage2
from report import write_report


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def verify_inputs(archive_path, output):
    output = Path(output)
    config_path = output / "experiment_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    smoke = json.loads((output / "gate1_smoke.json").read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010":
        raise ValueError("Wrong experiment ID")
    if config["dataset"]["test_enabled"] or config["dataset"]["historical_holdout_enabled"]:
        raise ValueError("Test/Holdout must stay disabled")
    if config["seeds"] != [42, 123, 2026, 3407, 7777] or config["batch_size"] != 16:
        raise ValueError("Frozen seed/batch protocol changed")
    if audit["manifest_sha256"] != MANIFEST_SHA or smoke["status"] != "PASS":
        raise ValueError("Gate 0 or Gate 1 did not pass")
    if smoke["preset"] != PRESET or smoke["model_parameters"] != 21668357:
        raise ValueError("Pretrained model provenance changed")
    if smoke["preset_artifacts"]["task.weights.h5"]["sha256"] != config["model"]["pretrained_task_weights_sha256"]:
        raise ValueError("Pretrained task weights SHA256 changed")
    if sha_file(archive_path) != ZIP_SHA:
        raise ValueError("V08 ZIP checksum changed")
    manifest = pd.read_csv(output / "data_manifest.csv")
    payload = manifest[["split", "class", "relative_path", "sha256"]].to_csv(index=False, lineterminator="\n")
    if hashlib.sha256(payload.encode()).hexdigest().upper() != MANIFEST_SHA:
        raise ValueError("Manifest checksum changed")
    counts = manifest.groupby("split").size().to_dict()
    if counts != {"train": 4790, "val": 1430, "test": 686}:
        raise ValueError(f"Split counts changed: {counts}")
    if [CLASSES[int(x)] for x in manifest.label] != manifest["class"].tolist():
        raise ValueError("Class index order changed")
    return config, manifest, sha_file(config_path)


def extract_train_val(archive_path, root, manifest):
    root = Path(root)
    needed = manifest.loc[manifest.split.isin(["train", "val"])]
    with zipfile.ZipFile(archive_path) as archive:
        for row in needed.itertuples(index=False):
            target = root / row.relative_path
            if target.is_file() and sha_file(target) == row.sha256:
                continue
            member = "CashewData_Split/" + row.relative_path
            raw = archive.read(member)
            if hashlib.sha256(raw).hexdigest().upper() != row.sha256:
                raise ValueError(f"Image SHA256 mismatch: {row.relative_path}")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    if not all((root / path).is_file() for path in needed.relative_path):
        raise RuntimeError("Train/Val extraction incomplete")
    return root


def dataset(frame, root, batch_size, seed=None):
    paths = [str(root / path) for path in frame.relative_path]
    labels = frame.label.to_numpy(dtype=np.int32)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if seed is not None:
        ds = ds.shuffle(len(frame), seed=seed, reshuffle_each_iteration=True)

    def decode(path, label):
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        image = tf.image.resize(image, (224, 224), method="bilinear")
        return tf.cast(image, tf.float32), label

    ds = ds.map(decode, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    return ds.with_options(opts)


def classifier_of(model):
    matches = [layer for layer in model.layers if isinstance(layer, keras_hub.models.DeiTImageClassifier)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one DeiTImageClassifier in checkpoint, got {len(matches)}")
    return matches[0]


class ValidationMetrics(keras.callbacks.Callback):
    def __init__(self, val_ds, val_frame, seed, stage):
        super().__init__()
        self.val_ds, self.val_frame, self.seed, self.stage = val_ds, val_frame, seed, stage
        self.records = []

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        logits = self.model.predict(self.val_ds, verbose=0)
        metrics, _ = score(self.val_frame, logits, self.seed, self.stage)
        row = {"epoch": epoch + 1, "learning_rate": float(self.model.optimizer.learning_rate.numpy()),
               "val_macro_f1": metrics["macro_f1"],
               "val_disease_to_healthy_count": metrics["disease_to_healthy_count"],
               "val_accuracy_from_predictions": metrics["accuracy"]}
        row.update({"val_recall_" + x["class"]: x["recall"] for x in metrics["per_class"]})
        logs.update({key: value for key, value in row.items() if key != "epoch"})
        self.records.append({**{"epoch": epoch + 1}, **{k: float(v) for k, v in logs.items()
                                                        if isinstance(v, (int, float, np.floating))}})


class EpochWarmupCosine(keras.callbacks.Callback):
    def __init__(self, start_lr, minimum_lr, max_epochs, warmup_epochs):
        super().__init__()
        self.start_lr, self.minimum_lr = start_lr, minimum_lr
        self.max_epochs, self.warmup_epochs = max_epochs, warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.start_lr * (epoch + 1) / self.warmup_epochs
        else:
            fraction = (epoch - self.warmup_epochs) / max(1, self.max_epochs - self.warmup_epochs - 1)
            lr = self.minimum_lr + 0.5 * (self.start_lr - self.minimum_lr) * (1 + math.cos(math.pi * fraction))
        self.model.optimizer.learning_rate.assign(float(lr))


def run_stage(model, train_ds, val_ds, val_frame, seed, stage, config, seed_dir):
    stage_cfg = config[stage]
    inventory = set_stage1(model, classifier_of(model)) if stage == "stage1" else set_stage2(model, classifier_of(model))
    write_json(seed_dir / f"trainable_{stage}.json", inventory)
    optimizer = keras.optimizers.AdamW(learning_rate=stage_cfg["learning_rate"],
                                      weight_decay=stage_cfg["weight_decay"],
                                      clipnorm=stage_cfg.get("clipnorm"))
    model.compile(optimizer=optimizer,
                  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[keras.metrics.SparseCategoricalAccuracy(name="accuracy")])
    checkpoint = seed_dir / "checkpoints" / f"{stage}_best.keras"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    validation = ValidationMetrics(val_ds, val_frame, seed, stage)
    callbacks = []
    if stage == "stage2":
        callbacks.append(EpochWarmupCosine(stage_cfg["learning_rate"],
                         stage_cfg["minimum_learning_rate"], stage_cfg["max_epochs"], 3))
    callbacks += [validation,
                  keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", mode="min",
                                                  save_best_only=True, verbose=1),
                  keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                                                patience=stage_cfg["early_stopping_patience"]) ]
    start = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=stage_cfg["max_epochs"],
              callbacks=callbacks, verbose=2)
    elapsed = time.perf_counter() - start
    history = pd.DataFrame(validation.records)
    history.to_csv(seed_dir / f"history_{stage}.csv", index=False)
    best_epoch = int(history.loc[history.val_loss.idxmin(), "epoch"])
    best = keras.models.load_model(checkpoint, compile=False)
    start_inference = time.perf_counter()
    logits = best.predict(val_ds, verbose=0)
    infer_seconds = time.perf_counter() - start_inference
    metrics, _ = export_validation(val_frame, logits, seed, stage, seed_dir)
    metrics.update({"best_epoch": best_epoch, "best_val_loss": float(history.val_loss.min()),
                    "epochs_ran": len(history), "training_seconds": elapsed,
                    "validation_inference_seconds": infer_seconds,
                    "inference_ms_per_image": 1000 * infer_seconds / len(val_frame),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": sha_file(checkpoint),
                    "trainable_parameters": sum(x["parameters"] for x in inventory)})
    write_json(seed_dir / f"metrics_{stage}.json", metrics)
    del best
    return metrics, checkpoint


def guardrail(metrics, tiny):
    rule = {"macro_f1": metrics["macro_f1"] >= tiny["macro_f1"] - .01,
            "disease_to_healthy": metrics["disease_to_healthy_count"] <= tiny["disease_to_healthy_count"],
            "healthy_recall": metrics["healthy_recall"] >= tiny["healthy_recall"] - .02,
            "not_cashew_leaf_recall": metrics["not_cashew_leaf_recall"] >= tiny["not_cashew_leaf_recall"] - .02}
    tiny_classes = {x["class"]: x for x in tiny["per_class"]}
    small_classes = {x["class"]: x for x in metrics["per_class"]}
    for klass in ("anthracnose", "leaf_miner", "red_rust"):
        rule[klass + "_recall"] = small_classes[klass]["recall"] >= tiny_classes[klass]["recall"] - .02
    return {"passed": all(rule.values()), "checks": rule}


def aggregate(output, config, stage_metrics, config_sha):
    tiny_data = json.loads((output / "baseline_tiny_008_per_seed.json").read_text(encoding="utf-8"))
    tiny = {int(x["seed"]): x for x in tiny_data["metrics"] if x["classifier"] == "A"}
    if set(tiny) != set(config["seeds"]):
        raise ValueError("Tiny A baseline missing seeds")
    rows, per_class, choices = [], [], []
    for seed in config["seeds"]:
        options = []
        for stage in ("stage1", "stage2"):
            metric = stage_metrics[seed][stage]
            checks = guardrail(metric, tiny[seed])
            write_json(output / f"seed_{seed}" / f"guardrail_{stage}.json", checks)
            row = {k: v for k, v in metric.items() if isinstance(v, (str, int, float))}
            row["guardrail_passed"] = checks["passed"]
            rows.append(row)
            for item in metric["per_class"]:
                per_class.append({"seed": seed, "stage": stage, **item})
            options.append((stage, metric, checks))
        eligible = [x for x in options if x[2]["passed"]]
        pool = eligible if eligible else options
        chosen = min(pool, key=lambda x: (x[1]["best_val_loss"], -x[1]["macro_f1"],
                                          0 if x[0] == "stage1" else 1))
        choices.append({"seed": seed, "stage": chosen[0], "eligible": chosen[2]["passed"],
                        "checkpoint_sha256": chosen[1]["checkpoint_sha256"],
                        "best_val_loss": chosen[1]["best_val_loss"], "macro_f1": chosen[1]["macro_f1"],
                        "disease_to_healthy_count": chosen[1]["disease_to_healthy_count"]})
    aggregate_dir = output / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(aggregate_dir / "validation_5seeds.csv", index=False)
    pd.DataFrame(per_class).to_csv(aggregate_dir / "per_class_validation.csv", index=False)
    ablation = []
    for seed in config["seeds"]:
        first, second = stage_metrics[seed]["stage1"], stage_metrics[seed]["stage2"]
        choice = next(x for x in choices if x["seed"] == seed)
        ablation.append({"seed": seed, "stage1_macro_f1": first["macro_f1"],
                         "stage2_macro_f1": second["macro_f1"],
                         "stage2_minus_stage1_macro_f1": second["macro_f1"] - first["macro_f1"],
                         "stage1_accuracy": first["accuracy"], "stage2_accuracy": second["accuracy"],
                         "stage2_minus_stage1_accuracy": second["accuracy"] - first["accuracy"],
                         "stage1_disease_to_healthy": first["disease_to_healthy_count"],
                         "stage2_disease_to_healthy": second["disease_to_healthy_count"],
                         "selected_stage": choice["stage"], "selected_stage_eligible": choice["eligible"]})
    pd.DataFrame(ablation).to_csv(aggregate_dir / "stage1_vs_stage2.csv", index=False)
    numeric = ["accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1",
               "log_loss", "disease_to_healthy_count", "disease_to_healthy_rate", "healthy_recall",
               "healthy_precision", "not_cashew_leaf_recall", "anthracnose_to_leaf_miner",
               "leaf_miner_to_anthracnose", "training_seconds", "checkpoint_bytes"]
    stats = df.groupby("stage")[numeric].agg(["mean", "std"])
    stats.columns = ["_".join(pair) for pair in stats.columns]
    stats.to_csv(aggregate_dir / "validation_mean_std.csv")
    tiny_df = pd.read_csv(output / "baseline_tiny_validation_008.csv")
    tiny_df = tiny_df[tiny_df.classifier == "A"]
    compare = []
    for stage in ("stage1", "stage2"):
        ours = df[df.stage == stage]
        for name in ("accuracy", "macro_f1", "disease_to_healthy_count"):
            compare.append({"stage": stage, "metric": name,
                            "tiny_mean": float(tiny_df[name].mean()), "tiny_sample_sd": float(tiny_df[name].std(ddof=1)),
                            "small_mean": float(ours[name].mean()), "small_sample_sd": float(ours[name].std(ddof=1)),
                            "small_minus_tiny_mean": float(ours[name].mean() - tiny_df[name].mean())})
    pd.DataFrame(compare).to_csv(aggregate_dir / "baseline_tiny_vs_small.csv", index=False)
    df[["seed", "stage", "disease_to_healthy_count", "disease_to_healthy_rate"]].to_csv(
        aggregate_dir / "disease_to_healthy.csv", index=False)
    real = []
    for seed in config["seeds"]:
        for stage in ("stage1", "stage2"):
            part = pd.read_csv(output / f"seed_{seed}" / f"real_prefix_val_{stage}.csv")
            if len(part) != 35:
                raise ValueError(f"Expected 35 real_ Validation images, got {len(part)} for {seed}/{stage}")
            real.append(part)
    pd.concat(real, ignore_index=True).to_csv(aggregate_dir / "real_prefix_validation.csv", index=False)
    eligible = [x for x in choices if x["eligible"]]
    selected = min(eligible, key=lambda x: (x["best_val_loss"], -x["macro_f1"], x["seed"])) if eligible else None
    stage123 = next(x for x in choices if x["seed"] == 123)
    tiny_f1 = float(tiny_df.macro_f1.mean())
    selected_f1 = float(np.mean([stage_metrics[x["seed"]][x["stage"]]["macro_f1"] for x in choices]))
    promoted = bool(selected and selected_f1 > tiny_f1 and stage123["eligible"]
                    and stage123["disease_to_healthy_count"] < 18)
    lock = {"experiment_id": config["experiment_id"], "status": "PROMOTED" if promoted else "NOT PROMOTED",
            "selection_basis": "Validation V08 only", "selected": selected,
            "per_seed_stage_choices": choices, "selected_stage_mean_macro_f1": selected_f1,
            "tiny_A_mean_macro_f1": tiny_f1, "seed_123_D2H_below_18": stage123["disease_to_healthy_count"] < 18,
            "seed_123_guardrails_pass": stage123["eligible"], "config_sha256": config_sha,
            "locked_at_utc": datetime.now(timezone.utc).isoformat(),
            "test_used": False, "historical_holdout_used": False}
    write_json(output / "selection" / "validation_selection_lock.json", lock)
    return lock


def figures(output, seeds):
    fig_dir = output / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    for seed in seeds:
        for stage, col in (("stage1", 0), ("stage2", 1)):
            frame = pd.read_csv(output / f"seed_{seed}" / f"history_{stage}.csv")
            axes[0, col].plot(frame.epoch, frame.loss, label=f"{seed} train")
            axes[0, col].plot(frame.epoch, frame.val_loss, linestyle="--", label=f"{seed} val")
            axes[1, col].plot(frame.epoch, frame.accuracy, label=f"{seed} train")
            axes[1, col].plot(frame.epoch, frame.val_accuracy, linestyle="--", label=f"{seed} val")
            axes[2, col].plot(frame.epoch, frame.val_macro_f1, label=str(seed))
            axes[0, col].set_title(stage + " loss")
            axes[1, col].set_title(stage + " accuracy")
            axes[2, col].set_title(stage + " Validation Macro F1")
    for ax in axes.flat:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "accuracy_loss_curves_5seeds.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    for j, seed in enumerate(seeds):
        for i, stage in enumerate(("stage1", "stage2")):
            cm = pd.read_csv(output / f"seed_{seed}" / f"confusion_matrix_val_{stage}.csv", index_col=0).to_numpy()
            axes[i, j].imshow(cm, cmap="Blues")
            axes[i, j].set_title(f"{seed} {stage}")
            axes[i, j].set_xticks(range(5), CLASSES, rotation=90, fontsize=7)
            axes[i, j].set_yticks(range(5), CLASSES, fontsize=7)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrices_5seeds.png", dpi=160)
    plt.close(fig)


def run(archive_path, output):
    output = Path(output)
    config, manifest, config_sha = verify_inputs(archive_path, output)
    tf.config.experimental.enable_op_determinism()
    work_root = Path(tempfile.gettempdir()) / "exp010_train_val" / "CashewData_Split"
    write_json(output / "training_runtime.json", {
        "tensorflow": tf.__version__, "keras": keras.__version__, "keras_hub": keras_hub.__version__,
        "gpu": [str(x) for x in tf.config.list_physical_devices("GPU")],
        "op_determinism_requested": True, "tf_data_deterministic": True,
        "mixed_precision": keras.mixed_precision.global_policy().name,
        "config_sha256": config_sha, "dataset_zip_sha256": ZIP_SHA,
        "train_val_work_root": str(work_root),
        "source_git_commit": (output / "source_git_commit.txt").read_text().strip()
                             if (output / "source_git_commit.txt").is_file() else None})
    root = extract_train_val(archive_path, work_root, manifest)
    train_frame = manifest[manifest.split == "train"].reset_index(drop=True)
    val_frame = manifest[manifest.split == "val"].reset_index(drop=True)
    val_ds = dataset(val_frame, root, config["batch_size"])
    all_metrics = {}
    write_json(output / "gate_status.json", {"gate0": "PASS_WITH_BASELINE_INFERENCE_PARITY_NOT_REPRODUCED",
        "gate1": "PASS_CURRENT_RUNTIME", "gate2": "IN_PROGRESS", "completed_seeds": [],
        "gate3": "NOT_STARTED_REQUIRES_GATE2", "gate4": "NOT_STARTED_REQUIRES_GATE3",
        "test_inference_executed": False, "historical_holdout_inference_executed": False})
    for seed in config["seeds"]:
        seed_dir = output / f"seed_{seed}"
        if seed_dir.exists():
            complete = seed_dir / "seed_complete.json"
            if not complete.is_file():
                raise FileExistsError(f"Incomplete seed output; preserve it for review before retry: {seed_dir}")
            record = json.loads(complete.read_text(encoding="utf-8"))
            if record.get("seed") != seed or record.get("status") != "PASS":
                raise ValueError(f"Invalid seed completion record: {complete}")
            prior = {}
            for stage in ("stage1", "stage2"):
                metric = json.loads((seed_dir / f"metrics_{stage}.json").read_text(encoding="utf-8"))
                checkpoint = seed_dir / "checkpoints" / f"{stage}_best.keras"
                if sha_file(checkpoint) != metric["checkpoint_sha256"]:
                    raise ValueError(f"Completed seed checkpoint changed: {checkpoint}")
                prior[stage] = metric
            all_metrics[seed] = prior
            write_json(output / "gate_status.json", {"gate0": "PASS_WITH_BASELINE_INFERENCE_PARITY_NOT_REPRODUCED",
                "gate1": "PASS_CURRENT_RUNTIME", "gate2": "IN_PROGRESS",
                "completed_seeds": list(all_metrics), "gate3": "NOT_STARTED_REQUIRES_GATE2",
                "gate4": "NOT_STARTED_REQUIRES_GATE3", "test_inference_executed": False,
                "historical_holdout_inference_executed": False})
            continue
        seed_dir.mkdir()
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        train_ds = dataset(train_frame, root, config["batch_size"], seed=seed)
        model, classifier = build_model(seed)
        if model.count_params() != 21668357:
            raise RuntimeError("Model parameter count changed")
        one = run_stage(model, train_ds, val_ds, val_frame, seed, "stage1", config, seed_dir)
        del model, classifier
        keras.backend.clear_session()
        stage1_model = keras.models.load_model(one[1], compile=False)
        two = run_stage(stage1_model, train_ds, val_ds, val_frame, seed, "stage2", config, seed_dir)
        all_metrics[seed] = {"stage1": one[0], "stage2": two[0]}
        write_json(seed_dir / "seed_complete.json", {"seed": seed, "status": "PASS",
                   "stage1_checkpoint_sha256": one[0]["checkpoint_sha256"],
                   "stage2_checkpoint_sha256": two[0]["checkpoint_sha256"]})
        write_json(output / "gate_status.json", {"gate0": "PASS_WITH_BASELINE_INFERENCE_PARITY_NOT_REPRODUCED",
            "gate1": "PASS_CURRENT_RUNTIME", "gate2": "IN_PROGRESS",
            "completed_seeds": list(all_metrics), "gate3": "NOT_STARTED_REQUIRES_GATE2",
            "gate4": "NOT_STARTED_REQUIRES_GATE3", "test_inference_executed": False,
            "historical_holdout_inference_executed": False})
        del stage1_model
        keras.backend.clear_session()
    lock = aggregate(output, config, all_metrics, config_sha)
    figures(output, config["seeds"])
    write_report(output, root)
    if lock["selected"]:
        from inference_one_image import predict
        first_val = manifest[manifest.split == "val"].iloc[0]
        result = predict(root / first_val.relative_path, output / "deployment_package")
        expected = pd.read_csv(output / f"seed_{lock['selected']['seed']}" /
                               f"predictions_val_{lock['selected']['stage']}.csv").iloc[0]
        if result["predicted_label"] != int(expected.predicted_label) or abs(result["confidence"] - expected.confidence) > 1e-4:
            raise RuntimeError("Staged inference parity failed")
        write_json(output / "deployment_package" / "reload_inference_smoke.json", result)
    write_json(output / "gate_status.json", {"gate0": "PASS_WITH_BASELINE_INFERENCE_PARITY_NOT_REPRODUCED",
        "gate1": "PASS_CURRENT_RUNTIME", "gate2": "PASS", "completed_seeds": config["seeds"],
        "gate3": "PASS", "gate4": "PASS" if lock["selected"] else "PARTIAL_NO_ELIGIBLE_CHECKPOINT",
        "test_inference_executed": False, "historical_holdout_inference_executed": False})
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_zip")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_zip, args.output), indent=2))

