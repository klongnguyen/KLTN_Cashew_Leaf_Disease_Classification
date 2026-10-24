"""EXP-011: separate, resumable DeiT-Small 30/30-epoch training.

Only Train and Validation members enter TensorFlow datasets. The V08 Test
members are read solely by the shared ZIP integrity audit, before this runner.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "vit_deit_small"
sys.path.insert(0, str(BASE))

import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

from dataset_audit import MANIFEST_SHA, ZIP_SHA, sha_file
from evaluate_validation import export_validation, score
from modeling import CLASSES, PRESET, build_model, set_stage1, set_stage2
from train import (EpochWarmupCosine, ValidationMetrics, aggregate, classifier_of,
                   dataset, extract_train_val)

EXPERIMENT_ID = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011"
SEEDS = [42, 123, 2026, 3407, 7777]
CONFIG_CHANGED = {
    "experiment_id", "source_experiment", "stage1.max_epochs",
    "stage1.early_stopping_patience", "stage2.max_epochs",
    "stage2.early_stopping_patience", "stage2.schedule",
}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(part, path)


def config_diff(a, b, prefix=""):
    keys = set(a) | set(b)
    changed = set()
    for key in keys:
        name = f"{prefix}.{key}" if prefix else key
        if key not in a or key not in b:
            changed.add(name)
        elif isinstance(a[key], dict) and isinstance(b[key], dict):
            changed |= config_diff(a[key], b[key], name)
        elif a[key] != b[key]:
            changed.add(name)
    return changed


def verify_protocol(output, source_root):
    output, source_root = Path(output), Path(source_root)
    config_file = output / "experiment_config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    old = json.loads((source_root / "training_results/current_dataset" /
                      "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010" /
                      "experiment_config.json").read_text(encoding="utf-8"))
    changed = config_diff(old, config)
    if changed != CONFIG_CHANGED:
        raise ValueError(f"EXP-010 to EXP-011 config diff changed: {sorted(changed)}")
    if config["experiment_id"] != EXPERIMENT_ID or config["seeds"] != SEEDS:
        raise ValueError("Wrong experiment ID or seed order")
    if config["batch_size"] != 16 or any(config[s]["max_epochs"] != 30 or
                                        config[s]["early_stopping_patience"] != 7
                                        for s in ("stage1", "stage2")):
        raise ValueError("Frozen EXP-011 epoch, patience or batch protocol changed")
    if config["dataset"]["test_enabled"] or config["dataset"]["historical_holdout_enabled"]:
        raise ValueError("Test/Holdout must remain disabled")
    if config["model"]["pretrained_preset"] != PRESET:
        raise ValueError("Pretrained source changed")
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    smoke = json.loads((output / "gate1_smoke.json").read_text(encoding="utf-8"))
    if (audit["experiment_id"] != EXPERIMENT_ID or
            audit["manifest_sha256"] != MANIFEST_SHA or smoke["status"] != "PASS"):
        raise ValueError("Gate 0 or Gate 1 did not pass")
    if smoke["preset_artifacts"]["task.weights.h5"]["sha256"] != config["model"]["pretrained_task_weights_sha256"]:
        raise ValueError("Pretrained task weights changed")
    manifest = pd.read_csv(output / "data_manifest.csv")
    payload = manifest[["split", "class", "relative_path", "sha256"]].to_csv(index=False, lineterminator="\n")
    if hashlib.sha256(payload.encode()).hexdigest().upper() != MANIFEST_SHA:
        raise ValueError("V08 manifest changed")
    if manifest.groupby("split").size().to_dict() != {"train": 4790, "val": 1430, "test": 686}:
        raise ValueError("V08 split counts changed")
    if [CLASSES[int(x)] for x in manifest.label] != manifest["class"].tolist():
        raise ValueError("V08 class order changed")
    return config, manifest, sha_file(config_file)


def load_inputs(archive, output, source_root):
    config, manifest, config_sha = verify_protocol(output, source_root)
    if sha_file(archive) != ZIP_SHA:
        raise ValueError("V08 ZIP SHA256 changed")
    if not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("EXP-011 requires a TensorFlow GPU")
    tf.config.experimental.enable_op_determinism()
    work_root = Path(tempfile.gettempdir()) / "exp011_train_val" / "CashewData_Split"
    root = extract_train_val(archive, work_root, manifest)
    return config, manifest, config_sha, root


def marker_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest().upper()


def validate_stage(seed_dir, stage, config_sha):
    marker_file = seed_dir / f"{stage}_complete.json"
    if not marker_file.is_file():
        return False
    marker = json.loads(marker_file.read_text(encoding="utf-8"))
    checkpoint = seed_dir / "checkpoints" / f"{stage}_best.keras"
    metrics_file = seed_dir / f"metrics_{stage}.json"
    if marker["config_sha256"] != config_sha:
        raise ValueError(f"Resume config mismatch: {marker_file}")
    if marker["checkpoint_sha256"] != sha_file(checkpoint):
        raise ValueError(f"Resume checkpoint SHA mismatch: {checkpoint}")
    if marker["metrics_sha256"] != marker_sha(metrics_file):
        raise ValueError(f"Resume metrics SHA mismatch: {metrics_file}")
    metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    if metrics["checkpoint_sha256"] != marker["checkpoint_sha256"]:
        raise ValueError(f"Metrics/checkpoint mismatch: {metrics_file}")
    return True


def backup_partial(path, output):
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = output / "_incomplete_backups" / f"{path.name}_{stamp}"
    destination.parent.mkdir(exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    shutil.move(str(path), str(destination))
    print(f"Preserved partial output: {destination}", flush=True)
    return destination


def backup_stage2(seed_dir, output):
    names = ["trainable_stage2.json", "history_stage2.csv", "metrics_stage2.json",
             "predictions_val_stage2.csv", "errors_val_stage2.csv",
             "confusion_matrix_val_stage2.csv", "real_prefix_val_stage2.csv"]
    paths = [seed_dir / name for name in names]
    paths.append(seed_dir / "checkpoints" / "stage2_best.keras")
    paths.append(seed_dir / "checkpoints" / "stage2_best.part.keras")
    present = [p for p in paths if p.exists()]
    if present:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = output / "_incomplete_backups" / f"{seed_dir.name}_stage2_{stamp}"
        dest.mkdir(parents=True, exist_ok=False)
        for path in present:
            shutil.move(str(path), str(dest / path.name))
        write_json(dest / "restart.json", {"reason": "Stage 2 incomplete",
                   "method": "restart Stage 2 from same-seed Stage 1 best; not exact epoch resume",
                   "stage1_checkpoint_sha256": sha_file(seed_dir / "checkpoints/stage1_best.keras")})
        print(f"Preserved partial Stage 2: {dest}", flush=True)


class AtomicBestCheckpoint(keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = Path(path)
        self.best = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        loss = float((logs or {})["val_loss"])
        if loss < self.best:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            part = self.path.with_name(self.path.stem + ".part.keras")
            self.model.save(part)
            sha_file(part)
            os.replace(part, self.path)
            self.best = loss
            print(f"  best val_loss={loss:.5f}; checkpoint={self.path}", flush=True)


class EpochProgress(keras.callbacks.Callback):
    def __init__(self, seed, stage, total):
        super().__init__()
        self.seed, self.stage, self.total = seed, stage.upper(), total

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        lr = float(self.model.optimizer.learning_rate.numpy())
        print(f"SEED {self.seed} — {self.stage}, epoch {epoch + 1}/{self.total} "
              f"val_loss={logs['val_loss']:.5f} val_macro_f1={logs['val_macro_f1']:.5f} "
              f"D2H={int(logs['val_disease_to_healthy_count'])} LR={lr:.3g}", flush=True)


def fit_stage(model, train_ds, val_ds, val_frame, seed, stage, config, seed_dir, config_sha):
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
    validation = ValidationMetrics(val_ds, val_frame, seed, stage)
    callbacks = []
    if stage == "stage2":
        callbacks.append(EpochWarmupCosine(stage_cfg["learning_rate"],
                         stage_cfg["minimum_learning_rate"], stage_cfg["max_epochs"], 3))
    callbacks.extend([validation, AtomicBestCheckpoint(checkpoint),
                      EpochProgress(seed, stage, stage_cfg["max_epochs"]),
                      keras.callbacks.EarlyStopping(monitor="val_loss", mode="min",
                          patience=stage_cfg["early_stopping_patience"], min_delta=0,
                          restore_best_weights=False)])
    print(f"SEED {seed} — {stage.upper()} start; max_epochs={stage_cfg['max_epochs']}; "
          f"patience={stage_cfg['early_stopping_patience']}", flush=True)
    started = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=stage_cfg["max_epochs"],
              callbacks=callbacks, verbose=0)
    elapsed = time.perf_counter() - started
    history = pd.DataFrame(validation.records)
    history.to_csv(seed_dir / f"history_{stage}.csv", index=False)
    best_epoch = int(history.loc[history.val_loss.idxmin(), "epoch"])
    best = keras.models.load_model(checkpoint, compile=False, safe_mode=True)
    started = time.perf_counter()
    logits = best.predict(val_ds, verbose=0)
    infer_seconds = time.perf_counter() - started
    metrics, _ = export_validation(val_frame, logits, seed, stage, seed_dir)
    metrics.update({"best_epoch": best_epoch, "best_val_loss": float(history.val_loss.min()),
                    "epochs_ran": len(history), "training_seconds": elapsed,
                    "validation_inference_seconds": infer_seconds,
                    "inference_ms_per_image": 1000 * infer_seconds / len(val_frame),
                    "checkpoint_bytes": checkpoint.stat().st_size,
                    "checkpoint_sha256": sha_file(checkpoint),
                    "trainable_parameters": sum(x["parameters"] for x in inventory)})
    metrics_file = seed_dir / f"metrics_{stage}.json"
    write_json(metrics_file, metrics)
    write_json(seed_dir / f"{stage}_complete.json", {
        "seed": seed, "stage": stage, "config_sha256": config_sha,
        "checkpoint_sha256": metrics["checkpoint_sha256"],
        "metrics_sha256": marker_sha(metrics_file),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    print(f"SEED {seed} — {stage.upper()} complete; best_epoch={best_epoch}; "
          f"macro_f1={metrics['macro_f1']:.5f}; checkpoint_sha256={metrics['checkpoint_sha256']}", flush=True)
    del best
    return metrics


def gate_status(output, completed):
    write_json(output / "gate_status.json", {
        "gate0": "PASS_WITH_PHASH_CANDIDATES_PENDING_GROUP_REVIEW",
        "gate1": "PASS_CURRENT_RUNTIME", "gate2": "PASS" if len(completed) == 5 else "IN_PROGRESS",
        "completed_seeds": completed, "gate3": "NOT_STARTED_REQUIRES_GATE2",
        "gate4": "NOT_STARTED_REQUIRES_GATE3",
        "test_inference_executed": False,
        "historical_holdout_inference_executed": False})


def train_seed(archive, output, source_root, seed):
    output = Path(output)
    config, manifest, config_sha, root = load_inputs(archive, output, source_root)
    if seed not in SEEDS:
        raise ValueError(f"Seed outside frozen protocol: {seed}")
    val_frame = manifest[manifest.split == "val"].reset_index(drop=True)
    train_frame = manifest[manifest.split == "train"].reset_index(drop=True)
    seed_dir = output / f"seed_{seed}"
    if seed_dir.exists() and (seed_dir / "seed_complete.json").is_file():
        if not (validate_stage(seed_dir, "stage1", config_sha) and
                validate_stage(seed_dir, "stage2", config_sha)):
            raise ValueError("Seed completion marker conflicts with stage markers")
        complete = json.loads((seed_dir / "seed_complete.json").read_text(encoding="utf-8"))
        if complete["config_sha256"] != config_sha:
            raise ValueError("Seed completion config mismatch")
        print(f"SEED {seed} complete and SHA256-verified; reused", flush=True)
        return
    if seed_dir.exists() and not (seed_dir / "stage1_complete.json").is_file():
        backup_partial(seed_dir, output)
    seed_dir.mkdir(exist_ok=True)
    stage1_done = validate_stage(seed_dir, "stage1", config_sha)
    if stage1_done:
        print(f"SEED {seed} Stage 1 SHA256-verified; reused", flush=True)
    if not stage1_done:
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        train_ds = dataset(train_frame, root, config["batch_size"], seed=seed)
        val_ds = dataset(val_frame, root, config["batch_size"])
        model, classifier = build_model(seed)
        if model.count_params() != 21668357:
            raise RuntimeError("Model parameter count changed")
        fit_stage(model, train_ds, val_ds, val_frame, seed, "stage1", config, seed_dir, config_sha)
        del model, classifier
        keras.backend.clear_session()
    if not validate_stage(seed_dir, "stage2", config_sha):
        backup_stage2(seed_dir, output)
        keras.backend.clear_session()
        keras.utils.set_random_seed(seed)
        train_ds = dataset(train_frame, root, config["batch_size"], seed=seed)
        val_ds = dataset(val_frame, root, config["batch_size"])
        stage1 = keras.models.load_model(seed_dir / "checkpoints/stage1_best.keras",
                                        compile=False, safe_mode=True)
        fit_stage(stage1, train_ds, val_ds, val_frame, seed, "stage2", config, seed_dir, config_sha)
        del stage1
        keras.backend.clear_session()
    else:
        print(f"SEED {seed} Stage 2 SHA256-verified; reused", flush=True)
    write_json(seed_dir / "seed_complete.json", {
        "seed": seed, "status": "PASS", "config_sha256": config_sha,
        "stage1_checkpoint_sha256": sha_file(seed_dir / "checkpoints/stage1_best.keras"),
        "stage2_checkpoint_sha256": sha_file(seed_dir / "checkpoints/stage2_best.keras"),
        "completed_at_utc": datetime.now(timezone.utc).isoformat()})
    completed = [s for s in SEEDS if (output / f"seed_{s}" / "seed_complete.json").is_file()
                 and validate_stage(output / f"seed_{s}", "stage1", config_sha)
                 and validate_stage(output / f"seed_{s}", "stage2", config_sha)]
    gate_status(output, completed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("verify", "train", "aggregate", "plots", "report", "export",
                                         "selftest-resume"))
    parser.add_argument("--zip", dest="archive")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-root")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    output = Path(args.output)
    if args.mode == "selftest-resume":
        from resume_smoke import run as resume_smoke
        resume_smoke(output)
        return
    if not args.source_root:
        parser.error("--source-root is required")
    config, _, config_sha = verify_protocol(output, args.source_root)
    if args.mode in ("verify", "train") and not args.archive:
        parser.error("--zip is required")
    if args.mode == "verify":
        if sha_file(args.archive) != ZIP_SHA:
            raise ValueError("V08 ZIP SHA256 mismatch")
        print(f"EXP-011 protocol PASS; config_sha256={config_sha}", flush=True)
    elif args.mode == "train":
        if args.seed is None:
            parser.error("--seed is required for train")
        train_seed(args.archive, output, args.source_root, args.seed)
    else:
        for seed in SEEDS:
            path = output / f"seed_{seed}"
            if not (path / "seed_complete.json").is_file():
                raise RuntimeError(f"Training incomplete: {path}")
            validate_stage(path, "stage1", config_sha)
            validate_stage(path, "stage2", config_sha)
        from results import run_results
        run_results(args.mode, output, config, config_sha, archive=args.archive)


if __name__ == "__main__":
    main()
