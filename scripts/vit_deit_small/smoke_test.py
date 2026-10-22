"""Gate 1 forward/backward and reload smoke. Never counts as a benchmark run."""

import argparse
import hashlib
import io
import json
import platform
from pathlib import Path

import keras
import keras_hub
import numpy as np
import pandas as pd
import tensorflow as tf

from modeling import PRESET, build_model, preprocess_parity, set_stage1, set_stage2


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def batch_from_manifest(dataset_root, manifest):
    frame = pd.read_csv(manifest)
    rows = [frame[(frame.split == "train") & (frame.label == label)].iloc[0]
            for label in (0, 1)]
    images = []
    for row in rows:
        raw = tf.io.read_file(str(Path(dataset_root) / row.relative_path))
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape((None, None, 3))
        images.append(tf.cast(tf.image.resize(image, (224, 224), method="bilinear"), tf.float32))
    return tf.stack(images), tf.constant([0, 1], dtype=tf.int32)


def gradient_step(model, images, labels, learning_rate):
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    weights = list(model.trainable_weights)
    with tf.GradientTape() as tape:
        logits = model(images, training=True)
        loss = loss_fn(labels, logits)
    grads = tape.gradient(loss, weights)
    details = []
    for weight, gradient in zip(weights, grads):
        details.append({"path": getattr(weight, "path", weight.name),
                        "gradient_present": gradient is not None,
                        "gradient_norm": float(tf.linalg.global_norm([gradient]).numpy()) if gradient is not None else None})
    if not np.isfinite(float(loss.numpy())) or any(g is None for g in grads):
        raise RuntimeError(f"Missing/non-finite gradient, loss={loss.numpy()}, gradients={details}")
    optimizer = keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=1e-4, clipnorm=1.0)
    groups = ("transformer_block_11", "transformer_block_12", "predictions")
    snapshots = {group: np.array(next(weight for weight in weights
                     if group in getattr(weight, "path", weight.name))) for group in groups
                 if any(group in getattr(weight, "path", weight.name) for weight in weights)}
    optimizer.apply_gradients(zip(grads, weights))
    updated = {group: bool(np.any(before != np.array(next(weight for weight in weights
               if group in getattr(weight, "path", weight.name))))) for group, before in snapshots.items()}
    if not updated or not all(updated.values()):
        raise RuntimeError(f"A trainable group did not update: {updated}")
    return {"loss": float(loss.numpy()), "output_shape": tuple(int(x) for x in logits.shape),
            "updated_groups": updated, "gradients": details}


def run(dataset_root, manifest, output, preset_cache):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if not (output / "dataset_audit.json").is_file():
        raise FileNotFoundError("Gate 0 dataset_audit.json is required")
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    if audit["manifest_sha256"] != "C699A9ACA707DA25F143DB1FC3C9B5133EA2858BCDC468A158E5644E500EC715":
        raise ValueError("Gate 0 manifest hash changed")
    keras.backend.clear_session()
    images, labels = batch_from_manifest(dataset_root, manifest)
    model, classifier = build_model(42)
    parity = preprocess_parity(model, classifier, images)
    head = set_stage1(model, classifier)
    backbone_probe = np.array(classifier.backbone.weights[0])
    stage1 = gradient_step(model, images, labels, 1e-4)
    if np.any(backbone_probe != np.array(classifier.backbone.weights[0])):
        raise RuntimeError("Stage 1 changed a frozen backbone weight")
    if stage1["output_shape"] != (2, 5):
        raise RuntimeError(f"Wrong Stage 1 logits: {stage1['output_shape']}")
    stage2_layers = set_stage2(model, classifier)
    stage2 = gradient_step(model, images, labels, 1e-5)
    if stage2["output_shape"] != (2, 5):
        raise RuntimeError(f"Wrong Stage 2 logits: {stage2['output_shape']}")
    parity_after = preprocess_parity(model, classifier, images)
    before_reload = np.asarray(model(images, training=False))
    saved = output / "_work" / "gate1_smoke_model.keras"
    saved.parent.mkdir(parents=True, exist_ok=True)
    model.save(saved)
    reloaded = keras.models.load_model(saved, compile=False, safe_mode=True)
    after_reload = np.asarray(reloaded(images, training=False))
    max_reload_error = float(np.max(np.abs(before_reload - after_reload)))
    if not np.allclose(before_reload, after_reload, rtol=1e-4, atol=1e-4):
        raise RuntimeError(f"Save/reload inference changed: {max_reload_error}")
    source = Path(preset_cache) / "models/keras/deit/keras/deit_small_distilled_patch16_224_imagenet/1"
    provenance = {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)} for p in source.glob("*") if p.is_file()}
    expected = {"config.json": "E3CABFB27BEDDB3C85BE90371FCCC14FAB94878051BD2587C42BBFC8D92EF26E",
                "model.weights.h5": "EB03ECF54381B14FEDC6120D30B696B7E848004BD6EE2A708DA973618C926893",
                "task.json": "DB83E029005D9B645AF578B407454AAF342DDC59D553DCA7C263666DF093E956",
                "task.weights.h5": "186714690069C17876BA8D6479F27E5C345AAC8A3B37574EE04B8B1D949AB72F"}
    if {name: item["sha256"] for name, item in provenance.items()} != expected:
        raise RuntimeError(f"Pretrained preset provenance changed: {provenance}")
    if head[0]["shape"] != (384, 5):
        raise RuntimeError("Expected DeiT-Small 384-D pre-head features")
    report = {"status": "PASS", "scope": "2 real Train images, one gradient update per stage; not benchmark",
        "preset": PRESET, "preset_artifacts": provenance,
        "python": platform.python_version(), "tensorflow": tf.__version__, "keras": keras.__version__,
        "keras_hub": keras_hub.__version__, "gpu": [str(x) for x in tf.config.list_physical_devices("GPU")],
        "model_parameters": int(model.count_params()), "backbone_parameters": int(classifier.backbone.count_params()),
        "stage1_trainable_parameters": sum(item["parameters"] for item in head),
        "stage2_trainable_parameters": sum(item["parameters"] for item in stage2_layers),
        "stage1_trainable": head, "stage2_trainable": stage2_layers,
        "preprocess_parity_before": parity, "preprocess_parity_after": parity_after,
        "stage1_gradient_step": stage1, "stage2_gradient_step": stage2,
        "save_reload_max_abs_logit_difference": max_reload_error,
        "checkpoint_sha256": sha(saved)}
    (output / "gate1_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root")
    parser.add_argument("manifest")
    parser.add_argument("output")
    parser.add_argument("preset_cache")
    args = parser.parse_args()
    report = run(args.dataset_root, args.manifest, args.output, args.preset_cache)
    print(json.dumps({k: v for k, v in report.items() if k not in ("stage1_gradient_step", "stage2_gradient_step",
                                                                 "stage1_trainable", "stage2_trainable")}, indent=2))
