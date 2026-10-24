"""Local architecture smoke for EXP-009 when ImageNet weights are unavailable.

This deliberately uses weights=None and never marks the notebook's mandatory
ImageNet pipeline smoke as passed. Colab must run the notebook's own smoke.
"""

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

NOTEBOOK = Path(__file__).with_name("resnet50_transfer_5seed_009.ipynb")
cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
model_code = "".join(cells[12]["source"]).split("if RUN_PIPELINE_SMOKE:", 1)[0]
scope = {
    "np": np, "tf": tf, "keras": keras, "layers": layers,
    "IMG_SIZE": 224, "NUM_CLASSES": 5, "HEAD_DROPOUT": 0.40,
    "WEIGHT_DECAY": 1e-4, "STAGE2_TRAINABLE_PREFIX": "conv5_block3_",
    "set_global_seed": keras.utils.set_random_seed,
    "hashlib": __import__("hashlib"), "json": json, "Path": Path,
}
exec(compile(model_code, str(NOTEBOOK) + ":cell12", "exec"), scope)
keras.utils.set_random_seed(42)
rgb = np.zeros((2, 224, 224, 3), dtype=np.float32)
rgb[0, 0, 0] = [255.0, 0.0, 0.0]
rgb[0, 0, 1] = [0.0, 128.0, 255.0]
rgb[1] = np.random.default_rng(42).uniform(0, 255, rgb[1].shape)
p = keras.Input((224, 224, 3))
preprocess = keras.Model(p, scope["resnet50_preprocessing"](p))
actual = preprocess(rgb).numpy()
reference = keras.applications.resnet50.preprocess_input(rgb.copy())
error = float(np.max(np.abs(actual - reference)))
assert error < 1e-4, error
assert np.allclose(actual[0, 0, 0], [-103.939, -116.779, 131.32], atol=1e-4)
print("CAFFE_PARITY_PASS", error, flush=True)

model, backbone = scope["build_transfer_model"](42, weights=None)
assert model.output_shape == (None, 5)
stage1 = scope["set_backbone_stage"](backbone, "stage1")
assert stage1 == 0
scope["compile_stage"](model, 1e-4)
old_optimizer = model.optimizer
x = tf.constant(rgb)
y = tf.constant([0, 1], dtype=tf.int32)
model.train_on_batch(x[:1], y[:1])
print("STAGE1_PASS", stage1, flush=True)
bn = [layer for layer in backbone.layers if isinstance(layer, layers.BatchNormalization)]
before = [(layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy()) for layer in bn]
stage2 = scope["set_backbone_stage"](backbone, "stage2")
assert 0 < stage2 < backbone.count_params()
assert all(layer.name.startswith("conv5_block3_") for layer in backbone.layers
           if layer.trainable and layer.weights)
assert all(not layer.trainable for layer in bn)
scope["compile_stage"](model, 3e-6)
assert model.optimizer is not old_optimizer
model.train_on_batch(x[:1], y[:1])
assert all(np.array_equal(m, layer.moving_mean.numpy()) and
           np.array_equal(v, layer.moving_variance.numpy())
           for (m, v), layer in zip(before, bn))
print("STAGE2_BN_PASS", stage2, flush=True)

prediction = model(x[:1], training=False).numpy()
assert prediction.shape == (1, 5) and np.isfinite(prediction).all()
assert np.allclose(prediction.sum(axis=1), 1.0, atol=1e-5)
with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    checkpoint = root / "smoke.weights.h5"
    package = root / "smoke.keras"
    input_file = root / "input.npy"
    output_file = root / "output.npy"
    model.save_weights(checkpoint)
    model.save(package)
    model.layers[-1].bias.assign_add(tf.one_hot(0, 5, on_value=20.0,
                                                off_value=0.0, dtype=tf.float32))
    assert not np.allclose(prediction, model(x[:1], training=False).numpy(), atol=1e-5)
    model.load_weights(checkpoint)
    assert np.allclose(prediction, model(x[:1], training=False).numpy(), atol=1e-5)
    np.save(input_file, rgb[:1])
    program = (
        "import sys,numpy as np,tensorflow as tf; from tensorflow import keras; "
        "keras.utils.set_random_seed(int(sys.argv[4])); "
        "tf.config.experimental.enable_op_determinism(); "
        "m=keras.models.load_model(sys.argv[1],compile=False,safe_mode=True); "
        "np.save(sys.argv[3],m(np.load(sys.argv[2]),training=False).numpy())"
    )
    subprocess.run([sys.executable, "-c", program, str(package),
                    str(input_file), str(output_file), "42"], check=True)
    clean_prediction = np.load(output_file)
    max_error = float(np.max(np.abs(prediction - clean_prediction)))
    assert np.allclose(prediction, clean_prediction, rtol=1e-3, atol=5e-4), max_error
    print("CLEAN_PROCESS_MAX_ABS_ERROR", max_error, flush=True)
print("CHECKPOINT_CLEAN_PROCESS_PASS", flush=True)
