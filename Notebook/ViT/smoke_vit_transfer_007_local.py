"""Offline structural smoke for ViT 007; no pretrained weights or real training.

The notebook itself requires the ImageNet preset and GPU in Colab. This local
check substitutes a tiny DeiT task only to exercise the nested preprocessing,
two-stage trainability, saving, and notebook-source packaging code.
"""

import base64
import gc
import json
import math
import random
import re
import tempfile
import time
import zlib
from pathlib import Path
from unittest.mock import patch

import keras
import keras_hub
import numpy as np
import tensorflow as tf

NOTEBOOK = Path(__file__).with_name("vit_transfer_5seed_007.ipynb")
notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
assert len(notebook["cells"]) == 21
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] != "code":
        continue
    assert cell["execution_count"] is None and cell["outputs"] == []
    compile(
        "".join(line for line in cell["source"]
                if not line.lstrip().startswith(("%", "!"))),
        f"cell {index}", "exec",
    )

export_source = "".join(notebook["cells"][-1]["source"])
payload = re.search(r'^NOTEBOOK_SOURCE_ZLIB_B64 = "([^"]+)"$',
                    export_source, re.MULTILINE).group(1)
snapshot = json.loads(zlib.decompress(base64.b64decode(payload)))
last = "".join(snapshot["cells"][-1]["source"])
snapshot["cells"][-1]["source"] = last.replace(
    'NOTEBOOK_SOURCE_ZLIB_B64 = ""',
    f'NOTEBOOK_SOURCE_ZLIB_B64 = "{payload}"', 1).splitlines(keepends=True)
assert (json.dumps(snapshot, ensure_ascii=False, indent=1) + "\n").encode() == NOTEBOOK.read_bytes()
print("NOTEBOOK_AND_ZIP_SOURCE_PASS", flush=True)

source = "".join(notebook["cells"][10]["source"])
model_code = source[source.index("def set_seed(seed):"):source.index("# Measures image read")]
scope = {
    "random": random, "np": np, "keras": keras, "keras_hub": keras_hub,
    "tf": tf, "time": time, "math": math, "gc": gc,
    "PRESET": "offline-smoke", "CLASS_NAMES": ["a", "b", "c", "d", "e"],
    "IMG_SIZE": 224, "WEIGHT_DECAY": 1e-4,
    "BACKBONE_LR": 5e-6, "MIN_LR": 5e-7, "FINETUNE_EPOCHS": 12,
}
exec(compile(model_code, "cell10_model_defs", "exec"), scope)

backbone = keras_hub.models.DeiTBackbone(
    image_shape=(224, 224, 3), patch_size=16, num_layers=1, num_heads=3,
    hidden_dim=48, intermediate_dim=96,
)
converter = keras_hub.layers.DeiTImageConverter(
    image_size=(224, 224), scale=1 / 255,
)
preprocessor = keras_hub.models.DeiTImageClassifierPreprocessor(
    image_converter=converter,
)
classifier = keras_hub.models.DeiTImageClassifier(
    backbone=backbone, num_classes=5, preprocessor=preprocessor,
    activation=None, head_dtype="float32",
)
with patch.object(keras_hub.models.DeiTImageClassifier, "from_preset",
                  return_value=classifier):
    model, task = scope["build_model"](42)

probe = tf.random.stateless_uniform((1, 224, 224, 3), seed=(42, 7),
                                    minval=0, maxval=255)
actual = np.asarray(model(probe, training=False))
manual = np.asarray(task(task.preprocessor(probe), training=False))
assert actual.shape == (1, 5) and np.allclose(actual, manual, atol=1e-5)
print("SINGLE_PREPROCESSOR_PASS", flush=True)

task.backbone.trainable = False
scope["compile_model"](model, 1e-4)
assert not task.backbone.trainable_weights
model.train_on_batch(probe, tf.constant([2]))
task.backbone.trainable = True
scope["compile_model"](model, 5e-6)
assert task.backbone.trainable_weights
model.train_on_batch(probe, tf.constant([2]))
print("TWO_STAGE_PASS", flush=True)

after_training = np.asarray(model(probe, training=False))
with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
    package = Path(tmp) / "offline-smoke.keras"
    model.save(package)
    reloaded = keras.models.load_model(package, compile=False, safe_mode=True)
    loaded = np.asarray(reloaded(probe, training=False))
    assert np.allclose(after_training, loaded, rtol=1e-4, atol=1e-4)
print("KERAS_RELOAD_PASS", flush=True)
