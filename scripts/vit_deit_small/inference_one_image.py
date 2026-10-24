"""Validation-locked, single-image inference from an EXP-010 staged checkpoint."""

import argparse
import json
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf

from dataset_audit import sha_file
from modeling import CLASSES


def predict(image_path, package_dir):
    package = Path(package_dir)
    meta = json.loads((package / "metadata.json").read_text(encoding="utf-8"))
    checkpoint = package / "deit_small_selected.keras"
    if sha_file(checkpoint) != meta["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA256 mismatch")
    image = tf.io.decode_image(tf.io.read_file(str(image_path)), channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.cast(tf.image.resize(image, (224, 224), method="bilinear"), tf.float32)
    model = keras.models.load_model(checkpoint, compile=False)
    logits = np.asarray(model(tf.expand_dims(image, 0), training=False))[0]
    probabilities = tf.nn.softmax(logits).numpy()
    index = int(np.argmax(probabilities))
    return {"image": str(image_path), "predicted_label": index,
            "predicted_class": CLASSES[index], "confidence": float(probabilities[index]),
            "probabilities": {klass: float(probabilities[i]) for i, klass in enumerate(CLASSES)},
            "checkpoint_sha256": meta["checkpoint_sha256"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("package_dir")
    args = parser.parse_args()
    print(json.dumps(predict(args.image, args.package_dir), indent=2))
