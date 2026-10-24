"""Write machine and dependency provenance without touching model or data."""

import argparse
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import tensorflow as tf


DEPENDENCIES = ("tensorflow", "keras", "keras-hub", "numpy", "pandas", "scipy",
                "scikit-learn", "pillow", "matplotlib", "h5py", "kagglehub")


def capture(output):
    versions = {name: importlib.metadata.version(name) for name in DEPENDENCIES}
    value = {"captured_utc": datetime.now(timezone.utc).isoformat(),
             "python": platform.python_version(), "platform": platform.platform(),
             "dependencies": versions,
             "tensorflow_physical_gpu": [str(x) for x in tf.config.list_physical_devices("GPU")],
             "tensorflow_physical_cpu": [str(x) for x in tf.config.list_physical_devices("CPU")],
             "mixed_precision_policy": tf.keras.mixed_precision.global_policy().name}
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(capture(args.output), indent=2))
