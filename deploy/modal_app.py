"""Modal deployment of the FastAPI backend (DenseNet121 x5 + YOLO26s, EXP-035).

Model binaries live in a Modal Volume (`cashew-models`), not in Git or in the image:

    modal volume create cashew-models
    modal volume put cashew-models EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04/pipeline_bundle /pipeline_bundle
    modal deploy deploy/modal_app.py

Endpoint: https://<workspace>--cashew-leaf-api-web.modal.run
"""
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ID = "EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04"
APP_DIR = "/root/app"

app = modal.App("cashew-leaf-api")

models = modal.Volume.from_name("cashew-models")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "python-multipart>=0.0.9",
        "pillow>=10.0.0",
        "numpy>=1.24.0",
        "tensorflow==2.21.0",
        "keras==3.15.1",
        "scikit-learn==1.6.1",
        "joblib>=1.3.0",
        "torch",
        "torchvision",
        "ultralytics>=8.4.0",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .env({"TF_CPP_MIN_LOG_LEVEL": "2", "YOLO_CONFIG_DIR": "/tmp/Ultralytics"})
    # Code is added last so editing the app does not rebuild the heavy layers.
    .add_local_dir(
        ROOT / "Application",
        f"{APP_DIR}/Application",
        ignore=["**/__pycache__", "**/*.pyc", "tools/**"],
    )
)


@app.function(
    image=image,
    volumes={f"{APP_DIR}/{BUNDLE_ID}": models},
    cpu=2.0,
    memory=4096,
    timeout=300,
    scaledown_window=300,  # stay warm 5 min after the last request
    max_containers=1,
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def web():
    import os
    import sys

    os.chdir(f"{APP_DIR}/Application")
    sys.path.insert(0, f"{APP_DIR}/Application")
    from server import app as fastapi_app  # loads + warms all models, verifies SHA-256

    return fastapi_app
