
# YOLO detection -> DenseNet121 crop classifier example.
# deployment_model.keras already contains DenseNet preprocessing.
from pathlib import Path
import json
import numpy as np
from PIL import Image
from tensorflow import keras

MODEL_DIR = Path("model_package")
classifier = keras.models.load_model(MODEL_DIR / "deployment_model.keras", compile=False)
label_map = json.loads((MODEL_DIR / "label_map.json").read_text())


def classify_crop(crop, img_size=224):
    if isinstance(crop, np.ndarray):
        crop = Image.fromarray(crop.astype(np.uint8))
    crop = crop.convert("RGB").resize((img_size, img_size))
    x = np.asarray(crop, dtype=np.float32)[None, ...]
    probs = classifier.predict(x, verbose=0)[0]
    pred_id = int(np.argmax(probs))
    return {"class_id": pred_id, "class_name": label_map[str(pred_id)],
            "confidence": float(probs[pred_id]), "probabilities": probs.tolist()}
