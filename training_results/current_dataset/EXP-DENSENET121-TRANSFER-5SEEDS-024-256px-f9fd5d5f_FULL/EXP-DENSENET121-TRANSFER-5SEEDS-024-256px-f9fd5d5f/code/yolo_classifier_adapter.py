from pathlib import Path
    import json
    import time
    import numpy as np
    import tensorflow as tf
    from PIL import Image, ImageOps
    from tensorflow import keras

    MODEL_DIR = Path("model_package")
    classifier = keras.models.load_model(MODEL_DIR / "deployment_model.keras", compile=False)
    label_map = json.loads((MODEL_DIR / "label_map.json").read_text(encoding="utf-8"))

    def classify_crop(crop):
        started = time.perf_counter()
        if isinstance(crop, np.ndarray):
            crop = Image.fromarray(crop.astype(np.uint8), mode="RGB")
        crop = ImageOps.exif_transpose(crop).convert("RGB")
        rgb = np.asarray(crop, dtype=np.float32)
        resized = tf.image.resize(rgb, [256, 256], method="bilinear").numpy()
        probability = classifier.predict(resized[None, ...], verbose=0)[0]
        class_id = int(np.argmax(probability))
        return {"class_id": class_id, "class_name": label_map[str(class_id)],
                "confidence": float(probability[class_id]),
                "probabilities": probability.tolist(),
                "elapsed_ms_including_preprocessing": 1000 * (time.perf_counter() - started)}
    