"""
DenseNet121 Transfer v014 Model Inference Service for Cashew Leaf Disease Classification.
"""
import io
import json
import time
import logging
from typing import Dict, Any, List

import numpy as np
from PIL import Image
from tensorflow import keras

from config import (
    MODEL_PATH,
    LABEL_MAP_PATH,
    METADATA_PATH,
    IMG_SIZE,
    MODEL_NAME,
    BEST_SEED,
    VAL_ACCURACY,
    VAL_LOSS,
    VAL_MACRO_F1,
    DISEASE_INFO,
)

logger = logging.getLogger("CashewLeafAI")
logging.basicConfig(level=logging.INFO)


class DenseNet121Service:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DenseNet121Service, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        logger.info(f"Loading best DenseNet121 model from: {MODEL_PATH}")
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

        # Load deployment model (contains internal DenseNet preprocessing)
        self.model = keras.models.load_model(str(MODEL_PATH), compile=False)

        # Load label map
        if LABEL_MAP_PATH.exists():
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)
        else:
            self.label_map = {
                "0": "anthracnose",
                "1": "healthy",
                "2": "leaf_miner",
                "3": "red_rust",
                "4": "not_cashew_leaf",
            }

        # Load metadata
        self.metadata = {}
        if METADATA_PATH.exists():
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)

        # Warm up model to eliminate initial inference latency
        logger.info("Warming up model with dummy input...")
        dummy = np.zeros((1, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        logger.info("Model loaded and warmed up successfully!")

    def predict_image(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs diagnosis on uploaded image bytes.
        Returns:
            - conclusion: top predicted class details (highest confidence)
            - probabilities_desc: all 5 classes sorted DESCENDING by confidence (Cao nhất lên đầu)
            - latency_ms: inference time
            - image_info: original image size, format
        """
        start_time = time.perf_counter()

        # Open and validate image
        try:
            image = Image.open(io.BytesIO(image_bytes))
            orig_width, orig_height = image.size
            orig_format = image.format or "JPEG"
            # Convert to RGB
            rgb_image = image.convert("RGB")
        except Exception as e:
            raise ValueError(f"Không thể đọc định dạng hình ảnh: {str(e)}")

        # Resize to model input size (224x224)
        resized_img = rgb_image.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
        x = np.asarray(resized_img, dtype=np.float32)[None, ...]

        # Predict
        probs = self.model.predict(x, verbose=0)[0]
        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Build list of classes with their confidence
        class_results: List[Dict[str, Any]] = []
        for idx_str, class_key in self.label_map.items():
            idx = int(idx_str)
            conf = float(probs[idx])
            info = DISEASE_INFO.get(class_key, {})
            class_results.append({
                "class_id": idx,
                "class_key": class_key,
                "label_vn": info.get("vn_name", class_key),
                "scientific_name": info.get("scientific_name", ""),
                "severity": info.get("severity", ""),
                "benchmark_precision": info.get("benchmark_precision", 0.0),
                "benchmark_recall": info.get("benchmark_recall", 0.0),
                "benchmark_f1": info.get("benchmark_f1", 0.0),
                "badge_color": info.get("badge_color", "primary"),
                "color_hex": info.get("color_hex", "#2563eb"),
                "icon": info.get("icon", "fa-leaf"),
                "confidence": conf,
                "confidence_percent": round(conf * 100, 2),
                "symptoms": info.get("symptoms", ""),
                "management": info.get("management", []),
            })

        # SẮP XẾP GIẢM DẦN THEO ĐỘ TIN CẬY (DESCENDING - Cao nhất lên trên đầu)
        probabilities_desc = sorted(class_results, key=lambda item: item["confidence"], reverse=True)

        # The one with the highest confidence on top is used as conclusion
        top_prediction = probabilities_desc[0]

        return {
            "success": True,
            "conclusion": top_prediction,
            # Sắp xếp giảm dần từ cao xuống thấp:
            "probabilities_desc": probabilities_desc,
            "latency_ms": latency_ms,
            "image_meta": {
                "width": orig_width,
                "height": orig_height,
                "format": orig_format,
            },
            "model_meta": {
                "name": MODEL_NAME,
                "seed": BEST_SEED,
                "val_accuracy": VAL_ACCURACY,
                "val_loss": VAL_LOSS,
                "val_macro_f1": VAL_MACRO_F1,
                "input_size": f"{IMG_SIZE}x{IMG_SIZE}",
            }
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "name": MODEL_NAME,
            "seed": BEST_SEED,
            "experiment_id": self.metadata.get("experiment_id", "EXP-DENSENET121-TRANSFER-5SEEDS-014"),
            "selection_source": self.metadata.get("selection_source", "validation only"),
            "val_accuracy": VAL_ACCURACY,
            "val_loss": VAL_LOSS,
            "val_macro_f1": VAL_MACRO_F1,
            "classes": [
                {
                    "class_key": k,
                    **DISEASE_INFO.get(k, {})
                }
                for k in self.label_map.values()
            ]
        }


# Global instance
model_service = DenseNet121Service()
