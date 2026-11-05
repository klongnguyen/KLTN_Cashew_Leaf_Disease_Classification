"""
EXP-034 Production Inference Service.
Model Architecture: DenseNet121 Ensemble (5 Seeds: 42, 123, 2026, 3407, 7777)
                   + YOLO26s Lesion Detector (best.pt)
                   + Stacker F3 (Logistic Regression C=3.0)
                   + Temperature Scaling (T=1.0135)
                   + Abstention Threshold (tau=0.5042)
"""
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import joblib
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from ultralytics import YOLO

from config import (
    EXP034_BUNDLE_DIR,
    EXP034_MODEL_NAME,
    EXP034_METRICS,
    EXP034_VERIFY_SHA256,
    DISEASE_INFO,
    ABSTAIN_MIN_MARGIN,
    BOX_CONF_PER_CLASS,
    CLASSIFIER_PREDOWNSCALE_MAX_SIDE,
    LEAF_MASK_FILTER,
    LEAF_MASK_MIN_INSIDE,
    MERGE_IOS,
    MERGE_IOU,
    TILE_MIN_LONG_SIDE,
    TILE_OVERLAP,
    TILE_SIZE,
    TILE_SOURCE_MAX_SIDE,
    YOLO_TILED_DISPLAY,
    FINAL_DECISION,
    DENSENET_ABSTAIN_THRESHOLD,
    VERIFY_CROP_CONTEXT,
    VERIFY_CROP_MIN_PX,
    VERIFY_HEALTHY_REJECT,
    VERIFY_MAX_CROPS,
    VERIFY_MIN_DISEASE_MASS,
    SECONDARY_CROP_MIN_CONF,
    OOD_GATE_ENABLED,
    OOD_REFERENCE_PATH,
    OOD_CALIBRATION_PATH,
    OOD_KNN_K,
    OOD_BANK_SEED,
    OOD_USE_ENERGY,
    OOD_BAND_MAX_P_NOT_CASHEW,
    OOD_BAND_MIN_TOP_CASHEW,
)
from diagnosis_report import (
    abstain_decision,
    build_report,
    format_boxes,
    probabilities_list,
    select_verified_boxes,
)
from image_utils import (
    assess_quality,
    compute_leaf_mask,
    damage_percent,
    load_image,
    merge_boxes,
    predownscale,
    tile_windows,
)

logger = logging.getLogger("CashewLeafAI.EXP034")
logging.basicConfig(level=logging.INFO)


# Register RandomHSVJitter custom layer from EXP-029/034 before loading Keras models
@keras.utils.register_keras_serializable(package="exp029", name="RandomHSVJitter")
class RandomHSVJitter(layers.Layer):
    """HSV augmentation layer used during training; acts as pure identity during inference."""
    def __init__(
        self,
        saturation_range=(0.6, 1.4),
        hue_delta=0.03,
        grayscale_probability=0.10,
        value_range=(0.0, 255.0),
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.saturation_range = tuple(float(v) for v in saturation_range)
        self.hue_delta = float(hue_delta)
        self.grayscale_probability = float(grayscale_probability)
        self.value_range = tuple(float(v) for v in value_range)
        self.seed = seed

    def call(self, inputs, training=False):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "saturation_range": list(self.saturation_range),
                "hue_delta": self.hue_delta,
                "grayscale_probability": self.grayscale_probability,
                "value_range": list(self.value_range),
                "seed": self.seed,
            }
        )
        return config


def prepare_input(image: Image.Image) -> np.ndarray:
    """Whole-image / crop preprocessing shared by the service and the OOD
    calibration tool: antialiased pre-shrink, then tf bilinear resize to
    256x256, float32 0..255 (the models rescale internally)."""
    src = predownscale(image, CLASSIFIER_PREDOWNSCALE_MAX_SIDE)
    return tf.image.resize(tf.cast(np.asarray(src), tf.float32), (256, 256)).numpy()


class EXP034Service:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EXP034Service, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    @staticmethod
    def _compute_sha256(filepath: Path) -> str:
        h = hashlib.sha256()
        with open(filepath, "rb") as f:
            while chunk := f.read(1024 * 1024):
                h.update(chunk)
        return h.hexdigest()

    def _verify_bundle_integrity(self):
        """Verifies bundle artifacts against artifact_sha256.csv."""
        csv_path = self.bundle_dir / "artifact_sha256.csv"
        if not csv_path.exists():
            logger.warning(f"Integrity check skipped: {csv_path} not found.")
            return

        lines = csv_path.read_text(encoding="utf-8").strip().splitlines()
        header = lines[0].split(",")
        for line in lines[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 3:
                rel_path = parts[0].strip()
                if rel_path == "artifact_sha256.csv":
                    continue
                expected_sha = parts[2].strip()
                target_file = self.bundle_dir / rel_path
                if not target_file.exists():
                    raise FileNotFoundError(f"Missing bundle artifact: {target_file}")
                actual_sha = self._compute_sha256(target_file)
                if actual_sha != expected_sha:
                    raise ValueError(
                        f"SHA256 mismatch for {rel_path}!\nExpected: {expected_sha}\nActual:   {actual_sha}"
                    )
        logger.info("All EXP-034 bundle artifacts verified successfully with exact SHA-256 match!")

    def _initialize(self):
        self.bundle_dir = Path(EXP034_BUNDLE_DIR)
        if not self.bundle_dir.exists():
            raise FileNotFoundError(f"EXP-034 bundle directory not found at: {self.bundle_dir}")

        if EXP034_VERIFY_SHA256:
            self._verify_bundle_integrity()

        # Load configurations
        self.config = json.loads((self.bundle_dir / "frozen_config.json").read_text(encoding="utf-8"))
        self.decision = json.loads((self.bundle_dir / "decision.json").read_text(encoding="utf-8"))
        self.classes: List[str] = self.config["classes"]
        self.detector_classes: List[str] = self.config["detector_classes"]
        self.seeds: List[int] = self.config["seeds"]
        self.rule: str = self.decision.get("selected_default_rule", "F3")
        self.temperature: float = float(self.config["temperature"][self.rule])
        self.abstain_threshold: float = float(self.config["abstain_thresholds"][self.rule])
        self.box_conf: float = float(self.config.get("box_conf", 0.25))
        self.box_nms_iou: float = float(self.config.get("box_nms_iou", 0.5))
        self.max_draw_boxes: int = int(self.config.get("max_draw_boxes", 60))

        # Load Stacker F3
        stacker_path = self.bundle_dir / "stacker.joblib"
        logger.info(f"Loading Stacker F3 from {stacker_path}...")
        self.stackers = joblib.load(stacker_path)
        self.stacker = self.stackers[self.rule]

        # Load YOLO detector
        yolo_path = self.bundle_dir / "best.pt"
        logger.info(f"Loading YOLO26s detector from {yolo_path}...")
        self.detector = YOLO(str(yolo_path))

        # Load 5 DenseNet121 models
        logger.info(f"Loading DenseNet121 ensemble across {len(self.seeds)} seeds ({self.seeds})...")
        self.models = []
        for seed in self.seeds:
            model_file = self.bundle_dir / f"seed_{seed}" / "deployment_model.keras"
            m = keras.models.load_model(str(model_file), compile=False)
            self.models.append(m)

        # Backbone feature models + softmax heads: one pass yields the
        # embedding (kNN), the logits (Energy) and the probabilities.
        self.feature_models = []
        self.heads = []
        for m in self.models:
            # Input of the softmax head = GAP embedding after dropout (identity
            # at inference). The nested backbone's own .output is not connected
            # to the outer graph in Keras 3, the head's input is.
            self.feature_models.append(keras.Model(m.inputs, m.get_layer("disease_output").input))
            W, b = m.get_layer("disease_output").get_weights()
            self.heads.append((W.astype(np.float64), b.astype(np.float64)))
        self.bank_index = self.seeds.index(OOD_BANK_SEED) if OOD_BANK_SEED in self.seeds else 0
        self._load_ood_reference()

        # Warm up ensemble and detector
        logger.info("Warming up EXP-034 ensemble & detector...")
        for size in self.BATCH_BUCKETS:
            self._forward(np.zeros((size, 256, 256, 3), dtype=np.float32))
        dummy_pil = Image.fromarray(np.zeros((640, 640, 3), dtype=np.uint8))
        _ = self.detector.predict(dummy_pil, conf=0.05, iou=0.7, imgsz=640, verbose=False)

        logger.info("EXP-034 Combined Pipeline successfully initialized and warmed up!")

    # ------------------------------------------------------------------
    # Detector passes
    # ------------------------------------------------------------------
    def _yolo_full_pass(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Single 640px pass, identical to the frozen training pipeline.

        Its boxes feed the stacker features; do not change its parameters.
        """
        res = self.detector.predict(image, conf=0.05, iou=0.7, imgsz=640, rect=False, verbose=False)[0]
        return self._parse_boxes(res, 0.0, 0.0, 1.0, "yolo")

    def _parse_boxes(self, res, off_x: float, off_y: float, inv_scale: float, source: str):
        out = []
        if len(res.boxes) == 0:
            return out
        for c, q, xy in zip(
            res.boxes.cls.cpu().numpy(), res.boxes.conf.cpu().numpy(), res.boxes.xyxy.cpu().numpy()
        ):
            ci = int(c)
            if ci >= len(self.detector_classes):
                continue
            out.append(
                {
                    "cls": self.detector_classes[ci],
                    "conf": float(q),
                    "xyxy": [
                        (float(xy[0]) + off_x) * inv_scale,
                        (float(xy[1]) + off_y) * inv_scale,
                        (float(xy[2]) + off_x) * inv_scale,
                        (float(xy[3]) + off_y) * inv_scale,
                    ],
                    "source": source,
                }
            )
        return out

    def _yolo_tiled_pass(self, image: Image.Image) -> Tuple[List[Dict[str, Any]], int]:
        """SAHI-style tiled pass on an antialiased copy (display boxes only)."""
        w, h = image.size
        if not YOLO_TILED_DISPLAY or max(w, h) < TILE_MIN_LONG_SIDE:
            return [], 0
        src = predownscale(image, TILE_SOURCE_MAX_SIDE)
        inv = w / float(src.width)
        windows = tile_windows(src.width, src.height, TILE_SIZE, TILE_OVERLAP)
        if len(windows) <= 1:
            return [], 0
        crops = [src.crop(win) for win in windows]
        min_conf = min(BOX_CONF_PER_CLASS.values())
        results = self.detector.predict(crops, conf=min_conf, iou=0.5, imgsz=TILE_SIZE, verbose=False)
        boxes: List[Dict[str, Any]] = []
        for (x0, y0, _, _), res in zip(windows, results):
            boxes.extend(self._parse_boxes(res, x0, y0, inv, "yolo_tile"))
        return boxes, len(windows)

    # ------------------------------------------------------------------
    # DenseNet helpers
    # ------------------------------------------------------------------
    def _densenet_batch(self, crops: List[Image.Image]) -> np.ndarray:
        """Ensemble probabilities for a batch of PIL images (256x256 input)."""
        return self._forward(np.stack([prepare_input(c) for c in crops]))["probs"]

    # Batches are padded to a few fixed sizes so TensorFlow compiles each
    # shape once (warmed up at startup) instead of on every new crop count.
    BATCH_BUCKETS = (1, 4, 9, 16)

    def _forward(self, x: np.ndarray) -> Dict[str, np.ndarray]:
        """One pass per seed through the DenseNet backbone.

        Returns the ensemble softmax ("probs", identical to the deployment
        models' output), the Energy score ("energy" = -logsumexp(logits),
        averaged over seeds; higher = more out-of-distribution) and the
        1024-d GAP embeddings of every seed ("feats", shape (n, seeds, 1024)).
        """
        n = len(x)
        size = next((b for b in self.BATCH_BUCKETS if b >= n), None)
        if size is None:  # larger than the biggest bucket: process in chunks
            big = self.BATCH_BUCKETS[-1]
            parts = [self._forward(x[i : i + big]) for i in range(0, n, big)]
            return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        if size > n:
            x = np.concatenate([x, np.zeros((size - n,) + x.shape[1:], dtype=x.dtype)])
        probs, energies, feats = [], [], []
        for fm, (W, b) in zip(self.feature_models, self.heads):
            # predict_on_batch: compiled, without predict()'s per-call overhead
            # (CPU bench: 105 ms vs 260 ms predict() vs 1124 ms eager call).
            f = np.asarray(fm.predict_on_batch(x), dtype=np.float64)
            z = f @ W + b
            zmax = z.max(axis=1, keepdims=True)
            lse = zmax[:, 0] + np.log(np.exp(z - zmax).sum(axis=1))
            probs.append(np.exp(z - lse[:, None]))
            energies.append(-lse)
            feats.append(f[:n].astype(np.float32))
        return {
            "probs": np.mean(probs, axis=0)[:n],
            "energy": np.mean(energies, axis=0)[:n],
            "feats": np.stack(feats, axis=1),
        }

    # ------------------------------------------------------------------
    # OOD gate (step 0, hard rule)
    # ------------------------------------------------------------------
    def _load_ood_reference(self) -> None:
        self.ood_bank = None
        self.ood_cal: Dict[str, Any] = {}
        if not OOD_GATE_ENABLED:
            return
        if OOD_REFERENCE_PATH.exists() and OOD_CALIBRATION_PATH.exists():
            ref = np.load(OOD_REFERENCE_PATH)
            bank = ref["bank"].astype(np.float32)
            if bank.ndim == 3:  # (N, seeds, 1024), already flip-averaged per seed
                self.ood_bank = self.gate_embedding(bank)
                self.ood_multi_seed = True
            else:  # legacy single-seed bank (N, 1024)
                self.ood_bank = bank / np.linalg.norm(bank, axis=1, keepdims=True)
                self.ood_multi_seed = False
            self.ood_cal = json.loads(OOD_CALIBRATION_PATH.read_text(encoding="utf-8"))
            logger.info(
                f"OOD gate: bank {bank.shape}, knn_threshold={self.ood_cal.get('knn_threshold')}, "
                f"knn_threshold_high={self.ood_cal.get('knn_threshold_high')}"
            )
        else:
            logger.warning(
                "OOD gate: reference bank/calibration not found -> only the not_cashew_leaf argmax "
                "rule is active. Run tools/build_ood_reference.py."
            )

    @staticmethod
    def flip_average(feats: np.ndarray, feats_flip: np.ndarray) -> np.ndarray:
        """Per-seed L2-normalised mean of an image's and its h-flip's embedding."""
        def nrm(a):
            return a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-12)
        return nrm(nrm(feats) + nrm(feats_flip))

    @staticmethod
    def gate_embedding(per_seed: np.ndarray) -> np.ndarray:
        """(..., seeds, 1024) -> flat vector whose dot product = mean cosine over seeds."""
        s = per_seed / np.maximum(np.linalg.norm(per_seed, axis=-1, keepdims=True), 1e-12)
        return (s.reshape(s.shape[:-2] + (-1,)) / np.sqrt(s.shape[-2])).astype(np.float32)

    def _knn_score(self, feats: np.ndarray, feats_flip: np.ndarray) -> Optional[float]:
        """feats / feats_flip: (seeds, 1024) embeddings of the image and its flip."""
        if self.ood_bank is None:
            return None
        if self.ood_multi_seed:
            q = self.gate_embedding(self.flip_average(feats, feats_flip))
        else:
            f = feats[self.bank_index]
            q = (f / max(1e-12, float(np.linalg.norm(f)))).astype(np.float32)
        sims = self.ood_bank @ q
        k = min(OOD_KNN_K, len(sims))
        kth = np.partition(-sims, k - 1)[k - 1]  # k-th largest similarity (negated)
        return float(1.0 + kth)  # cosine distance to the k-th nearest cashew leaf

    def _ood_gate(
        self, probs: np.ndarray, energy: float, feats: np.ndarray, feats_flip: np.ndarray
    ) -> Dict[str, Any]:
        """Three-zone hard gate: pass / band (conditional) / reject."""
        nc = self.classes.index("not_cashew_leaf")
        knn = self._knn_score(feats, feats_flip)
        t_low = self.ood_cal.get("knn_threshold")
        t_high = self.ood_cal.get("knn_threshold_high", t_low)
        t_e = self.ood_cal.get("energy_threshold")
        p_nc = float(probs[nc])
        top_cashew = float(np.max(np.delete(probs, nc)))
        confident_cashew = p_nc < OOD_BAND_MAX_P_NOT_CASHEW and top_cashew >= OOD_BAND_MIN_TOP_CASHEW

        reasons = []
        zone = "pass"
        if int(np.argmax(probs)) == nc:
            reasons.append(f"DenseNet toàn ảnh xếp lớp 'không phải lá điều' ({p_nc * 100:.1f}%)")
        if knn is not None and t_low is not None and knn > t_low:
            if knn > t_high:
                reasons.append(
                    f"đặc trưng ảnh khác xa mọi lá điều tham chiếu (khoảng cách kNN {knn:.3f} > ngưỡng cứng {t_high:.3f})"
                )
            elif not confident_cashew:
                reasons.append(
                    f"ảnh ở vùng nghi ngờ (kNN {knn:.3f} trong ({t_low:.3f}; {t_high:.3f}]) và DenseNet không đủ chắc "
                    f"là lá điều (bệnh/khỏe cao nhất {top_cashew * 100:.1f}% < {OOD_BAND_MIN_TOP_CASHEW * 100:.0f}% "
                    f"hoặc P(không phải lá điều) {p_nc * 100:.1f}% ≥ {OOD_BAND_MAX_P_NOT_CASHEW * 100:.0f}%)"
                )
            else:
                zone = "band"
        if OOD_USE_ENERGY and t_e is not None and energy > t_e:
            reasons.append(f"điểm bất thường Energy cao ({energy:.2f} > ngưỡng {t_e:.2f})")
        rejected = bool(reasons) and OOD_GATE_ENABLED
        return {
            "enabled": OOD_GATE_ENABLED,
            "calibrated": self.ood_bank is not None,
            "rejected": rejected,
            "zone": "reject" if rejected else zone,
            "reasons": reasons,
            "knn_distance": knn,
            "knn_threshold": t_low,
            "knn_threshold_high": t_high,
            "band_rule": {
                "max_p_not_cashew": OOD_BAND_MAX_P_NOT_CASHEW,
                "min_top_cashew": OOD_BAND_MIN_TOP_CASHEW,
                "top_cashew": top_cashew,
            },
            "energy": float(energy),
            "energy_threshold": t_e,
            "energy_used": OOD_USE_ENERGY,
            "p_not_cashew": p_nc,
        }

    @staticmethod
    def _context_window(w: int, h: int, xyxy: List[float]) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Option 1B: side = max(3 x long side of the box, 128 px), cut from the
        # full-resolution original so the crop carries surrounding leaf context.
        side = max(max(x2 - x1, y2 - y1) * VERIFY_CROP_CONTEXT, float(VERIFY_CROP_MIN_PX))
        side = min(side, float(min(w, h)))
        left = min(max(0.0, cx - side / 2.0), w - side)
        top = min(max(0.0, cy - side / 2.0), h - side)
        return (left, top, left + side, top + side)

    def _verify_boxes(self, image: Image.Image, boxes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
        """DenseNet verification of YOLO proposals.

        Boxes (highest YOLO confidence first) get a context window; a box that
        lies inside an existing window shares it (merged region). Each window
        is classified once by the DenseNet ensemble:
          - healthy dominant                      -> "rejected_healthy"
          - disease mass < VERIFY_MIN_DISEASE_MASS -> "unverified": DenseNet
            cannot read the fragment, the box keeps YOLO's label
          - otherwise the box gets DenseNet's disease (raw probability, not
            renormalised) -> "agree" / "differ" with YOLO.
        """
        w, h = image.size
        windows: List[Tuple[float, float, float, float]] = []
        assign: List[Optional[int]] = []
        for b in boxes:
            x1, y1, x2, y2 = b["xyxy"]
            k = next(
                (i for i, (l, t, r, bt) in enumerate(windows) if x1 >= l and y1 >= t and x2 <= r and y2 <= bt),
                None,
            )
            if k is None and len(windows) < VERIFY_MAX_CROPS:
                windows.append(self._context_window(w, h, b["xyxy"]))
                k = len(windows) - 1
            if k is None:  # cap reached: reuse a window holding the box centre
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                k = next(
                    (i for i, (l, t, r, bt) in enumerate(windows) if l <= cx <= r and t <= cy <= bt),
                    None,
                )
            assign.append(k)
        if not windows:
            return [], 0

        P = self._densenet_batch([image.crop(tuple(int(round(v)) for v in win)) for win in windows])
        d_idx = [self.classes.index(c) for c in self.detector_classes]
        h_idx = self.classes.index("healthy")

        out: List[Dict[str, Any]] = []
        for b, k in zip(boxes, assign):
            verify = {"yolo_cls": b["cls"], "yolo_conf": float(b["conf"]), "crop_id": k,
                      "crop_xyxy": None, "dense_conf": 0.0, "dense_disease_mass": None,
                      "dense_healthy": None, "status": "unverified"}
            cls, conf = b["cls"], None  # default: YOLO label, no DenseNet score
            if k is not None:
                p = P[k]
                pdis = p[d_idx]
                j = int(np.argmax(pdis))
                verify["crop_xyxy"] = [int(round(v)) for v in windows[k]]
                verify["dense_conf"] = float(pdis[j])
                verify["dense_disease_mass"] = float(pdis.sum())
                verify["dense_healthy"] = float(p[h_idx])
                if p[h_idx] >= VERIFY_HEALTHY_REJECT and p[h_idx] > pdis.max():
                    verify["status"] = "rejected_healthy"
                    cls = None
                elif pdis.sum() >= VERIFY_MIN_DISEASE_MASS:
                    cls, conf = self.detector_classes[j], float(pdis[j])
                    verify["status"] = "agree" if cls == b["cls"] else "differ"
            out.append({**b, "cls": cls, "conf": conf, "verify": verify})
        return out, len(windows)

    def _ood_rejection_result(self, image, orig_format, quality, densenet_probs, gate, t0) -> Dict[str, Any]:
        """Result returned when the OOD gate rejects the image (YOLO not run)."""
        w, h = image.size
        conclusion, _, plan = build_report("not_cashew_leaf", 1.0, False, None, [], [], 0.0, "image")
        conclusion.update(
            {
                "label_vn": "Từ Chối: Không Phải Lá Điều / Ảnh Ngoại Lai",
                "confidence_percent": None,
                "ood_rejected": True,
                "symptoms": (
                    "Ảnh bị từ chối ở cổng kiểm tra đầu vào, nên không khoanh vùng hay chẩn đoán bệnh "
                    "(kể cả khi trên lá có vết giống bệnh): " + "; ".join(gate["reasons"]) + "."
                ),
            }
        )
        plan = {
            **plan,
            "priority_note": "Hệ thống chỉ chẩn đoán bệnh trên lá điều. Ảnh này không giống lá điều trong dữ liệu huấn luyện.",
        }
        d_top = int(np.argmax(densenet_probs))
        return {
            "success": True,
            "conclusion": conclusion,
            "diseases_detected": [],
            "boxes": [],
            "total_lesion_boxes": 0,
            "total_damage_area_percent": 0.0,
            "treatment_plan": plan,
            "probabilities_desc": probabilities_list(self.classes, densenet_probs),
            "top2": [],
            "abstain": False,
            "quality": quality,
            "ood_gate": gate,
            "decision": {
                "final_source": "ood_gate",
                "densenet_label": self.classes[d_top],
                "densenet_conf": float(densenet_probs[d_top]),
                "stacker_label": None,
                "stacker_conf": None,
                "agree": None,
            },
            "localization": {"skipped": "ood_gate"},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "image_meta": {"width": w, "height": h, "format": orig_format},
            "model_meta": {
                "name": EXP034_MODEL_NAME,
                "version": "EXP-034",
                "engine": "exp034",
                "final_decision": "Cổng ngoại lai (DenseNet + kNN + Energy)",
                "seeds": self.seeds,
                "architecture": "Cổng ngoại lai → YOLO26s đề xuất vùng → DenseNet121 ×5 xác minh & kết luận",
            },
        }

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------
    def predict_image(
        self, image_bytes: bytes, sample_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        YOLO proposes, DenseNet decides:
        1. Decode + EXIF transpose; quality gate; coarse leaf mask.
        2. DenseNet121 x5 on the whole leaf -> final diagnosis (FINAL_DECISION
           = "densenet"); the frozen Stacker F3 is computed as a reference
           (or used as the final rule when FINAL_DECISION = "stacker").
        3. YOLO26s (640px pass + tiled pass) only proposes lesion regions.
        4. DenseNet classifies context crops around the proposals; each box is
           labelled by DenseNet and compared with YOLO's label.
        """
        t0 = time.perf_counter()
        image, orig_format = load_image(image_bytes)
        orig_w, orig_h = image.size

        leaf = compute_leaf_mask(image)
        quality = assess_quality(image, leaf)
        if quality["reject"]:
            raise ValueError(quality["reject"])

        # 1. Whole-image DenseNet ensemble (probs + Energy + embeddings). The
        #    h-flipped copy is used ONLY for the OOD embedding; the diagnosis
        #    uses the original image's probabilities (unchanged protocol).
        x = prepare_input(image)
        whole = self._forward(np.stack([x, x[:, ::-1, :]]))
        densenet_probs = whole["probs"][0]

        # 0. OOD gate -- HARD rule: on rejection YOLO is never run and nothing
        #    downstream (crop verification, stacker) can override it.
        gate = self._ood_gate(
            densenet_probs, float(whole["energy"][0]), whole["feats"][0], whole["feats"][1]
        )
        if gate["rejected"]:
            return self._ood_rejection_result(image, orig_format, quality, densenet_probs, gate, t0)

        # 2. YOLO 640 pass (frozen protocol) -> stacker reference features
        full_boxes = self._yolo_full_pass(image)
        feats = []
        for prefix in ("max_conf", "n_boxes", "area_frac"):
            for cls in self.detector_classes:
                sub = [b for b in full_boxes if b["cls"] == cls]
                if prefix == "max_conf":
                    feats.append(max((b["conf"] for b in sub), default=0.0))
                elif prefix == "n_boxes":
                    feats.append(float(len(sub)))
                else:
                    a = sum((b["xyxy"][2] - b["xyxy"][0]) * (b["xyxy"][3] - b["xyxy"][1]) for b in sub)
                    feats.append(min(a / max(1.0, float(orig_w * orig_h)), 1.0))
        feature = np.array(list(densenet_probs) + feats, dtype=float)
        fitted = self.stacker.predict_proba(feature[None, :])[0]
        p_stack = np.zeros(len(self.classes), dtype=float)
        p_stack[self.stacker[-1].classes_] = fitted
        p_stack = np.clip(p_stack, 1e-9, 1.0) ** (1.0 / self.temperature)
        p_stack = p_stack / p_stack.sum()

        # 3. Final decision
        use_dense = FINAL_DECISION.lower() != "stacker"
        p = densenet_probs if use_dense else p_stack
        threshold = DENSENET_ABSTAIN_THRESHOLD if use_dense else self.abstain_threshold
        order = np.argsort(-p)
        primary = self.classes[order[0]]
        confidence = float(p[order[0]])
        ranked = [(self.classes[i], float(p[i])) for i in order]
        abstain, abstain_reason = abstain_decision(ranked, threshold, ABSTAIN_MIN_MARGIN)

        # 4. YOLO proposals -> DenseNet verification
        n_tiles = n_crops = n_offleaf = 0
        shown_raw: List[Dict[str, Any]] = []
        confirmed: List[str] = []
        vstats: Dict[str, int] = {}
        if not abstain and primary != "not_cashew_leaf":
            tiled, n_tiles = self._yolo_tiled_pass(image) if primary != "healthy" else ([], 0)
            cands = [
                b for b in full_boxes + tiled
                if b["conf"] >= BOX_CONF_PER_CLASS.get(b["cls"], self.box_conf)
            ]
            cands = merge_boxes(cands, MERGE_IOU, MERGE_IOS)
            if LEAF_MASK_FILTER and leaf is not None:
                on_leaf = [b for b in cands if leaf.box_inside_fraction(b["xyxy"]) >= LEAF_MASK_MIN_INSIDE]
                n_offleaf = len(cands) - len(on_leaf)
                cands = on_leaf
            cands.sort(key=lambda b: -b["conf"])
            verified, n_crops = self._verify_boxes(image, cands)

            if primary == "healthy":
                # Whole leaf says healthy: only flag a conflict, draw nothing.
                diseased = {b["verify"]["crop_id"] for b in verified
                            if b["verify"]["status"] in ("agree", "differ")
                            and b["verify"]["dense_conf"] >= SECONDARY_CROP_MIN_CONF}
                if len(diseased) >= 2:
                    abstain = True
                    abstain_reason = (
                        "DenseNet toàn lá kết luận khỏe nhưng xác nhận dấu hiệu bệnh ở "
                        f"{len(diseased)} vùng YOLO đề xuất"
                    )
                vstats = {"proposed": len(verified)}
            else:
                shown_raw, confirmed, vstats = select_verified_boxes(verified, primary, self.max_draw_boxes)

        boxes = format_boxes(image, shown_raw)
        damage, basis = damage_percent(boxes, orig_w, orig_h, leaf)
        conclusion, diseases, plan = build_report(
            primary, confidence, abstain, abstain_reason, boxes, confirmed, damage, basis
        )

        s_top = int(np.argmax(p_stack))
        d_top = int(np.argmax(densenet_probs))
        return {
            "success": True,
            "conclusion": conclusion,
            "diseases_detected": diseases,
            "boxes": boxes,
            "total_lesion_boxes": len(boxes),
            "total_damage_area_percent": conclusion.get("total_damage_area_percent", 0.0),
            "treatment_plan": plan,
            "probabilities_desc": probabilities_list(self.classes, p),
            "top2": ranked[:2],
            "abstain": abstain,
            "quality": quality,
            "ood_gate": gate,
            "decision": {
                "final_source": "densenet" if use_dense else "stacker",
                "densenet_label": self.classes[d_top],
                "densenet_conf": float(densenet_probs[d_top]),
                "stacker_label": self.classes[s_top],
                "stacker_conf": float(p_stack[s_top]),
                "agree": d_top == s_top,
            },
            "localization": {
                "full_pass_boxes": len(full_boxes),
                "tiles": n_tiles,
                "verify_crops": n_crops,
                "removed_off_leaf": n_offleaf,
                "leaf_mask": leaf is not None,
                "verification": vstats,
            },
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "image_meta": {"width": orig_w, "height": orig_h, "format": orig_format},
            "model_meta": {
                "name": EXP034_MODEL_NAME,
                "version": "EXP-034",
                "engine": "exp034",
                "final_decision": "DenseNet121 ×5 (toàn lá)" if use_dense else "Stacker F3",
                "seeds": self.seeds,
                "abstain_threshold": threshold,
                "abstain_min_margin": ABSTAIN_MIN_MARGIN,
                "test_accuracy": EXP034_METRICS["test_accuracy"],
                "test_macro_f1": EXP034_METRICS["test_macro_f1"],
                "holdout_accuracy": EXP034_METRICS["holdout_accuracy"],
                "architecture": "YOLO26s đề xuất vùng → DenseNet121 ×5 xác minh từng vùng & kết luận",
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "engine": "exp034",
            "name": EXP034_MODEL_NAME,
            "version": "EXP-034",
            "final_decision": FINAL_DECISION,
            "rule": self.rule,
            "seeds": self.seeds,
            "val_accuracy": EXP034_METRICS["validation_clean_accuracy"],
            "test_accuracy": EXP034_METRICS["test_accuracy"],
            "test_macro_f1": EXP034_METRICS["test_macro_f1"],
            "holdout_accuracy": EXP034_METRICS["holdout_accuracy"],
            "classes": [
                {
                    "class_key": k,
                    **DISEASE_INFO.get(k, {}),
                }
                for k in self.classes
            ],
        }
