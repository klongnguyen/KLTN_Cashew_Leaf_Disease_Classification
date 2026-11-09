"""
Legacy engine: DenseNet121 Transfer EXP-014 classifier + colour-cue
"suspected regions" (Cách 1).

The diagnosis comes only from the classifier. The colour heuristic is not a
classifier: its regions are drawn as unscored suspected areas of the predicted
disease and never change the label or the confidence.
"""
import json
import time
import logging
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter
from tensorflow import keras

from config import (
    MODEL_PATH,
    LABEL_MAP_PATH,
    METADATA_PATH,
    FALLBACK_MODEL_PATH,
    FALLBACK_LABEL_MAP,
    IMG_SIZE,
    MODEL_NAME,
    BEST_SEED,
    VAL_ACCURACY,
    VAL_LOSS,
    VAL_MACRO_F1,
    DISEASE_INFO,
    ALLOW_SCRATCH_FALLBACK,
    SCRATCH_FALLBACK_NAME,
    LEGACY_ABSTAIN_THRESHOLD,
    ABSTAIN_MIN_MARGIN,
    CLASSIFIER_PREDOWNSCALE_MAX_SIDE,
    LEAF_MASK_FILTER,
    LEAF_MASK_MIN_INSIDE,
)
from diagnosis_report import (
    DISEASE_CLASSES,
    abstain_decision,
    build_report,
    format_boxes,
    probabilities_list,
)
from image_utils import (
    assess_quality,
    compute_leaf_mask,
    damage_percent,
    load_image,
    predownscale,
)

logger = logging.getLogger("CashewLeafAI")
logging.basicConfig(level=logging.INFO)


def _binary_dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Fast 2D morphological disk dilation in pure NumPy."""
    if radius <= 0:
        return mask.copy()
    h, w = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy * dy + dx * dx <= radius * radius + 1:
                out |= padded[radius + dy : radius + dy + h, radius + dx : radius + dx + w]
    return out


def _binary_erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Fast 2D morphological disk erosion in pure NumPy."""
    return ~_binary_dilate(~mask, radius)


def _binary_close(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological closing (dilation followed by erosion) to fill internal holes."""
    return _binary_erode(_binary_dilate(mask, radius), radius)


def _binary_open(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Morphological opening (erosion followed by dilation) to remove speckle noise."""
    return _binary_dilate(_binary_erode(mask, radius), radius)


def _connected_components(
    mask: np.ndarray, min_area: int = 6, use_4conn: bool = False
) -> List[Dict[str, Any]]:
    """Connected component labeling via BFS in pure NumPy (supports 4-conn and 8-conn)."""
    h, w = mask.shape
    visited = np.zeros((h, w), dtype=bool)
    comps: List[Dict[str, Any]] = []
    ys, xs = np.nonzero(mask)
    for sy, sx in zip(ys, xs):
        if visited[sy, sx]:
            continue
        q = deque([(int(sy), int(sx))])
        visited[sy, sx] = True
        pixels: List[Tuple[int, int]] = []
        min_x = max_x = int(sx)
        min_y = max_y = int(sy)
        while q:
            cy, cx = q.popleft()
            pixels.append((cy, cx))
            if cx < min_x:
                min_x = cx
            if cx > max_x:
                max_x = cx
            if cy < min_y:
                min_y = cy
            if cy > max_y:
                max_y = cy
            if use_4conn:
                neighbors = ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1))
            else:
                neighbors = (
                    (cy - 1, cx - 1), (cy - 1, cx), (cy - 1, cx + 1),
                    (cy, cx - 1),                   (cy, cx + 1),
                    (cy + 1, cx - 1), (cy + 1, cx), (cy + 1, cx + 1),
                )
            for ny, nx in neighbors:
                if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx] and mask[ny, nx]:
                    visited[ny, nx] = True
                    q.append((ny, nx))
        if len(pixels) >= min_area:
            comps.append(
                {
                    "area": len(pixels),
                    "bbox": [min_x, min_y, max_x, max_y],
                    "pixels": pixels,
                }
            )
    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def _merge_boxes_by_class(
    boxes: List[Dict[str, Any]],
    max_gap_x: int,
    max_gap_y: int,
    min_fill_density: float = 0.25,
    scale: float = 1.0,
) -> List[Dict[str, Any]]:
    """
    Agglomeratively merges nearby or overlapping bounding boxes of the same disease class,
    guarded by a minimum fill-density check so boxes never expand across healthy leaf tissue.
    """
    merged = [dict(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        new_list: List[Dict[str, Any]] = []
        used = [False] * len(merged)
        for i in range(len(merged)):
            if used[i]:
                continue
            cur = merged[i]
            bx1, by1, bx2, by2 = cur["xyxy"]
            for j in range(i + 1, len(merged)):
                if used[j] or merged[j]["cls"] != cur["cls"]:
                    continue
                ox1, oy1, ox2, oy2 = merged[j]["xyxy"]
                gap_x = max(0, max(bx1, ox1) - min(bx2, ox2))
                gap_y = max(0, max(by1, oy1) - min(by2, oy2))
                if gap_x <= max_gap_x and gap_y <= max_gap_y:
                    nx1, ny1 = min(bx1, ox1), min(by1, oy1)
                    nx2, ny2 = max(bx2, ox2), max(by2, oy2)
                    total_a = cur["pixel_area"] + merged[j]["pixel_area"]
                    rect_a_scaled = max(1.0, (nx2 - nx1) * (ny2 - ny1) * (scale * scale))
                    if (total_a / rect_a_scaled) >= min_fill_density:
                        cur["conf"] = max(cur["conf"], merged[j]["conf"])
                        cur["pixel_area"] = total_a
                        bx1, by1, bx2, by2 = nx1, ny1, nx2, ny2
                        cur["xyxy"] = [bx1, by1, bx2, by2]
                        used[j] = True
                        changed = True
            new_list.append(cur)
        merged = new_list
    return merged


class DenseNet121Service:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DenseNet121Service, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        primary_ok = MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000
        if primary_ok:
            active_model_path = MODEL_PATH
            using_fallback = False
            logger.info(f"Loading DenseNet121 EXP-014 model from: {active_model_path}")
        elif ALLOW_SCRATCH_FALLBACK and FALLBACK_MODEL_PATH.exists() and FALLBACK_MODEL_PATH.stat().st_size > 1_000_000:
            active_model_path = FALLBACK_MODEL_PATH
            using_fallback = True
            logger.warning(
                "ALLOW_SCRATCH_FALLBACK=True: serving the obsolete EXP-DENSENET121-SCRATCH-001 "
                "(dataset v01) checkpoint. Results are for debugging only."
            )
        else:
            size = MODEL_PATH.stat().st_size if MODEL_PATH.exists() else 0
            raise FileNotFoundError(
                f"Legacy model not available: {MODEL_PATH} "
                f"({'missing' if not MODEL_PATH.exists() else f'{size} bytes — Git LFS stub, run git lfs pull'}). "
                "The obsolete scratch checkpoint is not served silently "
                "(set ALLOW_SCRATCH_FALLBACK=True in config.py only for debugging)."
            )
        self.using_fallback = using_fallback
        self.display_name = SCRATCH_FALLBACK_NAME if using_fallback else MODEL_NAME

        # Load model (contains internal preprocessing/rescaling)
        self.model = keras.models.load_model(str(active_model_path), compile=False)

        # Optional Class Activation Mapping (CAM) extractor from last 4D feature layer
        self.cam_extractor = None
        self.cam_weights = None
        try:
            last_conv_layer = None
            dense_weights = None
            for layer in reversed(self.model.layers):
                if dense_weights is None and isinstance(layer, keras.layers.Dense):
                    w = layer.get_weights()
                    if w and len(w[0].shape) == 2 and w[0].shape[1] == len(FALLBACK_LABEL_MAP):
                        dense_weights = w[0]
                out_shape = getattr(layer, "output_shape", None)
                if out_shape is None and hasattr(layer, "output"):
                    out_shape = layer.output.shape
                if out_shape is not None and len(out_shape) == 4 and out_shape[1] and out_shape[1] >= 4:
                    last_conv_layer = layer
                    break
            if last_conv_layer is not None:
                self.cam_extractor = keras.Model(
                    inputs=self.model.inputs,
                    outputs=last_conv_layer.output,
                )
                self.cam_weights = dense_weights
                logger.info(f"Initialized DenseNet121 spatial CAM extractor on layer '{last_conv_layer.name}'")
        except Exception as e:
            logger.info(f"Using direct multi-cue segmentation without separate CAM submodel: {e}")

        # Load label map and metadata corresponding to active model
        if using_fallback:
            self.label_map = FALLBACK_LABEL_MAP
            self.metadata = {
                "experiment_id": "EXP-DENSENET121-SCRATCH-001",
                "selection_source": "validation only (fallback checkpoint in repo)",
            }
        else:
            # Never guess the class order: EXP-014 uses
            # [anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf].
            if not LABEL_MAP_PATH.exists():
                raise FileNotFoundError(f"label_map.json not found next to the model: {LABEL_MAP_PATH}")
            with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
                self.label_map = json.load(f)

            self.metadata = {}
            if METADATA_PATH.exists():
                with open(METADATA_PATH, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)

        # Reverse map: class_key -> class_idx
        self.class_to_idx = {v: int(k) for k, v in self.label_map.items()}

        # Check for optional YOLO detector
        self.detector = None
        yolo_pt_candidates = [
            MODEL_PATH.parent.parent.parent / "yolo_negative_v03" / "EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED" / "weights" / "best.pt",
            MODEL_PATH.parent.parent.parent.parent / "weights" / "best.pt",
            Path(__file__).resolve().parent / "weights" / "best.pt",
        ]
        for pt_path in yolo_pt_candidates:
            if pt_path.exists() and pt_path.stat().st_size > 1_000_000:
                try:
                    from ultralytics import YOLO
                    self.detector = YOLO(str(pt_path))
                    logger.info(f"Loaded YOLO detector from {pt_path}")
                    break
                except Exception as e:
                    logger.warning(f"Could not load YOLO from {pt_path}: {e}")

        # Determine expected model input size dynamically
        in_shape = getattr(self.model, "input_shape", None)
        if in_shape and len(in_shape) == 4 and in_shape[1] is not None:
            self.input_size = int(in_shape[1])
        else:
            self.input_size = IMG_SIZE
        logger.info(f"Model input resolution: {self.input_size}x{self.input_size}")

        # Warm up model to eliminate initial inference latency
        logger.info("Warming up model with dummy input...")
        dummy = np.zeros((1, self.input_size, self.input_size, 3), dtype=np.float32)
        _ = self.model.predict(dummy, verbose=0)
        logger.info("Model loaded and warmed up successfully!")

    def _compute_spatial_saliency(self, x_input: np.ndarray, target_size: Tuple[int, int]) -> Optional[np.ndarray]:
        """Computes normalized 2D spatial activation heatmap from DenseNet121 conv features."""
        if self.cam_extractor is None:
            return None
        try:
            fmap = self.cam_extractor.predict(x_input, verbose=0)[0]  # (Hf, Wf, C)
            saliency = np.mean(np.maximum(0.0, fmap), axis=-1)
            s_min, s_max = float(np.min(saliency)), float(np.max(saliency))
            if s_max > s_min + 1e-6:
                saliency = (saliency - s_min) / (s_max - s_min)
            else:
                return None
            sal_img = Image.fromarray((saliency * 255.0).astype(np.uint8)).resize(
                target_size, Image.Resampling.BILINEAR
            )
            return np.asarray(sal_img, dtype=np.float32) / 255.0
        except Exception:
            return None

    def _adaptive_lesion_segmentation(
        self,
        rgb_image: Image.Image,
        global_probs: Dict[str, float],
        x_input: Optional[np.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Cách 1 — Multi-Cue Leaf Foreground Masking + Chromatic Deviation +
        Connected-Component Labeling & Density-Guarded Box Merging:
        1. Isolates the cashew leaf blade from studio/sky light backgrounds, dark shadows,
           and non-leaf border objects (such as fingers holding the leaf).
        2. Computes the healthy leaf chlorophyll baseline (median R, G, B on healthy leaf tissue)
           and detects pixel-level necrotic/rust/silvery deviations.
        3. Applies morphological opening/closing and 8-connected component labeling to find
           exact lesion boundaries (both small circular spots and large contiguous blights).
        4. Merges proximate components of the same class with a fill-density guard so boxes
           hug lesion boundaries tightly without spilling into healthy leaf areas.
        """
        orig_w, orig_h = rgb_image.size
        target_max = 280
        scale = target_max / max(orig_w, orig_h)
        sw = max(32, int(round(orig_w * scale)))
        sh = max(32, int(round(orig_h * scale)))
        small = rgb_image.resize((sw, sh), Image.Resampling.BILINEAR)
        arr = np.asarray(small, dtype=np.float32)

        r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
        intensity = (r + g + b) / 3.0
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        chroma = max_c - min_c
        sat = chroma / (max_c + 1e-5)

        # Step 1: Background rejection (white/gray studio backdrop, bright sky, deep shadows)
        # Preserve warm tan leaf necrosis where R > G > B and R - B >= 14
        is_warm_tan = (r > g) & (g > b) & ((r - b) >= 14) & (intensity < 205)
        bg_light = (~is_warm_tan) & (
            ((b >= r - 4) & (intensity > 135))
            | ((min_c > 165) & (chroma < 28))
            | (intensity > 212)
        )
        bg_dark = intensity < 28

        # Step 2: Strict Healthy Green Leaf Blade (chlorophyll dominance: G > R + 4 and G > B + 12)
        green_raw = (
            (~bg_light)
            & (~bg_dark)
            & (g > r + 4)
            & (g > b + 12)
            & (g > 38)
            & (intensity < 205)
        )
        green_comps = _connected_components(green_raw, min_area=max(20, int(sw * sh * 0.005)))
        if not green_comps:
            return []

        main_green = np.zeros((sh, sw), dtype=bool)
        max_green_area = green_comps[0]["area"]
        for gc in green_comps:
            if gc["area"] >= max_green_area * 0.18:
                for py, px in gc["pixels"]:
                    main_green[py, px] = True

        r_med = float(np.median(r[main_green]))
        g_med = float(np.median(g[main_green]))
        b_med = float(np.median(b[main_green]))
        rg_base = r_med - g_med  # strongly negative on healthy leaf blade (-20 to -38)
        rb_base = r_med - b_med

        # Step 3: Build Leaf Foreground Masks (Interior & Full Extended Blade)
        # Reject skin/bark border artifacts (low G-B excess where R > G)
        skin_or_dark_bg = ((g - b) < 16) & (r > g) & (intensity < 115) & (~main_green)
        close_radius = max(5, int(max(sw, sh) * 0.05))
        leaf_interior = (
            _binary_close(main_green, radius=close_radius)
            & (~bg_light)
            & (~bg_dark)
            & (~skin_or_dark_bg)
        )

        # Extended full leaf mask: captures large marginal/apical necrotic zones attached to main_green
        warm_necrotic_tissue = (
            (~bg_light)
            & (~bg_dark)
            & (~skin_or_dark_bg)
            & (r >= g * 0.98)
            & ((r - b) >= 14)
            & ((g - b) >= 6)
            & (intensity > 42)
            & (intensity < 202)
        )
        candidate_leaf = main_green | warm_necrotic_tissue
        leaf_comps = _connected_components(candidate_leaf, min_area=40)
        leaf_full = np.zeros((sh, sw), dtype=bool)
        for lc in leaf_comps:
            stride = max(1, len(lc["pixels"]) // 250)
            if any(main_green[py, px] for py, px in lc["pixels"][::stride]):
                for py, px in lc["pixels"]:
                    leaf_full[py, px] = True
        leaf_full = _binary_close(leaf_full, radius=2) & (~bg_light) & (~skin_or_dark_bg)

        # Step 4: Detect abnormal lesion pixels relative to healthy leaf blade & veins
        healthy_blade_or_vein = main_green | (
            leaf_interior & (g >= r - 2) & (g > b + 20) & ((r - g) < rg_base + 24)
        )
        rg_shift = (r - g) - rg_base
        rb_shift = (r - b) - rb_base

        # Optional DenseNet121 spatial saliency map to weight confidence
        saliency_map = (
            self._compute_spatial_saliency(x_input, (sw, sh))
            if x_input is not None
            else None
        )

        # Separate dark-brown scorched necrotic blight (Anthracnose) from warm orange-rust / silvery mines
        dark_necrotic_mask = (
            leaf_full
            & (~healthy_blade_or_vein)
            & (r > g * 1.18)
            & ((g - b) < 28)
            & (intensity < 118)
            & ((r - b) >= 20)
        )
        dark_necrotic_mask = _binary_open(dark_necrotic_mask, radius=1)
        dark_necrotic_mask = _binary_close(dark_necrotic_mask, radius=3)

        total_leaf_pixels = max(1, int(np.sum(leaf_full)))
        min_comp_area = max(9, int(total_leaf_pixels * 0.0013))

        comps_dark = _connected_components(
            dark_necrotic_mask, min_area=max(22, int(total_leaf_pixels * 0.006))
        )
        used_dark = np.zeros((sh, sw), dtype=bool)
        for cd in comps_dark:
            min_x, min_y, max_x, max_y = cd["bbox"]
            if max_x >= int(sw * 0.94) and min_x >= int(sw * 0.82):
                continue
            used_dark[
                max(0, min_y - 4) : min(sh, max_y + 5),
                max(0, min_x - 4) : min(sw, max_x + 5),
            ] = True

        lesion_warm = (
            leaf_full
            & (~healthy_blade_or_vein)
            & (~used_dark)
            & (r >= g - 1)
            & (rg_shift > 22.0)
            & ((r - b) >= 14)
        )
        lesion_silvery = (
            leaf_interior
            & (~healthy_blade_or_vein)
            & (~used_dark)
            & (rg_shift > 16.0)
            & (sat < 0.35)
            & (intensity > 80)
            & (intensity < 198)
        )

        lesion_other = _binary_open(lesion_warm | lesion_silvery, radius=1)
        lesion_other = _binary_close(lesion_other, radius=1)
        comps_other = _connected_components(lesion_other, min_area=min_comp_area)

        disease_order = ["anthracnose", "leaf_miner", "red_rust"]
        top_disease = max(disease_order, key=lambda k: global_probs.get(k, 0.0))
        top_prob = global_probs.get(top_disease, 0.0)

        # Step 4b: High-Sensitivity Pinpoint Red Rust / Algal Spot Detector (Cephaleuros virescens)
        # Dedicated to early-stage pinpoint chlorotic/algal speckles that are smaller than standard min_comp_area
        comps_rust_spots: List[Dict[str, Any]] = []
        is_rust_relevant = (top_disease == "red_rust") or (
            global_probs.get("red_rust", 0.0) >= 0.20
        )
        if is_rust_relevant:
            blur_r = 3
            im_small_pil = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            im_blur = np.asarray(
                im_small_pil.filter(ImageFilter.BoxBlur(radius=blur_r)),
                dtype=np.float32,
            )
            local_int_diff = intensity - np.mean(im_blur, axis=-1)

            pinpoint_rust = (
                main_green
                & (~used_dark)
                & (
                    ((local_int_diff >= 7.0) & (rg_shift >= 5.5) & (rb_shift >= 9.0))
                    | ((r >= g - 1) & (rg_shift >= 16.0) & ((r - b) >= 12))
                )
                & (intensity < 205)
            )
            pinpoint_rust = _binary_open(pinpoint_rust, radius=1)

            raw_rust_comps = _connected_components(
                pinpoint_rust, min_area=4, use_4conn=True
            )
            for rc in raw_rust_comps:
                min_x, min_y, max_x, max_y = rc["bbox"]
                bw = max_x - min_x + 1
                bh = max_y - min_y + 1
                aspect = max(bw, bh) / max(1, min(bw, bh))
                # Filter out long skinny leaf veins/midrib ridges
                if aspect > 3.0 and max(bw, bh) >= int(max(sw, sh) * 0.08):
                    continue
                comps_rust_spots.append(rc)

        raw_boxes: List[Dict[str, Any]] = []

        all_candidate_comps = [(True, c) for c in comps_dark] + [
            (False, c) for c in comps_other
        ]

        for is_dark_comp, comp in all_candidate_comps:
            min_x, min_y, max_x, max_y = comp["bbox"]
            # Reject human thumb/fingers pressed against the right border
            if max_x >= int(sw * 0.94) and min_x >= int(sw * 0.82):
                continue

            pts = comp["pixels"]
            ys = np.array([p[0] for p in pts])
            xs = np.array([p[1] for p in pts])
            c_int, c_sat = intensity[ys, xs], sat[ys, xs]

            area_ratio = comp["area"] / total_leaf_pixels
            silvery_frac = float(np.mean((c_sat < 0.28) & (c_int > 105)))
            sal_boost = (
                float(np.mean(saliency_map[ys, xs])) * 0.04
                if saliency_map is not None
                else 0.02
            )

            if is_dark_comp:
                cls_name = "anthracnose"
                conf = min(0.98, max(0.93, global_probs.get("anthracnose", 0.94)) + sal_boost)
            elif top_disease == "leaf_miner" and top_prob >= 0.55:
                cls_name = "leaf_miner"
                conf = min(0.98, max(0.92, top_prob) + sal_boost)
            elif top_disease == "anthracnose" and top_prob >= 0.55:
                cls_name = "anthracnose"
                conf = min(0.98, max(0.93, top_prob) + sal_boost)
            elif silvery_frac > 0.22 and area_ratio >= 0.012:
                cls_name = "leaf_miner"
                conf = min(0.97, max(0.90, global_probs.get("leaf_miner", 0.91)) + sal_boost)
            elif top_prob >= 0.38:
                cls_name = top_disease
                base_c = top_prob - (0.01 if area_ratio > 0.008 else 0.03)
                conf = min(0.98, max(0.88, base_c + sal_boost))
            else:
                if area_ratio > 0.04:
                    cls_name = "anthracnose"
                elif silvery_frac > 0.18:
                    cls_name = "leaf_miner"
                else:
                    cls_name = "red_rust"
                conf = 0.89 + sal_boost

            pad_x = max(2, int(orig_w * 0.01))
            pad_y = max(2, int(orig_h * 0.01))
            x1 = max(0, int(round(min_x / scale)) - pad_x)
            y1 = max(0, int(round(min_y / scale)) - pad_y)
            x2 = min(orig_w, int(round((max_x + 1) / scale)) + pad_x)
            y2 = min(orig_h, int(round((max_y + 1) / scale)) + pad_y)

            raw_boxes.append(
                {
                    "cls": cls_name,
                    "conf": float(conf),
                    "xyxy": [x1, y1, x2, y2],
                    "pixel_area": comp["area"],
                }
            )

        # Append pinpoint red rust spot candidates
        for comp in comps_rust_spots:
            min_x, min_y, max_x, max_y = comp["bbox"]
            pad_x = max(2, int(orig_w * 0.012))
            pad_y = max(2, int(orig_h * 0.012))
            x1 = max(0, int(round(min_x / scale)) - pad_x)
            y1 = max(0, int(round(min_y / scale)) - pad_y)
            x2 = min(orig_w, int(round((max_x + 1) / scale)) + pad_x)
            y2 = min(orig_h, int(round((max_y + 1) / scale)) + pad_y)
            conf = min(0.99, max(0.92, global_probs.get("red_rust", 0.94)))
            raw_boxes.append(
                {
                    "cls": "red_rust",
                    "conf": float(conf),
                    "xyxy": [x1, y1, x2, y2],
                    "pixel_area": comp["area"],
                }
            )

        # Step 5: Smart Class-Specific Box Merging with Fill-Density Guard
        anth_or_miner = [
            b for b in raw_boxes if b["cls"] in ("anthracnose", "leaf_miner")
        ]
        rust_boxes = [b for b in raw_boxes if b["cls"] == "red_rust"]

        merged_am = _merge_boxes_by_class(
            anth_or_miner,
            max_gap_x=int(orig_w * 0.06),
            max_gap_y=int(orig_h * 0.06),
            min_fill_density=0.22,
            scale=scale,
        )
        merged_rust = _merge_boxes_by_class(
            rust_boxes,
            max_gap_x=int(orig_w * 0.025),
            max_gap_y=int(orig_h * 0.025),
            min_fill_density=0.18,
            scale=scale,
        )

        final_boxes = merged_am + merged_rust
        final_boxes.sort(key=lambda b: b["pixel_area"], reverse=True)
        return final_boxes[:12]

    def predict_image(
        self, image_bytes: bytes, sample_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Legacy diagnosis (DenseNet121 EXP-014 only):
        - the label and confidence come from the classifier, nothing else;
        - abstain on low confidence or small top-1/top-2 margin;
        - colour-cue regions are shown only as unscored "suspected regions"
          of the predicted disease; they never change the diagnosis.
        """
        t0 = time.perf_counter()
        image, orig_format = load_image(image_bytes)
        orig_w, orig_h = image.size
        leaf = compute_leaf_mask(image)
        quality = assess_quality(image)
        if quality["reject"]:
            raise ValueError(quality["reject"])

        dim = getattr(self, "input_size", IMG_SIZE)
        cls_src = predownscale(image, CLASSIFIER_PREDOWNSCALE_MAX_SIDE)
        x = np.asarray(cls_src.resize((dim, dim), Image.Resampling.BILINEAR), dtype=np.float32)[None, ...]
        probs = self.model.predict(x, verbose=0)[0]

        classes = [self.label_map[str(i)] for i in range(len(self.label_map))]
        global_probs = {c: float(probs[i]) for i, c in enumerate(classes)}
        ranked = sorted(global_probs.items(), key=lambda kv: -kv[1])
        primary, confidence = ranked[0]
        abstain, reason = abstain_decision(ranked, LEGACY_ABSTAIN_THRESHOLD, ABSTAIN_MIN_MARGIN)

        raw: List[Dict[str, Any]] = []
        if not abstain and primary in DISEASE_CLASSES:
            regions = self._adaptive_lesion_segmentation(image, global_probs, x_input=x)
            for r in regions:
                raw.append(
                    {"cls": primary, "conf": None, "xyxy": r["xyxy"], "source": "heuristic"}
                )
            if LEAF_MASK_FILTER and leaf is not None:
                raw = [b for b in raw if leaf.box_inside_fraction(b["xyxy"]) >= LEAF_MASK_MIN_INSIDE]
        boxes = format_boxes(image, raw)
        damage, basis = damage_percent(boxes, orig_w, orig_h, leaf)
        conclusion, diseases, plan = build_report(
            primary, confidence, abstain, reason, boxes, [], damage, basis
        )

        warning = None
        if self.using_fallback:
            warning = (
                "Đang dùng checkpoint dự phòng EXP-DENSENET121-SCRATCH-001 (dataset v01 cũ). "
                "Kết quả chỉ để debug, không dùng để chẩn đoán."
            )
        return {
            "success": True,
            "conclusion": conclusion,
            "diseases_detected": diseases,
            "boxes": boxes,
            "total_lesion_boxes": len(boxes),
            "total_damage_area_percent": conclusion.get("total_damage_area_percent", 0.0),
            "treatment_plan": plan,
            "probabilities_desc": probabilities_list(classes, probs),
            "top2": ranked[:2],
            "abstain": abstain,
            "quality": quality,
            "localization": {"method": "heuristic_color_regions", "leaf_mask": leaf is not None},
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "image_meta": {"width": orig_w, "height": orig_h, "format": orig_format},
            "model_meta": {
                "name": self.display_name,
                "engine": "legacy",
                "engine_warning": warning,
                "seed": BEST_SEED,
                "abstain_threshold": LEGACY_ABSTAIN_THRESHOLD,
                "abstain_min_margin": ABSTAIN_MIN_MARGIN,
                "val_accuracy": None if self.using_fallback else VAL_ACCURACY,
                "val_loss": None if self.using_fallback else VAL_LOSS,
                "val_macro_f1": None if self.using_fallback else VAL_MACRO_F1,
                "input_size": f"{dim}x{dim}",
                "architecture": "DenseNet121 (nhãn) + vùng nghi ngờ theo màu sắc (không chấm điểm)",
            },
        }

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "engine": "legacy",
            "name": self.display_name,
            "is_fallback_checkpoint": self.using_fallback,
            "seed": BEST_SEED,
            "experiment_id": self.metadata.get("experiment_id", "EXP-DENSENET121-TRANSFER-5SEEDS-014"),
            "selection_source": self.metadata.get("selection_source", "validation only"),
            "val_accuracy": None if self.using_fallback else VAL_ACCURACY,
            "val_loss": None if self.using_fallback else VAL_LOSS,
            "val_macro_f1": None if self.using_fallback else VAL_MACRO_F1,
            "classes": [{"class_key": k, **DISEASE_INFO.get(k, {})} for k in self.label_map.values()],
        }


# The service is no longer instantiated at import time: engine.py decides
# whether to load this legacy engine or EXP-034 (which only reuses the static
# helpers above and must not pay for loading the legacy model).
def get_legacy_service() -> DenseNet121Service:
    return DenseNet121Service()
