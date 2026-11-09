"""
Shared image utilities for the inference services.

- load_image: decode + EXIF transpose (server frame == browser frame).
- predownscale: antialiased shrink of very large photos before the classifier.
- compute_leaf_mask: coarse leaf-blade mask (used to reject boxes on the
  background and to measure damage relative to the leaf, not the photo).
- assess_quality: input resolution check (reject < 64px, warn < 224px).
- tile_windows, merge_boxes: tiled YOLO inference helpers.
- damage_percent: union of boxes over the leaf area.
"""
import io
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage as ndi
from scipy.spatial import ConvexHull

from config import QUALITY


# ----------------------------------------------------------------------------
# Decoding
# ----------------------------------------------------------------------------
def load_image(image_bytes: bytes) -> Tuple[Image.Image, str]:
    """Decode bytes, apply EXIF orientation and convert to RGB."""
    try:
        raw = Image.open(io.BytesIO(image_bytes))
        fmt = raw.format or "JPEG"
        img = ImageOps.exif_transpose(raw).convert("RGB")
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Không thể đọc định dạng hình ảnh: {e}")
    return img, fmt


def predownscale(img: Image.Image, max_side: int) -> Image.Image:
    """Antialiased shrink so that the long side is <= max_side (aspect kept).

    tf.image.resize (bilinear, antialias=False) samples only a sparse grid of
    pixels when shrinking a 4000px photo straight to 256px, which produces
    heavy aliasing that the classifier never saw in training. Shrinking first
    with a proper low-pass filter removes that artefact.
    """
    if not max_side or max(img.size) <= max_side:
        return img
    s = max_side / float(max(img.size))
    size = (max(1, round(img.width * s)), max(1, round(img.height * s)))
    return img.resize(size, Image.Resampling.LANCZOS)


# ----------------------------------------------------------------------------
# Leaf mask
# ----------------------------------------------------------------------------
@dataclass
class LeafMask:
    mask: np.ndarray  # bool (h, w) at working resolution
    scale: float      # working pixels per original pixel

    @property
    def fraction(self) -> float:
        return float(self.mask.mean())

    def box_inside_fraction(self, xyxy: Sequence[float]) -> float:
        h, w = self.mask.shape
        x1 = int(np.clip(np.floor(xyxy[0] * self.scale), 0, w - 1))
        y1 = int(np.clip(np.floor(xyxy[1] * self.scale), 0, h - 1))
        x2 = int(np.clip(np.ceil(xyxy[2] * self.scale), x1 + 1, w))
        y2 = int(np.clip(np.ceil(xyxy[3] * self.scale), y1 + 1, h))
        return float(self.mask[y1:y2, x1:x2].mean())


def _disk(radius: int) -> np.ndarray:
    r = max(1, int(radius))
    y, x = np.ogrid[-r : r + 1, -r : r + 1]
    return (x * x + y * y) <= r * r


def _rgb_to_lab(arr: np.ndarray) -> np.ndarray:
    """sRGB (0..255) -> CIE Lab (D65)."""
    c = arr / 255.0
    c = np.where(c > 0.04045, ((c + 0.055) / 1.055) ** 2.4, c / 12.92)
    m = np.array(
        [[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]],
        dtype=np.float32,
    )
    xyz = c @ m.T / np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16.0 / 116.0)
    L = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return np.stack([L, a, b], axis=-1)


def _kmeans(x: np.ndarray, k: int, iters: int = 12) -> np.ndarray:
    """Tiny deterministic k-means (k-means++-like farthest-point init)."""
    centers = [x[0]]
    for _ in range(1, k):
        d = np.min([np.sum((x - c) ** 2, axis=1) for c in centers], axis=0)
        centers.append(x[int(np.argmax(d))])
    c = np.array(centers)
    for _ in range(iters):
        lab = np.argmin(((x[:, None, :] - c[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            if np.any(lab == j):
                c[j] = x[lab == j].mean(0)
    return c


def compute_leaf_mask(img: Image.Image, work_max_side: int = 384) -> Optional[LeafMask]:
    """Coarse leaf mask = everything that is not background and is connected
    to the green blade.

    The background colour model is learnt from the non-green pixels of the
    image border (paper, sky, soil, table...), clustered in Lab. A pixel is
    foreground when it is far (Delta E) from every background cluster.
    Foreground components that contain enough green are kept, then closed
    and hole-filled so that lesions of any colour (white dots, silvery mines,
    brown/black necrosis, orange rust) stay inside the mask.

    Returns None when the mask cannot be trusted (too little green, or the
    leaf fills the whole frame so there is no background to learn); callers
    must then skip mask-based filtering.
    """
    s = min(1.0, work_max_side / float(max(img.size)))
    small = img if s >= 1.0 else img.resize(
        (max(8, round(img.width * s)), max(8, round(img.height * s))),
        Image.Resampling.BILINEAR,
    )
    s_x = small.width / float(img.width)
    arr = np.asarray(small, dtype=np.float32)
    h, w = arr.shape[:2]
    side = max(h, w)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    green = (g > r + 2) & (g > b + 4) & (g > 30)
    if green.mean() < 0.03:
        return None

    lab = _rgb_to_lab(arr)
    bw = max(2, int(side * 0.04))
    border = np.zeros((h, w), dtype=bool)
    border[:bw, :] = border[-bw:, :] = True
    border[:, :bw] = border[:, -bw:] = True
    bg_samples = lab[border & ~green]
    if bg_samples.shape[0] < 0.15 * border.sum():
        return None  # leaf touches most of the frame: nothing to learn
    if bg_samples.shape[0] > 3000:
        idx = np.linspace(0, bg_samples.shape[0] - 1, 3000).astype(int)
        bg_samples = bg_samples[idx]
    centers = _kmeans(bg_samples.astype(np.float32), k=3)

    dist = np.min(
        np.stack([np.sqrt(((lab - c) ** 2).sum(-1)) for c in centers]), axis=0
    )
    fg = (dist > 18.0) | green
    fg = ndi.binary_opening(fg, structure=_disk(1))

    labels, n = ndi.label(fg)
    if n == 0:
        return None
    idx = np.arange(1, n + 1)
    sizes = ndi.sum(fg, labels, index=idx)
    greens = ndi.sum(green, labels, index=idx)
    keep = idx[(sizes >= sizes.max() * 0.10) & (greens >= 0.10 * sizes)]
    if keep.size == 0:
        return None
    # Cashew blades are obovate (nearly convex): the convex hull of each kept
    # component recovers heavily diseased margins whose colour drifted towards
    # the background (e.g. dense white speckling on a grey/beige backdrop).
    hull_img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(hull_img)
    for lbl in keep:
        ys, xs = np.nonzero(labels == lbl)
        pts = np.stack([xs, ys], axis=1).astype(np.float64)
        try:
            hull = ConvexHull(pts)
            poly = [tuple(map(float, pts[v])) for v in hull.vertices]
            draw.polygon(poly, fill=255)
        except Exception:  # degenerate (collinear) component
            hull_img.paste(255, mask=Image.fromarray((labels == lbl).astype(np.uint8) * 255))
    leaf = np.asarray(hull_img) > 0
    leaf = ndi.binary_dilation(leaf, structure=_disk(max(1, int(side * 0.01))))
    if leaf.mean() > 0.97:
        return None
    return LeafMask(mask=leaf, scale=s_x)


# ----------------------------------------------------------------------------
# Quality gate
# ----------------------------------------------------------------------------
def assess_quality(img: Image.Image) -> Dict[str, Any]:
    """Input size check only: {"reject": str|None, "warnings": [...], "metrics": {...}}.

    Blur / exposure / leaf-coverage are NOT checked here: the user is asked on
    the upload panel to provide a sharp, well-lit, unobstructed photo.
    """
    q = QUALITY
    w, h = img.size
    short = min(w, h)
    warnings: List[str] = []
    reject = None
    if short < q["min_side_reject"]:
        reject = (
            f"Ảnh quá nhỏ ({w}×{h}px). Cần cạnh ngắn tối thiểu "
            f"{q['min_side_reject']}px để chẩn đoán."
        )
    elif short < q["min_side_warn"]:
        warnings.append(
            f"Độ phân giải thấp ({w}×{h}px): các đốm bệnh nhỏ có thể không được phát hiện. "
            f"Nên dùng ảnh có cạnh ngắn ≥ {q['min_side_warn']}px."
        )
    return {"reject": reject, "warnings": warnings, "metrics": {"width": w, "height": h}}


# ----------------------------------------------------------------------------
# Tiling + box merging
# ----------------------------------------------------------------------------
def _starts(length: int, tile: int, step: int) -> List[int]:
    if length <= tile:
        return [0]
    out = list(range(0, length - tile, step))
    out.append(length - tile)
    return sorted(set(out))


def tile_windows(w: int, h: int, tile: int, overlap: float) -> List[Tuple[int, int, int, int]]:
    step = max(1, int(round(tile * (1.0 - overlap))))
    return [
        (x, y, min(w, x + tile), min(h, y + tile))
        for y in _starts(h, tile, step)
        for x in _starts(w, tile, step)
    ]


def _area(b: Sequence[float]) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _inter(a: Sequence[float], b: Sequence[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    i = _inter(a, b)
    u = _area(a) + _area(b) - i
    return i / u if u > 0 else 0.0


def merge_boxes(
    boxes: List[Dict[str, Any]], iou_thr: float, ios_thr: float
) -> List[Dict[str, Any]]:
    """Class-wise greedy NMS over the union of full-image and tile detections.

    A box is also suppressed when it lies mostly inside a higher-confidence box
    of the same class (intersection / smaller area > ios_thr): that removes the
    truncated duplicates produced where a lesion crosses a tile border.
    """
    kept: List[Dict[str, Any]] = []
    for box in sorted(boxes, key=lambda b: -b["conf"]):
        ok = True
        for k in kept:
            if k["cls"] != box["cls"]:
                continue
            if iou(box["xyxy"], k["xyxy"]) > iou_thr:
                ok = False
                break
            # Containment rule only involves tile boxes (border-cut duplicates);
            # nested boxes from the single full pass are kept as YOLO gave them.
            if "yolo_tile" not in (box.get("source"), k.get("source")):
                continue
            small = min(_area(box["xyxy"]), _area(k["xyxy"]))
            if small > 0 and _inter(box["xyxy"], k["xyxy"]) / small > ios_thr:
                ok = False
                break
        if ok:
            kept.append(box)
    return kept


# ----------------------------------------------------------------------------
# Damage estimate
# ----------------------------------------------------------------------------
def damage_percent(
    boxes: List[Dict[str, Any]], w: int, h: int, leaf: Optional[LeafMask]
) -> Tuple[float, str]:
    """Union of boxes as % of the leaf area (or of the image if no leaf mask).

    Boxes are rectangles, so this is an upper bound of the lesion area; it is
    still far closer than summing overlapping box areas over the whole photo.
    """
    if not boxes or w <= 0 or h <= 0:
        return 0.0, ("leaf" if leaf is not None else "image")
    if leaf is not None:
        mh, mw = leaf.mask.shape
        s = leaf.scale
        base = leaf.mask
    else:
        s = min(1.0, 512.0 / max(w, h))
        mw, mh = max(1, round(w * s)), max(1, round(h * s))
        base = np.ones((mh, mw), dtype=bool)
    cover = np.zeros((mh, mw), dtype=bool)
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        gx1 = int(np.clip(np.floor(x1 * s), 0, mw))
        gy1 = int(np.clip(np.floor(y1 * s), 0, mh))
        gx2 = int(np.clip(np.ceil(x2 * s), 0, mw))
        gy2 = int(np.clip(np.ceil(y2 * s), 0, mh))
        cover[gy1:gy2, gx1:gx2] = True
    denom = max(1, int(base.sum()))
    pct = 100.0 * float((cover & base).sum()) / denom
    return round(min(100.0, pct), 2), ("leaf" if leaf is not None else "image")
