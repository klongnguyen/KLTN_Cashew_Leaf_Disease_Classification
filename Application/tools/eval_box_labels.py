"""
Box-level label study for the YOLO -> crop -> DenseNet design (roadmap steps 3-4).

For every image of a YOLO split (labels available) this script
  1. generates candidate boxes exactly like the web app (640 pass + tiled pass,
     per-class conf threshold, class-wise merge, leaf-mask filter), keeps the
     TOP_K most confident (the web verifies at most 16 crops per image);
  2. matches each candidate to the ground-truth boxes (greedy, IoU >= 0.5)
     -> gt_cls, or "background" for a false positive;
  3. classifies one crop PER BOX (no window sharing) for several padding
     configurations with the DenseNet x5 ensemble and stores the raw 5-class
     probabilities;
  4. stores the whole-image DenseNet probabilities and the image-level
     ground-truth disease set.

Nothing is decided here: thresholds/rules are chosen afterwards on Validation
from the CSVs (tools/analyze_box_labels.py) and only reported on Test.

Usage (from Application/):
    python tools/eval_box_labels.py --root <YOLO dataset dir> --split val --out <dir>
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import BOX_CONF_PER_CLASS, LEAF_MASK_MIN_INSIDE, MERGE_IOS, MERGE_IOU  # noqa: E402
from exp034_service import EXP034Service  # noqa: E402
from image_utils import compute_leaf_mask, iou, load_image, merge_boxes  # noqa: E402

# (name, padding factor on the box's long side, minimum crop side in px)
CROP_CONFIGS = [("p125_m64", 1.25, 64), ("p150_m96", 1.5, 96), ("p200_m128", 2.0, 128), ("p300_m128", 3.0, 128)]
TOP_K = 16
MATCH_IOU = 0.5
CLASSES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]


def read_gt(label_path: Path, names, w: int, h: int):
    out = []
    if not label_path.exists():
        return out
    for line in label_path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        c, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
        out.append({"cls": names[c], "xyxy": [(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h]})
    return out


def square_window(w, h, xyxy, pad, min_px):
    x1, y1, x2, y2 = xyxy
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    side = min(max(max(x2 - x1, y2 - y1) * pad, float(min_px)), float(min(w, h)))
    left = min(max(0.0, cx - side / 2), w - side)
    top = min(max(0.0, cy - side / 2), h - side)
    return tuple(int(round(v)) for v in (left, top, left + side, top + side))


def candidates(svc, image, leaf):
    full = svc._yolo_full_pass(image)
    tiled, _ = svc._yolo_tiled_pass(image)
    c = [b for b in full + tiled if b["conf"] >= BOX_CONF_PER_CLASS.get(b["cls"], svc.box_conf)]
    c = merge_boxes(c, MERGE_IOU, MERGE_IOS)
    if leaf is not None:
        c = [b for b in c if leaf.box_inside_fraction(b["xyxy"]) >= LEAF_MASK_MIN_INSIDE]
    c.sort(key=lambda b: -b["conf"])
    return c


def match(cands, gts):
    used, res = set(), []
    for b in cands:  # already sorted by confidence
        best, bi = 0.0, None
        for i, g in enumerate(gts):
            if i in used:
                continue
            v = iou(b["xyxy"], g["xyxy"])
            if v > best:
                best, bi = v, i
        if bi is not None and best >= MATCH_IOU:
            used.add(bi)
            res.append((gts[bi]["cls"], best))
        else:
            res.append(("background", best))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sample", type=int, default=0, help="stratified sample size (by filename class prefix), seed 2026")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    svc = EXP034Service()
    names = svc.detector_classes
    imgs = sorted(p for p in (root / "images" / a.split).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if a.sample and a.sample < len(imgs):
        import random
        from collections import defaultdict
        groups = defaultdict(list)
        for q in imgs:
            groups[q.stem.rsplit("_", 1)[0]].append(q)
        rng = random.Random(2026)
        picked = []
        for key in sorted(groups):
            g = groups[key]
            picked += rng.sample(g, max(1, round(a.sample * len(g) / len(imgs))))
        imgs = sorted(picked)
        print("stratified sample:", {k: len(v) for k, v in groups.items()}, "->", len(imgs), flush=True)
    if a.limit:
        imgs = imgs[: a.limit]

    box_f = open(out / f"boxes_{a.split}.csv", "w", newline="", encoding="utf-8")
    img_f = open(out / f"images_{a.split}.csv", "w", newline="", encoding="utf-8")
    bw = csv.writer(box_f)
    iw = csv.writer(img_f)
    prob_cols = [f"{cfg}_{c}" for cfg, _, _ in CROP_CONFIGS for c in CLASSES]
    bw.writerow(["image", "rank", "source", "yolo_cls", "yolo_conf", "gt_cls", "match_iou", "box_w", "box_h", *prob_cols])
    iw.writerow(["image", "w", "h", "n_gt", "gt_set", "n_cand", *[f"whole_{c}" for c in CLASSES]])

    t0 = time.time()
    for n, p in enumerate(imgs, 1):
        image, _ = load_image(p.read_bytes())
        w, h = image.size
        gts = read_gt(root / "labels" / a.split / (p.stem + ".txt"), names, w, h)
        leaf = compute_leaf_mask(image)
        whole = svc._densenet_batch([image])[0]
        cands = candidates(svc, image, leaf)[:TOP_K]
        m = match(cands, gts)
        crops = [image.crop(square_window(w, h, b["xyxy"], pad, mpx)) for b in cands for _, pad, mpx in CROP_CONFIGS]
        probs = svc._densenet_batch(crops) if crops else np.zeros((0, len(CLASSES)))
        k = len(CROP_CONFIGS)
        for r, (b, (gcls, mi)) in enumerate(zip(cands, m)):
            x1, y1, x2, y2 = b["xyxy"]
            pr = probs[r * k:(r + 1) * k].reshape(-1)
            bw.writerow([p.name, r, b["source"], b["cls"], round(b["conf"], 4), gcls, round(mi, 3),
                         round(x2 - x1, 1), round(y2 - y1, 1), *[round(float(v), 5) for v in pr]])
        iw.writerow([p.name, w, h, len(gts), "|".join(sorted({g["cls"] for g in gts})), len(cands),
                     *[round(float(v), 5) for v in whole]])
        if n % 20 == 0:
            box_f.flush(); img_f.flush()
            el = time.time() - t0
            print(f"[{a.split}] {n}/{len(imgs)}  {el / n:.2f}s/img  eta {el / n * (len(imgs) - n) / 60:.1f} min", flush=True)
    box_f.close(); img_f.close()
    print("done", a.split, len(imgs), f"{(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
