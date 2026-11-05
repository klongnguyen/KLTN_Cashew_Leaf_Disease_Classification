"""
Build the cashew-leaf reference bank and calibrate the three-zone OOD gate.

Protocol (no OOD data and no Test data are used to choose a threshold):
  1. Bank   : for every TRAIN image of the four cashew classes, the DenseNet
              embeddings of all 5 seeds, each the L2-normalised mean of the
              image and its horizontal flip  -> array (N, seeds, 1024).
  2. Score  : 1 - mean cosine over seeds to the k-th nearest bank image.
  3. Thresh.: knn_threshold      = --quantile      (0.99)  of cashew VALIDATION
              knn_threshold_high = --quantile-high (0.999) of cashew VALIDATION
              (scores in between -> accepted only if DenseNet is confidently
              cashew, see config.OOD_BAND_*).
  4. Report : acceptance on cashew Validation, rejection on OOD sets
              (Validation not_cashew_leaf + optional zips) -- reported only.

Usage (from Application/):
  ..\\.venv\\Scripts\\python.exe tools\\build_ood_reference.py ^
      --train-dir "D:\\...\\CashewData_Split\\train" ^
      --val-dir   "D:\\...\\CashewData_Split\\val" ^
      --ood-zip   "D:\\...\\Corn_Merge_Data.zip" --ood-zip-max 150
Memory note: on an 8 GB machine run it with the web server stopped.
"""
import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

from config import (  # noqa: E402
    OOD_BAND_MAX_P_NOT_CASHEW,
    OOD_BAND_MIN_TOP_CASHEW,
    OOD_CALIBRATION_PATH,
    OOD_DIR,
    OOD_KNN_K,
    OOD_REFERENCE_PATH,
)
from exp034_service import EXP034Service, prepare_input  # noqa: E402
from image_utils import load_image  # noqa: E402

CASHEW = ("anthracnose", "healthy", "leaf_miner", "red_rust")
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(folder: Path, classes, max_per_class=None):
    items = []
    for c in classes:
        files = sorted(p for p in (folder / c).glob("*") if p.suffix.lower() in EXTS)
        if max_per_class:
            files = files[:: max(1, len(files) // max_per_class)][:max_per_class]
        items += [(c, p.read_bytes) for p in files]
    return items


def list_zip(zpath: Path, max_per_dir: int):
    zf = zipfile.ZipFile(zpath)
    by_dir = {}
    for n in sorted(zf.namelist()):
        if Path(n).suffix.lower() in EXTS:
            by_dir.setdefault(str(Path(n).parent).replace("\\", "/"), []).append(n)
    items = []
    for d, names in by_dir.items():
        pick = names[:: max(1, len(names) // max_per_dir)][:max_per_dir]
        items += [(f"{zpath.stem}:{Path(d).name}", (lambda n=n: zf.read(n))) for n in pick]
    return items


def embed(svc, items, batch=8, label=""):
    """Returns per-seed flip-averaged embeddings (n, seeds, 1024), probs, tags."""
    feats, probs, tags, bad = [], [], [], 0
    t0 = time.time()
    for i in range(0, len(items), batch):
        xs, ts = [], []
        for tag, reader in items[i : i + batch]:
            try:
                img, _ = load_image(reader())
                xs.append(prepare_input(img))
                ts.append(tag)
            except Exception:  # noqa: BLE001
                bad += 1
        if not xs:
            continue
        k = len(xs)
        out = svc._forward(np.stack(xs + [a[:, ::-1, :] for a in xs]))
        feats.append(svc.flip_average(out["feats"][:k], out["feats"][k:]).astype(np.float32))
        probs.append(out["probs"][:k])
        tags += ts
        done = min(i + batch, len(items))
        if done % (batch * 25) == 0 or done == len(items):
            print(f"  [{label}] {done}/{len(items)}  {time.time() - t0:.0f}s", flush=True)
    return np.concatenate(feats), np.concatenate(probs), np.array(tags), bad


def knn_dist(bank_flat, per_seed, k):
    q = EXP034Service.gate_embedding(per_seed)
    out = np.empty(len(q))
    for i in range(0, len(q), 256):
        sims = q[i : i + 256] @ bank_flat.T
        out[i : i + 256] = 1.0 - (-np.partition(-sims, k - 1, axis=1)[:, k - 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True, type=Path)
    ap.add_argument("--val-dir", required=True, type=Path)
    ap.add_argument("--ood-zip", action="append", default=[], type=Path)
    ap.add_argument("--ood-zip-max", type=int, default=150, help="images per folder inside each zip")
    ap.add_argument("--bank-max-per-class", type=int, default=0, help="0 = all train images")
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--quantile-high", type=float, default=0.999)
    args = ap.parse_args()

    svc = EXP034Service()
    classes = svc.classes
    nc = classes.index("not_cashew_leaf")

    print("Bank (train, cashew classes)...", flush=True)
    bank, _, _, bad_b = embed(svc, list_images(args.train_dir, CASHEW, args.bank_max_per_class or None), label="bank")
    bank_flat = EXP034Service.gate_embedding(bank)

    print("In-distribution Validation (cashew classes)...", flush=True)
    f_id, p_id, t_id, bad_v = embed(svc, list_images(args.val_dir, CASHEW), label="val-cashew")
    k_id = knn_dist(bank_flat, f_id, OOD_KNN_K)
    t_low = float(np.quantile(k_id, args.quantile))
    t_high = float(np.quantile(k_id, args.quantile_high))

    def gate(knn, probs):
        top = np.delete(probs, nc, axis=1).max(1)
        ok = (probs[:, nc] < OOD_BAND_MAX_P_NOT_CASHEW) & (top >= OOD_BAND_MIN_TOP_CASHEW)
        return (probs.argmax(1) == nc) | (knn > t_high) | ((knn > t_low) & ~ok)

    report = {"in_distribution": {}, "ood": {}}
    rej = gate(k_id, p_id)
    report["in_distribution"]["all_cashew_val"] = {"n": int(len(rej)), "accepted": round(float(1 - rej.mean()), 4)}
    for c in CASHEW:
        m = t_id == c
        report["in_distribution"][c] = {"n": int(m.sum()), "accepted": round(float(1 - rej[m].mean()), 4)}

    ood_sets = [("val:not_cashew_leaf", list_images(args.val_dir, ("not_cashew_leaf",)))]
    for z in args.ood_zip:
        items = list_zip(z, args.ood_zip_max)
        for tag in sorted({t for t, _ in items}):
            ood_sets.append((tag, [it for it in items if it[0] == tag]))
    for name, items in ood_sets:
        print(f"OOD set {name} ({len(items)})...", flush=True)
        f_o, p_o, _, _ = embed(svc, items, label=name)
        r = gate(knn_dist(bank_flat, f_o, OOD_KNN_K), p_o)
        report["ood"][name] = {"n": int(len(r)), "rejected": round(float(r.mean()), 4)}

    OOD_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OOD_REFERENCE_PATH, bank=bank.astype(np.float16))
    cal = {
        "embedding": "5 seeds, per-seed L2-normalised mean of image + h-flip; distance = 1 - mean cosine over seeds",
        "knn_threshold": round(t_low, 6),
        "knn_threshold_high": round(t_high, 6),
        "quantile": args.quantile,
        "quantile_high": args.quantile_high,
        "knn_k": OOD_KNN_K,
        "band_rule": {"max_p_not_cashew": OOD_BAND_MAX_P_NOT_CASHEW, "min_top_cashew": OOD_BAND_MIN_TOP_CASHEW},
        "energy_threshold": None,
        "energy_used": False,
        "bank_size": int(len(bank)),
        "bank_source": str(args.train_dir),
        "calibration_source": str(args.val_dir),
        "skipped_unreadable": {"bank": bad_b, "val": bad_v},
        "protocol": "Both thresholds = quantiles of in-distribution cashew Validation scores only; OOD sets are reported, never used for tuning.",
        "report": report,
    }
    OOD_CALIBRATION_PATH.write_text(json.dumps(cal, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(cal, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
