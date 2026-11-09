"""
Analysis of tools/eval_box_labels.py outputs (roadmap steps 3-5).

Box target: the GT disease of a matched box, or "background" for a YOLO false
positive (the verifier should reject it). Labelling rules compared:
  yolo        : keep YOLO's label, never reject
  dense       : DenseNet crop argmax over the 3 diseases (web rule: keeps YOLO's
                label when disease mass < MASS, rejects when healthy dominates)
  switch@t    : DenseNet label only when its top disease probability >= t,
                otherwise YOLO's label; healthy-dominant crops are rejected
Selection (crop config, rule, t, healthy-reject h) maximises the accuracy over
ALL boxes (true boxes must get the right disease, false-positive boxes must be
rejected) on VALIDATION only. Macro-F1 is reported but not used to select: with
"background" as one of four classes it over-rewards rejecting a few false boxes
while many true boxes are lost. Test is reported once (+ McNemar vs YOLO).

Image level (step 5): disease set = classes with >= n_min kept boxes;
"healthy" = no kept box. Compared with the GT set of the YOLO label file.
"""
import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

DIS = ["anthracnose", "leaf_miner", "red_rust"]
CFGS = ["p125_m64", "p150_m96", "p200_m128", "p300_m128"]
TARGETS = DIS + ["background"]
HEALTHY_REJECT = 0.50
MASS = 0.50


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    for r in rows:
        r["yolo_conf"] = float(r["yolo_conf"])
        for k in list(r):
            if k.startswith("p") and "_m" in k:
                r[k] = float(r[k])
    return rows


def decide(r, cfg, rule, t=None, h=HEALTHY_REJECT):
    """Returns the predicted target (a disease or 'background')."""
    if rule == "yolo":
        return r["yolo_cls"]
    pd = {c: r[f"{cfg}_{c}"] for c in DIS}
    ph = r[f"{cfg}_healthy"]
    top = max(pd, key=pd.get)
    if h is not None and ph >= h and ph > pd[top]:
        return "background"
    if rule == "dense":
        return top if sum(pd.values()) >= MASS else r["yolo_cls"]
    if rule == "switch":
        return top if pd[top] >= t else r["yolo_cls"]
    raise ValueError(rule)


def macro_f1(pairs, labels):
    f1s, per = [], {}
    for c in labels:
        tp = sum(1 for g, p in pairs if g == c and p == c)
        fp = sum(1 for g, p in pairs if g != c and p == c)
        fn = sum(1 for g, p in pairs if g == c and p != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per[c] = {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f, 4), "support": tp + fn}
        if tp + fn:
            f1s.append(f)
    return sum(f1s) / len(f1s), per


def mcnemar(a_ok, b_ok):
    b = sum(1 for x, y in zip(a_ok, b_ok) if x and not y)
    c = sum(1 for x, y in zip(a_ok, b_ok) if y and not x)
    n = b + c
    if n == 0:
        return b, c, 1.0
    k = min(b, c)  # exact two-sided binomial
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n * 2
    return b, c, min(1.0, p)


def evaluate(rows, cfg, rule, t=None, h=HEALTHY_REJECT):
    pairs = [(r["gt_cls"], decide(r, cfg, rule, t, h)) for r in rows]
    mf, per = macro_f1(pairs, TARGETS)
    acc_true = [p == g for g, p in pairs if g != "background"]
    bg = [p == "background" for g, p in pairs if g == "background"]
    return {
        "acc_all_boxes": round(sum(g == p for g, p in pairs) / max(1, len(pairs)), 4),
        "macro_f1": round(mf, 4),
        "label_acc_true_boxes": round(sum(acc_true) / max(1, len(acc_true)), 4),
        "true_boxes_wrongly_rejected": round(sum(1 for g, p in pairs if g != "background" and p == "background") / max(1, len(acc_true)), 4),
        "background_rejected": round(sum(bg) / max(1, len(bg)), 4),
        "per_class": per,
    }, pairs


def image_level(rows, imgs, cfg, rule, t, n_min, h=HEALTHY_REJECT):
    by = defaultdict(list)
    for r in rows:
        by[r["image"]].append(decide(r, cfg, rule, t, h))
    res = []
    for im in imgs:
        gt = set(filter(None, im["gt_set"].split("|")))
        cnt = Counter(p for p in by.get(im["image"], []) if p != "background")
        pred = {c for c, n in cnt.items() if n >= n_min}
        res.append((gt, pred, im["image"]))
    out = {"exact_match": round(sum(g == p for g, p, _ in res) / len(res), 4)}
    for c in DIS:
        tp = sum(1 for g, p, _ in res if c in g and c in p)
        fp = sum(1 for g, p, _ in res if c not in g and c in p)
        fn = sum(1 for g, p, _ in res if c in g and c not in p)
        out[c] = {"precision": round(tp / (tp + fp), 4) if tp + fp else None,
                  "recall": round(tp / (tp + fn), 4) if tp + fn else None}
    neg = [(g, p) for g, p, _ in res if not g]
    out["healthy_or_negative_images"] = {"n": len(neg), "predicted_no_disease": round(sum(not p for _, p in neg) / len(neg), 4) if neg else None}
    multi = [(g, p) for g, p, _ in res if len(g) >= 2]
    out["multi_disease_images"] = {"n": len(multi), "exact": round(sum(g == p for g, p in multi) / len(multi), 4) if multi else None}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    val, test = load(d / "boxes_val.csv"), load(d / "boxes_test.csv") if (d / "boxes_test.csv").exists() else None
    ival = list(csv.DictReader(open(d / "images_val.csv", encoding="utf-8")))
    report = {"n_val_boxes": len(val), "val_target_counts": Counter(r["gt_cls"] for r in val)}

    # 1) Grid on VALIDATION
    grid = []
    base, _ = evaluate(val, None, "yolo")
    grid.append(("-", "yolo", None, None, base))
    for cfg in CFGS:
        for h in (0.5, 0.7, 0.8, 0.9, 0.95, None):
            m, _ = evaluate(val, cfg, "dense", None, h)
            grid.append((cfg, "dense", None, h, m))
            for t in [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:
                m, _ = evaluate(val, cfg, "switch", t, h)
                grid.append((cfg, "switch", t, h, m))
    report["val_grid"] = [{"cfg": c, "rule": r, "t": t, "h": h, **{k: v for k, v in m.items() if k != "per_class"}}
                          for c, r, t, h, m in grid]
    # ties -> the most conservative setting (YOLO first, then higher t / h)
    best = max(grid, key=lambda g: (g[4]["acc_all_boxes"], g[1] == "yolo", g[2] or 0, g[3] or 1))
    report["val_selected"] = {"cfg": best[0], "rule": best[1], "t": best[2], "h": best[3], **best[4]}
    report["val_yolo"] = base
    best_verifier = max((g for g in grid if g[1] != "yolo"),
                        key=lambda g: (g[4]["acc_all_boxes"], g[2] or 0, g[3] or 1))
    report["val_best_verifier"] = {"cfg": best_verifier[0], "rule": best_verifier[1], "t": best_verifier[2],
                                   "h": best_verifier[3], **best_verifier[4]}
    best = best_verifier  # the deployed design keeps the DenseNet verification step (Fig. 3.15)

    # 2) Image level on VALIDATION: choose n_min
    report["val_image_level"] = {}
    for n_min in (1, 2, 3):
        report["val_image_level"][f"yolo_n{n_min}"] = image_level(val, ival, None, "yolo", None, n_min)
        report["val_image_level"][f"selected_n{n_min}"] = image_level(val, ival, best[0], best[1], best[2], n_min, best[3])

    # 3) TEST once with the selected setting
    if test:
        itest = list(csv.DictReader(open(d / "images_test.csv", encoding="utf-8")))
        ty, py = evaluate(test, None, "yolo")
        ts, ps = evaluate(test, best[0], best[1], best[2], best[3])
        report["test_yolo"] = ty
        report["test_selected"] = ts
        ok_y = [g == p for g, p in py]
        ok_s = [g == p for g, p in ps]
        b, c, p = mcnemar(ok_y, ok_s)
        report["test_mcnemar_yolo_vs_selected"] = {"yolo_only_correct": b, "selected_only_correct": c, "p_value": p}
        report["test_image_level"] = {
            f"{name}_n{n}": image_level(test, itest, cf, ru, tt, n, hh)
            for n in (1, 2, 3)
            for name, cf, ru, tt, hh in (("yolo", None, "yolo", None, None), ("selected", best[0], best[1], best[2], best[3]))
        }
    (d / "box_label_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    print("VAL top settings by accuracy over all boxes:")
    for g in sorted(report["val_grid"], key=lambda g: -g["acc_all_boxes"])[:8]:
        print(f"  {g['cfg']:10} {g['rule']:7} t={str(g['t']):5} h={str(g['h']):5} acc_all={g['acc_all_boxes']:.4f} "
              f"label_acc={g['label_acc_true_boxes']:.4f} wrongrej={g['true_boxes_wrongly_rejected']:.4f} bgrej={g['background_rejected']:.4f}")
    print("VAL YOLO only      :", {k: report["val_yolo"][k] for k in ("acc_all_boxes", "label_acc_true_boxes")})
    print("VAL best verifier  :", {k: report["val_best_verifier"][k] for k in ("cfg", "rule", "t", "h", "acc_all_boxes")})
    if test:
        print("TEST yolo    :", {k: v for k, v in report["test_yolo"].items() if k != "per_class"})
        print("TEST selected:", {k: v for k, v in report["test_selected"].items() if k != "per_class"})
        print("TEST McNemar :", report["test_mcnemar_yolo_vs_selected"])


if __name__ == "__main__":
    main()
