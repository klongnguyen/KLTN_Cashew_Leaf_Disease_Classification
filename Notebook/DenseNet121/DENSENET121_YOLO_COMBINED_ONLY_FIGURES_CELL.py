# DenseNet121 + YOLO only (F3): statistical figures from already computed results.
# Run after the localization/evaluation cell. This cell never fits or selects a model.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_classes = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
_class_index = {name: i for i, name in enumerate(_classes)}
_root = Path(OUT_DIR) if "OUT_DIR" in globals() else None
if _root is None:
    _candidates = [p for p in Path("/content/training_results/current_dataset").glob(
        "EXP-DENSENET121-YOLO*") if (p / "aggregate" / "fusion_results_partB.csv").is_file()]
    if len(_candidates) != 1:
        raise RuntimeError(f"Set OUT_DIR to the intended EXP-032/033 result folder: {_candidates}")
    _root = _candidates[0]
_aggregate = _root / "aggregate"
_output = _aggregate / "figures" / "combined_only"
_output.mkdir(parents=True, exist_ok=True)

def _read(name):
    path = _aggregate / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()

_metrics = _read("fusion_results_partB.csv")
if _metrics.empty or not {"F3", "test", "holdout"}.issubset(
        set(_metrics.get("rule", [])) | set(_metrics.get("split", []))):
    raise RuntimeError("Run Part B/EXP-033 evaluation before this figure cell")
_metrics = _metrics[(_metrics.rule == "F3") &
                    (_metrics.split.isin(["test", "holdout", "holdout_n124"]))].copy()
_errors = _read("error_taxonomy.csv")
_errors = _errors[(_errors.part == "B") & (_errors.rule == "F3")]
_risk = _read("risk_coverage.csv")
_risk = _risk[(_risk.part == "B") & (_risk.rule == "F3")]
_abstention = _read("abstention.csv")
_abstention = _abstention[(_abstention.part == "B") & (_abstention.rule == "F3")]
_localization = _read("localization_test.csv")
_config_path = _aggregate / "frozen_config.json"
_config = json.loads(_config_path.read_text(encoding="utf-8")) if _config_path.is_file() else {}
_threshold = _config.get("abstain_thresholds", {}).get("F3")
_color = "#1baf7a"
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10})

def _wilson(correct, total):
    if total == 0:
        return np.nan, np.nan
    z = 1.9599639845
    p = correct / total
    divisor = 1 + z*z/total
    center = (p + z*z/(2*total)) / divisor
    radius = z*np.sqrt((p*(1-p) + z*z/(4*total))/total) / divisor
    return center-radius, center+radius

def _save(fig, name):
    fig.tight_layout()
    path = _output / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Đã lưu:", path)

def _predictions(split):
    # The live EXP-032/033 runtime keeps calibrated per-image probabilities.
    if "PART_PREDICTIONS" in globals() and "B" in PART_PREDICTIONS and split in PART_PREDICTIONS["B"]:
        frame, by_rule = PART_PREDICTIONS["B"][split]
        labels, probs = by_rule["F3"]
        result = frame[["evidence_key", "true_label"]].reset_index(drop=True).copy()
        result["predicted_label"] = [_classes[int(i)] for i in labels]
        for i, cls in enumerate(_classes):
            result[f"prob_{cls}"] = probs[:, i]
        return result
    path = _aggregate / f"predictions_{split}_F3.csv"
    return pd.read_csv(path) if path.is_file() else None

_test_pred, _hold_pred = _predictions("test"), _predictions("holdout")
for _split, _frame in (("test", _test_pred), ("holdout", _hold_pred)):
    if _frame is not None:
        if _frame.evidence_key.duplicated().any():
            raise ValueError(f"Duplicate {_split} prediction keys")
        _probs = _frame[[f"prob_{name}" for name in _classes]].to_numpy(dtype=float)
        if not np.isfinite(_probs).all() or not np.allclose(_probs.sum(axis=1), 1, atol=1e-4):
            raise ValueError(f"Invalid {_split} probabilities")
        if not np.array_equal(np.array(_classes)[_probs.argmax(axis=1)],
                              _frame.predicted_label.to_numpy()):
            raise ValueError(f"{_split} labels do not match probabilities")

# Accuracy and Wilson 95% intervals; gap CI uses independent image bootstrap if available.
_acc = _metrics.set_index("split")
_splits = [split for split in ("test", "holdout", "holdout_n124") if split in _acc.index]
_accuracy_rows = []
for _split in _splits:
    _row = _acc.loc[_split]
    _accuracy_rows.append({"split": _split, "n": int(_row.n), "correct": int(_row.correct),
        "accuracy": float(_row.accuracy), "wilson_low": float(_row.wilson_low),
        "wilson_high": float(_row.wilson_high), "ece15": float(_row.ece15)})
_gap = float(_acc.loc["test"].accuracy - _acc.loc["holdout"].accuracy)
_gap_low = _gap_high = np.nan
if _test_pred is not None and _hold_pred is not None:
    _rng = np.random.default_rng(32033)
    _test_ok = (_test_pred.true_label == _test_pred.predicted_label).to_numpy(dtype=float)
    _hold_ok = (_hold_pred.true_label == _hold_pred.predicted_label).to_numpy(dtype=float)
    _ti = _rng.integers(0, len(_test_ok), size=(10000, len(_test_ok)))
    _hi = _rng.integers(0, len(_hold_ok), size=(10000, len(_hold_ok)))
    _distribution = _test_ok[_ti].mean(axis=1) - _hold_ok[_hi].mean(axis=1)
    _gap_low, _gap_high = np.quantile(_distribution, [.025, .975])
pd.DataFrame(_accuracy_rows).to_csv(_output / "accuracy_wilson.csv", index=False)
pd.DataFrame([{"test_minus_holdout": _gap, "bootstrap_low": _gap_low,
               "bootstrap_high": _gap_high, "resamples": 10000 if np.isfinite(_gap_low) else 0}]).to_csv(
                   _output / "test_holdout_gap.csv", index=False)
fig, (ax, gap_ax) = plt.subplots(1, 2, figsize=(11, 4.5), gridspec_kw={"width_ratios": [2, 1]})
for i, record in enumerate(_accuracy_rows):
    mid = 100 * record["accuracy"]
    ax.errorbar(i, mid, yerr=[[mid - 100*record["wilson_low"]],
                              [100*record["wilson_high"] - mid]],
                fmt="o", capsize=5, color=_color, markersize=8)
ax.set(xticks=range(len(_splits)), xticklabels=["Test", "Holdout 126", "Holdout 124"][:len(_splits)],
       ylim=(0, 100), ylabel="Độ chính xác (%)",
       title="Mô hình kết hợp: độ chính xác và khoảng Wilson 95%")
ax.grid(axis="y", alpha=.25)
if np.isfinite(_gap_low):
    gap_ax.errorbar(0, 100*_gap, yerr=[[100*(_gap-_gap_low)], [100*(_gap_high-_gap)]],
                    fmt="o", capsize=5, color=_color)
else:
    gap_ax.scatter(0, 100*_gap, color=_color)
    gap_ax.text(.5, .05, "Không có xác suất từng ảnh\nđể tính khoảng bootstrap",
                transform=gap_ax.transAxes, ha="center", fontsize=8)
gap_ax.axhline(10, color="grey", ls="--", label="Mốc 10 điểm")
gap_ax.set(xlim=(-.5, .5), xticks=[0], xticklabels=["Test − Holdout"],
           ylabel="Chênh lệch (điểm %)", title="Khoảng cách giữa hai tập")
gap_ax.legend(fontsize=8)
_save(fig, "01_accuracy_and_gap.png")

# Class recall uses the row support of each saved F3 confusion matrix.
_class_rows = []
for _split in ("test", "holdout"):
    _cm = pd.read_csv(_aggregate / f"confusion_B_{_split}_F3.csv", index_col=0)
    _cm = _cm.reindex(index=_classes, columns=_classes, fill_value=0)
    for _class in _classes:
        _support = int(_cm.loc[_class].sum())
        if _support:
            _correct = int(_cm.loc[_class, _class])
            _low, _high = _wilson(_correct, _support)
            _class_rows.append({"split": _split, "class": _class, "support": _support,
                "correct": _correct, "recall": _correct/_support,
                "wilson_low": _low, "wilson_high": _high})
_class_table = pd.DataFrame(_class_rows)
_class_table.to_csv(_output / "class_recall_wilson.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
for ax, _split in zip(axes, ("test", "holdout")):
    _rows = _class_table[_class_table.split == _split]
    x = np.arange(len(_rows))
    ax.bar(x, _rows.recall.to_numpy(), color=_color, width=.7)
    midpoint = _rows.recall.to_numpy(dtype=float)
    ax.errorbar(x, midpoint,
                yerr=np.vstack((midpoint - _rows.wilson_low.to_numpy(dtype=float),
                                _rows.wilson_high.to_numpy(dtype=float) - midpoint)),
                fmt="none", ecolor="black", capsize=3)
    ax.set(xticks=x, xticklabels=[f"{r['class']}\n(n={r['support']})" for _, r in _rows.iterrows()],
           ylim=(0, 1), ylabel="Recall theo lớp", title=f"{_split.title()}: lớp nào dễ bị bỏ sót?")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=.2)
_save(fig, "02_class_recall.png")

# Confusion matrices show counts and row percentages on a shared 0–100% scale.
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, _split in zip(axes, ("test", "holdout")):
    _cm = pd.read_csv(_aggregate / f"confusion_B_{_split}_F3.csv", index_col=0)
    _cm = _cm.reindex(index=_classes, columns=_classes, fill_value=0)
    counts = _cm.to_numpy(dtype=int)
    support = counts.sum(axis=1, keepdims=True)
    frac = np.divide(counts, support, out=np.zeros_like(counts, dtype=float), where=support > 0)
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{counts[i,j]}\n{frac[i,j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if frac[i,j] > .55 else "black")
    ax.set(xticks=range(5), xticklabels=_classes, yticks=range(5), yticklabels=_classes,
           xlabel="Nhãn dự đoán", ylabel="Nhãn thật", title=f"{_split.title()}: ma trận nhầm lẫn")
    ax.tick_params(axis="x", labelrotation=45)
_save(fig, "03_confusion_matrices.png")

_error_rows = _errors[_errors.split.isin(["test", "holdout"])][
    ["split", "missed_disease", "wrong_disease_type", "false_alarm"]].copy()
_error_rows.to_csv(_output / "error_types.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4.5))
_error_names = [("missed_disease", "Bỏ sót bệnh", "#777777"),
                ("wrong_disease_type", "Nhầm bệnh", "#eb6834"),
                ("false_alarm", "Báo bệnh giả", "#2a78d6")]
for i, (_field, _label, _shade) in enumerate(_error_names):
    ax.bar(np.arange(2) + (i-1)*.25,
           [int(_error_rows[_error_rows.split == s].iloc[0][_field]) for s in ("test", "holdout")],
           width=.24, label=_label, color=_shade)
ax.set(xticks=[0, 1], xticklabels=["Test", "Holdout"], ylabel="Số ảnh lỗi",
       title="Mô hình kết hợp: ba loại lỗi cần đọc riêng")
ax.legend()
ax.grid(axis="y", alpha=.2)
_save(fig, "04_error_types.png")

# A reliability diagram needs per-image probabilities; ECE is still shown when absent.
_bin_rows = []
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, _split, _pred in zip(axes, ("test", "holdout"), (_test_pred, _hold_pred)):
    _ece = float(_acc.loc[_split].ece15)
    if _pred is not None:
        probs = _pred[[f"prob_{name}" for name in _classes]].to_numpy(dtype=float)
        confidence = probs.max(axis=1)
        correct = (_pred.true_label == _pred.predicted_label).to_numpy()
        x, y = [], []
        for b in range(15):
            low, high = b/15, (b+1)/15
            chosen = (confidence >= low) & ((confidence <= high) if b == 14 else (confidence < high))
            if chosen.any():
                mean_conf, observed = float(confidence[chosen].mean()), float(correct[chosen].mean())
                x.append(mean_conf)
                y.append(observed)
                _bin_rows.append({"split": _split, "bin": b, "n": int(chosen.sum()),
                                  "mean_confidence": mean_conf, "observed_accuracy": observed})
        ax.plot(x, y, "o-", color=_color)
    else:
        ax.text(.5, .5, "Thiếu xác suất từng ảnh\nchỉ có ECE tổng hợp",
                transform=ax.transAxes, ha="center", va="center")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Độ tin cậy trung bình",
           ylabel="Tỷ lệ dự đoán đúng", title=f"{_split.title()}: ECE={_ece:.3f}")
pd.DataFrame(_bin_rows, columns=["split", "bin", "n", "mean_confidence",
    "observed_accuracy"]).to_csv(_output / "reliability_bins.csv", index=False)
_save(fig, "05_calibration.png")

_risk_rows = _risk[_risk.split.isin(["test", "holdout"])][
    ["split", "threshold", "coverage", "risk"]].copy()
_risk_rows.to_csv(_output / "risk_coverage.csv", index=False)
_abstention_rows = _abstention[_abstention.split.isin(["test", "holdout"])][
    ["split", "threshold", "coverage", "selective_accuracy", "referred_to_expert"]].copy()
_abstention_rows.to_csv(_output / "abstention.csv", index=False)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
for ax, _split in zip(axes, ("test", "holdout")):
    view = _risk_rows[_risk_rows.split == _split].dropna(subset=["risk"])
    ax.plot(view.coverage, 100*view.risk, color=_color)
    point = _abstention_rows[_abstention_rows.split == _split]
    if not point.empty and np.isfinite(point.iloc[0].selective_accuracy):
        ax.scatter(point.iloc[0].coverage, 100*(1-point.iloc[0].selective_accuracy),
                   marker="x", color="black", s=90, label="Ngưỡng đã khóa")
    ax.set(xlim=(0, 1), xlabel="Tỷ lệ ảnh tự kết luận",
           ylabel="Lỗi trong ảnh được nhận (%)",
           title=f"{_split.title()}: từ chối ảnh khó giúp giảm lỗi?")
    ax.legend()
_save(fig, "06_risk_coverage.png")

if _test_pred is not None and _hold_pred is not None:
    _confidence_rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharex=True, sharey=True)
    for ax, _split, _pred in zip(axes, ("test", "holdout"), (_test_pred, _hold_pred)):
        probs = _pred[[f"prob_{name}" for name in _classes]].to_numpy(dtype=float)
        confidence = probs.max(axis=1)
        correct = (_pred.true_label == _pred.predicted_label).to_numpy()
        _confidence_rows.extend({"split": _split, "evidence_key": key,
            "confidence": float(q), "correct": bool(ok)}
            for key, q, ok in zip(_pred.evidence_key, confidence, correct))
        ax.hist([confidence[correct], confidence[~correct]], bins=np.linspace(0, 1, 16),
                stacked=True, color=[_color, "#eb6834"], label=["Đúng", "Sai"])
        ax.set(xlim=(0, 1), xlabel="Độ tin cậy", ylabel="Số ảnh",
               title=f"{_split.title()}: độ tin cậy của ảnh đúng và sai")
        ax.legend()
    pd.DataFrame(_confidence_rows).to_csv(_output / "confidence_by_image.csv", index=False)
    _save(fig, "07_confidence_distribution.png")

_selected = globals().get("SELECTED_DEFAULT")
if _selected is None and (_aggregate / "exp033_summary.json").is_file():
    _selected = "F3"  # EXP-033 summary declares the combined method.
if _selected == "F3" and not _localization.empty and "mode" in _localization.columns:
    _loc = _localization[(_localization["mode"] == "fused_display") &
                         (_localization["subset"].isin(["all", "real_"]))].copy()
    if not _loc.empty:
        _loc.to_csv(_output / "fused_display_localization.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, subset in zip(axes, ("all", "real_")):
            view = _loc[_loc["subset"] == subset].set_index("class")
            names = [cls for cls in ("anthracnose", "leaf_miner", "red_rust") if cls in view.index]
            x = np.arange(len(names))
            for i, (field, label, shade) in enumerate((
                    ("precision50", "Precision", "#2a78d6"),
                    ("recall50", "Recall", "#eb6834"),
                    ("f1_50", "F1", "#1baf7a"))):
                ax.bar(x + (i-1)*.25, view.loc[names, field].to_numpy(dtype=float),
                       width=.24, label=label, color=shade)
            count = int(view.images.iloc[0]) if len(view) else 0
            ax.set(xticks=x, xticklabels=names, ylim=(0, 1),
                   ylabel="Chỉ số box tại IoU 0,5",
                   title=f"YOLO Test {'real_' if subset == 'real_' else 'toàn bộ'} (n={count})")
            ax.tick_params(axis="x", labelrotation=20)
        axes[0].legend()
        _save(fig, "08_fused_display_localization.png")
else:
    print("Bỏ qua hình định vị: bảng fused_display hiện có không được xác nhận là của F3.")

print("Chỉ phân tích F3 (DenseNet121 + YOLO). Holdout là đánh giá mô tả, không phải blind benchmark.")
print("McNemar cần hai phương pháp trên cùng ảnh nên không áp dụng cho cell chỉ một mô hình.")
print("Ngưỡng từ chối đã khóa:", _threshold, "| Hình/CSV:", _output)
