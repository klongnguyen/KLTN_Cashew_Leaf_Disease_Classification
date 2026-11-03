# EXP-033 standard report: tables, paired tests, figures, and qualitative audit.
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from PIL import ImageFont
from scipy.stats import binomtest
try:
    from IPython.display import display, Markdown
except ImportError:  # Local dry-run outside Jupyter.
    Markdown = str
    def display(value):
        print(value if isinstance(value, str) else f"Rendered {type(value).__name__}")

METHODS = {"F0": "DenseNet một mình", "F3b": "YOLO một mình",
           "F3": "DenseNet + YOLO"}
APPENDIX = {"F3a": "Stacker không YOLO"}
COLORS = {"F0": "#2a78d6", "F3b": "#eb6834", "F3": "#1baf7a",
          "F3a": "#777777"}
FIGURES = AGGREGATE / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10})

def save_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def markdown_table(frame):
    columns = [str(c) for c in frame.columns]
    body = [[str(v).replace("|", "/").replace("\n", " ") for v in row]
            for row in frame.itertuples(index=False, name=None)]
    return "\n".join(["| " + " | ".join(columns) + " |",
                      "|" + "|".join(["---"] * len(columns)) + "|"] +
                     ["| " + " | ".join(row) + " |" for row in body])

rows = pd.DataFrame([r for r in ALL_RESULTS if r["part"] == "B"])
errors = pd.DataFrame([r for r in ALL_TAXONOMY if r["part"] == "B"])
abstention = pd.DataFrame([r for r in ALL_ABSTENTION if r["part"] == "B"])
curves = pd.DataFrame([r for r in ALL_CURVES if r["part"] == "B"])
assert not rows.empty and set(METHODS).issubset(set(rows.rule))
assert set(("validation_clean", "test", "holdout", "holdout_n124")).issubset(set(rows.split))
assert SELECTED_DEFAULT == "F3"

# Save human-facing tables and a separate appendix control.
metric_cols = ["split", "method", "n", "correct", "accuracy", "wilson_low",
               "wilson_high", "macro_f1", "top2_accuracy", "ece15"] + [
                   f"recall_{c}" for c in CLASSES] + [
                   "missed_disease", "wrong_disease_type", "false_alarm"]
tables = {}
for split, name in (("test", "final_test"), ("holdout", "real_holdout_n126"),
                    ("holdout_n124", "real_holdout_n124"),
                    ("validation_clean", "validation_grouped_oof")):
    frame = rows[(rows.split == split) & rows.rule.isin(METHODS)].copy()
    frame["method"] = frame.rule.map(METHODS)
    frame = frame.merge(errors[errors.split == split][
        ["rule", "missed_disease", "wrong_disease_type", "false_alarm"]],
        on="rule", validate="one_to_one")
    frame = frame[metric_cols]
    if split.startswith("holdout"):
        frame["recall_not_cashew_leaf"] = np.nan  # No positive examples in Holdout.
    frame.to_csv(AGGREGATE / f"{name}.csv", index=False)
    tables[split] = frame
appendix = rows[rows.rule == "F3a"].copy()
appendix["method"] = APPENDIX["F3a"]
appendix.to_csv(AGGREGATE / "appendix_stacker_no_yolo.csv", index=False)
abstention["method"] = abstention.rule.map({**METHODS, **APPENDIX})
abstention.to_csv(AGGREGATE / "abstention_report.csv", index=False)
for split in ("test", "holdout", "holdout_n124"):
    display(Markdown(f"### {'Test' if split == 'test' else 'Holdout 09'} "
                     f"{'(n=124)' if split == 'holdout_n124' else ''}\n\n" +
                     markdown_table(tables[split].round(4))))

# Recompute every per-seed accuracy from the saved, aligned probability rows.
seed_rows = []
for seed in SEEDS:
    for split in ("validation", "test", "holdout"):
        frame = pd.read_csv(OUT_DIR / f"seed_{seed}" / f"{split}_predictions.csv")
        unique_keys(frame, f"seed {seed} {split}")
        assert set(frame.evidence_key) == set(INVENTORY[split].evidence_key)
        y = frame.true_label.map(CLASS_INDEX).to_numpy()
        pred = frame[PROB_COLUMNS].to_numpy().argmax(1)
        seed_rows.append({"seed": seed, "split": split, "n": len(frame),
                          "accuracy": float(np.mean(pred == y)),
                          "correct": int(np.sum(pred == y))})
per_seed = pd.DataFrame(seed_rows)
per_seed.to_csv(AGGREGATE / "per_seed_results.csv", index=False)
display(Markdown("### Kết quả từng seed\n\n" + markdown_table(per_seed.round(4))))

# The two comparison families were declared before reading Test/Holdout.
paired_rows = []
gap_rows = []
rng = np.random.default_rng(33033)
for split in ("test", "holdout"):
    frame, predictions = PART_PREDICTIONS["B"][split]
    y = frame.true_label.map(CLASS_INDEX).to_numpy()
    ids = rng.integers(0, len(y), size=(10000, len(y)))
    for challenger, baseline in (("F3", "F0"), ("F3", "F3a")):
        a = predictions[challenger][0] == y
        b = predictions[baseline][0] == y
        delta = a.astype(float) - b.astype(float)
        distribution = delta[ids].mean(axis=1)
        better, worse = int(np.sum(a & ~b)), int(np.sum(~a & b))
        paired_rows.append({"split": split, "challenger": challenger,
                            "baseline": baseline, "n": len(y),
                            "delta_accuracy": float(delta.mean()),
                            "bootstrap_low": float(np.quantile(distribution, .025)),
                            "bootstrap_high": float(np.quantile(distribution, .975)),
                            "b_challenger_only_correct": better,
                            "c_baseline_only_correct": worse,
                            "mcnemar_exact_p": float(binomtest(min(better, worse),
                                n=better + worse, p=.5).pvalue) if better + worse else 1.0})
test_frame, test_preds = PART_PREDICTIONS["B"]["test"]
hold_frame, hold_preds = PART_PREDICTIONS["B"]["holdout"]
test_y = test_frame.true_label.map(CLASS_INDEX).to_numpy()
hold_y = hold_frame.true_label.map(CLASS_INDEX).to_numpy()
test_ids = rng.integers(0, len(test_y), size=(10000, len(test_y)))
hold_ids = rng.integers(0, len(hold_y), size=(10000, len(hold_y)))
for rule in METHODS:
    test_ok = (test_preds[rule][0] == test_y).astype(float)
    hold_ok = (hold_preds[rule][0] == hold_y).astype(float)
    distribution = test_ok[test_ids].mean(axis=1) - hold_ok[hold_ids].mean(axis=1)
    gap_rows.append({"method": rule, "test_minus_holdout": float(test_ok.mean() - hold_ok.mean()),
                     "bootstrap_low": float(np.quantile(distribution, .025)),
                     "bootstrap_high": float(np.quantile(distribution, .975)),
                     "test_n": len(test_y), "holdout_n": len(hold_y)})
paired = pd.DataFrame(paired_rows)
gaps = pd.DataFrame(gap_rows)
paired.to_csv(AGGREGATE / "paired_tests.csv", index=False)
gaps.to_csv(AGGREGATE / "test_holdout_gap.csv", index=False)
display(Markdown("### So sánh ghép cặp\n\n" + markdown_table(paired.round(4))))

# Training curves are available only for models trained by this notebook.
fig, axes = plt.subplots(2, len(SEEDS), figsize=(19, 7), sharey="row")
history_rows = []
for column, seed in enumerate(SEEDS):
    meta = json.loads((OUT_DIR / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))
    cursor = 0
    for stage in meta.get("stages", []):
        history = stage.get("history") or {}
        length = len(history.get("loss", []))
        if length:
            epochs = np.arange(cursor + 1, cursor + length + 1)
            history_rows.extend({"seed": seed, "stage": stage["stage"],
                "epoch_global": int(epochs[i]), "epoch_stage": i + 1,
                "loss": float(history["loss"][i]),
                "val_loss": float(history["val_loss"][i]),
                "accuracy": float(history["accuracy"][i]),
                "val_accuracy": float(history["val_accuracy"][i])}
                for i in range(length))
            for row, key, val_key in ((0, "loss", "val_loss"),
                                      (1, "accuracy", "val_accuracy")):
                axes[row, column].plot(epochs, history[key], color="#2a78d6", label="Huấn luyện")
                axes[row, column].plot(epochs, history[val_key], color="#eb6834", label="Thẩm định")
            cursor += length
            axes[0, column].axvline(cursor + .5, color="grey", ls="--", lw=.8)
            axes[1, column].axvline(cursor + .5, color="grey", ls="--", lw=.8)
    if cursor == 0:
        for row in (0, 1):
            axes[row, column].text(.5, .5, "Không có lịch sử epoch\n(model tái sử dụng)",
                                   transform=axes[row, column].transAxes, ha="center")
    axes[0, column].set_title(f"Seed {seed}: loss theo hai giai đoạn")
    axes[1, column].set_title(f"Seed {seed}: accuracy theo hai giai đoạn")
    for row in (0, 1):
        axes[row, column].set_xlabel("Vòng huấn luyện")
        axes[row, column].grid(alpha=.2)
axes[0, 0].set_ylabel("Hàm mất mát")
axes[1, 0].set_ylabel("Độ chính xác")
axes[0, 0].legend(loc="best")
pd.DataFrame(history_rows, columns=["seed", "stage", "epoch_global", "epoch_stage",
    "loss", "val_loss", "accuracy", "val_accuracy"]).to_csv(
        AGGREGATE / "training_history.csv", index=False)
save_figure(fig, "01_training_curves.png")

fig, ax = plt.subplots(figsize=(8, 5))
for i, split in enumerate(("validation", "test", "holdout")):
    group = per_seed[per_seed.split == split]
    ax.scatter(np.full(len(group), i) + np.linspace(-.12, .12, len(group)),
               group.accuracy * 100, color="#2a78d6", s=35)
    ensemble_split = "validation_clean" if split == "validation" else split
    ensemble = rows[(rows.split == ensemble_split) & (rows.rule == "F0")].iloc[0]
    ax.hlines(ensemble.accuracy * 100, i - .28, i + .28, color="#1baf7a", lw=2)
ax.set(xticks=range(3), xticklabels=["Validation", "Test", "Holdout"],
       ylabel="Độ chính xác (%)", title="Năm seed và trung bình xác suất của DenseNet")
ax.text(.01, .01, "Holdout: đánh giá mô tả; Validation ensemble dùng phần sạch",
        transform=ax.transAxes, fontsize=8)
ax.grid(axis="y", alpha=.2)
save_figure(fig, "02_per_seed_accuracy.png")

fig, axes = plt.subplots(1, 3, figsize=(17, 5), gridspec_kw={"width_ratios": [1, 1, .7]})
for ax, split in zip(axes[:2], ("test", "holdout")):
    for i, rule in enumerate(METHODS):
        row = rows[(rows.split == split) & (rows.rule == rule)].iloc[0]
        mid = 100 * row.accuracy
        ax.errorbar(i, mid, yerr=[[mid - 100 * row.wilson_low],
                                  [100 * row.wilson_high - mid]], fmt="o", capsize=4,
                    color=COLORS[rule], markersize=8)
    ax.set(xticks=range(3), xticklabels=list(METHODS.values()),
           ylim=(0, 100), ylabel="Độ chính xác (%)", title=f"{split.title()}: ba cách dự đoán")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=.2)
for i, rule in enumerate(METHODS):
    row = gaps[gaps.method == rule].iloc[0]
    axes[2].errorbar(i, 100 * row.test_minus_holdout,
                     yerr=[[100 * (row.test_minus_holdout - row.bootstrap_low)],
                           [100 * (row.bootstrap_high - row.test_minus_holdout)]],
                     fmt="o", capsize=4, color=COLORS[rule])
axes[2].axhline(10, ls="--", color="grey", label="Mục tiêu 10 điểm")
axes[2].set(xticks=range(3), xticklabels=list(METHODS.values()),
            ylabel="Test trừ Holdout (điểm %)", title="Khoảng cách giữa hai tập")
axes[2].tick_params(axis="x", labelrotation=25)
axes[2].legend(fontsize=8)
save_figure(fig, "03_main_accuracy_and_gap.png")

fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
pd.DataFrame([{"split": split, "class": cls,
               "n": int((INVENTORY[split].true_label == cls).sum())}
              for split in ("test", "holdout") for cls in CLASSES]).to_csv(
                  AGGREGATE / "class_support.csv", index=False)
for ax, split in zip(axes, ("test", "holdout")):
    counts = INVENTORY[split].true_label.value_counts()
    present = [c for c in CLASSES if counts.get(c, 0) > 0]
    x = np.arange(len(present))
    for i, rule in enumerate(METHODS):
        row = rows[(rows.split == split) & (rows.rule == rule)].iloc[0]
        ax.bar(x + (i - 1) * .25, [row[f"recall_{c}"] for c in present],
               width=.24, color=COLORS[rule], label=METHODS[rule])
    ax.set(xticks=x, xticklabels=[f"{c}\n(n={counts[c]})" for c in present],
           ylim=(0, 1), ylabel="Recall theo lớp", title=f"{split.title()}: bệnh nào dễ bị bỏ sót?")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=.2)
axes[0].legend(fontsize=8)
save_figure(fig, "04_class_recall.png")

fig, axes = plt.subplots(2, 3, figsize=(19, 10))
for row_index, split in enumerate(("test", "holdout")):
    for col_index, rule in enumerate(METHODS):
        confusion_frame = pd.read_csv(AGGREGATE / f"confusion_B_{split}_{rule}.csv", index_col=0)
        values = confusion_frame.to_numpy(dtype=int)
        denom = values.sum(axis=1, keepdims=True)
        normalized = np.divide(values, denom, out=np.zeros_like(values, dtype=float), where=denom > 0)
        ax = axes[row_index, col_index]
        ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        for r in range(len(CLASSES)):
            for c in range(len(CLASSES)):
                ax.text(c, r, f"{values[r,c]}\n{normalized[r,c]:.0%}",
                        ha="center", va="center", fontsize=8,
                        color="white" if normalized[r,c] > .55 else "black")
        ax.set(xticks=range(5), xticklabels=CLASSES, yticks=range(5), yticklabels=CLASSES,
               xlabel="Dự đoán", ylabel="Nhãn thật", title=f"{split.title()}: {METHODS[rule]}")
        ax.tick_params(axis="x", labelrotation=45)
save_figure(fig, "05_confusion_matrices.png")

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, split in zip(axes, ("test", "holdout")):
    x = np.arange(3)
    bottom = np.zeros(3)
    for field, label, color in (("missed_disease", "Bỏ sót bệnh", "#777777"),
                                ("wrong_disease_type", "Nhầm bệnh", "#eb6834"),
                                ("false_alarm", "Báo bệnh giả", "#2a78d6")):
        vals = [errors[(errors.split == split) & (errors.rule == rule)].iloc[0][field]
                for rule in METHODS]
        ax.bar(x, vals, bottom=bottom, label=label, color=color)
        bottom += np.asarray(vals)
    ax.set(xticks=x, xticklabels=list(METHODS.values()), ylabel="Số ảnh lỗi",
           title=f"{split.title()}: ba loại lỗi")
    ax.tick_params(axis="x", labelrotation=25)
axes[0].legend(fontsize=8)
save_figure(fig, "06_error_types.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
calibration_rows = []
for ax, split in zip(axes, ("test", "holdout")):
    frame, predictions = PART_PREDICTIONS["B"][split]
    y = frame.true_label.map(CLASS_INDEX).to_numpy()
    for rule in METHODS:
        labels, probs = predictions[rule]
        confidence = probs.max(axis=1)
        correct = labels == y
        xs, ys = [], []
        for b in range(15):
            low, high = b / 15, (b + 1) / 15
            chosen = (confidence >= low) & ((confidence <= high) if b == 14 else (confidence < high))
            if chosen.any():
                observed = float(correct[chosen].mean())
                average = float(confidence[chosen].mean())
                xs.append(average)
                ys.append(observed)
                calibration_rows.append({"split": split, "method": rule, "bin": b,
                    "n": int(chosen.sum()), "mean_confidence": average, "observed_accuracy": observed})
        ece = rows[(rows.split == split) & (rows.rule == rule)].iloc[0].ece15
        ax.plot(xs, ys, "o-", color=COLORS[rule], label=f"{METHODS[rule]} (ECE {ece:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set(title=f"{split.title()}: tin cậy có khớp tần suất đúng?",
           xlabel="Confidence trung bình", ylabel="Tỷ lệ đúng", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=8)
pd.DataFrame(calibration_rows).to_csv(AGGREGATE / "reliability_bins.csv", index=False)
save_figure(fig, "07_reliability.png")

fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True, sharey=True)
for ax, split in zip(axes, ("test", "holdout")):
    for rule in METHODS:
        view = curves[(curves.split == split) & (curves.rule == rule)].dropna(subset=["risk"])
        ax.plot(view.coverage, 100 * view.risk, color=COLORS[rule], label=METHODS[rule])
        chosen = abstention[(abstention.split == split) & (abstention.rule == rule)].iloc[0]
        if np.isfinite(chosen.selective_accuracy):
            ax.scatter(chosen.coverage, 100 * (1 - chosen.selective_accuracy),
                       color=COLORS[rule], marker="x", s=70)
    ax.set(title=f"{split.title()}: từ chối ảnh khó để giảm lỗi",
           xlabel="Tỷ lệ ảnh tự kết luận", ylabel="Lỗi trong ảnh được nhận (%)",
           xlim=(0, 1))
    ax.legend(fontsize=8)
save_figure(fig, "08_risk_coverage.png")

for baseline, filename, title in (("F0", "09_paired_vs_densenet.png", "DenseNet một mình"),
                                   ("F3a", "09_appendix_vs_no_yolo.png", "stacker không YOLO")):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for i, split in enumerate(("test", "holdout")):
        row = paired[(paired.split == split) & (paired.baseline == baseline)].iloc[0]
        ax.plot([100 * row.bootstrap_low, 100 * row.bootstrap_high], [i, i],
                lw=3, color="#1baf7a")
        ax.scatter(100 * row.delta_accuracy, i, color="#1baf7a", s=60)
        ax.text(100 * row.bootstrap_high + .2, i,
                f"p={row.mcnemar_exact_p:.3g}; b={row.b_challenger_only_correct}, c={row.c_baseline_only_correct}",
                va="center", fontsize=8)
    ax.axvline(0, ls="--", color="grey")
    ax.set(yticks=[0, 1], yticklabels=["Test", "Holdout"],
           xlabel="Chênh lệch accuracy (điểm %), bootstrap 95%",
           title=f"DenseNet + YOLO so với {title}")
    ax.invert_yaxis()
    save_figure(fig, filename)

# Fixed-C bootstrap refits measure stacker instability, not a new selection.
refit_rng = np.random.default_rng(33034)
hold_x = matrix(holdB, "F3")
val_x = matrix(cleanB, "F3")
val_y = cleanB.true_label.map(CLASS_INDEX).to_numpy()
hold_y = holdB.true_label.map(CLASS_INDEX).to_numpy()
refit_rows = []
for repetition in range(200):
    for attempt in range(100):
        sample = refit_rng.integers(0, len(cleanB), size=len(cleanB))
        if len(np.unique(val_y[sample])) == len(CLASSES):
            break
    else:
        raise RuntimeError("Cannot bootstrap all five Validation classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=choiceB["F3"], max_iter=1500))
    model.fit(val_x[sample], val_y[sample])
    probs = model.predict_proba(hold_x)
    result = np.zeros((len(holdB), len(CLASSES)))
    result[:, model[-1].classes_] = probs
    prediction = apply_temperature(result, configB["temperature"]["F3"]).argmax(1)
    refit_rows.append({"replicate": repetition, "holdout_accuracy": float(np.mean(prediction == hold_y))})
refit = pd.DataFrame(refit_rows)
refit.to_csv(AGGREGATE / "stacker_refit_variability.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(100 * refit.holdout_accuracy, bins=15, color="#1baf7a", edgecolor="white")
ax.axvline(100 * refit.holdout_accuracy.mean(), color="black", ls="--")
ax.set(xlabel="Độ chính xác Holdout (%)", ylabel="Số lần fit lại",
       title="Kết quả Holdout thay đổi khi fit lại stacker trên Validation")
save_figure(fig, "10_stacker_refit_variability.png")
print("200 refit: mean", refit.holdout_accuracy.mean(), "min", refit.holdout_accuracy.min(),
      "max", refit.holdout_accuracy.max())

localization = pd.DataFrame(LOCALIZATION)
localization.to_csv(AGGREGATE / "localization_report.csv", index=False)
macro_rows = []
for (mode, subset), group in localization.groupby(["mode", "subset"]):
    macro_rows.append({"mode": mode, "subset": subset, "images": int(group.images.iloc[0]),
        "macro_precision50": float(group.precision50.mean()),
        "macro_recall50": float(group.recall50.mean()),
        "macro_f1_50": float(group.f1_50.mean()),
        "mAP50": float(group.ap50.mean()) if group.ap50.notna().any() else np.nan,
        "mAP50_95": float(group.ap50_95.mean()) if group.ap50_95.notna().any() else np.nan})
pd.DataFrame(macro_rows).to_csv(AGGREGATE / "localization_macro.csv", index=False)
fig, ax = plt.subplots(figsize=(8, 5))
pr_rows = []
for cls in DETECTOR_CLASSES:
    pred, matched = match_boxes(BOX_FRAMES["yolo_test"][BOX_FRAMES["yolo_test"].cls == cls],
                                GT_TEST[GT_TEST.cls == cls])
    count = int((GT_TEST.cls == cls).sum())
    if not len(pred) or not count:
        continue
    recall = np.cumsum(matched) / count
    precision = np.cumsum(matched) / np.arange(1, len(pred) + 1)
    ax.plot(recall, precision, label=cls)
    pr_rows.extend({"class": cls, "confidence": float(pred.iloc[i].conf),
                    "recall": float(recall[i]), "precision": float(precision[i])}
                   for i in range(len(pred)))
pd.DataFrame(pr_rows).to_csv(AGGREGATE / "localization_pr_curve.csv", index=False)
ax.set(xlabel="Recall của box", ylabel="Precision của box", xlim=(0, 1), ylim=(0, 1),
       title="YOLO Test: đường precision–recall của từng bệnh")
ax.legend()
save_figure(fig, "11_localization_pr.png")

# Draw on the original image first, then draw scaled edges once more for crisp panels.
BOX_COLORS = {"anthracnose": (42, 120, 214), "leaf_miner": (235, 104, 52),
              "red_rust": (27, 175, 122)}
font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")

def font(size):
    return ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default()

def annotated_tile(path, prediction):
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    width, height = image.size
    line = max(3, round(.004 * max(width, height)))
    source_boxes = prediction["boxes"]
    drawn = ImageDraw.Draw(image)
    for box in source_boxes:
        color = BOX_COLORS[box["cls"]]
        thickness = line * (2 if box["cls"] == prediction["label"] else 1)
        drawn.rectangle(box["xyxy"], outline=color, width=thickness)
        label = f"{box['cls']} {box['conf']:.2f}"
        label_size = max(14, round(.018 * max(width, height)))
        label_font = font(label_size)
        left = max(0, round(box["xyxy"][0]))
        top = max(0, round(box["xyxy"][1] - 1.5 * label_size))
        bounds = drawn.textbbox((left, top), label, font=label_font)
        drawn.rectangle(bounds, fill=color)
        drawn.text((left, top), label, fill="white", font=label_font)
    if source_boxes:
        pixels = np.asarray(image)
        assert any(np.all(pixels == color, axis=2).any() for color in BOX_COLORS.values())
    image.thumbnail((300, 270), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (320, 335), "white")
    x = (320 - image.width) // 2
    tile.paste(image, (x, 0))
    tile_draw = ImageDraw.Draw(tile)
    for box in source_boxes:
        color = BOX_COLORS[box["cls"]]
        scaled = [x + box["xyxy"][0] * image.width / width,
                  box["xyxy"][1] * image.height / height,
                  x + box["xyxy"][2] * image.width / width,
                  box["xyxy"][3] * image.height / height]
        tile_draw.rectangle(scaled, outline=color,
                            width=max(3, round(line * image.width / width)) *
                                  (2 if box["cls"] == prediction["label"] else 1))
    state = "cần chuyên gia kiểm tra" if prediction["abstain"] else (
        "không phát hiện bệnh" if prediction["label"] in ("healthy", "not_cashew_leaf")
        else "có box")
    tile_draw.text((5, 277), f"{prediction['label']}  {prediction['confidence']:.2f}",
                   fill="black", font=font(14))
    tile_draw.text((5, 297), f"Từ chối: {prediction['abstain']} | {state}",
                   fill="black", font=font(12))
    if source_boxes:
        assert any(np.all(np.asarray(tile) == color, axis=2).any() for color in BOX_COLORS.values())
    return tile

def panel(keys, filename):
    canvas = Image.new("RGB", (960, 670), "white")
    for i, key in enumerate(keys[:6]):
        path = INVENTORY["holdout"].set_index("evidence_key").loc[key, "path"]
        result = pipeline.predict(path)
        canvas.paste(annotated_tile(path, result), ((i % 3) * 320, (i // 3) * 335))
    destination = FIGURES / filename
    canvas.save(destination, dpi=(200, 200))
    display(canvas)

for cls in CLASSES[:4]:
    keys = sorted(INVENTORY["holdout"][INVENTORY["holdout"].true_label == cls].evidence_key)[:6]
    panel(keys, f"12_holdout_{cls}.png")
wrong = sorted(hold_frame.evidence_key[hold_preds["F3"][0] != hold_y])[:6]
panel(wrong, "12_holdout_six_errors.png")
pd.DataFrame([{"panel": cls, "evidence_key": key}
    for cls in CLASSES[:4]
    for key in sorted(INVENTORY["holdout"][INVENTORY["holdout"].true_label == cls].evidence_key)[:6]] +
    [{"panel": "combined_errors", "evidence_key": key} for key in wrong]).to_csv(
        AGGREGATE / "qualitative_panel_keys.csv", index=False)

# Predeclared traffic-light criteria; no rule is selected from these outcomes.
combined_test = rows[(rows.split == "test") & (rows.rule == "F3")].iloc[0]
combined_hold = rows[(rows.split == "holdout") & (rows.rule == "F3")].iloc[0]
gap = gaps[gaps.method == "F3"].iloc[0]
vs_dense_test = paired[(paired.split == "test") & (paired.baseline == "F0")].iloc[0]
vs_dense_hold = paired[(paired.split == "holdout") & (paired.baseline == "F0")].iloc[0]
hold_support = INVENTORY["holdout"].true_label.value_counts()
disease_n = int(sum(hold_support.get(c, 0) for c in DETECTOR_CLASSES))
missed = int(errors[(errors.split == "holdout") & (errors.rule == "F3")].iloc[0].missed_disease)

def traffic(value, target, low=None, direction="at_least"):
    passes = value >= target if direction == "at_least" else value <= target
    if not passes:
        return "FAIL"
    if low is not None and direction == "at_least" and low < target:
        return "BORDERLINE"
    return "PASS"

criteria = [
    {"criterion": "Test accuracy >=95%", "status": traffic(combined_test.accuracy, .95,
        combined_test.wilson_low), "estimate": combined_test.accuracy,
        "interval_low": combined_test.wilson_low, "interval_high": combined_test.wilson_high},
    {"criterion": "Holdout accuracy >=90%", "status": traffic(combined_hold.accuracy, .90,
        combined_hold.wilson_low), "estimate": combined_hold.accuracy,
        "interval_low": combined_hold.wilson_low, "interval_high": combined_hold.wilson_high},
    {"criterion": "Test-Holdout gap <10 points", "status": traffic(gap.test_minus_holdout, .10,
        direction="at_most"), "estimate": gap.test_minus_holdout,
        "interval_low": gap.bootstrap_low, "interval_high": gap.bootstrap_high},
    {"criterion": "Holdout missed-disease <=3%", "status": traffic(missed / disease_n, .03,
        direction="at_most"), "estimate": missed / disease_n,
        "interval_low": np.nan, "interval_high": np.nan},
    {"criterion": "Combined beats DenseNet on Holdout", "status": "PASS" if
        vs_dense_hold.bootstrap_low > 0 else "FAIL", "estimate": vs_dense_hold.delta_accuracy,
        "interval_low": vs_dense_hold.bootstrap_low,
        "interval_high": vs_dense_hold.bootstrap_high},
    {"criterion": "Combined Test loss <=1 point", "status": traffic(vs_dense_test.delta_accuracy,
        -.01), "estimate": vs_dense_test.delta_accuracy,
        "interval_low": vs_dense_test.bootstrap_low,
        "interval_high": vs_dense_test.bootstrap_high},
]
for split, record in (("test", combined_test), ("holdout", combined_hold)):
    available = [c for c in CLASSES if (INVENTORY[split].true_label == c).any()]
    for cls in available:
        support = int((INVENTORY[split].true_label == cls).sum())
        recall = float(record[f"recall_{cls}"])
        class_correct = int(round(recall * support))
        class_low, class_high = wilson(class_correct, support)
        criteria.append({"criterion": f"{split} recall {cls} >=85% (n={support})",
                         "status": traffic(recall, .85, class_low), "estimate": recall,
                         "interval_low": class_low, "interval_high": class_high})
criteria = pd.DataFrame(criteria)
criteria.to_csv(AGGREGATE / "traffic_light.csv", index=False)
verdict = "ĐẠT" if (criteria.status == "PASS").all() else (
    "CẦN THẬN TRỌNG" if not (criteria.status == "FAIL").any() else "CHƯA ĐẠT")
disclosure = ("Holdout 09 is a descriptive audit, not a blind benchmark; the combined design "
              "was first explored on it in EXP-032, so confirmation needs new field images.")
summary = {"experiment_id": EXPERIMENT_ID, "proposed_method": "combined",
           "verdict": verdict, "criteria": criteria.to_dict(orient="records"),
           "source_model_reuse": REUSE_032_MODELS, "holdout_exposed": True,
           "test_accuracy": float(combined_test.accuracy),
           "holdout_accuracy": float(combined_hold.accuracy),
           "holdout_n124_accuracy": float(rows[(rows.split == "holdout_n124") &
               (rows.rule == "F3")].iloc[0].accuracy),
           "disclosure": disclosure, "localization_pass_fail": "report only"}
write_json(AGGREGATE / "exp033_summary.json", summary)
readme = ["# EXP-033 — DenseNet + YOLO", "", f"Kết luận: **{verdict}**.",
          f"Test {combined_test.accuracy:.1%} (Wilson {combined_test.wilson_low:.1%}–{combined_test.wilson_high:.1%}); "
          f"Holdout {combined_hold.accuracy:.1%} (Wilson {combined_hold.wilson_low:.1%}–{combined_hold.wilson_high:.1%}).",
          "", "## Tiêu chí", "", markdown_table(criteria.round(4)), "",
          "## Giới hạn", "", disclosure,
          "Holdout chỉ có 126 ảnh, trong đó 19 ảnh healthy; chất lượng định vị YOLO cần đọc riêng trong localization_report.csv.",
          "Mô hình kết hợp được định trước; không chọn lại theo Test/Holdout.",
          "", "## Tệp", "", "Các CSV và hình PNG nằm trong aggregate/; "
          "pipeline_consistency.json và cashew_pipeline.py được đóng gói trong cả hai ZIP."]
(AGGREGATE / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
display(Markdown("### Kết luận ngắn\n\n" + "\n\n".join(readme[2:5]) +
                 "\n\n**Đèn tiêu chí:**\n\n" + markdown_table(criteria.round(4)) +
                 "\n\n" + disclosure))
