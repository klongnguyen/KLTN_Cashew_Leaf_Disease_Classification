# EXP-032 — render validation figures from existing aggregate CSVs.
# Run after the evaluation/localization cell, before the ANALYSIS snapshot upload cell.
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if "OUT_DIR" in globals():
    figure_root = Path(OUT_DIR)
else:
    candidates = [p for p in Path("/content/training_results/current_dataset").glob(
        "EXP-DENSENET121-YOLO-PIPELINE-032*") if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Set OUT_DIR to the intended EXP-032 result folder: {candidates}")
    figure_root = candidates[0]
aggregate = figure_root / "aggregate"
if not aggregate.is_dir():
    raise FileNotFoundError(f"No aggregate results in {figure_root}")
figures = aggregate / "figures"
figures.mkdir(exist_ok=True)
classes = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
rules = ["F0", "F1", "F2", "F3", "F3a", "F3b"]

def read_table(name):
    path = aggregate / name
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()

results = pd.concat([read_table("fusion_results_partA.csv"),
                     read_table("fusion_results_partB.csv")], ignore_index=True)
if results.empty:
    raise RuntimeError("No fusion results yet; run Part A or B first")
boot = read_table("paired_bootstrap.csv")
risk = read_table("risk_coverage.csv")
localization = read_table("localization_test.csv")
part = "B" if "B" in set(results.part) else "A"
if "SELECTED_DEFAULT" in globals() and part == "B":
    selected = SELECTED_DEFAULT
elif (aggregate / "exp032_summary.json").is_file() and part == "B":
    selected = json.loads((aggregate / "exp032_summary.json").read_text(
        encoding="utf-8")).get("selected_default_rule") or "F0"
else:
    selected = "F0"
if selected not in rules:
    raise RuntimeError(f"Unknown selected rule: {selected}")
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 170, "font.size": 10})
made = []

def save(fig, name):
    fig.tight_layout()
    destination = figures / name
    fig.savefig(destination, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    made.append(destination)

# Wilson intervals describe each accuracy separately; they are not the paired rule comparison.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
for ax, split in zip(axes, ("test", "holdout")):
    sub = results[results.split == split]
    for offset, current_part, color in ((-0.11, "A", "#4776b6"),
                                        (0.11, "B", "#c46b32")):
        group = sub[sub.part == current_part].set_index("rule")
        if group.empty:
            continue
        present = [r for r in rules if r in group.index]
        vals = group.loc[present]
        if not {"accuracy", "wilson_low", "wilson_high"}.issubset(vals.columns):
            continue
        x = np.array([rules.index(r) for r in present], dtype=float) + offset
        y = vals.accuracy.to_numpy(dtype=float) * 100
        err = np.vstack((y - vals.wilson_low.to_numpy(dtype=float) * 100,
                         vals.wilson_high.to_numpy(dtype=float) * 100 - y))
        ax.errorbar(x, y, yerr=err, fmt="o", capsize=3, color=color,
                    label=f"Part {current_part}")
    ax.set(title=f"{split.title()} accuracy (Wilson 95% CI)", xlabel="Fusion rule",
           xticks=range(len(rules)), xticklabels=rules)
    ax.grid(axis="y", alpha=.25)
    if split == "holdout":
        ax.text(.02, .02, "Descriptive audit; not a blind benchmark",
                transform=ax.transAxes, fontsize=8)
    ax.legend(loc="best")
axes[0].set_ylabel("Accuracy (%)")
save(fig, "accuracy_wilson.png")

if not boot.empty:
    sub = boot[boot.split.isin(["test", "holdout"])].copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True)
    for ax, split in zip(axes, ("test", "holdout")):
        view = sub[sub.split == split]
        labels = [f"{p} / {r}" for p in ("A", "B") for r in rules[1:]
                  if ((view.part == p) & (view.rule == r)).any()]
        for y, label in enumerate(labels):
            p, r = label.split(" / ")
            row = view[(view.part == p) & (view.rule == r)].iloc[0]
            lo, mid, hi = [100 * float(row[k]) for k in
                           ("bootstrap_low", "delta_accuracy", "bootstrap_high")]
            ax.plot([lo, hi], [y, y], color="#4776b6" if p == "A" else "#c46b32", lw=2)
            ax.plot(mid, y, "o", color="#4776b6" if p == "A" else "#c46b32")
        ax.axvline(0, color="black", lw=1, ls="--")
        ax.set(yticks=range(len(labels)), yticklabels=labels,
               title=f"{split.title()}: paired bootstrap, 10,000 resamples",
               xlabel="Accuracy difference vs F0 (percentage points)")
        ax.grid(axis="x", alpha=.25)
        if split == "holdout":
            ax.text(.02, .02, "Descriptive audit; not a blind benchmark",
                    transform=ax.transAxes, fontsize=8)
    axes[1].invert_yaxis()
    save(fig, "paired_bootstrap_vs_f0.png")

def confusion(part_name, split, rule):
    path = aggregate / f"confusion_{part_name}_{split}_{rule}.csv"
    return pd.read_csv(path, index_col=0) if path.is_file() else None

matrix_test = confusion(part, "test", selected)
matrix_hold = confusion(part, "holdout", selected)
if matrix_test is not None or matrix_hold is not None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, split, matrix in zip(axes, ("test", "holdout"),
                                 (matrix_test, matrix_hold)):
        if matrix is None:
            ax.set_axis_off()
            continue
        values = matrix.to_numpy(dtype=int)
        ax.imshow(values, cmap="Blues")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, str(values[i, j]), ha="center", va="center",
                        color="white" if values[i, j] > values.max() / 2 else "black")
        ax.set(xticks=range(len(matrix.columns)), xticklabels=matrix.columns,
               yticks=range(len(matrix.index)), yticklabels=matrix.index,
               xlabel="Predicted", ylabel="True",
               title=f"Part {part} {selected} — {split}" +
                     (" (descriptive audit)" if split == "holdout" else ""))
        ax.tick_params(axis="x", labelrotation=45)
    save(fig, "confusion_selected_rule.png")

recall_cols = [f"recall_{name}" for name in classes]
if set(recall_cols).issubset(results.columns):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, split in zip(axes, ("test", "holdout")):
        subset = results[(results.part == part) & (results.split == split)]
        for rule, color in (("F0", "#4776b6"), (selected, "#c46b32")):
            row = subset[subset.rule == rule]
            if row.empty:
                continue
            ax.plot(classes, row.iloc[0][recall_cols].to_numpy(dtype=float), "o-",
                    label=rule, color=color)
            if rule == selected:
                break
        ax.set(title=f"Part {part}: {split} recall by class" +
                     (" (descriptive audit)" if split == "holdout" else ""),
               ylim=(0, 1.05), ylabel="Recall", xlabel="True class")
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=.25)
        ax.legend()
    save(fig, "recall_by_class.png")

if not risk.empty:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, split in zip(axes, ("test", "holdout")):
        for rule, color in (("F0", "#4776b6"), (selected, "#c46b32")):
            view = risk[(risk.part == part) & (risk.split == split) &
                        (risk.rule == rule)].dropna(subset=["coverage", "risk"])
            if not view.empty:
                ax.plot(view.coverage, 100 * view.risk, label=rule, color=color)
            if rule == selected:
                break
        ax.set(title=f"Part {part}: {split} risk–coverage" +
                     (" (descriptive audit)" if split == "holdout" else ""),
               xlabel="Coverage (fraction accepted)")
        ax.grid(alpha=.25)
        ax.legend()
    axes[0].set_ylabel("Error among accepted (%)")
    save(fig, "risk_coverage.png")

if "ece15" in results.columns:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for ax, split in zip(axes, ("test", "holdout")):
        view = results[results.split == split]
        x = np.arange(len(rules))
        for offset, current_part, color in ((-.18, "A", "#4776b6"),
                                            (.18, "B", "#c46b32")):
            group = view[view.part == current_part].set_index("rule")
            vals = [100 * float(group.loc[r, "ece15"]) if r in group.index
                    else np.nan for r in rules]
            if np.isfinite(vals).any():
                ax.bar(x + offset, vals, width=.35, label=f"Part {current_part}",
                       color=color)
        ax.set(xticks=x, xticklabels=rules, title=f"{split.title()} ECE (15 bins)" +
               (" — descriptive audit" if split == "holdout" else ""),
               xlabel="Fusion rule")
        ax.grid(axis="y", alpha=.25)
        ax.legend()
    axes[0].set_ylabel("Expected calibration error (%)")
    save(fig, "calibration_ece15.png")

if not localization.empty:
    view = localization[localization.subset == "all"].copy()
    if "mode" not in view.columns:
        view["mode"] = "detector_only"
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(3)
    for offset, mode, label, color in ((-.17, "detector_only", "Detector", "#4776b6"),
                                       (.17, "fused_display", "Displayed", "#c46b32")):
        vals = []
        for cls in ("anthracnose", "leaf_miner", "red_rust"):
            row = view[(view["class"] == cls) & (view["mode"] == mode)]
            vals.append(float(row.iloc[0].f1_50) if not row.empty else np.nan)
        if np.isfinite(vals).any():
            ax.bar(x + offset, vals, width=.32, label=label, color=color)
    ax.set(xticks=x, xticklabels=["anthracnose", "leaf_miner", "red_rust"],
           ylim=(0, 1), ylabel="Box F1 at IoU 0.5",
           title="YOLO Test localization: raw detector vs shown boxes")
    ax.legend()
    ax.grid(axis="y", alpha=.25)
    save(fig, "localization_f1.png")

note = ("# EXP-032 validation figures\n\n"
        "Generated from aggregate CSVs without fitting or selecting any parameters.\n\n"
        "Wilson intervals describe single-rule accuracy. The paired bootstrap figure "
        "shows the 95% interval of each rule's difference from F0. Holdout figures "
        "are descriptive audits, not blind benchmarks. Localization uses labeled "
        "YOLO Test only; Holdout has no box ground truth. The risk–coverage curves "
        "sweep thresholds for illustration and do not retune the frozen abstention "
        "threshold. The ECE bar chart summarizes calibration error; a reliability diagram "
        "requires per-image calibrated probabilities, which the aggregate tables "
        "do not retain.\n")
(figures / "README.md").write_text(note, encoding="utf-8")
print("Saved", len(made), "validation figures to", figures)
for path in made:
    print(path.name)
