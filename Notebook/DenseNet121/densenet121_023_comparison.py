# Validation-only comparison. Baseline bytes come from the EXP-022 ANALYSIS ZIP.
BASELINE_SOURCE_ZLIB_B64 = "__BASELINE_B64__"
T95_DF4 = 2.7764451051977987


def mean_sd_ci(values):
    values = np.asarray(values, dtype=float)
    if len(values) != 5 or not np.isfinite(values).all():
        raise ValueError("Five finite seed values required")
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    half = T95_DF4 * sd / np.sqrt(5)
    return {"mean": mean, "sample_sd": sd, "ci95_low": mean-half, "ci95_high": mean+half}


def load_baseline():
    payload = json.loads(zlib.decompress(base64.b64decode(BASELINE_SOURCE_ZLIB_B64)))
    if payload["archive"] != "EXP-DENSENET121-TRANSFER-5SEEDS-022_ANALYSIS.zip":
        raise ValueError("Wrong EXP-022 baseline source")
    provenance = OUT_DIR / "provenance" / "exp022_validation_baseline"
    provenance.mkdir(parents=True, exist_ok=True)
    for relative, content in payload["files"].items():
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if digest != payload["sha256"][relative]:
            raise ValueError(f"EXP-022 source CSV hash mismatch: {relative}")
        target = provenance / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Existing baseline file differs: {target}")
        target.write_text(content, encoding="utf-8")
    save_json(provenance / "source.json", {key: payload[key] for key in
              ("archive", "archive_sha256", "sha256")})
    old = pd.read_csv(provenance / "aggregate" / "validation_results_5seeds.csv")
    if old["seed"].tolist() != SEEDS or not (old["disease_count"] == 915).all():
        raise ValueError("EXP-022 baseline does not match V10 seeds/denominator")
    return old, provenance


def compare_validation():
    old, provenance = load_baseline()
    new = pd.read_csv(AGGREGATE_DIR / "validation_results_5seeds.csv")
    if new["seed"].tolist() != SEEDS or not (new["disease_count"] == 915).all():
        raise ValueError("EXP-023 Validation rows do not match V10")
    paired = pd.DataFrame({"seed": SEEDS})
    for prefix, frame in (("exp022", old), ("exp023", new)):
        for metric in ("val_macro_f1", "val_accuracy", "false_healthy_rate", "false_healthy_count"):
            paired[f"{prefix}_{metric}"] = frame[metric].to_numpy()
        paired[f"{prefix}_healthy_recall"] = [
            float(pd.read_csv(
                (provenance if prefix == "exp022" else OUT_DIR) /
                f"seed_{seed}/validation/classification_report.csv", index_col=0
            ).loc["healthy", "recall"]) for seed in SEEDS]
    metrics = ("val_macro_f1", "healthy_recall", "false_healthy_rate")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[f"exp023_{metric}"] - paired[f"exp022_{metric}"]
    paired.to_csv(AGGREGATE_DIR / "comparison_exp022_exp023_by_seed.csv", index=False)
    summary = pd.DataFrame([
        {"metric": metric, "series": series, **mean_sd_ci(paired[f"{series}_{metric}"])}
        for metric in metrics for series in ("exp022", "exp023", "delta")
    ])
    summary.to_csv(AGGREGATE_DIR / "comparison_exp022_exp023_summary.csv", index=False)
    summary.to_csv(AGGREGATE_DIR / "confidence_intervals_95.csv", index=False)
    return paired, summary, provenance


def draw_validation_comparison(paired):
    fig, ax = plt.subplots(figsize=(7, 5), facecolor="white")
    for x, prefix, color in ((0, "exp022", "#64748b"), (1, "exp023", "#2563eb")):
        y = paired[f"{prefix}_val_macro_f1"].to_numpy() * 100
        stats = mean_sd_ci(y)
        ax.scatter(x + np.linspace(-.08, .08, 5), y, color=color)
        ax.errorbar(x, stats["mean"], yerr=[[stats["mean"]-stats["ci95_low"]],
                    [stats["ci95_high"]-stats["mean"]]], fmt="s", color="black", capsize=5)
    ax.set_xticks([0, 1], ["EXP-022", "EXP-023 +Brightness"])
    ax.set_ylabel("Validation Macro-F1 (%)"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(PANEL_DIR / "validation_macro_f1_ci95.png", dpi=180)
    display(fig); plt.close(fig)
    for metric, label, name in (
        ("val_macro_f1", "Macro-F1 (percentage points)", "paired_delta_macro_f1_ci95.png"),
        ("healthy_recall", "Healthy recall (percentage points)", "paired_delta_healthy_recall_ci95.png"),
        ("false_healthy_rate", "Disease → Healthy (pp; 915 diseased images/seed)",
         "paired_delta_disease_to_healthy_ci95.png")):
        y = paired[f"delta_{metric}"].to_numpy() * 100
        stats = mean_sd_ci(y)
        fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
        ax.axhline(0, color="black", linewidth=1)
        ax.plot(range(5), y, "o", color="#2563eb", markersize=8)
        ax.errorbar(5, stats["mean"], yerr=[[stats["mean"]-stats["ci95_low"]],
                    [stats["ci95_high"]-stats["mean"]]], fmt="s", color="#dc2626", capsize=5)
        ax.set_xticks(range(6), [str(s) for s in SEEDS] + ["Mean ± CI95"])
        ax.set_ylabel("EXP-023 − EXP-022: " + label); ax.grid(axis="y", alpha=.25)
        fig.tight_layout(); fig.savefig(PANEL_DIR / name, dpi=180)
        display(fig); plt.close(fig)
    per_class = pd.read_csv(AGGREGATE_DIR / "per_class_validation_by_seed.csv")
    healthy = per_class[per_class["class"] == "healthy"].set_index("seed").loc[SEEDS]
    fig, ax = plt.subplots(figsize=(8, 4), facecolor="white")
    x = np.arange(5)
    ax.plot(x, healthy["precision"]*100, "o-", label="Healthy precision")
    ax.plot(x, healthy["recall"]*100, "s-", label="Healthy recall")
    ax.set_xticks(x, [str(s) for s in SEEDS]); ax.set_ylabel("Validation (%)")
    ax.legend(); ax.grid(alpha=.25); fig.tight_layout()
    fig.savefig(PANEL_DIR / "healthy_precision_recall_by_seed.png", dpi=180)
    display(fig); plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 4), facecolor="white")
    for i, name in enumerate(CLASS_NAMES):
        values = per_class[per_class["class"] == name].set_index("seed").loc[SEEDS, "recall"]*100
        stats = mean_sd_ci(values)
        ax.scatter(i + np.linspace(-.08, .08, 5), values, alpha=.7)
        ax.errorbar(i, stats["mean"], yerr=[[stats["mean"]-stats["ci95_low"]],
                    [stats["ci95_high"]-stats["mean"]]], fmt="s", color="black", capsize=4)
    ax.set_xticks(range(5), CLASS_NAMES, rotation=25, ha="right")
    ax.set_ylabel("EXP-023 Validation recall (%)"); ax.grid(axis="y", alpha=.25)
    fig.tight_layout(); fig.savefig(PANEL_DIR / "per_class_recall_ci95.png", dpi=180)
    display(fig); plt.close(fig)
    errors = pd.read_csv(OUT_DIR / f"seed_{deployment_seed}/validation/error_cases.csv")
    errors = errors.sort_values("confidence", ascending=False).head(12)
    fig, axes = plt.subplots(3, 4, figsize=(12, 10), facecolor="white")
    for ax, (_, row) in zip(axes.flat, errors.iterrows()):
        with Image.open(row["filename"]) as image: ax.imshow(image.convert("RGB"))
        ax.set_title(f"{row['true_label']} → {row['predicted_label']}\n{row['confidence']:.2f}", fontsize=9)
        ax.axis("off")
    for ax in list(axes.flat)[len(errors):]: ax.axis("off")
    fig.suptitle(f"Validation errors, selected seed {deployment_seed}; inspect only")
    fig.tight_layout(); fig.savefig(PANEL_DIR / "validation_errors_seed_selected.png", dpi=180)
    display(fig); plt.close(fig)


if RUN_FULL_TRAINING:
    paired, comparison_summary, provenance = compare_validation()
    draw_validation_comparison(paired)
    display(paired, comparison_summary)
    per_class = pd.read_csv(AGGREGATE_DIR / "per_class_validation_by_seed.csv")
    old_per_class = []
    for seed in SEEDS:
        report = pd.read_csv(provenance / f"seed_{seed}/validation/classification_report.csv", index_col=0)
        for name in ("anthracnose", "leaf_miner", "red_rust"):
            old_per_class.append({"class": name, "recall": report.loc[name, "recall"]})
    old_mean = pd.DataFrame(old_per_class).groupby("class")["recall"].mean()
    new_mean = per_class.groupby("class")["recall"].mean()
    def value(metric, series):
        return comparison_summary.query("metric == @metric and series == @series")["mean"].iloc[0]
    guardrails = {
        "macro_f1_nonnegative_delta": bool(value("val_macro_f1", "exp023") >= value("val_macro_f1", "exp022")),
        "disease_to_healthy_below_baseline_plus_0_20pp":
            bool(value("false_healthy_rate", "exp023") <= value("false_healthy_rate", "exp022") + .002),
        "disease_recall_floor_minus_1pp": bool(all(
            new_mean[name] >= old_mean[name] - .01
            for name in ("anthracnose", "leaf_miner", "red_rust")))}
    save_json(OUT_DIR / "validation_decision_note.json", {
        "status": "REVIEW_REQUIRED" if all(guardrails.values()) else "RESEARCH_ONLY",
        "guardrails": guardrails, "automatic_promotion": False,
        "limitations": "Historical paired seeds on one split; only 3 real_ Healthy in Validation"})
else:
    print("EXP-022 comparison PENDING until five-seed Validation finishes")
