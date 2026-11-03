# EXP-034: training diagnostics and final report for the combined model only.
import matplotlib.pyplot as plt
from PIL import ImageFont
try:
    from IPython.display import display, Markdown
except ImportError:  # Local notebook smoke test without Jupyter.
    def display(value):
        print(value)
    def Markdown(value):
        return value

FIGURES = AGGREGATE / "figures"
FIGURES.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 200, "font.size": 10})

def save_training_figure(fig, name):
    fig.tight_layout()
    fig.savefig(FIGURES / name, dpi=200, bbox_inches="tight")
    plt.show()
    plt.close(fig)

def markdown_table(frame):
    columns = [str(name) for name in frame.columns]
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join("---" for _ in columns) + "|"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(item).replace("|", "/") for item in row) + " |")
    return "\n".join(lines)

# These are training diagnostics for the DenseNet component inside this one pipeline.
training_rows, history_rows = [], []
for seed in SEEDS:
    meta = json.loads((OUT_DIR / f"seed_{seed}" / "seed_summary.json").read_text(encoding="utf-8"))
    selected = meta["selected_stage"]
    for stage in meta["stages"]:
        training_rows.append({"seed": seed, "stage": stage["stage"],
            "selected": stage["stage"] == selected, "epochs": int(stage["epochs"]),
            "val_loss": float(stage["val_loss"]), "seconds": float(stage["seconds"]),
            "model_sha256": meta["deployment_sha256"]})
        for epoch, loss in enumerate(stage["history"]["loss"], start=1):
            history_rows.append({"seed": seed, "stage": stage["stage"], "epoch": epoch,
                "loss": float(loss), "val_loss": float(stage["history"]["val_loss"][epoch-1]),
                "accuracy": float(stage["history"]["accuracy"][epoch-1]),
                "val_accuracy": float(stage["history"]["val_accuracy"][epoch-1])})
training_table = pd.DataFrame(training_rows)
history_table = pd.DataFrame(history_rows)
training_table.to_csv(AGGREGATE / "combined_training_stages.csv", index=False)
history_table.to_csv(AGGREGATE / "combined_training_history.csv", index=False)
display(Markdown("### Huấn luyện phần DenseNet trong pipeline kết hợp\n\n" +
                 markdown_table(training_table.round(4))))

fig, axes = plt.subplots(2, len(SEEDS), figsize=(19, 7), sharey="row")
for column, seed in enumerate(SEEDS):
    cursor = 0
    for stage in ("stage1", "stage2"):
        view = history_table[(history_table.seed == seed) & (history_table.stage == stage)]
        x = np.arange(cursor + 1, cursor + len(view) + 1)
        for row, metric, validation in ((0, "loss", "val_loss"),
                                        (1, "accuracy", "val_accuracy")):
            axes[row, column].plot(x, view[metric], color="#2a78d6", label="Huấn luyện")
            axes[row, column].plot(x, view[validation], color="#eb6834", label="Thẩm định")
        cursor += len(view)
        if stage == "stage1":
            for row in (0, 1):
                axes[row, column].axvline(cursor + .5, color="grey", ls="--", lw=.8)
    axes[0, column].set_title(f"Seed {seed}: hàm mất mát")
    axes[1, column].set_title(f"Seed {seed}: độ chính xác")
    for row in (0, 1):
        axes[row, column].set_xlabel("Vòng huấn luyện")
        axes[row, column].grid(alpha=.2)
axes[0, 0].set_ylabel("Hàm mất mát")
axes[1, 0].set_ylabel("Độ chính xác")
axes[0, 0].legend(fontsize=8)
save_training_figure(fig, "00_densenet_component_training_curves.png")

cv = pd.read_csv(AGGREGATE / "stacker_cv_partB.csv")
cv = cv[cv.rule == "F3"].copy()
assert len(cv) == 5 and choiceB["F3"] in set(cv.C)
cv.to_csv(AGGREGATE / "combined_stacker_cv.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(cv.C, cv.cv_log_loss, "o-", color="#1baf7a")
ax.axvline(choiceB["F3"], color="grey", ls="--", label=f"C đã khóa = {choiceB['F3']}")
ax.set(xscale="log", xlabel="C của logistic regression",
       ylabel="Log loss ngoài fold", title="Chọn C bằng Validation theo nhóm pHash")
ax.legend()
ax.grid(alpha=.2)
save_training_figure(fig, "00_combined_stacker_grouped_cv.png")

# A fixed-C bootstrap refit measures the stability of this combined stacker only.
refit_rng = np.random.default_rng(34034)
train_x = matrix(cleanB, "F3")
train_y = cleanB.true_label.map(CLASS_INDEX).to_numpy()
hold_x = matrix(holdB, "F3")
hold_y = holdB.true_label.map(CLASS_INDEX).to_numpy()
refit_rows = []
for repeat in range(200):
    for attempt in range(100):
        sample = refit_rng.integers(0, len(cleanB), size=len(cleanB))
        if len(np.unique(train_y[sample])) == len(CLASSES):
            break
    else:
        raise RuntimeError("Bootstrap sample did not contain all five classes")
    model = make_pipeline(StandardScaler(), LogisticRegression(C=choiceB["F3"], max_iter=1500))
    model.fit(train_x[sample], train_y[sample])
    raw = np.zeros((len(holdB), len(CLASSES)))
    raw[:, model[-1].classes_] = model.predict_proba(hold_x)
    pred = apply_temperature(raw, configB["temperature"]["F3"]).argmax(1)
    refit_rows.append({"replicate": repeat, "holdout_accuracy": float(np.mean(pred == hold_y))})
refit = pd.DataFrame(refit_rows)
refit.to_csv(AGGREGATE / "combined_stacker_refit.csv", index=False)
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(refit.holdout_accuracy * 100, bins=15, color="#1baf7a", edgecolor="white")
ax.axvline(refit.holdout_accuracy.mean() * 100, ls="--", color="black")
ax.set(xlabel="Độ chính xác Holdout (%)", ylabel="Số lần fit lại",
       title="Độ ổn định của phần kết hợp qua 200 lần fit lại")
save_training_figure(fig, "00_combined_stacker_refit.png")

# Final tables contain only the combined rule, including OOF Validation and both Holdout views.
metrics = pd.DataFrame(ALL_RESULTS)
metrics = metrics[(metrics.part == "B") & (metrics.rule == "F3") &
                  (metrics.split.isin(["validation_clean", "test", "holdout", "holdout_n124"]))].copy()
errors = pd.DataFrame(ALL_TAXONOMY)
errors = errors[(errors.part == "B") & (errors.rule == "F3")]
abstention = pd.DataFrame(ALL_ABSTENTION)
abstention = abstention[(abstention.part == "B") & (abstention.rule == "F3")]
table = metrics.merge(errors[["split", "missed_disease", "wrong_disease_type", "false_alarm"]],
                      on="split", validate="one_to_one")
table = table.merge(abstention[["split", "threshold", "coverage", "selective_accuracy",
                               "referred_to_expert"]], on="split", validate="one_to_one")
table = table.drop(columns=["part", "rule"])
table.loc[table.split.str.startswith("holdout"), "recall_not_cashew_leaf"] = np.nan
table.to_csv(AGGREGATE / "combined_results.csv", index=False)
display(Markdown("### Kết quả cuối của DenseNet121 + YOLO\n\n" +
                 markdown_table(table.round(4)) +
                 "\n\nValidation là dự đoán ngoài fold theo nhóm pHash. "
                 "Holdout 09 đã được xem trước, chỉ là đánh giá mô tả."))

# Draw deterministic qualitative panels using the frozen combined pipeline.
_report_bundle = OUT_DIR / "pipeline_bundle"
_report_consistency = AGGREGATE / "pipeline_consistency.json"
if ("pipeline" not in globals() or
        Path(getattr(pipeline, "bundle_dir", "")).resolve() != _report_bundle.resolve()):
    if not _report_consistency.is_file() or not (_report_bundle / "cashew_pipeline.py").is_file():
        raise RuntimeError(
            "Run the pipeline consistency cell immediately before this report cell; "
            "the frozen pipeline bundle is not ready. DenseNet training does not need to be repeated.")
    _report_gate = json.loads(_report_consistency.read_text(encoding="utf-8"))
    if _report_gate.get("status") != "PASS":
        raise RuntimeError("Pipeline consistency did not pass; inspect that cell before reporting")
    import importlib.util
    _report_spec = importlib.util.spec_from_file_location(
        "cashew_pipeline_exp034_report", _report_bundle / "cashew_pipeline.py")
    _report_module = importlib.util.module_from_spec(_report_spec)
    _report_spec.loader.exec_module(_report_module)
    pipeline = _report_module.CashewPipeline(_report_bundle)

box_colors = {"anthracnose": (42, 120, 214), "leaf_miner": (235, 104, 52),
              "red_rust": (27, 175, 122)}
font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
def panel_font(size):
    return ImageFont.truetype(str(font_path), size) if font_path.is_file() else ImageFont.load_default()

def annotated_tile(path, result):
    with Image.open(path) as handle:
        image = ImageOps.exif_transpose(handle).convert("RGB")
    w, h = image.size
    line = max(3, round(.004 * max(w, h)))
    draw = ImageDraw.Draw(image)
    for box in result["boxes"]:
        color = box_colors[box["cls"]]
        width = line * (2 if box["cls"] == result["label"] else 1)
        draw.rectangle(box["xyxy"], outline=color, width=width)
        label = f"{box['cls']} {box['conf']:.2f}"
        size = max(14, round(.018 * max(w, h)))
        pos = (max(0, round(box["xyxy"][0])), max(0, round(box["xyxy"][1] - 1.5*size)))
        bounds = draw.textbbox(pos, label, font=panel_font(size))
        draw.rectangle(bounds, fill=color)
        draw.text(pos, label, fill="white", font=panel_font(size))
    image.thumbnail((300, 270), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (320, 335), "white")
    xoff = (320-image.width)//2
    tile.paste(image, (xoff, 0))
    draw = ImageDraw.Draw(tile)
    for box in result["boxes"]:
        color = box_colors[box["cls"]]
        rect = [xoff + box["xyxy"][0]*image.width/w, box["xyxy"][1]*image.height/h,
                xoff + box["xyxy"][2]*image.width/w, box["xyxy"][3]*image.height/h]
        draw.rectangle(rect, outline=color, width=max(3, round(line*image.width/w)) *
                       (2 if box["cls"] == result["label"] else 1))
    state = "cần chuyên gia kiểm tra" if result["abstain"] else (
        "không phát hiện bệnh" if result["label"] in ("healthy", "not_cashew_leaf") else "có box")
    draw.text((5, 277), f"{result['label']} {result['confidence']:.2f}",
              fill="black", font=panel_font(14))
    draw.text((5, 297), f"Từ chối: {result['abstain']} | {state}",
              fill="black", font=panel_font(12))
    if result["boxes"]:
        pixels = np.asarray(tile)
        assert any(np.all(pixels == color, axis=2).any() for color in box_colors.values())
    return tile

def save_panel(keys, name):
    canvas = Image.new("RGB", (960, 670), "white")
    inventory = INVENTORY["holdout"].set_index("evidence_key")
    for i, key in enumerate(keys[:6]):
        path = inventory.loc[key, "path"]
        canvas.paste(annotated_tile(path, pipeline.predict(path)), ((i % 3)*320, (i // 3)*335))
    canvas.save(FIGURES / name, dpi=(200, 200))
    display(canvas)

panel_keys = []
for cls in CLASSES[:4]:
    keys = sorted(INVENTORY["holdout"][INVENTORY["holdout"].true_label == cls].evidence_key)[:6]
    save_panel(keys, f"holdout_{cls}.png")
    panel_keys.extend({"panel": cls, "evidence_key": key} for key in keys)
hold_frame, hold_predictions = PART_PREDICTIONS["B"]["holdout"]
hold_true = hold_frame.true_label.map(CLASS_INDEX).to_numpy()
wrong = sorted(hold_frame.evidence_key[hold_predictions["F3"][0] != hold_true])[:6]
save_panel(wrong, "holdout_combined_errors.png")
panel_keys.extend({"panel": "errors", "evidence_key": key} for key in wrong)
pd.DataFrame(panel_keys).to_csv(AGGREGATE / "combined_panel_keys.csv", index=False)

test = table[table.split == "test"].iloc[0]
holdout = table[table.split == "holdout"].iloc[0]
gap = float(test.accuracy - holdout.accuracy)
summary = {"experiment_id": EXPERIMENT_ID, "model": "DenseNet121 + YOLO26s",
    "status": "synthetic dry run" if DRY_RUN else "real run",
    "seeds": SEEDS, "selected_C": float(choiceB["F3"]),
    "temperature": float(configB["temperature"]["F3"]),
    "abstain_threshold": float(configB["abstain_thresholds"]["F3"]),
    "box_conf": float(BOX_CONF), "test_accuracy": float(test.accuracy),
    "test_wilson_95": [float(test.wilson_low), float(test.wilson_high)],
    "holdout_accuracy": float(holdout.accuracy),
    "holdout_wilson_95": [float(holdout.wilson_low), float(holdout.wilson_high)],
    "test_minus_holdout": gap,
    "holdout_n": int(holdout.n),
    "source_hashes": {"split13": DATASET_ARCHIVE_SHA256,
                      "detector_weights": DETECTOR_WEIGHTS_SHA256}}
write_json(AGGREGATE / "combined_summary.json", summary)
readme = ["# DenseNet121 + YOLO26s — EXP-034", "",
          "Báo cáo này chỉ đánh giá pipeline kết hợp. Không chọn hoặc so sánh phương pháp khác.",
          f"Test: {test.accuracy:.1%} (Wilson 95% {test.wilson_low:.1%}–{test.wilson_high:.1%}).",
          f"Holdout 09: {holdout.accuracy:.1%} (Wilson 95% {holdout.wilson_low:.1%}–{holdout.wilson_high:.1%}).",
          f"Khoảng cách Test–Holdout: {gap*100:.1f} điểm phần trăm.",
          f"C={choiceB['F3']}; nhiệt độ={configB['temperature']['F3']:.4f}; "
          f"ngưỡng từ chối={configB['abstain_thresholds']['F3']:.4f}; BOX_CONF={BOX_CONF:.2f}.",
          "Validation là ngoài fold theo nhóm pHash. Holdout là đánh giá mô tả đã được xem trong EXP-032, không phải blind benchmark.",
          "Holdout không có ground truth box; chất lượng box chỉ đo trên YOLO Test.",
          "Các bảng CSV và hình PNG nằm trong aggregate/. Hai ZIP chứa pipeline và báo cáo consistency."]
if DRY_RUN:
    readme.append("DRY RUN tổng hợp: detector giả lập, 2 seed, 1 epoch mỗi giai đoạn; không phải kết quả thật.")
(AGGREGATE / "README.md").write_text("\n\n".join(readme) + "\n", encoding="utf-8")
display(Markdown("### Kết luận ngắn\n\n" + "\n\n".join(readme[1:8])))
