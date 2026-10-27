"""Build EXP-018 from the frozen EXP-014 notebook without carrying old outputs."""

import base64
import json
import re
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "densenet121_transfer_5seed_014.ipynb"
TARGET = HERE / "densenet121_transfer_5seed_018.ipynb"
book = json.loads(SOURCE.read_text(encoding="utf-8"))


def change(cell, old, new, count=1):
    body = "".join(book["cells"][cell]["source"])
    found = body.count(old)
    if found != count:
        raise RuntimeError(f"cell {cell}: expected {count} occurrence(s), found {found}: {old[:80]}")
    book["cells"][cell]["source"] = body.replace(old, new).splitlines(keepends=True)


def set_cell(cell, body):
    book["cells"][cell]["source"] = body.splitlines(keepends=True)


def markdown(body):
    return {"cell_type": "markdown", "metadata": {}, "source": body.splitlines(keepends=True)}


def code(body):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": body.splitlines(keepends=True)}


for cell in book["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

set_cell(0, """# EXP-DENSENET121-TRANSFER-5SEEDS-018 — Split 9

DenseNet121 ImageNet transfer learning, five fixed seeds, two stages, based on
EXP-014. This is a **new** experiment on `CashewData_Split (9).zip` and
`RL_Cashew_holdout (2).zip`. No EXP-014 weights or metrics are reused.

The stage/seed selection rule remains Validation loss with macro-F1 tie-break,
so the training recipe stays comparable. Different dataset versions must not be
pooled in a paired statistical comparison. The external holdout contains four
true classes; `not_cashew_leaf` has no true support there.

**Run 1:** `RUN_FULL_TRAINING=True`, `RUN_FINAL_TEST=False`,
`RUN_REAL_HOLDOUT=False`. The pipeline audits both archives, trains five seeds,
locks Validation selection, and exports Validation artifacts.

**Run 2:** change only the three flags to `False, True, True` and run all cells.
This evaluates the locked checkpoints once on Test and external holdout, then
refreshes the reports and ZIPs. Never use either result to tune the recipe.

**Colab account:** the configuration finds the project through its `MyDrive`
shortcut. Set `FORCE_REMOUNT_DRIVE=True` only if the runtime still mounts the
previous account; use the two Drive path overrides if the shortcut has an
unusual location or the shared project is read-only.

**Status:** code prepared; numerical training/evaluation results are pending execution.
""")

config = "".join(book["cells"][4]["source"])
start = config.index('EXPECTED_COUNTS = {')
end = config.index('# ------------------------------------------------------------\n# Fixed five-seed', start)
counts = '''EXPECTED_COUNTS = {
    "anthracnose": {"train": 982, "val": 316, "test": 122, "total": 1420},
    "healthy": {"train": 828, "val": 216, "test": 118, "total": 1162},
    "leaf_miner": {"train": 875, "val": 264, "test": 132, "total": 1271},
    "not_cashew_leaf": {"train": 1102, "val": 314, "test": 156, "total": 1572},
    "red_rust": {"train": 1127, "val": 335, "test": 158, "total": 1620},
}
EXPECTED_TOTALS = {"train": 4914, "val": 1445, "test": 686, "total": 7045}
STRICT_DATASET_CHECK = True
EXPECTED_REAL_FILENAME_COUNTS = {"train": 236, "val": 50, "test": 28}
REQUIRED_ABSENT_IMAGES = []  # V08-specific removals are not asserted for V09.

'''
config = config[:start] + counts + config[end:]
config = config.replace('NOTEBOOK_VERSION = "014"', 'NOTEBOOK_VERSION = "018"')
config = config.replace('CONFIGURATION — EXP-014', 'CONFIGURATION — EXP-018')
config = config.replace('NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 014 .ipynb on Drive.',
                        'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 018 .ipynb on Drive.')
config = config.replace('EXP-DENSENET121-TRANSFER-5SEEDS-014', 'EXP-DENSENET121-TRANSFER-5SEEDS-018')
config = config.replace('Cashew_dataV08_real_val', 'Cashew_dataV09_more_real')
config = config.replace('CashewData_Split (8).zip', 'CashewData_Split (9).zip')
config = config.replace('DATASET_ARCHIVE_OVERRIDE = None  # Optional exact ZIP path, useful on Kaggle.',
                        'DATASET_ARCHIVE_OVERRIDE = None  # Optional exact ZIP path.\n'
                        'DRIVE_PROJECT_DIR_OVERRIDE = None  # Mounted shortcut path, if auto-discovery fails.\n'
                        'DRIVE_RESULTS_ROOT_OVERRIDE = None  # Writable folder in the new account, if needed.\n'
                        'FORCE_REMOUNT_DRIVE = False  # True if this Colab runtime still mounts the old account.')
config = config.replace('EXPECTED_MANIFEST_SHA256 = "1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b"',
                        'EXPECTED_MANIFEST_SHA256 = None  # Archive SHA256 + saved manifest freeze this new dataset.')
config = config.replace('RL_Cashew_holdout.zip', 'RL_Cashew_holdout (2).zip')
config = config.replace('BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165',
                        '47A5722660E2C8DC7CC04C9FF720E1F889C2481EC99112B41BC0606958A7FDD7')
config = config.replace('{"anthracnose": 13, "healthy": 5, "leaf_miner": 24, "red_rust": 7}',
                        '{"anthracnose": 46, "healthy": 20, "leaf_miner": 36, "red_rust": 25}')
config = config.replace('historical 48-image set', 'new 127-image archive')
config = config.replace('Historical, already inspected 48-image set.',
                        'New 127-image archive; evaluate after Validation lock.')
config = config.replace('Evaluation requires a completed EXP-014 Validation lock.',
                        'Evaluation requires a completed EXP-018 Validation lock.')
config = config.replace('drive.mount("/content/drive")',
                        'drive.mount("/content/drive", force_remount=FORCE_REMOUNT_DRIVE)')
old_drive_block = '''    DRIVE_PROJECT_DIR = Path("/content/drive/MyDrive/Cashew_Leaf_model_result")
    DRIVE_DATASET_ARCHIVE = DRIVE_PROJECT_DIR / "Dataset" / DATASET_ARCHIVE_NAME
    if not DRIVE_PROJECT_DIR.is_dir():
        raise FileNotFoundError(DRIVE_PROJECT_DIR)
    RESULTS_ROOT = DRIVE_PROJECT_DIR / "training_results" / "current_dataset"'''
new_drive_block = '''    drive_root = Path("/content/drive/MyDrive")
    if DRIVE_PROJECT_DIR_OVERRIDE is not None:
        DRIVE_PROJECT_DIR = Path(DRIVE_PROJECT_DIR_OVERRIDE).expanduser()
    else:
        # The shortcut in this account may have a different display name.
        direct = drive_root / "Cashew_Leaf_model_result"
        candidates = [direct] if (direct / "Dataset" / DATASET_ARCHIVE_NAME).is_file() else []
        if not candidates:
            candidates = [p for p in drive_root.iterdir() if p.is_dir() and
                          (p / "Dataset" / DATASET_ARCHIVE_NAME).is_file()]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Found {len(candidates)} project shortcuts containing {DATASET_ARCHIVE_NAME}. "
                "Set DRIVE_PROJECT_DIR_OVERRIDE to the mounted shortcut path. "
                "The new account needs access to the original folder.")
        DRIVE_PROJECT_DIR = candidates[0]
    DRIVE_DATASET_ARCHIVE = DRIVE_PROJECT_DIR / "Dataset" / DATASET_ARCHIVE_NAME
    if not DRIVE_DATASET_ARCHIVE.is_file():
        raise FileNotFoundError(f"Dataset ZIP not accessible via shortcut: {DRIVE_DATASET_ARCHIVE}")
    RESULTS_ROOT = (Path(DRIVE_RESULTS_ROOT_OVERRIDE).expanduser() if
                    DRIVE_RESULTS_ROOT_OVERRIDE is not None else
                    DRIVE_PROJECT_DIR / "training_results" / "current_dataset")
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    write_probe = RESULTS_ROOT / f".{EXPERIMENT_ID}.write_probe"
    try:
        write_probe.write_text("ok", encoding="utf-8")
    except OSError as exc:
        raise PermissionError(
            f"Cannot write to {RESULTS_ROOT}. Grant access to the original folder "
            "or set DRIVE_RESULTS_ROOT_OVERRIDE to a writable folder in this account.") from exc
    finally:
        write_probe.unlink(missing_ok=True)
    print("Colab project shortcut:", DRIVE_PROJECT_DIR)
    print("Results directory:", RESULTS_ROOT)'''
if old_drive_block not in config:
    raise RuntimeError("Original Colab path block changed")
config = config.replace(old_drive_block, new_drive_block)
set_cell(4, config)

change(6, '75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7',
       '1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4')
change(6, 'split_08', 'split_09', 2)
change(6, 'Split (8)', 'Split (9)', 11)
change(8, 'Split (8)', 'Split (9)', 4)
change(8, 'if DATASET_MANIFEST_SHA256 != EXPECTED_MANIFEST_SHA256:',
       'if EXPECTED_MANIFEST_SHA256 is not None and DATASET_MANIFEST_SHA256 != EXPECTED_MANIFEST_SHA256:')
change(8, 'The 35 real_', 'The 50 real_', 0) if False else None
change(16, 'The 35 real_ Validation filenames changed',
       'The 50 real_ Validation filenames changed')
change(9, 'CashewData_Split (8) - Class Distribution',
       'CashewData_Split (9) - Class Distribution')
change(14, '"baseline_source": "EXP-DENSENET121-SCRATCH-5SEEDS-013 source notebook",',
       '"baseline_source": "EXP-014 protocol; no checkpoints or prior metrics reused",')
change(14, '"comparison_scope": "method comparison; ImageNet + two stages versus scratch",',
       '"comparison_scope": "new V09 dataset; compare only with runs sharing the exact manifest",')
change(14, '"holdout_protocol": "previously exposed 48-image historical comparison only",',
       '"holdout_protocol": "new 127-image four-class archive; Validation lock precedes evaluation",')
change(14, 'Existing EXP-014 recipe differs.', 'Existing EXP-018 recipe differs.')
change(18, 'if RUN_FINAL_TEST:\n    assert_selection_lock()\n    test_ds =',
       'if RUN_FINAL_TEST and (AGGREGATE_DIR / "test_results_5seeds.csv").is_file():\n'
       '    assert_selection_lock()\n'
       '    test_results_df = pd.read_csv(AGGREGATE_DIR / "test_results_5seeds.csv")\n'
       '    if test_results_df["seed"].tolist() != SEEDS or any(\n'
       '            not (OUT_DIR / f"seed_{seed}" / "test" / "predictions.csv").is_file()\n'
       '            for seed in SEEDS):\n'
       '        raise RuntimeError("Existing Final Test artifacts are incomplete.")\n'
       '    print("Final Test already completed; reading frozen results.")\n'
       'elif RUN_FINAL_TEST:\n'
       '    assert_selection_lock()\n'
       '    if any((OUT_DIR / f"seed_{seed}" / "test" / "metrics.json").exists() for seed in SEEDS):\n'
       '        raise RuntimeError("Partial Test exists; inspect before retrying to avoid re-scoring.")\n'
       '    test_ds =')
change(17, 'retrospective comparison because EXP-013 Test\nresults were already available during EXP-014 design.',
       'evaluation on V09 only after the EXP-018 Validation lock.')
set_cell(19, """## External real holdout

The new archive has 127 images (46 anthracnose, 20 healthy, 36 leaf miner,
25 red rust). The notebook verifies its SHA-256, checks exact byte overlap
against all three internal splits, then evaluates the same five locked models.
Its four-class macro metrics and 127-image support are reported separately
from the five-class internal Test. Exact bytes alone cannot rule out related
views of the same leaf or tree.
""")
change(20, 'Split (8)', 'Split (9)', 3)
change(20, 'Historical external classification evaluation embedded in EXP-014.',
       'External classification evaluation for EXP-018.')
change(20,
       'if Path("/content/drive/MyDrive").is_dir():\n'
       '        candidate = Path("/content/drive/MyDrive/Cashew_Leaf_model_result/Dataset") / archive_name',
       'if RUNTIME == "Colab":\n'
       '        candidate = DRIVE_PROJECT_DIR / "Dataset" / archive_name')
change(20, 'on Colab add the supplied Dataset folder to My Drive.',
       'on Colab check the project shortcut and original-folder permissions.')
change(20, 'expected_overlap = {("test", "red_rust/IMG_3244.jpg")}',
       'expected_overlap = set()  # Verified against both local ZIPs before authoring EXP-018.')
change(20, '# One known Test duplicate is excluded from scoring, not concealed.',
       '# Exact overlap is unexpected for this frozen V09/holdout02 pair.')
change(20, 'real_holdout_014', 'real_holdout_018', 3)
change(20, 'if len(holdout_manifest) != 48:', 'if len(holdout_manifest) != 127:')
change(20, 'Historical holdout must have 48 scored images',
       'Holdout02 must have 127 scored images after exact-overlap audit')
change(20, '"status": "HISTORICAL_EXPOSED", "evaluated_images": 48,',
       '"status": "EVALUATED_SEPARATELY", "evaluated_images": 127,')
change(20, '"excluded_exact_test_duplicate": 1,', '"excluded_exact_test_duplicate": 0,')
change(20, '"use": "descriptive comparison only; not new blind field validation",',
       '"use": "external four-class descriptive evaluation; no Test/holdout tuning",')
change(20, 'Real holdout PENDING (RUN_REAL_HOLDOUT=False; historical exposed set)',
       'Real holdout PENDING (RUN_REAL_HOLDOUT=False)')
change(22, '"historical_holdout_exposed": True,',
       '"holdout_archive": REAL_HOLDOUT_ARCHIVE_NAME,')
change(18, 'restoring the EXP-014 trained checkpoint.',
       'restoring the EXP-018 trained checkpoint.')
change(22, 'Do not overwrite EXP-014 checkpoints.', 'Do not overwrite EXP-018 checkpoints.')
set_cell(25, """## Completion check

- Both archive SHA-256 values and the V09 class/real_ counts match.
- The dataset audit reports corrupt images, exact cross-split matches and pHash candidates.
- Holdout02 contains 127 images and has no exact-byte overlap with Train/Val/Test.
- Pipeline smoke and all five Validation runs finish before the selection lock.
- Test and holdout are run only in a later evaluation pass with the locked checkpoints.
- Reports, panels, manifests, notebook copy and FULL/ANALYSIS ZIPs are refreshed after that pass.
""")

preflight = r'''# 06b. External archive preflight before training
# This is structural/data-integrity inspection only: no model predictions or tuning.
if REAL_HOLDOUT_ARCHIVE_OVERRIDE is not None:
    holdout_preflight_path = Path(REAL_HOLDOUT_ARCHIVE_OVERRIDE)
elif RUNTIME == "Windows":
    holdout_preflight_path = Path(r"D:\Study\KhoaLuanTotNghiep\Dataset") / REAL_HOLDOUT_ARCHIVE_NAME
elif RUNTIME == "Colab":
    holdout_preflight_path = DRIVE_PROJECT_DIR / "Dataset" / REAL_HOLDOUT_ARCHIVE_NAME
else:
    matches = sorted(Path("/kaggle/input").rglob(REAL_HOLDOUT_ARCHIVE_NAME))
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected one {REAL_HOLDOUT_ARCHIVE_NAME}: {matches}")
    holdout_preflight_path = matches[0]
if not holdout_preflight_path.is_file():
    raise FileNotFoundError(holdout_preflight_path)
if sha256_file(holdout_preflight_path).upper() != REAL_HOLDOUT_ARCHIVE_SHA256:
    raise ValueError("Holdout02 ZIP SHA-256 mismatch")
split_hashes = set(valid_df["sha256"].str.lower())
holdout_rows = []
with zipfile.ZipFile(holdout_preflight_path) as holdout_zip:
    for member in holdout_zip.infolist():
        if member.is_dir():
            continue
        parts = Path(member.filename.replace("\\", "/")).parts
        if len(parts) != 3 or parts[0] != "RL_Cashew_holdout" or parts[1] not in REAL_HOLDOUT_EXPECTED_COUNTS:
            raise ValueError(f"Unexpected holdout ZIP member: {member.filename}")
        digest = hashlib.sha256()
        with holdout_zip.open(member) as stream:
            for block in iter(lambda: stream.read(1 << 20), b""):
                digest.update(block)
        holdout_rows.append({"class": parts[1], "relative_path": "/".join(parts[1:]),
                             "sha256": digest.hexdigest(),
                             "overlap_with_split": digest.hexdigest() in split_hashes})
holdout_preflight_df = pd.DataFrame(holdout_rows)
observed = holdout_preflight_df["class"].value_counts().to_dict()
if observed != REAL_HOLDOUT_EXPECTED_COUNTS or len(holdout_preflight_df) != 127:
    raise ValueError(f"Holdout02 class counts changed: {observed}")
if holdout_preflight_df["sha256"].duplicated().any() or holdout_preflight_df["overlap_with_split"].any():
    raise ValueError("Holdout02 has duplicate bytes internally or with Split 9")
holdout_preflight_df.to_csv(OUT_DIR / "holdout_preflight_manifest.csv", index=False)
near_rows = []
split_phashes = [(row.relative_path, row.split, int(row.phash64))
                 for row in valid_df.itertuples(index=False)]
with zipfile.ZipFile(holdout_preflight_path) as holdout_zip:
    for member in holdout_zip.infolist():
        if member.is_dir():
            continue
        with holdout_zip.open(member) as stream, Image.open(stream) as image:
            fingerprint = phash64(ImageOps.exif_transpose(image))
        for relative, split_name, split_fingerprint in split_phashes:
            distance = (fingerprint ^ split_fingerprint).bit_count()
            if distance <= 4:
                near_rows.append({"holdout_path": member.filename, "split_path": relative,
                                  "split": split_name, "phash_hamming": distance})
pd.DataFrame(near_rows, columns=["holdout_path", "split_path", "split",
                                 "phash_hamming"]).to_csv(
    OUT_DIR / "holdout_split_near_duplicate_candidates.csv", index=False)
(OUT_DIR / "holdout_preflight_audit.json").write_text(json.dumps({
    "archive": REAL_HOLDOUT_ARCHIVE_NAME, "archive_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,
    "class_counts": observed, "images": len(holdout_preflight_df),
    "exact_byte_overlap_with_train_val_test": 0,
    "phash_hamming_le_4_candidates": len(near_rows),
    "limitation": "Near duplicates and shared capture sessions require manual review.",
}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("Holdout02 preflight PASS:", observed, "exact-byte overlap: 0")
'''
book["cells"].insert(9, markdown("## New holdout archive preflight\n\nValidate the revised holdout before GPU training.\n"))
book["cells"].insert(10, code(preflight))

report_code = r'''# 20. Standard analysis artifacts for the statistical-analysis plan
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

STANDARD_COLUMNS = ["experiment_id", "architecture", "training_mode", "dataset_version",
                    "dataset_sha256", "split", "seed", "image_id", "relative_path",
                    "true_label", "predicted_label", "true_class_idx", "predicted_class_idx",
                    "correct", "confidence"] + [f"prob_{name}" for name in CLASS_NAMES]

def normalize_prediction_table(seed, split, path):
    frame = pd.read_csv(path)
    if split == "real_holdout":
        relative = frame["relative_path"].astype(str)
        truth = frame["class"].astype(str)
        predicted = frame["predicted_class"].astype(str)
        for name in CLASS_NAMES:
            frame[f"prob_{name}"] = frame[f"probability_{name}"]
    else:
        relative = frame["filename"].astype(str).map(
            lambda p: Path(p).relative_to(DATASET_PATH).as_posix())
        truth, predicted = frame["true_label"].astype(str), frame["predicted_label"].astype(str)
    normalized = pd.DataFrame({
        "experiment_id": EXPERIMENT_ID, "architecture": MODEL_NAME,
        "training_mode": TRAINING_MODE, "dataset_version": DATASET_VERSION,
        "dataset_sha256": DATASET_ZIP_SHA256, "split": split, "seed": seed,
        "image_id": relative, "relative_path": relative,
        "true_label": truth, "predicted_label": predicted,
        "true_class_idx": truth.map({name: i for i, name in enumerate(CLASS_NAMES)}),
        "predicted_class_idx": predicted.map({name: i for i, name in enumerate(CLASS_NAMES)}),
        "correct": frame["correct"].astype(bool),
        "confidence": frame["confidence"].astype(float),
    })
    for name in CLASS_NAMES:
        normalized[f"prob_{name}"] = frame[f"prob_{name}"].astype(float)
    if normalized[["true_class_idx", "predicted_class_idx"]].isna().any().any():
        raise ValueError(f"Unknown class in {path}")
    normalized[["true_class_idx", "predicted_class_idx"]] = normalized[
        ["true_class_idx", "predicted_class_idx"]].astype(int)
    probabilities = normalized[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy()
    if (len(normalized) == 0 or not np.isfinite(probabilities).all() or
            not np.allclose(probabilities.sum(axis=1), 1, atol=1e-4)):
        raise ValueError(f"Invalid probabilities in {path}")
    if normalized["image_id"].duplicated().any():
        raise ValueError(f"Duplicate image_id in {path}")
    return normalized[STANDARD_COLUMNS]

def plot_matrix(csv_path, png_path, title, normalized=False):
    matrix = pd.read_csv(csv_path, index_col=0).loc[CLASS_NAMES, CLASS_NAMES].to_numpy()
    fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
    ax.imshow(matrix, cmap=MATRIX_CMAP, vmin=0, vmax=1 if normalized else max(1, matrix.max()))
    ax.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    for row in range(NUM_CLASSES):
        for col in range(NUM_CLASSES):
            value = f"{matrix[row, col]:.2f}" if normalized else str(int(matrix[row, col]))
            ax.text(col, row, value, ha="center", va="center", fontsize=9)
    ax.set(xlabel="Predicted", ylabel="True", title=title)
    fig.tight_layout(); fig.savefig(png_path, dpi=180); plt.close(fig)

def calibration_outputs(predictions, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    confidence = predictions["confidence"].to_numpy(dtype=float)
    correct = predictions["correct"].to_numpy(dtype=bool)
    truth = predictions["true_class_idx"].to_numpy(dtype=int)
    probs = predictions[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)
    edges = np.linspace(0, 1, 11)
    rows = []
    for i in range(10):
        mask = (confidence >= edges[i]) & (confidence < edges[i+1] if i < 9 else confidence <= 1)
        n = int(mask.sum())
        rows.append({"bin": i, "lower": edges[i], "upper": edges[i+1], "n": n,
                     "accuracy": float(correct[mask].mean()) if n else np.nan,
                     "mean_confidence": float(confidence[mask].mean()) if n else np.nan})
    bins = pd.DataFrame(rows)
    bins.to_csv(output_dir / "calibration_bins.csv", index=False)
    ece = float(sum(row["n"] / len(predictions) * abs(row["accuracy"] - row["mean_confidence"])
                    for row in rows if row["n"]))
    brier = float(np.mean(np.sum((probs - np.eye(NUM_CLASSES)[truth]) ** 2, axis=1)))
    save_json(output_dir / "calibration_metrics.json", {"ece_10_bins": ece,
              "multiclass_brier": brier, "n": len(predictions),
              "note": "Diagnostic only; no calibration fit or threshold tuning."})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), facecolor="white")
    used = bins[bins["n"] > 0]
    axes[0].plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
    axes[0].plot(used["mean_confidence"], used["accuracy"], "o-", color="#2563eb")
    axes[0].set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy",
                title=f"Reliability (ECE={ece:.3f})")
    axes[1].hist(confidence[correct], bins=edges, alpha=.7, label="Correct")
    axes[1].hist(confidence[~correct], bins=edges, alpha=.7, label="Incorrect")
    axes[1].set(xlabel="Confidence", ylabel="Images", title="Confidence distribution")
    axes[1].legend(); fig.tight_layout()
    fig.savefig(output_dir / "reliability_confidence.png", dpi=180); plt.close(fig)
    return {"ece_10_bins": ece, "multiclass_brier": brier}

def roc_pr_outputs(predictions, output_dir):
    truth = predictions["true_class_idx"].to_numpy(dtype=int)
    probs = predictions[[f"prob_{name}" for name in CLASS_NAMES]].to_numpy(dtype=float)
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor="white")
    for class_id, name in enumerate(CLASS_NAMES):
        binary = (truth == class_id).astype(int)
        if binary.sum() == 0 or binary.sum() == len(binary):
            rows.append({"class": name, "support": int(binary.sum()), "roc_auc": np.nan,
                         "average_precision": np.nan, "status": "UNDEFINED_NO_POSITIVE_OR_NEGATIVE"})
            continue
        fpr, tpr, _ = roc_curve(binary, probs[:, class_id])
        precision, recall, _ = precision_recall_curve(binary, probs[:, class_id])
        roc_auc = float(auc(fpr, tpr))
        ap = float(average_precision_score(binary, probs[:, class_id]))
        rows.append({"class": name, "support": int(binary.sum()), "roc_auc": roc_auc,
                     "average_precision": ap, "status": "OK"})
        axes[0].plot(fpr, tpr, label=f"{name} ({roc_auc:.3f})")
        axes[1].plot(recall, precision, label=f"{name} ({ap:.3f})")
    axes[0].plot([0, 1], [0, 1], "--", color="gray")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC one-vs-rest")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR one-vs-rest")
    for ax in axes:
        ax.legend(fontsize=8); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(output_dir / "roc_pr_ovr.png", dpi=180); plt.close(fig)
    pd.DataFrame(rows).to_csv(output_dir / "roc_pr_ovr_metrics.csv", index=False)

analysis_splits = [("validation", OUT_DIR, AGGREGATE_DIR)]
if (AGGREGATE_DIR / "test_results_5seeds.csv").is_file():
    analysis_splits.append(("test", OUT_DIR, AGGREGATE_DIR))
if (OUT_DIR / "real_holdout" / "evaluation_completed.json").is_file():
    analysis_splits.append(("real_holdout", OUT_DIR / "real_holdout", OUT_DIR / "real_holdout" / "aggregate"))

artifact_rows = []
for split, root, aggregate in analysis_splits:
    per_class_rows, calibration_rows = [], []
    for seed in SEEDS:
        seed_output = root / f"seed_{seed}" / (split if split != "real_holdout" else "")
        source_csv = seed_output / "predictions.csv"
        normalized = normalize_prediction_table(seed, split, source_csv)
        expected_n = {"validation": EXPECTED_TOTALS["val"], "test": EXPECTED_TOTALS["test"],
                      "real_holdout": 127}[split]
        if len(normalized) != expected_n:
            raise ValueError(f"{split} seed {seed}: expected {expected_n} predictions")
        normalized.to_csv(seed_output / "predictions_standard.csv", index=False)
        normalized[~normalized["correct"]].to_csv(seed_output / "error_cases_standard.csv", index=False)
        raw = seed_output / "confusion_matrix.csv"
        norm = seed_output / "confusion_matrix_normalized.csv"
        for src, suffix, is_normalized in ((raw, "raw", False), (norm, "normalized", True)):
            target_csv = seed_output / f"confusion_matrix_{suffix}.csv"
            if src != target_csv:
                target_csv.write_bytes(src.read_bytes())
            target_png = seed_output / f"confusion_matrix_{suffix}.png"
            plot_matrix(target_csv, target_png, f"{split} · seed {seed}", is_normalized)
        y_true = normalized["true_class_idx"].to_numpy()
        y_pred = normalized["predicted_class_idx"].to_numpy()
        report = classification_report(y_true, y_pred, labels=np.arange(NUM_CLASSES),
                                       target_names=CLASS_NAMES, output_dict=True, zero_division=0)
        for name in CLASS_NAMES:
            supported = int(report[name]["support"]) > 0
            per_class_rows.append({"split": split, "seed": seed, "class": name,
                                   "support": int(report[name]["support"]),
                                   "precision": report[name]["precision"] if supported else np.nan,
                                   "recall": report[name]["recall"] if supported else np.nan,
                                   "f1": report[name]["f1-score"] if supported else np.nan})
        cal = calibration_outputs(normalized, seed_output)
        calibration_rows.append({"split": split, "seed": seed, "n": len(normalized), **cal})
        roc_pr_outputs(normalized, seed_output)
        artifact_rows.append({"split": split, "seed": seed, "rows": len(normalized),
                              "probability_columns": NUM_CLASSES, "status": "COMPLETE"})
    per_class = pd.DataFrame(per_class_rows)
    per_class.to_csv(aggregate / f"per_class_{split}_by_seed.csv", index=False)
    per_class.groupby("class")[["precision", "recall", "f1"]].agg(["mean", "std"]).to_csv(
        aggregate / f"per_class_{split}_mean_std.csv")
    pd.DataFrame(calibration_rows).to_csv(aggregate / f"{split}_calibration_5seeds.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    heat = per_class.pivot(index="class", columns="seed", values="f1").reindex(CLASS_NAMES)
    im = ax.imshow(heat, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(SEEDS)), SEEDS)
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set(xlabel="Seed", title=f"{split}: per-class F1")
    for row in range(NUM_CLASSES):
        for col in range(len(SEEDS)):
            value = heat.iloc[row, col]
            ax.text(col, row, f"{value:.2f}" if np.isfinite(value) else "N/A",
                    ha="center", va="center")
    fig.colorbar(im, ax=ax, label="F1")
    fig.tight_layout(); fig.savefig(PANEL_DIR / f"per_class_f1_{split}_heatmap.png", dpi=180); plt.close(fig)
    result_path = (aggregate / "real_holdout_results_5seeds.csv" if split == "real_holdout"
                   else aggregate / f"{split}_results_5seeds.csv")
    results = pd.read_csv(result_path)
    metric_map = ({"accuracy": "accuracy", "macro_recall": "macro_recall_present_classes",
                   "macro_f1": "macro_f1_present_classes"} if split == "real_holdout" else
                  {"accuracy": "val_accuracy", "macro_recall": "val_macro_recall",
                   "macro_f1": "val_macro_f1", "balanced_accuracy": "val_balanced_accuracy"}
                  if split == "validation" else
                  {"accuracy": "accuracy", "macro_recall": "macro_recall",
                   "macro_f1": "macro_f1", "balanced_accuracy": "balanced_accuracy"})
    metric_rows = [{"metric": label, "mean": float(results[col].mean()),
                    "sd_seed": float(results[col].std(ddof=1)), "n_seeds": len(SEEDS)}
                   for label, col in metric_map.items()]
    pd.DataFrame(metric_rows).to_csv(aggregate / f"{split}_metric_mean_seed_sd.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.5), facecolor="white")
    ax.bar([row["metric"] for row in metric_rows], [row["mean"] for row in metric_rows],
           yerr=[row["sd_seed"] for row in metric_rows], capsize=5, color="#3b82f6")
    ax.set(ylim=(0, 1.05), ylabel="Score", title=f"{split}: five-seed mean ± sample SD")
    fig.tight_layout(); fig.savefig(PANEL_DIR / f"{split}_metrics_mean_seed_sd.png", dpi=180); plt.close(fig)

pd.DataFrame(artifact_rows).to_csv(AGGREGATE_DIR / "standard_output_inventory.csv", index=False)
if "test" in [s[0] for s in analysis_splits] and "real_holdout" in [s[0] for s in analysis_splits]:
    # Different support and true-class sets; a descriptive domain-gap chart only.
    test_scores = pd.read_csv(AGGREGATE_DIR / "test_results_5seeds.csv").set_index("seed")
    hold_scores = pd.read_csv(OUT_DIR / "real_holdout" / "aggregate" /
                              "real_holdout_results_5seeds.csv").set_index("seed")
    if "macro_f1_present_classes" in hold_scores.columns:
        gap = pd.DataFrame({"seed": SEEDS, "test_5class_macro_f1":
                            [test_scores.loc[s, "macro_f1"] for s in SEEDS],
                            "holdout_4class_macro_f1":
                            [hold_scores.loc[s, "macro_f1_present_classes"] for s in SEEDS]})
        gap.to_csv(AGGREGATE_DIR / "domain_gap_descriptive.csv", index=False)
        ax = gap.set_index("seed").plot.bar(figsize=(9, 4.5), ylim=(0, 1), rot=0,
                                             title="Internal Test vs external holdout (different class support)")
        ax.set_ylabel("Macro F1"); ax.figure.tight_layout()
        ax.figure.savefig(PANEL_DIR / "domain_gap_descriptive.png", dpi=180); plt.close(ax.figure)
print("Standard artifacts:", pd.DataFrame(artifact_rows))
'''
book["cells"].insert(-2, markdown("## Statistical analysis exports\n\nCreate a consistent per-image schema, class tables and diagnostic figures.\n"))
book["cells"].insert(-2, code(report_code))

latency_code = r'''# Batch-1 deployment latency on the Validation input path, never on Test/holdout
sample_path = Path(val_paths[0])
for _ in range(5):
    predict_field_image(sample_path)
latency_ms = [predict_field_image(sample_path)["elapsed_ms_including_preprocessing"]
              for _ in range(30)]
latency_summary = {
    "experiment_id": EXPERIMENT_ID, "deployment_seed": deployment_seed,
    "batch_size": 1, "warmup_runs": 5, "measured_runs": 30,
    "measurement": "wall-clock EXIF decode + RGB conversion + bilinear resize + Keras predict + argmax",
    "device": [device.name for device in tf.config.list_physical_devices("GPU")] or ["CPU"],
    "mean_ms": float(np.mean(latency_ms)), "median_ms": float(np.median(latency_ms)),
    "p95_ms": float(np.percentile(latency_ms, 95)),
    "sample_sd_ms": float(np.std(latency_ms, ddof=1)),
    "parameters": int(reloaded.count_params()),
    "deployment_model_bytes": deployment_path.stat().st_size,
    "note": "One Validation image repeated; platform-specific timing, not a cross-device benchmark.",
}
save_json(AGGREGATE_DIR / "deployment_efficiency.json", latency_summary)
pd.DataFrame({"run": np.arange(1, 31), "latency_ms": latency_ms}).to_csv(
    AGGREGATE_DIR / "deployment_latency_batch1.csv", index=False)
print("Deployment batch-1 latency:", latency_summary)
'''
deployment_index = next(i for i, c in enumerate(book["cells"])
                        if c["cell_type"] == "code" and
                        'print("Deployment seed", deployment_seed' in "".join(c["source"]))
book["cells"].insert(deployment_index + 1, markdown(
    "## Deployment timing\n\nMeasure batch-1 end-to-end inference on a Validation image after warm-up.\n"))
book["cells"].insert(deployment_index + 2, code(latency_code))

archive = r'''# 22–23. Report, artifact audit, notebook copy, FULL/ANALYSIS ZIP
import base64
import zlib

assert_selection_lock()
test_status = "COMPLETE" if (AGGREGATE_DIR / "test_results_5seeds.csv").is_file() else "PENDING"
holdout_status = "COMPLETE" if (OUT_DIR / "real_holdout" / "evaluation_completed.json").is_file() else "PENDING"
save_json(OUT_DIR / "execution_status.json", {
    "experiment_id": EXPERIMENT_ID, "training": "COMPLETE", "validation": "COMPLETE",
    "test": test_status, "real_holdout": holdout_status,
    "dataset_zip_sha256": DATASET_ZIP_SHA256,
    "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
    "holdout_zip_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,
    "deployment_seed": deployment_seed,
})
readme = f"""# {EXPERIMENT_ID}

Five-seed DenseNet121 ImageNet transfer, two-stage protocol from EXP-014.
Dataset: {DATASET_ARCHIVE_NAME} ({DATASET_ZIP_SHA256}).
Manifest: {DATASET_MANIFEST_SHA256}. Holdout: {REAL_HOLDOUT_ARCHIVE_NAME}
({REAL_HOLDOUT_ARCHIVE_SHA256}). No prior checkpoints are reused.

Selection: lowest Validation loss, macro-F1 tie-break; seed and checkpoints
frozen in `validation_selection_lock.json` before Test/holdout. Test={test_status};
external holdout={holdout_status}. Test has {EXPECTED_TOTALS['test']} images and
five true classes. Holdout has 127 images and four true classes. These macro-F1
values have different denominators and are reported separately.

Training: `aggregate/training_summary_5seeds.csv`, per-seed stage histories and
`figures/panels/training_curves_5seeds_panel.png`.
Evaluation: `seed_*/validation|test/predictions_standard.csv` (plus holdout),
`classification_report.csv`, raw/normalized confusion CSV/PNG, calibration
bins/ECE/Brier, ROC/PR OVR CSV/PNG, errors, per-class tables and heatmaps.
`aggregate/standard_output_inventory.csv` lists completed outputs.
The `real_` filename subgroup is a source proxy, not verified capture metadata.
Seed mean ± sample SD describes training variation, not an image-sampling CI.
No Test or holdout score is used for model/threshold selection.
Exact-byte overlap is absent in the supplied archives. Near-duplicate and
same-leaf/tree independence still need manual review of audit candidates.

Deployment: Validation-selected `.keras` model and inference adapter in
`model_package/` and `code/`. Notebook source in `notebook/`.
Batch-1 deployment timing and method: `aggregate/deployment_efficiency.json`.
"""
(OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

required = ["experiment_config.json", "dataset_manifest.csv", "dataset_audit_summary.json",
            "holdout_preflight_audit.json", "validation_selection_lock.json",
            "aggregate/validation_results_5seeds.csv", "aggregate/validation_mean_std_summary.csv",
            "aggregate/standard_output_inventory.csv", "aggregate/deployment_efficiency.json",
            "model_package/deployment_model.keras"]
for seed in SEEDS:
    required += [f"seed_{seed}/best.weights.h5", f"seed_{seed}/validation/predictions_standard.csv",
                 f"seed_{seed}/validation/confusion_matrix_raw.png",
                 f"seed_{seed}/validation/confusion_matrix_normalized.png"]
if test_status == "COMPLETE":
    required += ["aggregate/test_results_5seeds.csv", "aggregate/test_mean_std_summary.csv"]
    required += [f"seed_{seed}/test/predictions_standard.csv" for seed in SEEDS]
if holdout_status == "COMPLETE":
    required += ["real_holdout/dataset_audit.json",
                 "real_holdout/aggregate/real_holdout_results_5seeds.csv"]
    required += [f"real_holdout/seed_{seed}/predictions_standard.csv" for seed in SEEDS]
missing = [item for item in required if not (OUT_DIR / item).is_file() or
           (OUT_DIR / item).stat().st_size == 0]
audit = {"archive_version": "v1", "experiment_id": EXPERIMENT_ID,
         "required_count": len(required), "missing": missing,
         "test": test_status, "real_holdout": holdout_status,
         "status": "PASS" if not missing else "FAIL"}
save_json(OUT_DIR / "audit_required_artifacts.json", audit)
if missing:
    raise RuntimeError(f"Required outputs missing: {missing}")

NOTEBOOK_SOURCE_ZLIB_B64 = ""
notebook_dir = OUT_DIR / "notebook"
notebook_dir.mkdir(exist_ok=True)
notebook_copy = notebook_dir / "densenet121_transfer_5seed_018.ipynb"
source_book = json.loads(zlib.decompress(base64.b64decode(NOTEBOOK_SOURCE_ZLIB_B64)))
for cell in source_book["cells"]:
    if cell["cell_type"] == "code" and any(
            line.startswith('NOTEBOOK_SOURCE_ZLIB_B64 = ""') for line in cell["source"]):
        cell["source"] = [line.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                                       f'NOTEBOOK_SOURCE_ZLIB_B64 = "{NOTEBOOK_SOURCE_ZLIB_B64}"', 1)
                          for line in cell["source"]]
notebook_copy.write_text(json.dumps(source_book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")

save_json(OUT_DIR / "analysis_manifest.json", {
    "archive_version": "v1", "experiment_id": EXPERIMENT_ID,
    "dataset_zip_sha256": DATASET_ZIP_SHA256,
    "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
    "holdout_zip_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,
    "class_order": CLASS_NAMES, "seeds": SEEDS,
    "test": test_status, "real_holdout": holdout_status,
})
manifest_rows = []
for path in sorted(OUT_DIR.rglob("*")):
    if path.is_file() and path.name not in {"FILE_MANIFEST.csv", "CHECKSUMS.sha256"}:
        manifest_rows.append({"path": path.relative_to(OUT_DIR).as_posix(),
                              "size_bytes": path.stat().st_size, "sha256": file_sha256(path)})
pd.DataFrame(manifest_rows).to_csv(OUT_DIR / "FILE_MANIFEST.csv", index=False)
(OUT_DIR / "CHECKSUMS.sha256").write_text(
    "\n".join(f"{row['sha256']}  {row['path']}" for row in manifest_rows) + "\n", encoding="utf-8")

analysis_suffixes = {".csv", ".json", ".md", ".png", ".svg", ".txt", ".sha256", ".ipynb", ".py"}
for target, analysis_only in ((RESULTS_ROOT / f"{EXPERIMENT_ID}_FULL.zip", False),
                              (RESULTS_ROOT / f"{EXPERIMENT_ID}_ANALYSIS.zip", True)):
    temporary = target.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(OUT_DIR)
            if analysis_only and ("model_package" in relative.parts or
                                  path.suffix.lower() not in analysis_suffixes):
                continue
            archive.write(path, arcname=(Path(EXPERIMENT_ID) / relative).as_posix())
    with zipfile.ZipFile(temporary) as archive:
        if (archive.testzip() is not None or
                f"{EXPERIMENT_ID}/notebook/{notebook_copy.name}" not in archive.namelist()):
            raise RuntimeError(f"Archive verification failed: {temporary}")
    temporary.replace(target)
    print("Verified:", target, f"{target.stat().st_size / 2**20:.1f} MiB")
'''

archive_index = next(i for i, c in enumerate(book["cells"])
                     if c["cell_type"] == "code" and "NOTEBOOK_SOURCE_ZLIB_B64" in "".join(c["source"]))
set_cell(archive_index, archive)

# Update remaining prose and identifier references without changing generated code.
for cell in book["cells"]:
    if cell["cell_type"] == "markdown":
        body = "".join(cell["source"])
        body = body.replace("EXP-014", "EXP-018").replace("Split (8)", "Split (9)")
        cell["source"] = body.splitlines(keepends=True)

empty_source = json.dumps(book, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
compressed = base64.b64encode(zlib.compress(empty_source, 9)).decode("ascii")
archive_body = "".join(book["cells"][archive_index]["source"])
archive_body = archive_body.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                                    f'NOTEBOOK_SOURCE_ZLIB_B64 = "{compressed}"', 1)
set_cell(archive_index, archive_body)
TARGET.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(book['cells'])} cells, {TARGET.stat().st_size} bytes")
