"""Generate the clean EXP-023 notebook from the EXP-022 ANALYSIS source."""
import base64
import copy
import hashlib
import json
import textwrap
import zipfile
import zlib
from pathlib import Path

ARCHIVE = Path(r"C:\Users\KIM LONG\Downloads\EXP-DENSENET121-TRANSFER-5SEEDS-022_ANALYSIS.zip")
OUTPUT = Path(__file__).with_name("densenet121_transfer_photometric_5seeds_023.ipynb")
OLD = "EXP-DENSENET121-TRANSFER-5SEEDS-022"
NEW = "EXP-DENSENET121-TRANSFER-5SEEDS-023"
SEEDS = [42, 123, 2026, 3407, 7777]


def get(book, index):
    return "".join(book["cells"][index]["source"])


def put(book, index, value):
    book["cells"][index]["source"] = value.splitlines(keepends=True)


def gated(value, condition, message):
    return f"if {condition}:\n" + textwrap.indent(value.rstrip() + "\n", "    ") + f"else:\n    print({message!r})\n"


def cell(kind, content):
    result = {"cell_type": kind, "metadata": {}, "source": content.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


with zipfile.ZipFile(ARCHIVE) as archive:
    book = json.loads(archive.read(f"{OLD}/notebook/densenet121_transfer_5seed_022.ipynb"))
    baseline_names = ["aggregate/validation_results_5seeds.csv"] + [
        f"seed_{seed}/validation/classification_report.csv" for seed in SEEDS]
    baseline = {name: archive.read(f"{OLD}/{name}").decode("utf-8") for name in baseline_names}
    archive_sha = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()

for c in book["cells"]:
    c["metadata"] = {}
    if c["cell_type"] == "code":
        c["outputs"], c["execution_count"] = [], None

put(book, 0, """# EXP-023 — DenseNet121 + RandomBrightness(0.08)

This is a controlled Validation comparison with historical EXP-022 on frozen Split (10).
The model recipe adds **only** brightness after the existing contrast augmentation.
Five independent ImageNet initializations, both training stages, Validation checkpoint
selection and seed selection follow EXP-022.

Default: audit + smoke only. To train: set RUN_SMOKE_ONLY=False and
RUN_FULL_TRAINING=True, then Run all. Final Test and Holdout (4) are separately
opt-in after the Validation lock. VIEW_ONLY renders saved Validation files without
opening dataset ZIPs. No EXP-023 results exist in this notebook before execution.
""")
put(book, 2, get(book, 2).replace("import os\n", "import os\nimport base64\nimport zlib\n") + '''
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
''')
put(book, 3, get(book, 3).replace(
    'if RUNTIME == "Colab" and not gpus:\n    raise RuntimeError("Colab has no GPU. Select a GPU runtime and rerun from the first cell.")',
    'if RUNTIME == "Colab" and not gpus:\n    print("CPU runtime detected; VIEW_ONLY can continue, model smoke/training requires GPU.")'))

s = get(book, 4).replace("CONFIGURATION — EXP-022", "CONFIGURATION — EXP-023")
s = s.replace('NOTEBOOK_VERSION = "022"', 'NOTEBOOK_VERSION = "023"').replace(OLD, NEW)
s = s.replace("Optional path to this exact 022 .ipynb", "Optional path to this exact 023 .ipynb")
s = s.replace('EXPECTED_MANIFEST_SHA256 = None  # Archive SHA256 + saved manifest freeze this new dataset.',
              'EXPECTED_MANIFEST_SHA256 = "9c1279776dd831ffa23bfa35960f7a3512eae88016f23abf067c4915d963c09e"')
s = s.replace('EXPECTED_MANIFEST_SHA256 = "9c1279776dd831ffa23bfa35960f7a3512eae88016f23abf067c4915d963c09e"',
              'EXPECTED_ZIP_SHA256 = "b24f0faa5e918fb43d7d16500c6b8234df1cd771cc4a27eb2354051bce84edac"\n'
              'EXPECTED_MANIFEST_SHA256 = "9c1279776dd831ffa23bfa35960f7a3512eae88016f23abf067c4915d963c09e"')
s = s.replace("RUN_PIPELINE_SMOKE = True", "RUN_PIPELINE_SMOKE = True\nVIEW_ONLY = False")
s = s.replace("RUN_FULL_TRAINING = True\nRUN_FINAL_TEST = True\nRUN_REAL_HOLDOUT = True",
              "RUN_SMOKE_ONLY = True\nRUN_FULL_TRAINING = False\nRUN_FINAL_TEST = False\nRUN_REAL_HOLDOUT = False")
s = s.replace('if not RUN_FULL_TRAINING and not (OUT_DIR / "validation_selection_lock.json").is_file():\n    raise FileNotFoundError("Evaluation requires a completed EXP-022 Validation lock.")',
'''if VIEW_ONLY and (RUN_SMOKE_ONLY or RUN_FULL_TRAINING or RUN_FINAL_TEST or RUN_REAL_HOLDOUT):
    raise ValueError("VIEW_ONLY requires all run flags False")
if RUNTIME == "Colab" and not gpus and not VIEW_ONLY:
    raise RuntimeError("Select a Colab GPU runtime for model smoke/training")
if RUN_SMOKE_ONLY and (RUN_FULL_TRAINING or RUN_FINAL_TEST or RUN_REAL_HOLDOUT):
    raise ValueError("Disable RUN_SMOKE_ONLY explicitly for training/evaluation")
if (RUN_FINAL_TEST or RUN_REAL_HOLDOUT or VIEW_ONLY) and not (RUN_FULL_TRAINING or (OUT_DIR / "validation_selection_lock.json").is_file()):
    raise FileNotFoundError("A completed EXP-023 Validation lock is required")''')
s = s.replace('candidates = [direct] if (direct / "Dataset" / DATASET_ARCHIVE_NAME).is_file() else []',
              'candidates = [direct] if ((direct / "training_results" / "current_dataset" / EXPERIMENT_ID).is_dir() if VIEW_ONLY else (direct / "Dataset" / DATASET_ARCHIVE_NAME).is_file()) else []')
s = s.replace('(p / "Dataset" / DATASET_ARCHIVE_NAME).is_file()]',
              '((p / "training_results" / "current_dataset" / EXPERIMENT_ID).is_dir() if VIEW_ONLY else (p / "Dataset" / DATASET_ARCHIVE_NAME).is_file())]')
s = s.replace('if not DRIVE_DATASET_ARCHIVE.is_file():', 'if not VIEW_ONLY and not DRIVE_DATASET_ARCHIVE.is_file():')
s = s.replace('    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)',
'''    if VIEW_ONLY:
        if not RESULTS_ROOT.is_dir(): raise FileNotFoundError(RESULTS_ROOT)
    else:
        RESULTS_ROOT.mkdir(parents=True, exist_ok=True)''')
s = s.replace('    directory.mkdir(parents=True, exist_ok=True)',
'''    if VIEW_ONLY:
        if not directory.is_dir(): raise FileNotFoundError(directory)
    else:
        directory.mkdir(parents=True, exist_ok=True)''')
probe_start = s.index('    write_probe = RESULTS_ROOT')
probe_end = s.index('    print("Colab project shortcut:"', probe_start)
s = s[:probe_start] + '    if not VIEW_ONLY:\n' + textwrap.indent(s[probe_start:probe_end], '    ') + s[probe_end:]
put(book, 4, s)

for i in (5, 6, 8, 11, 12):
    put(book, i, gated(get(book, i), "not VIEW_ONLY", f"Cell {i} skipped in VIEW_ONLY"))
put(book, 10, 'print("Holdout preflight deferred to opt-in evaluation after Validation lock.")\n')

s = get(book, 14)
s = s.replace('AUGMENTATION_CONFIG = {"horizontal_flip": True, "rotation_factor": 0.08,\n                       "zoom_factor": 0.10, "contrast_factor": 0.10,\n                       "brightness": False}',
'''BRIGHTNESS_FACTOR = 0.08
BRIGHTNESS_VALUE_RANGE = (0.0, 255.0)
if not hasattr(layers, "RandomBrightness"):
    raise RuntimeError("This Keras runtime lacks RandomBrightness")
AUGMENTATION_CONFIG = {"horizontal_flip": True, "rotation_factor": 0.08,
                       "zoom_factor": 0.10, "contrast_factor": 0.10,
                       "brightness_enabled": True, "brightness_factor": BRIGHTNESS_FACTOR,
                       "brightness_value_range": list(BRIGHTNESS_VALUE_RANGE)}''')
s = s.replace('        layers.RandomContrast(0.10, seed=seed),',
'''        layers.RandomContrast(0.10, seed=seed),
        layers.RandomBrightness(factor=BRIGHTNESS_FACTOR,
                                value_range=BRIGHTNESS_VALUE_RANGE, seed=seed + 1100),''')
s = s.replace('    initial_backbone = [w.numpy().copy() for w in backbone.weights[:2]]',
'''    augmentation = model.get_layer(f"train_only_augmentation_seed_{SEEDS[0]}")
    qa_paths = []
    for class_name in CLASS_NAMES:
        subset = train_meta[train_meta["class"] == class_name].head(3)
        if len(subset) != 3:
            raise AssertionError(f"Need 3 Train images for {class_name}")
        qa_paths += subset["path"].tolist()
    qa_input = tf.stack([decode_image(tf.constant(str(path)), tf.constant(0))[0]
                         for path in qa_paths])
    original = qa_input.numpy()
    for repeat in range(2):
        if not np.allclose(augmentation(qa_input, training=False).numpy(), original, atol=1e-5):
            raise AssertionError(f"Augmentation changed inference image (pass {repeat})")
    draws = [augmentation(qa_input, training=True).numpy() for _ in range(3)]
    if not all(np.isfinite(d).all() and d.min() >= -1e-4 and d.max() <= 255.0001 for d in draws):
        raise AssertionError("Augmentation produced nonfinite/out-of-range pixels")
    if not any(np.mean(np.abs(d - original)) > .1 for d in draws):
        raise AssertionError("Train augmentation inactive")
    if not any(np.mean(np.abs(draws[i] - draws[j])) > .1 for i in range(3) for j in range(i+1, 3)):
        raise AssertionError("Train augmentation not stochastic")
    qa_dir = OUT_DIR / "figures" / "augmentation"
    qa_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(15, 4, figsize=(13, 34), facecolor="white")
    for row, path in enumerate(qa_paths):
        for col, pixels in enumerate([original] + draws):
            axes[row, col].imshow(np.clip(pixels[row], 0, 255).astype(np.uint8))
            axes[row, col].axis("off")
            if row == 0: axes[row, col].set_title("Train original" if col == 0 else f"Train draw {col}")
            if col == 0: axes[row, col].set_ylabel(f"{CLASS_NAMES[row//3]} / {row%3+1}")
    fig.tight_layout()
    qa_path = qa_dir / "photometric_qa_5classes.png"
    fig.savefig(qa_path, dpi=130, bbox_inches="tight")
    display(fig); plt.close(fig)
    initial_backbone = [w.numpy().copy() for w in backbone.weights[:2]]''')
s = s.replace("if stage2_params <= 0 or stage2_params >= backbone.count_params():",
              "if stage2_params != 2_129_920:")
s = s.replace('    bn_means = [layer.moving_mean.numpy().copy() for layer in bn_layers]',
              '    if len(bn_layers) != 121: raise AssertionError("Expected 121 BatchNorm layers")\n    bn_means = [layer.moving_mean.numpy().copy() for layer in bn_layers]')
s = s.replace('"weights_and_model_reload": "PASS", "imagenet_initialization": "PASS",',
              '"weights_and_model_reload": "PASS", "imagenet_initialization": "PASS",\n        "inference_augmentation_identity": "PASS", "train_augmentation_stochastic": "PASS",')
put(book, 14, gated(s, "not VIEW_ONLY", "Model smoke skipped in VIEW_ONLY"))

s = get(book, 16).replace("EXP-018 protocol; no checkpoints or prior metrics reused",
                           "EXP-022 metrics reference; no checkpoints reused")
s = s.replace("new V10 dataset; compare only with runs sharing the exact manifest",
              "+RandomBrightness(0.08) versus historical EXP-022 on identical V10")
s = s.replace("Existing EXP-022 recipe differs", "Existing EXP-023 recipe differs")
s = s.replace("run once after five-seed Validation selection lock in the same execution",
              "opt-in exposed historical Test after five-seed Validation lock")
put(book, 16, '''if VIEW_ONLY:
    config_path = OUT_DIR / "experiment_config.json"
    experiment_config = json.loads(config_path.read_text(encoding="utf-8"))
    CONFIG_SHA256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    DATASET_ZIP_SHA256 = experiment_config["dataset"]["archive_sha256"]
    DATASET_MANIFEST_SHA256 = experiment_config["dataset"]["manifest_sha256"]
    CLASS_NAMES = experiment_config["dataset"]["class_names"]
    NUM_CLASSES = len(CLASS_NAMES)
    if experiment_config.get("code_revision_sha256") != CODE_REVISION_SHA256:
        raise ValueError("VIEW_ONLY notebook revision differs from saved EXP-023 recipe")
    if DATASET_ZIP_SHA256 != EXPECTED_ZIP_SHA256 or DATASET_MANIFEST_SHA256 != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Saved dataset fingerprint differs")
else:
''' + textwrap.indent(s.rstrip() + '''
save_json(OUT_DIR / "execution_status.json", {
    "experiment_id": EXPERIMENT_ID, "training": "PENDING", "validation": "PENDING",
    "test": "PENDING", "real_holdout": "PENDING",
    "smoke": "PASS" if RUN_PIPELINE_SMOKE else "NOT_EXECUTED"})
''', "    "))

s = get(book, 17)
s = s.replace('    save_json(output / "metrics.json", metrics)',
'''    healthy_mask = y_true == CLASS_NAMES.index("healthy")
    diseased_prediction = np.isin(y_pred, disease_ids)
    metrics.update(
        healthy_to_diseased_count=int(np.sum(healthy_mask & diseased_prediction)),
        healthy_count=int(healthy_mask.sum()),
        healthy_to_diseased_rate=float(np.sum(healthy_mask & diseased_prediction) / healthy_mask.sum()))
    if metrics["healthy_count"] != 216:
        raise ValueError("Healthy Validation denominator changed")
    save_json(output / "metrics.json", metrics)''', 1)
s = s.replace('else:\n    training_summary_df = pd.read_csv(AGGREGATE_DIR / "training_summary_5seeds.csv")',
                           'elif VIEW_ONLY or RUN_FINAL_TEST or RUN_REAL_HOLDOUT:\n    training_summary_df = pd.read_csv(AGGREGATE_DIR / "training_summary_5seeds.csv")')
at = s.index("ranked = validation_results_df.sort_values(")
put(book, 17, s[:at] + gated(s[at:], "RUN_FULL_TRAINING or VIEW_ONLY or RUN_FINAL_TEST or RUN_REAL_HOLDOUT", "Training/Validation PENDING"))

s = get(book, 18).replace('if len(predictions) != len(val_labels):',
                           'if len(predictions) != EXPECTED_TOTALS["val"]:')
s = s.replace('"leaf_miner_recall",\n                   "false_healthy_rate"]',
              '"leaf_miner_recall",\n                   "false_healthy_rate", "healthy_to_diseased_rate"]')
s = s.replace('("other_filename", predictions[~predictions["is_real_filename"]])):',
              '("other_filename", predictions[~predictions["is_real_filename"]]),\n                            ("real_healthy_filename", predictions[predictions["is_real_filename"] &\n                                (predictions["true_label"] == "healthy")])):')
s = s.replace('    report = pd.read_csv(val_dir / "classification_report.csv", index_col=0)',
'''    real_healthy_row = real_rows[-1]
    if real_healthy_row["total"] != 3:
        raise ValueError("Expected exactly 3 real_ Healthy Validation images")
    report = pd.read_csv(val_dir / "classification_report.csv", index_col=0)''')
s = s.replace("training_curves_5seeds_panel.png", "accuracy_loss_curves_5seeds.png")
s = s.replace('f"validation_confusion_matrices{suffix}_5seeds_panel.png"',
              'f"validation_confusion_matrices_5seeds{suffix}.png"')
put(book, 18, gated(s, "RUN_FULL_TRAINING", "Validation figures PENDING"))
put(book, 20, get(book, 20).replace("EXP-022 trained checkpoint", "EXP-023 trained checkpoint"))
put(book, 22, get(book, 22).replace("EXP-022", "EXP-023"))
for i in (24, 26):
    put(book, i, gated(get(book, i), "RUN_FULL_TRAINING", f"Cell {i} PENDING"))
put(book, 29, gated(get(book, 29), "RUN_FULL_TRAINING or RUN_FINAL_TEST or RUN_REAL_HOLDOUT",
                    "Statistical artifacts PENDING"))

BASELINE = base64.b64encode(zlib.compress(json.dumps({
    "archive": ARCHIVE.name, "archive_sha256": archive_sha, "files": baseline,
    "sha256": {name: hashlib.sha256(text.encode()).hexdigest()
               for name, text in baseline.items()}
}).encode(), 9)).decode()

# The larger comparison/visualization code is in a plain Python source file;
# embed it to keep the delivered notebook self-contained.
comparison_template = Path(__file__).with_name("densenet121_023_comparison.py").read_text(encoding="utf-8")
comparison_code = comparison_template.replace("__BASELINE_B64__", BASELINE)
book["cells"].insert(19, cell("markdown", "## Validation comparison with EXP-022\n\n"
    "Per-seed CSVs copied from the EXP-022 ANALYSIS archive, with SHA-256 checks. "
    "Student-t CI95 covers run variation on this one split.\n"))
book["cells"].insert(20, cell("code", comparison_code))

export_template = Path(__file__).with_name("densenet121_023_export.py").read_text(encoding="utf-8")
put(book, 32, export_template)
view_template = Path(__file__).with_name("densenet121_023_view.py").read_text(encoding="utf-8")
book["cells"].append(cell("markdown", "## VIEW_ONLY — saved EXP-023 Validation artifacts\n"))
book["cells"].append(cell("code", view_template))
put(book, 9, "## Holdout access\n\nHoldout (4) is opened only in the opt-in evaluation cell after the five-seed Validation lock.\n")
put(book, 23, get(book, 23).replace("All five locked models are scored.",
    "When RUN_REAL_HOLDOUT=True after the Validation lock, all five frozen models are scored."))
put(book, 33, """## Completion check

- Dataset V10 ZIP and manifest hashes, class order and counts are verified.
- Train-only contact sheet and model smoke pass before full training.
- All five seeds complete both stages; checkpoint and deployment seed use Validation only.
- EXP-022 raw per-seed CSVs are hashed and paired with EXP-023 on the same Validation split.
- Final Test and Holdout (4) are opt-in and marked as exposed historical evaluations.
- EXP-023 FULL and ANALYSIS archives contain a clean notebook copy, metrics, figures,
  provenance and checksum manifest; FULL also contains checkpoints and model binaries.
""")

revision_material = ("".join(get(book, i) for i in (14, 17, 20, 32, 35))).encode("utf-8")
revision_sha = hashlib.sha256(revision_material).hexdigest()
put(book, 4, get(book, 4).replace('NOTEBOOK_VERSION = "023"',
    f'NOTEBOOK_VERSION = "023"\nCODE_REVISION_SHA256 = "{revision_sha}"'))
put(book, 16, get(book, 16).replace(
    '"experiment_id": EXPERIMENT_ID, "notebook_version": NOTEBOOK_VERSION,',
    '"experiment_id": EXPERIMENT_ID, "notebook_version": NOTEBOOK_VERSION,\n        "code_revision_sha256": CODE_REVISION_SHA256,') + '''
if not VIEW_ONLY:
    save_json(ENV_DIR / "runtime.json", {
        "python": sys.version, "tensorflow": tf.__version__,
        "keras": getattr(keras, "__version__", "tf.keras"),
        "platform": platform.platform(), "gpu": [str(g) for g in gpus],
        "cuda_build": str(tf.sysconfig.get_build_info().get("cuda_version", "unknown")),
        "seeds": SEEDS, "code_revision_sha256": CODE_REVISION_SHA256,
        "code_revision_method": "SHA256 of generated model/smoke, train, comparison, export, and VIEW_ONLY cells"})
''')

snapshot = copy.deepcopy(book)
snapshot["cells"][32]["source"] = ['NOTEBOOK_SOURCE_ZLIB_B64 = ""\n'] + snapshot["cells"][32]["source"]
source_b64 = base64.b64encode(zlib.compress(json.dumps(snapshot, ensure_ascii=False).encode(), 9)).decode()
put(book, 32, 'NOTEBOOK_SOURCE_ZLIB_B64 = "' + source_b64 + '"\n' + get(book, 32))
OUTPUT.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {OUTPUT}, {len(book['cells'])} cells, {OUTPUT.stat().st_size:,} bytes")
print("EXP-022 source SHA256:", archive_sha)
