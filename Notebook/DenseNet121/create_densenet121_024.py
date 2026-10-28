"""Build EXP-024: same Split (11) recipe at 224 and 256 pixels."""

import base64
import copy
import hashlib
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "densenet121_transfer_photometric_5seeds_023.ipynb"
TARGET = HERE / "densenet121_transfer_split11_resize_024.ipynb"
book = json.loads(SOURCE.read_text(encoding="utf-8"))


def get(index):
    return "".join(book["cells"][index]["source"])


def put(index, source):
    book["cells"][index]["source"] = source.splitlines(keepends=True)


def replace(index, old, new, expected=1):
    source = get(index)
    count = source.count(old)
    if count != expected:
        raise AssertionError(f"cell {index}: {old[:80]!r}: expected {expected}, got {count}")
    put(index, source.replace(old, new))


for cell in book["cells"]:
    source = "".join(cell["source"])
    source = source.replace("EXP-DENSENET121-TRANSFER-5SEEDS-023", "EXP-DENSENET121-TRANSFER-5SEEDS-024")
    source = source.replace("EXP-023", "EXP-024")
    source = source.replace("Split (10)", "Split (11)")
    source = source.replace("split_10", "split_11")
    source = source.replace("Cashew_dataV10_more_real", "Cashew_dataV11")
    source = source.replace("CashewData_Split (10).zip", "CashewData_Split (11).zip")
    source = source.replace("RL_Cashew_holdout (4).zip", "RL_Cashew_holdout (5).zip")
    source = source.replace("Holdout (4)", "Holdout (5)")
    source = source.replace("Holdout04", "Holdout05")
    source = source.replace("holdout04", "holdout05")
    source = source.replace("real_holdout_022", "real_holdout_024")
    source = source.replace("densenet121_transfer_photometric_5seeds_023.ipynb", TARGET.name)
    cell["source"] = source.splitlines(keepends=True)
    cell["metadata"] = {}
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None

put(0, """# EXP-024 — DenseNet121 on Split (11), resize comparison

The architecture and two-stage ImageNet transfer recipe follow EXP-022; the
training safeguards and RandomBrightness(0.08) follow EXP-023. Split (11) and
Holdout (5) are new frozen archives. The default input is **256×256**.

To measure the effect of resize, run this notebook twice on the same Split (11):
first with `IMG_SIZE = 224`, then with `IMG_SIZE = 256`. Each size has a separate
result namespace that includes the code revision. Both use the same five seeds
and Validation-only selection.
The second run writes paired Validation deltas and 95% confidence intervals.
Test and Holdout (5) are optional, only after Validation is locked. Do not use
them to choose image size or any hyperparameter. VIEW_ONLY reads saved results.

The ZIP checksums below identify the local archives inspected on 28 Oct 2026.
Run All starts five fresh ImageNet transfer runs at 256×256. No checkpoint from
EXP-022, EXP-023, or an earlier EXP-024 run is loaded for training.
Training and performance results are pending until this notebook is executed.
""")

replace(4, 'NOTEBOOK_VERSION = "023"', 'NOTEBOOK_VERSION = "024"')
replace(4, 'EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-024"',
        'EXPERIMENT_ID = f"EXP-DENSENET121-TRANSFER-5SEEDS-024-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"')
# IMG_SIZE must be defined before the experiment ID is constructed.
replace(4, 'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 023 .ipynb on Drive.',
        'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 024 .ipynb on Drive.\nIMG_SIZE = 256  # Use 224 for the matched control run.')
replace(4, 'SEEDS = [42, 123, 2026, 3407, 7777]\nIMG_SIZE = 224',
        'SEEDS = [42, 123, 2026, 3407, 7777]\nif IMG_SIZE not in (224, 256):\n    raise ValueError("EXP-024 compares only 224 and 256 pixels")')
replace(4, '"anthracnose": {"train": 982, "val": 316, "test": 122, "total": 1420}',
        '"anthracnose": {"train": 982, "val": 309, "test": 120, "total": 1411}')
replace(4, '"healthy": {"train": 834, "val": 216, "test": 118, "total": 1168}',
        '"healthy": {"train": 830, "val": 222, "test": 118, "total": 1170}')
replace(4, '"leaf_miner": {"train": 875, "val": 264, "test": 132, "total": 1271}',
        '"leaf_miner": {"train": 875, "val": 267, "test": 130, "total": 1272}')
replace(4, '"red_rust": {"train": 1127, "val": 335, "test": 158, "total": 1620}',
        '"red_rust": {"train": 1127, "val": 335, "test": 161, "total": 1623}')
replace(4, 'EXPECTED_TOTALS = {"train": 4920, "val": 1445, "test": 686, "total": 7051}',
        'EXPECTED_TOTALS = {"train": 4916, "val": 1447, "test": 685, "total": 7048}')
replace(4, 'EXPECTED_REAL_FILENAME_COUNTS = {"train": 242, "val": 50, "test": 28}',
        'EXPECTED_REAL_FILENAME_COUNTS = {"train": 238, "val": 54, "test": 28}')
replace(4, 'b24f0faa5e918fb43d7d16500c6b8234df1cd771cc4a27eb2354051bce84edac',
        '5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49')
replace(4, 'EXPECTED_MANIFEST_SHA256 = "9c1279776dd831ffa23bfa35960f7a3512eae88016f23abf067c4915d963c09e"',
        'EXPECTED_MANIFEST_SHA256 = None  # Frozen at first audited run; ZIP SHA256 is checked above.')
replace(4, '3B340EA830418638FA610FB2176DCC171739CC7FE0BD8E8430C8ED2A614717CC',
        '8D5C3758B5869C3AE19238FA8D78E53AD15BB40544F4C70F632602896D80045F')
replace(17, '''    if all(path.is_file() for path in (selected_path, summary_path, validation_path,
                                        stage1_dir / "best.weights.h5", stage2_dir / "best.weights.h5")):
        training_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
        validation_rows.append(json.loads(validation_path.read_text(encoding="utf-8")))
        print("Reusing completed seed", seed)
        continue
''', '')
replace(17, 'Partial seed output needs review: {seed_dir}',
        'Seed output already exists: {seed_dir}. Use a new experiment ID for a fresh run')
replace(4, '120-image four-class Holdout (5)', '120-image four-class Holdout (5)', expected=0)
replace(6, 'EXPECTED_ZIP_SHA256 = "b24f0faa5e918fb43d7d16500c6b8234df1cd771cc4a27eb2354051bce84edac"',
        '# EXPECTED_ZIP_SHA256 is defined in the configuration cell.')
replace(17, 'if metrics["healthy_count"] != 216:',
        'if metrics["healthy_count"] != EXPECTED_COUNTS["healthy"]["val"]:')
replace(18, 'if real_healthy_row["total"] != 3:',
        'if real_healthy_row["total"] != int(((valid_df["split"] == "val") &\n            (valid_df["class"] == "healthy") & valid_df["is_real_filename"]).sum()):')
replace(18, 'Expected exactly 3 real_ Healthy Validation images',
        'real_ Healthy Validation count changed')
replace(18, 'The 50 real_ Validation filenames changed',
        'The real_ Validation filename count changed')
replace(21, '686-image internal Test', '685-image internal Test')
replace(16, 'if DATASET_ZIP_SHA256 != EXPECTED_ZIP_SHA256 or DATASET_MANIFEST_SHA256 != EXPECTED_MANIFEST_SHA256:',
        'if DATASET_ZIP_SHA256 != EXPECTED_ZIP_SHA256 or (EXPECTED_MANIFEST_SHA256 is not None and DATASET_MANIFEST_SHA256 != EXPECTED_MANIFEST_SHA256):')
replace(16, '"counts": EXPECTED_TOTALS, "class_names": EXPECTED_CLASSES,',
        '"counts": EXPECTED_TOTALS, "class_names": EXPECTED_CLASSES,')
replace(16, '"seeds": SEEDS, "input": "raw RGB float32 0..255, bilinear resize 224x224",',
        '"seeds": SEEDS, "input": f"raw RGB float32 0..255, bilinear resize {IMG_SIZE}x{IMG_SIZE}",')
replace(16, '"comparison_scope": "+RandomBrightness(0.08) versus historical EXP-022 on identical V10",',
        '"comparison_scope": "224 versus 256 pixels on frozen Split (11); same model recipe and seeds",')
replace(16, '"baseline_source": "EXP-022 metrics reference; no checkpoints reused",',
        '"baseline_source": "EXP-022 architecture; EXP-023 augmentation; fresh checkpoints",')
replace(16, '"holdout_protocol": "120-image four-class Holdout (5); Validation lock precedes evaluation",',
        '"holdout_protocol": "120-image four-class Holdout (5); Validation lock precedes evaluation",')
replace(26, '"input": "RGB float32 raw 0..255, 224x224, bilinear resize",',
        '"input": f"RGB float32 raw 0..255, {IMG_SIZE}x{IMG_SIZE}, bilinear resize",')
replace(26, '    (CODE_PACKAGE_DIR / "yolo_classifier_adapter.py").write_text(yolo_adapter_code, encoding="utf-8")',
        '    yolo_adapter_code = yolo_adapter_code.replace("[224, 224]", f"[{IMG_SIZE}, {IMG_SIZE}]")\n    (CODE_PACKAGE_DIR / "yolo_classifier_adapter.py").write_text(yolo_adapter_code, encoding="utf-8")')
replace(26, 'Do not overwrite EXP-022 checkpoints.', 'Do not overwrite EXP-024 checkpoints.')
replace(24, 'frozen V10/holdout05 pair', 'frozen V11/holdout05 pair')
replace(31, '"real_holdout": 120}[split]',
        '"real_holdout": sum(REAL_HOLDOUT_EXPECTED_COUNTS.values())}[split]')
replace(32, 'holdout_status = "EXPOSED_HOLDOUT_04"', 'holdout_status = "EXPOSED_HOLDOUT_05"')
replace(32, '"DenseNet121 + RandomBrightness(0.08) versus historical EXP-022 on frozen V10.\\n\\n"',
        'f"DenseNet121 + RandomBrightness(0.08), Split (11), {IMG_SIZE}x{IMG_SIZE}.\\n\\n"')
replace(32, '"Validation has only 3 real_ Healthy images. Paired-seed CI is not a field-population CI.\\n"',
        '"Paired-seed CI on one split is not a field-population CI.\\n"')
put(19, """## Same-split Validation comparison: 224 versus 256

Run both sizes to compare the five paired seeds. This cell waits for the other
size's results; rerun this cell after both Validation runs finish. The reported
delta is `256 − 224`. A confidence interval spanning zero does not establish
an improvement. Test and Holdout are excluded from choosing image size.
""")

put(20, '''# Compare only finished, matching Split (11) Validation runs.
def compare_image_sizes():
    other_size = 224 if IMG_SIZE == 256 else 256
    other_id = f"EXP-DENSENET121-TRANSFER-5SEEDS-024-{other_size}px-{CODE_REVISION_SHA256[:8]}"
    other_dir = RESULTS_ROOT / other_id
    other_config_path = other_dir / "experiment_config.json"
    other_lock = other_dir / "validation_selection_lock.json"
    if not (other_config_path.is_file() and other_lock.is_file()):
        print(f"224/256 comparison PENDING: finish the {other_size}px Validation run first")
        return
    other_config = json.loads(other_config_path.read_text(encoding="utf-8"))
    for field in ("archive_sha256", "manifest_sha256", "counts", "class_names"):
        if other_config["dataset"][field] != experiment_config["dataset"][field]:
            raise ValueError(f"Cannot compare different datasets: {field}")
    if other_config["seeds"] != experiment_config["seeds"]:
        raise ValueError("Cannot pair different seeds")
    for field in ("model", "optimizer", "weight_decay", "batch_size", "loss",
                  "stage1", "stage2", "selection_rule", "deployment_rule"):
        if other_config[field] != experiment_config[field]:
            raise ValueError(f"Cannot isolate resize; recipe differs: {field}")
    own = pd.read_csv(AGGREGATE_DIR / "validation_results_5seeds.csv").set_index("seed")
    other = pd.read_csv(other_dir / "aggregate" / "validation_results_5seeds.csv").set_index("seed")
    if sorted(own.index.tolist()) != sorted(SEEDS) or sorted(other.index.tolist()) != sorted(SEEDS):
        raise ValueError("Both runs must finish all five seeds")
    runs = {IMG_SIZE: own, other_size: other}
    metrics = ("val_macro_f1", "val_accuracy", "false_healthy_rate",
               "healthy_to_diseased_rate", "inference_time_ms_per_image")
    paired = pd.DataFrame({"seed": SEEDS})
    for metric in metrics:
        for size in (224, 256):
            paired[f"{size}px_{metric}"] = runs[size].loc[SEEDS, metric].to_numpy()
        paired[f"delta_256_minus_224_{metric}"] = paired[f"256px_{metric}"] - paired[f"224px_{metric}"]
    paired.to_csv(AGGREGATE_DIR / "resize_224_vs_256_by_seed.csv", index=False)
    from scipy.stats import t
    summary = []
    for metric in metrics:
        delta = paired[f"delta_256_minus_224_{metric}"].to_numpy(dtype=float)
        mean, sd = float(delta.mean()), float(delta.std(ddof=1))
        half = float(t.ppf(.975, len(delta)-1) * sd / np.sqrt(len(delta)))
        summary.append({"metric": metric, "mean_delta_256_minus_224": mean,
                        "sd_delta": sd, "ci95_low": mean-half, "ci95_high": mean+half,
                        "n_paired_seeds": len(delta)})
    summary = pd.DataFrame(summary)
    summary.to_csv(AGGREGATE_DIR / "resize_224_vs_256_summary.csv", index=False)
    display(paired, summary)
    save_json(OUT_DIR / "validation_decision_note.json", {
        "comparison": "256 minus 224 on identical Split (11), paired five seeds",
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "selection_source": "Validation only", "automatic_promotion": False,
        "limitation": "Five seeds on one split; CI estimates run variation, not population shift."})

if RUN_FULL_TRAINING or VIEW_ONLY:
    compare_image_sizes()
else:
    print("Resize comparison PENDING until Validation finishes")
''')

put(23, """## Optional external real Holdout (5)

After the Validation lock, this cell checks the 120-image archive against all
Train/Validation/Test bytes and scores all five frozen checkpoints. Four-class
metrics are separate from the five-class Test. The holdout is not used to
select the resize size, checkpoint, seed, or threshold.
""")
put(33, """## Completion check

- Split (11) and Holdout (5) ZIP checksums, classes and counts are verified.
- The dataset audit checks corrupt files and cross-split byte overlap.
- Train-only augmentation and model smoke pass before full training.
- Five seeds finish both stages; checkpoint and deployment seed use Validation.
- The 224/256 comparison runs only when both arms have identical dataset and recipe.
- Test and Holdout (5) are opt-in after the Validation lock.
""")
put(35, '''# Saved Validation artifacts; no dataset, training, Test or Holdout access.
if VIEW_ONLY:
    from IPython.display import Image as DisplayImage
    for name in ("validation_results_5seeds.csv", "validation_mean_std_summary.csv",
                 "per_class_validation_by_seed.csv", "real_validation_by_seed.csv",
                 "resize_224_vs_256_by_seed.csv", "resize_224_vs_256_summary.csv"):
        path = AGGREGATE_DIR / name
        if path.is_file():
            print(name); display(pd.read_csv(path))
    for path in (PANEL_DIR / "accuracy_loss_curves_5seeds.png",
                 PANEL_DIR / "validation_confusion_matrices_5seeds.png"):
        if path.is_file(): display(DisplayImage(filename=str(path)))
''')

# EXP-023's embedded notebook must be regenerated rather than carried into EXP-024.
source = get(32)
source = re.sub(r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]+"\n', '', source, count=1)
if source.startswith('NOTEBOOK_SOURCE_ZLIB_B64'):
    raise AssertionError("Could not remove EXP-023 embedded source")
put(32, source)
revision_material = "".join(get(book_index) for book_index in (14, 17, 20, 32, 35)).encode()
revision_sha = hashlib.sha256(revision_material).hexdigest()
put(4, re.sub(r'CODE_REVISION_SHA256 = "[a-f0-9]+"',
              f'CODE_REVISION_SHA256 = "{revision_sha}"', get(4), count=1))
snapshot = copy.deepcopy(book)
snapshot["cells"][32]["source"] = ['NOTEBOOK_SOURCE_ZLIB_B64 = ""\n'] + snapshot["cells"][32]["source"]
embedded = base64.b64encode(zlib.compress(json.dumps(snapshot, ensure_ascii=False).encode(), 9)).decode()
put(32, f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"\n' + get(32))
TARGET.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(book['cells'])} cells, {TARGET.stat().st_size:,} bytes")
