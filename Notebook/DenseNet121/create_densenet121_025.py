"""Create an evaluation-only notebook for frozen EXP-024 DenseNet121 checkpoints."""

import base64
import copy
import json
import re
import textwrap
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "densenet121_transfer_split11_resize_024.ipynb"
TARGET = HERE / "densenet121_eval_024_checkpoints_025.ipynb"
source = json.loads(SOURCE.read_text(encoding="utf-8"))


def src(index):
    return "".join(source["cells"][index]["source"])


def cell(kind, body):
    item = {"cell_type": kind, "metadata": {}, "source": body.splitlines(keepends=True)}
    if kind == "code":
        item.update(execution_count=None, outputs=[])
    return item


book = {"cells": [], "metadata": source.get("metadata", {}), "nbformat": 4,
        "nbformat_minor": source.get("nbformat_minor", 5)}
add = lambda kind, body: book["cells"].append(cell(kind, body))

add("markdown", """# EXP-025 — Final Test and Holdout (5) from frozen EXP-024 checkpoints

Evaluation only: this notebook does **not train**. It verifies the source
EXP-024 Validation lock, dataset fingerprint and all five checkpoint SHA-256
values before scoring the 685-image Final Test and 120-image Holdout (5).
Both evaluations write to a separate EXP-025 result directory. The Validation
selected deployment seed stays fixed; Test and Holdout do not tune the model.

The pinned source is the completed 024 256px run in the shared Google Drive
project: [EXP-024 result folder](https://drive.google.com/drive/folders/13BEQoXLIdhanQ1zSLRawhAeoBwAQji-M).
The notebook finds it through the mounted project shortcut. Set
`SOURCE_DIR_OVERRIDE` to that mounted folder, or `SOURCE_FULL_ZIP_OVERRIDE` to
its FULL archive if automatic discovery fails. An ANALYSIS archive has no checkpoints.
The Split (11) and Holdout (5) ZIPs must also be mounted, or given through the
archive override paths. Run all cells in order.
""")
imports = src(2).replace("import zipfile\n", "import zipfile\nimport stat\nimport tempfile\n")
add("code", imports)
add("code", src(3).replace(
    'if RUNTIME == "Colab" and not gpus:\n    print("CPU runtime detected; VIEW_ONLY can continue, model smoke/training requires GPU.")',
    'if not gpus:\n    print("CPU evaluation is available, but may be slow.")'))

add("code", '''# EXP-025 configuration: 024 checkpoints are read-only inputs.
SOURCE_DIR_OVERRIDE = None       # e.g. /content/drive/MyDrive/Cashew_Leaf_model_result/training_results/current_dataset/EXP-...024-256px-f9fd5d5f
SOURCE_FULL_ZIP_OVERRIDE = None  # mounted Drive path to SOURCE_EXP_ID + "_FULL.zip", or attached Kaggle input
SOURCE_EXP_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-024-256px-f9fd5d5f"
SOURCE_DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/13BEQoXLIdhanQ1zSLRawhAeoBwAQji-M"
DATASET_ARCHIVE_OVERRIDE = None
REAL_HOLDOUT_ARCHIVE_OVERRIDE = None
DRIVE_PROJECT_DIR_OVERRIDE = None
DRIVE_RESULTS_ROOT_OVERRIDE = None
FORCE_REMOUNT_DRIVE = False
RUN_FINAL_TEST = True
RUN_REAL_HOLDOUT = True
RUN_FULL_TRAINING = False
RUN_PIPELINE_SMOKE = False
RUN_SMOKE_ONLY = False
VIEW_ONLY = False

DATASET_ARCHIVE_NAME = "CashewData_Split (11).zip"
REAL_HOLDOUT_ARCHIVE_NAME = "RL_Cashew_holdout (5).zip"
EXPECTED_ZIP_SHA256 = "5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49"
REAL_HOLDOUT_ARCHIVE_SHA256 = "8D5C3758B5869C3AE19238FA8D78E53AD15BB40544F4C70F632602896D80045F"
REAL_HOLDOUT_EXPECTED_COUNTS = {"anthracnose": 46, "healthy": 13, "leaf_miner": 36, "red_rust": 25}
EXPECTED_CLASSES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
EXPECTED_COUNTS = {
    "anthracnose": {"train": 982, "val": 309, "test": 120, "total": 1411},
    "healthy": {"train": 830, "val": 222, "test": 118, "total": 1170},
    "leaf_miner": {"train": 875, "val": 267, "test": 130, "total": 1272},
    "not_cashew_leaf": {"train": 1102, "val": 314, "test": 156, "total": 1572},
    "red_rust": {"train": 1127, "val": 335, "test": 161, "total": 1623},
}
EXPECTED_TOTALS = {"train": 4916, "val": 1447, "test": 685, "total": 7048}
EXPECTED_REAL_FILENAME_COUNTS = {"train": 238, "val": 54, "test": 28}
REQUIRED_ABSENT_IMAGES = []
STRICT_DATASET_CHECK = True
SEEDS = [42, 123, 2026, 3407, 7777]
IMG_SIZE = 256
MODEL_NAME = "DenseNet121"
EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-025-256px"

if RUNTIME == "Windows":
    PROJECT_ROOT = Path(r"D:\Study\KhoaLuanTotNghiep\Resource")
    RESULTS_ROOT = PROJECT_ROOT / "training_results" / "current_dataset"
elif RUNTIME == "Kaggle":
    PROJECT_ROOT = Path("/kaggle/working")
    RESULTS_ROOT = PROJECT_ROOT / "training_results" / "current_dataset"
elif RUNTIME == "Colab":
    from google.colab import drive
    drive.mount("/content/drive", force_remount=FORCE_REMOUNT_DRIVE)
    PROJECT_ROOT = Path("/content")
    drive_root = Path("/content/drive/MyDrive")
    if DRIVE_PROJECT_DIR_OVERRIDE is not None:
        DRIVE_PROJECT_DIR = Path(DRIVE_PROJECT_DIR_OVERRIDE).expanduser()
    else:
        direct = drive_root / "Cashew_Leaf_model_result"
        candidates = [direct] if (direct / "Dataset" / DATASET_ARCHIVE_NAME).is_file() else []
        if not candidates:
            candidates = [p for p in drive_root.iterdir() if p.is_dir() and
                          (p / "Dataset" / DATASET_ARCHIVE_NAME).is_file()]
        if len(candidates) != 1:
            raise FileNotFoundError("Set DRIVE_PROJECT_DIR_OVERRIDE to the project shortcut")
        DRIVE_PROJECT_DIR = candidates[0]
    DRIVE_DATASET_ARCHIVE = DRIVE_PROJECT_DIR / "Dataset" / DATASET_ARCHIVE_NAME
    RESULTS_ROOT = (Path(DRIVE_RESULTS_ROOT_OVERRIDE).expanduser() if
                    DRIVE_RESULTS_ROOT_OVERRIDE is not None else
                    DRIVE_PROJECT_DIR / "training_results" / "current_dataset")
else:
    raise RuntimeError(f"Unsupported runtime: {RUNTIME}")

def _source_candidates():
    roots = [RESULTS_ROOT]
    if RUNTIME == "Kaggle": roots.append(Path("/kaggle/input"))
    found = []
    for root in roots:
        if not root.is_dir(): continue
        found.extend(p.parent for p in root.rglob("validation_selection_lock.json")
                     if p.parent.name == SOURCE_EXP_ID)
    return sorted(set(found))

def _extract_full_archive(archive_path):
    archive_path = Path(archive_path).expanduser()
    if not archive_path.is_file(): raise FileNotFoundError(archive_path)
    if not archive_path.name.endswith("_FULL.zip"):
        raise ValueError("Use EXP-024 FULL.zip; ANALYSIS.zip lacks checkpoints")
    cache = PROJECT_ROOT / "source_024_cache" / archive_path.stem
    marker = cache / ".source_zip_sha256"
    digest = file_sha256(archive_path)
    if cache.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != digest:
            raise RuntimeError(f"Source cache changed: {cache}")
    else:
        cache.mkdir(parents=True)
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (cache / member.filename).resolve()
                if stat.S_ISLNK(member.external_attr >> 16) or (target != cache.resolve() and cache.resolve() not in target.parents):
                    raise ValueError(f"Unsafe source ZIP member: {member.filename}")
            archive.extractall(cache)
        marker.write_text(digest + "\\n", encoding="utf-8")
    dirs = [p.parent for p in cache.rglob("validation_selection_lock.json")]
    if len(dirs) != 1: raise RuntimeError(f"Expected one EXP-024 run in {archive_path}; found {dirs}")
    return dirs[0]

if SOURCE_DIR_OVERRIDE is not None and SOURCE_FULL_ZIP_OVERRIDE is not None:
    raise ValueError("Set only one source override")
if SOURCE_DIR_OVERRIDE is not None:
    SOURCE_DIR = Path(SOURCE_DIR_OVERRIDE).expanduser()
elif SOURCE_FULL_ZIP_OVERRIDE is not None:
    SOURCE_DIR = _extract_full_archive(SOURCE_FULL_ZIP_OVERRIDE)
else:
    candidates = _source_candidates()
    if len(candidates) == 1:
        SOURCE_DIR = candidates[0]
    elif not candidates:
        zip_roots = [RESULTS_ROOT] + ([Path("/kaggle/input")] if RUNTIME == "Kaggle" else [])
        archives = sorted(set(p for root in zip_roots if root.is_dir()
                              for p in root.rglob(SOURCE_EXP_ID + "_FULL.zip")))
        if len(archives) != 1:
            raise FileNotFoundError("Cannot find exactly one 024 256px FULL archive; set SOURCE_DIR_OVERRIDE or SOURCE_FULL_ZIP_OVERRIDE")
        SOURCE_DIR = _extract_full_archive(archives[0])
    else:
        raise RuntimeError(f"Multiple EXP-024 sources: {candidates}. Set SOURCE_DIR_OVERRIDE")
if not SOURCE_DIR.is_dir(): raise FileNotFoundError(SOURCE_DIR)
print("Source EXP-024:", SOURCE_DIR)
print("EXP-025 output root:", RESULTS_ROOT)
''')

add("code", '''# Verify the immutable 024 source before reading any Test or Holdout labels.
source_config_path = SOURCE_DIR / "experiment_config.json"
source_lock_path = SOURCE_DIR / "validation_selection_lock.json"
source_validation_path = SOURCE_DIR / "aggregate" / "validation_results_5seeds.csv"
for path in (source_config_path, source_lock_path, source_validation_path):
    if not path.is_file(): raise FileNotFoundError(path)
source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
selection_lock = json.loads(source_lock_path.read_text(encoding="utf-8"))
if source_config["experiment_id"] != SOURCE_EXP_ID:
    raise ValueError("Source is not the pinned EXP-024 256px run")
if SOURCE_DIR.name != source_config["experiment_id"]:
    raise ValueError("Source folder and config experiment ID differ")
if source_config["dataset"]["archive"] != DATASET_ARCHIVE_NAME or source_config["dataset"]["archive_sha256"] != EXPECTED_ZIP_SHA256:
    raise ValueError("Source 024 is not the frozen Split (11) archive")
if source_config["dataset"]["counts"] != EXPECTED_TOTALS or source_config["dataset"]["class_names"] != EXPECTED_CLASSES:
    raise ValueError("Source class order or split counts changed")
if source_config["dataset"]["real_filename_counts"] != EXPECTED_REAL_FILENAME_COUNTS:
    raise ValueError("Source real_ counts changed")
if source_config["seeds"] != SEEDS or "256x256" not in source_config["input"]:
    raise ValueError("Source 024 is not the five-seed 256px run")
if source_config["model"]["backbone"] != "DenseNet121" or source_config["model"]["augmentation"].get("brightness_factor") != 0.08:
    raise ValueError("Source model recipe differs from EXP-024")
if selection_lock["experiment_id"] != source_config["experiment_id"] or selection_lock["config_sha256"] != file_sha256(source_config_path):
    raise ValueError("Source Validation lock does not match source config")
if selection_lock["dataset_sha256"] != EXPECTED_ZIP_SHA256 or selection_lock["dataset_manifest_sha256"] != source_config["dataset"]["manifest_sha256"]:
    raise ValueError("Source Validation lock dataset fingerprint changed")
if selection_lock["class_names"] != EXPECTED_CLASSES or selection_lock["seeds"] != SEEDS:
    raise ValueError("Source Validation lock class order/seeds changed")
if selection_lock["validation_sha256"] != file_sha256(source_validation_path):
    raise ValueError("Source Validation CSV changed")
validation_results_df = pd.read_csv(source_validation_path)
if validation_results_df["seed"].tolist() != SEEDS:
    raise ValueError("Source Validation lacks the five expected seeds")
ranked = validation_results_df.sort_values(["val_loss", "val_macro_f1", "seed"], ascending=[True, False, True])
deployment_seed = int(ranked.iloc[0]["seed"])
if deployment_seed != int(selection_lock["deployment_seed"]):
    raise ValueError("Validation-selected deployment seed changed")
for seed in SEEDS:
    checkpoint = SOURCE_DIR / f"seed_{seed}" / "best.weights.h5"
    stage_path = SOURCE_DIR / f"seed_{seed}" / "stage_selection.json"
    if not checkpoint.is_file() or not stage_path.is_file():
        raise FileNotFoundError(f"Missing EXP-024 checkpoint or stage selection for seed {seed}")
    if file_sha256(checkpoint) != selection_lock["selected_checkpoint_sha256"][str(seed)]:
        raise ValueError(f"EXP-024 checkpoint SHA256 changed for seed {seed}")

IMG_SIZE = 256
BATCH_SIZE = int(source_config["batch_size"])
HEAD_DROPOUT = float(source_config["model"]["head_dropout"])
WEIGHT_DECAY = float(source_config["weight_decay"])
EXPECTED_MANIFEST_SHA256 = source_config["dataset"]["manifest_sha256"]
DATASET_ZIP_SHA256 = EXPECTED_ZIP_SHA256
CLASS_NAMES = EXPECTED_CLASSES
NUM_CLASSES = len(CLASS_NAMES)

OUT_DIR = RESULTS_ROOT / EXPERIMENT_ID
AGGREGATE_DIR = OUT_DIR / "aggregate"
PANEL_DIR = OUT_DIR / "figures" / "panels"
for directory in (OUT_DIR, AGGREGATE_DIR, PANEL_DIR): directory.mkdir(parents=True, exist_ok=True)
provenance = {
    "experiment_id": EXPERIMENT_ID,
    "source_experiment_id": source_config["experiment_id"],
    "source_config_sha256": file_sha256(source_config_path),
    "source_validation_lock_sha256": file_sha256(source_lock_path),
    "source_validation_sha256": file_sha256(source_validation_path),
    "source_checkpoint_sha256": selection_lock["selected_checkpoint_sha256"],
    "dataset_zip_sha256": EXPECTED_ZIP_SHA256,
    "dataset_manifest_sha256": EXPECTED_MANIFEST_SHA256,
    "holdout_zip_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,
    "deployment_seed": deployment_seed,
    "training": "NONE", "final_test": RUN_FINAL_TEST, "holdout": RUN_REAL_HOLDOUT,
}
provenance_path = OUT_DIR / "source_024_provenance.json"
if provenance_path.exists() and json.loads(provenance_path.read_text(encoding="utf-8")) != provenance:
    raise ValueError("EXP-025 already points to a different EXP-024 source")
save_json(provenance_path, provenance)
print("Verified 024 Validation lock and five checkpoint hashes; selected seed:", deployment_seed)
''')

add("markdown", "## Audit the frozen Split (11) ZIP")
dataset_code = src(6).replace('    EXPECTED_ZIP_SHA256 = "5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49"',
                              '    # EXPECTED_ZIP_SHA256 comes from the verified EXP-024 source.')
add("code", dataset_code)
add("code", src(8))
add("code", '''if DATASET_MANIFEST_SHA256 != EXPECTED_MANIFEST_SHA256:
    raise ValueError("Split (11) file manifest differs from trained EXP-024 dataset")
print("Verified source dataset manifest:", DATASET_MANIFEST_SHA256)
''')
add("code", src(5))
add("code", src(12))

model_code = src(14).split("    if RUN_PIPELINE_SMOKE:", 1)[0]
if "def build_transfer_model" not in model_code or ".fit(" in model_code:
    raise AssertionError("Could not isolate EXP-024 architecture without training")
add("markdown", "## Rebuild the architecture and load only frozen 024 weights")
add("code", model_code)

test_code = src(22)
if 'output.mkdir(exist_ok=True)' not in test_code:
    raise AssertionError("Test output directory creation changed in EXP-024")
test_code = test_code.replace('output.mkdir(exist_ok=True)',
                              'output.mkdir(parents=True, exist_ok=True)')
for old, new in [
    ('(OUT_DIR / "validation_selection_lock.json")', '(SOURCE_DIR / "validation_selection_lock.json")'),
    ('file_sha256(config_path)', 'file_sha256(source_config_path)'),
    ('file_sha256(OUT_DIR / f"seed_{seed}" / "best.weights.h5")',
     'file_sha256(SOURCE_DIR / f"seed_{seed}" / "best.weights.h5")'),
    ('(OUT_DIR / f"seed_{seed}" / "stage_selection.json")',
     '(SOURCE_DIR / f"seed_{seed}" / "stage_selection.json")'),
    ('model.load_weights(OUT_DIR / f"seed_{seed}" / "best.weights.h5")',
     'model.load_weights(SOURCE_DIR / f"seed_{seed}" / "best.weights.h5")'),
]:
    if old not in test_code: raise AssertionError(f"Test patch missing {old}")
    test_code = test_code.replace(old, new)
add("markdown", "## Final Test — five frozen EXP-024 models")
add("code", test_code)

holdout_code = src(24).replace("real_holdout_024", "real_holdout_025")
old = 'checkpoint_paths = {seed: OUT_DIR / f"seed_{seed}" / "best.weights.h5" for seed in SEEDS}'
new = 'checkpoint_paths = {seed: SOURCE_DIR / f"seed_{seed}" / "best.weights.h5" for seed in SEEDS}'
if old not in holdout_code: raise AssertionError("Holdout checkpoint source not found")
holdout_code = holdout_code.replace(old, new)
add("markdown", "## Holdout (5) — separate four-class evaluation")
add("code", holdout_code)

add("code", '''# EXP-025 result summary and archive; no checkpoint bytes are copied.
assert_selection_lock()
if not (AGGREGATE_DIR / "test_results_5seeds.csv").is_file():
    raise FileNotFoundError("Final Test results are incomplete")
if not (OUT_DIR / "real_holdout" / "evaluation_completed.json").is_file():
    raise FileNotFoundError("Holdout results are incomplete")
summary = {
    "experiment_id": EXPERIMENT_ID, "source_experiment_id": source_config["experiment_id"],
    "selection_source": "EXP-024 Validation only", "deployment_seed": deployment_seed,
    "training": "NOT_RUN", "final_test_images": EXPECTED_TOTALS["test"],
    "holdout_images": sum(REAL_HOLDOUT_EXPECTED_COUNTS.values()),
    "final_test": "COMPLETE", "holdout": "COMPLETE",
    "source_validation_lock_sha256": file_sha256(source_lock_path),
}
save_json(OUT_DIR / "evaluation_summary.json", summary)
display(pd.read_csv(AGGREGATE_DIR / "test_results_5seeds.csv"))
display(pd.read_csv(OUT_DIR / "real_holdout" / "aggregate" / "real_holdout_results_5seeds.csv"))
notebook_copy = OUT_DIR / "notebook" / "densenet121_eval_024_checkpoints_025.ipynb"
notebook_copy.parent.mkdir(exist_ok=True)
snapshot = json.loads(zlib.decompress(base64.b64decode(NOTEBOOK_SOURCE_ZLIB_B64)))
notebook_copy.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1) + "\\n", encoding="utf-8")
analysis_zip = RESULTS_ROOT / f"{EXPERIMENT_ID}_ANALYSIS.zip"
with zipfile.ZipFile(analysis_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
    for path in sorted(OUT_DIR.rglob("*")):
        if path.is_file():
            archive.write(path, arcname=f"{EXPERIMENT_ID}/{path.relative_to(OUT_DIR).as_posix()}")
print("EXP-025 evaluation complete:", analysis_zip, "SHA256", file_sha256(analysis_zip))
''')
add("markdown", """## Interpretation

EXP-025 evaluates 024 checkpoints without retraining or changing the
Validation-selected seed. Final Test has five true classes; Holdout (5) has
four true classes, so compare their reports separately. Holdout has no true
`not_cashew_leaf` support. These metrics describe the frozen 024 models and
must not be used to choose a new checkpoint or resize size.
""")

for index, item in enumerate(book["cells"]):
    if item["cell_type"] == "code":
        body = "".join(item["source"])
        if ".fit(" in body or "RUN_FULL_TRAINING = True" in body:
            raise AssertionError(f"Training code leaked into cell {index}")
        if index != 0 and "!nvidia-smi" not in body:
            compile(body, f"cell-{index}", "exec")

# Embed a copy that is readable after export without relying on local paths.
snapshot = copy.deepcopy(book)
last_code = next(i for i in range(len(book["cells"])-1, -1, -1)
                 if book["cells"][i]["cell_type"] == "code")
snapshot["cells"][last_code]["source"] = ['NOTEBOOK_SOURCE_ZLIB_B64 = ""\n'] + snapshot["cells"][last_code]["source"]
embedded = base64.b64encode(zlib.compress(json.dumps(snapshot, ensure_ascii=False).encode(), 9)).decode()
book["cells"][last_code]["source"] = [f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"\n'] + book["cells"][last_code]["source"]
TARGET.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(book['cells'])} cells, {TARGET.stat().st_size:,} bytes")
