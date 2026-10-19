"""Build a standalone EXP-015 real-holdout evaluator for locked EXP-014 weights."""

import copy
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE_NOTEBOOK = HERE / "densenet121_transfer_5seed_014.ipynb"
TARGET = HERE / "densenet121_transfer_5seed_015.ipynb"
SOURCE = json.loads(SOURCE_NOTEBOOK.read_text(encoding="utf-8"))
CELLS = []


def add(kind, source):
    cell = {
        "cell_type": kind,
        "id": hashlib.sha256(f"{len(CELLS)}:{source}".encode()).hexdigest()[:8],
        "metadata": {},
        "source": source.splitlines(keepends=True),
    }
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    CELLS.append(cell)


add("markdown", """# EXP-DENSENET121-TRANSFER-5SEEDS-015

**Mục tiêu:** đánh giá real holdout bằng đúng năm checkpoint đã huấn luyện và
khóa theo Validation trong `EXP-DENSENET121-TRANSFER-5SEEDS-014`.

Đây là **lượt đánh giá của mô hình 014**, không phải lần huấn luyện mới. Notebook
không thay đổi weights, seed triển khai, hyperparameter hay kết quả 014. Kết quả
015 được lưu trong thư mục riêng. `RUN_REAL_HOLDOUT=True`; Final Test vẫn tắt.

Holdout này từng được xem trong quá trình nghiên cứu và có một ảnh trùng byte
với Test V08. Notebook loại ảnh trùng trước khi chấm 48 ảnh còn lại. Kết quả
là so sánh lịch sử có giới hạn, chưa phải phép thử thực địa mới độc lập.

**Chạy trên Colab:** đặt FULL ZIP 014 hoặc thư mục kết quả 014 tại
`MyDrive/Cashew_Leaf_model_result/training_results/current_dataset/`, cùng
`RL_Cashew_holdout.zip` tại `MyDrive/Cashew_Leaf_model_result/Dataset/`, chọn
runtime GPU rồi Run all. ANALYSIS ZIP không chứa checkpoints.
""")

add("code", """# 01. Imports
import gc
import hashlib
import io
import json
import os
import platform
import random
import shutil
import sys
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image
from IPython.display import display
from sklearn.metrics import accuracy_score

print('Python:', sys.version.split()[0], '| TensorFlow:', tf.__version__)
print('GPU:', tf.config.list_physical_devices('GPU'))
""")

add("code", r'''# 02. Locked evaluation configuration and Drive paths
SOURCE_EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-014"
EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-015"
RUN_FULL_TRAINING = False
RUN_FINAL_TEST = False
RUN_REAL_HOLDOUT = True
SEEDS = [42, 123, 2026, 3407, 7777]
CLASS_NAMES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE, BATCH_SIZE = 224, 16
HEAD_DROPOUT, WEIGHT_DECAY = 0.40, 1e-4
STAGE1_LR, STAGE1_EPOCHS = 3e-4, 12
STAGE2_LR, STAGE2_EPOCHS = 1e-5, 30
EXPECTED_DATASET_SHA256 = "75c5928b2f6432aca901bc16831e91f9aa4bdacc1e2ff536753d71da60c147e7"
EXPECTED_MANIFEST_SHA256 = "1810d7e654fede9c8f6a9d08a4078be7882492db2d2d842a19b5320b9aae356b"
REAL_HOLDOUT_ARCHIVE_NAME = "RL_Cashew_holdout.zip"
REAL_HOLDOUT_ARCHIVE_SHA256 = "BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165"
REAL_HOLDOUT_EXPECTED_COUNTS = {"anthracnose": 13, "healthy": 5, "leaf_miner": 24, "red_rust": 7}
REAL_HOLDOUT_ARCHIVE_OVERRIDE = None
SOURCE_FULL_ZIP_OVERRIDE = None

if RUN_FULL_TRAINING or RUN_FINAL_TEST or not RUN_REAL_HOLDOUT:
    raise RuntimeError("EXP-015 is the locked 014 checkpoint real-holdout evaluation only")
if Path("/content").is_dir():
    from google.colab import drive
    drive.mount("/content/drive")
    project_root = Path("/content/drive/MyDrive/Cashew_Leaf_model_result")
    RESULTS_ROOT = project_root / "training_results" / "current_dataset"
    CACHE_ROOT = Path("/content/exp015_source_cache")
elif os.name == "nt":
    RESULTS_ROOT = Path(r"D:\Study\KhoaLuanTotNghiep\Resource\training_results\current_dataset")
    CACHE_ROOT = RESULTS_ROOT / "_exp015_source_cache"
else:
    RESULTS_ROOT = Path("/kaggle/working/training_results/current_dataset")
    CACHE_ROOT = Path("/kaggle/working/exp015_source_cache")
OUT_DIR = RESULTS_ROOT / EXPERIMENT_ID
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("EXP-014 results root:", RESULTS_ROOT)
print("EXP-015 output:", OUT_DIR)
''')

add("markdown", """## Nguồn checkpoint 014 và kiểm tra khóa

Ưu tiên thư mục kết quả 014. Nếu chỉ có FULL ZIP, notebook trích đúng config,
manifest, bảng kết quả và năm checkpoint cần cho đánh giá. Mọi checksum được
đối chiếu với `validation_selection_lock.json` trước khi đọc holdout.
""")

add("code", r'''# 03. Locate EXP-014 results without changing the trained run
def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")

required_relatives = [
    "experiment_config.json", "validation_selection_lock.json", "dataset_manifest.csv",
    "execution_status.json",
    "aggregate/training_summary_5seeds.csv", "aggregate/validation_results_5seeds.csv",
    "aggregate/real_validation_by_seed.csv",
]
for seed in SEEDS:
    required_relatives += [f"seed_{seed}/best.weights.h5",
                           f"seed_{seed}/stage_selection.json",
                           f"seed_{seed}/stage1/summary.json",
                           f"seed_{seed}/stage2/summary.json"]
optional_relatives = ["model_package/deployment_model.keras",
                      "environment/environment.json",
                      "aggregate/test_results_5seeds.csv"]
source_folder = RESULTS_ROOT / SOURCE_EXPERIMENT_ID
if all((source_folder / relative).is_file() for relative in required_relatives):
    SOURCE_DIR = source_folder
    SOURCE_FULL_ZIP_SHA256 = None
    print("Using completed EXP-014 result folder:", SOURCE_DIR)
else:
    full_zip = (Path(SOURCE_FULL_ZIP_OVERRIDE) if SOURCE_FULL_ZIP_OVERRIDE is not None
                else RESULTS_ROOT / f"{SOURCE_EXPERIMENT_ID}_FULL.zip")
    if not full_zip.is_file():
        missing = [relative for relative in required_relatives
                   if not (source_folder / relative).is_file()]
        raise FileNotFoundError(f"EXP-014 folder is incomplete: {missing}; FULL ZIP missing: {full_zip}")
    SOURCE_FULL_ZIP_SHA256 = file_sha256(full_zip)
    SOURCE_DIR = CACHE_ROOT / SOURCE_FULL_ZIP_SHA256[:16]
    marker = SOURCE_DIR / ".full_zip_sha256"
    if SOURCE_DIR.exists():
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != SOURCE_FULL_ZIP_SHA256:
            raise RuntimeError(f"Unverified EXP-014 extraction exists: {SOURCE_DIR}")
    else:
        SOURCE_DIR.mkdir(parents=True)
        wanted = set(required_relatives + optional_relatives)
        with zipfile.ZipFile(full_zip) as archive:
            for relative in wanted:
                member = f"{SOURCE_EXPERIMENT_ID}/{relative}"
                if member not in archive.namelist():
                    if relative in required_relatives:
                        raise FileNotFoundError(f"FULL ZIP lacks {member}")
                    continue
                safe = PurePosixPath(relative)
                if safe.is_absolute() or ".." in safe.parts:
                    raise ValueError(f"Unsafe archive member: {member}")
                target = SOURCE_DIR.joinpath(*safe.parts)
                if SOURCE_DIR.resolve() not in target.resolve().parents:
                    raise ValueError(f"Archive extraction escaped cache: {target}")
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
        marker.write_text(SOURCE_FULL_ZIP_SHA256 + "\n", encoding="utf-8")
    print("Using verified FULL ZIP extraction:", SOURCE_DIR)
missing = [relative for relative in required_relatives if not (SOURCE_DIR / relative).is_file()]
if missing:
    raise FileNotFoundError(f"Required EXP-014 source artifacts missing: {missing}")
''')

add("code", r'''# 04. Check exact recipe, Validation lock, dataset manifest and five checkpoints
config_path = SOURCE_DIR / "experiment_config.json"
lock_path = SOURCE_DIR / "validation_selection_lock.json"
experiment_config = json.loads(config_path.read_text(encoding="utf-8"))
source_lock = json.loads(lock_path.read_text(encoding="utf-8"))
source_status = json.loads((SOURCE_DIR / "execution_status.json").read_text(encoding="utf-8"))
if (source_status.get("five_seed_training") != "COMPLETED" or
        source_status.get("pipeline_smoke") != "PASS"):
    raise ValueError("EXP-014 five-seed training or pipeline smoke is not complete")
if experiment_config["experiment_id"] != SOURCE_EXPERIMENT_ID:
    raise ValueError("This is not the trained EXP-014 configuration")
if source_lock["experiment_id"] != SOURCE_EXPERIMENT_ID:
    raise ValueError("This is not the EXP-014 Validation lock")
if file_sha256(config_path) != source_lock["config_sha256"]:
    raise ValueError("EXP-014 configuration changed after Validation selection")
dataset = experiment_config["dataset"]
if (dataset["name"] != "Cashew_dataV08_real_val" or
        dataset["archive"] != "CashewData_Split (8).zip" or
        dataset["archive_sha256"] != EXPECTED_DATASET_SHA256 or
        dataset["manifest_sha256"] != EXPECTED_MANIFEST_SHA256 or
        dataset["counts"] != {"train": 4790, "val": 1430, "test": 686, "total": 6906} or
        dataset["real_filename_counts"] != {"train": 135, "val": 35, "test": 28} or
        dataset["class_names"] != CLASS_NAMES or experiment_config["seeds"] != SEEDS):
    raise ValueError("EXP-014 dataset, class order or seed list differs from the frozen recipe")
expected_model = {"backbone": "DenseNet121", "weights": "imagenet", "include_top": False,
                  "pooling": "avg", "head_dropout": HEAD_DROPOUT,
                  "preprocessing": "inside model: /255 then ImageNet mean/std",
                  "augmentation": {"horizontal_flip": True, "rotation_factor": 0.08,
                                   "zoom_factor": 0.10, "contrast_factor": 0.10,
                                   "brightness": False},
                  "backbone_call": "backbone(x, training=False)"}
if experiment_config["model"] != expected_model:
    raise ValueError("Model architecture/preprocessing differs from trained EXP-014")
expected_stage1 = {"lr": STAGE1_LR, "max_epochs": STAGE1_EPOCHS,
                   "early_patience": 4, "reduce_patience": 2,
                   "min_lr": 1e-6, "trainable_backbone": "none"}
expected_stage2 = {"lr": STAGE2_LR, "max_epochs": STAGE2_EPOCHS,
                   "early_patience": 8, "reduce_patience": 4,
                   "min_lr": 1e-7, "trainable_backbone": "conv5_* non-BatchNorm only"}
if (experiment_config["input"] != "raw RGB float32 0..255, bilinear resize 224x224" or
        experiment_config["batch_size"] != BATCH_SIZE or
        experiment_config["weight_decay"] != WEIGHT_DECAY or
        experiment_config["optimizer"] != "AdamW" or
        experiment_config["loss"] != "SparseCategoricalCrossentropy" or
        experiment_config["stage1"] != expected_stage1 or
        experiment_config["stage2"] != expected_stage2 or
        experiment_config["selection_rule"] != "lowest Validation loss, higher Validation macro-F1, Stage 1 on tie" or
        experiment_config["deployment_rule"] != "lowest Validation loss, higher macro-F1, lower seed on tie"):
    raise ValueError("EXP-014 training recipe differs from the locked 015 evaluator")
if (source_lock["dataset_sha256"] != EXPECTED_DATASET_SHA256 or
        source_lock["dataset_manifest_sha256"] != EXPECTED_MANIFEST_SHA256 or
        source_lock["class_names"] != CLASS_NAMES or source_lock["seeds"] != SEEDS):
    raise ValueError("Validation lock has a different dataset/class/seed identity")
validation_path = SOURCE_DIR / "aggregate/validation_results_5seeds.csv"
if file_sha256(validation_path) != source_lock["validation_sha256"]:
    raise ValueError("EXP-014 Validation results changed after selection")
validation_results_df = pd.read_csv(validation_path)
training_summary_df = pd.read_csv(SOURCE_DIR / "aggregate/training_summary_5seeds.csv")
source_environment_path = SOURCE_DIR / "environment/environment.json"
source_environment = (json.loads(source_environment_path.read_text(encoding="utf-8"))
                      if source_environment_path.is_file() else {})
if (validation_results_df["seed"].tolist() != SEEDS or
        training_summary_df["seed"].tolist() != SEEDS):
    raise ValueError("All five EXP-014 seeds must be complete and ordered")
ranked = validation_results_df.sort_values(["val_loss", "val_macro_f1", "seed"],
                                            ascending=[True, False, True])
deployment_seed = int(ranked.iloc[0]["seed"])
if deployment_seed != source_lock["deployment_seed"]:
    raise ValueError("EXP-014 deployment seed changed after the Validation lock")
manifest_df = pd.read_csv(SOURCE_DIR / "dataset_manifest.csv")
manifest_records = manifest_df[["relative_path", "label", "sha256"]].to_dict("records")
manifest_digest = hashlib.sha256(json.dumps(
    manifest_records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
if (manifest_digest != EXPECTED_MANIFEST_SHA256 or len(manifest_df) != 6906 or
        manifest_df["split"].value_counts().to_dict() != {"train": 4790, "val": 1430, "test": 686}):
    raise ValueError("EXP-014 Split (8) manifest failed the canonical check")
stage_selection = {}
for seed in SEEDS:
    checkpoint = SOURCE_DIR / f"seed_{seed}/best.weights.h5"
    actual = file_sha256(checkpoint)
    selected = json.loads((SOURCE_DIR / f"seed_{seed}/stage_selection.json").read_text(
        encoding="utf-8"))
    if actual != source_lock["selected_checkpoint_sha256"][str(seed)] or actual != selected["checkpoint_sha256"]:
        raise ValueError(f"EXP-014 checkpoint for seed {seed} changed after Validation")
    if selected["selected_stage"] not in ("stage1", "stage2"):
        raise ValueError(f"Invalid selected stage for seed {seed}")
    stage_selection[seed] = selected
eval_config = {
    "experiment_id": EXPERIMENT_ID, "type": "evaluation_only_of_locked_014_checkpoints",
    "source_experiment_id": SOURCE_EXPERIMENT_ID,
    "source_config_sha256": source_lock["config_sha256"],
    "source_validation_lock_sha256": file_sha256(lock_path),
    "source_validation_results_sha256": source_lock["validation_sha256"],
    "source_full_zip_sha256": SOURCE_FULL_ZIP_SHA256,
    "source_checkpoint_sha256": source_lock["selected_checkpoint_sha256"],
    "deployment_seed": deployment_seed,
    "holdout_archive_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,
    "run_full_training": False, "run_final_test": False, "run_real_holdout": True,
    "holdout_status": "historical_previously_exposed",
}
eval_config_path = OUT_DIR / "evaluation_config.json"
if eval_config_path.exists():
    if json.loads(eval_config_path.read_text(encoding="utf-8")) != eval_config:
        raise ValueError("Existing EXP-015 evaluation source/config differs")
else:
    save_json(eval_config_path, eval_config)
shutil.copyfile(config_path, OUT_DIR / "source_014_experiment_config.json")
shutil.copyfile(lock_path, OUT_DIR / "source_014_validation_selection_lock.json")
print("EXP-014 source integrity PASS; Validation-selected deployment seed:", deployment_seed)
display(validation_results_df)
''')

seed_source = "".join(SOURCE["cells"][5]["source"])
model_source = "".join(SOURCE["cells"][12]["source"]).split("if RUN_PIPELINE_SMOKE:", 1)[0]
decode_source = "".join(SOURCE["cells"][10]["source"]).split("def decode_image(path, label):", 1)[1]
decode_source = "def decode_image(path, label):" + decode_source.split("def make_dataset(", 1)[0]
add("markdown", """## Nạp lại đúng đồ thị DenseNet121 của 014

Đoạn dựng model, preprocessing và bộ đọc ảnh dưới đây được sao chép từ source
notebook `014`. `weights=None` chỉ dùng lúc dựng đồ thị; ngay sau đó notebook
nạp checkpoint đã train và xác minh SHA. Không khởi tạo mô hình mới để train.
""")
add("code", seed_source + "\n" + model_source + "\n" + decode_source + r'''
def load_selected_model(seed):
    selected = stage_selection[seed]
    model, backbone = build_transfer_model(seed, weights=None)
    set_backbone_stage(backbone, selected["selected_stage"])
    checkpoint = SOURCE_DIR / f"seed_{seed}/best.weights.h5"
    if file_sha256(checkpoint) != source_lock["selected_checkpoint_sha256"][str(seed)]:
        raise ValueError(f"Checkpoint seed {seed} changed before inference")
    model.load_weights(checkpoint)
    return model

# Fixed, nonrandom input checks output shape and optional exported-model parity.
keras.backend.clear_session()
probe_model = load_selected_model(deployment_seed)
if int(probe_model.count_params()) != int(training_summary_df.iloc[0]["total_parameters"]):
    raise RuntimeError("Reconstructed EXP-014 model has a different parameter count")
parity_input = tf.reshape(tf.linspace(0.0, 255.0, IMG_SIZE * IMG_SIZE * 3),
                          (1, IMG_SIZE, IMG_SIZE, 3))
parity_model = keras.Model((parity_raw := keras.Input((IMG_SIZE, IMG_SIZE, 3))),
                           densenet_preprocessing(parity_raw))
parity_error = float(tf.reduce_max(tf.abs(
    parity_model(parity_input) - keras.applications.densenet.preprocess_input(parity_input)
)).numpy())
if parity_error >= 1e-5:
    raise RuntimeError(f"DenseNet preprocessing differs from EXP-014: {parity_error}")
probe_input = tf.zeros((2, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32)
probe_prob = probe_model(probe_input, training=False).numpy()
if (probe_prob.shape != (2, NUM_CLASSES) or not np.isfinite(probe_prob).all() or
        not np.allclose(probe_prob.sum(axis=1), 1.0, atol=1e-5)):
    raise RuntimeError("Reconstructed EXP-014 model failed the inference smoke check")
if any(layer.trainable for layer in probe_model.get_layer("densenet121").layers
       if isinstance(layer, layers.BatchNormalization)):
    raise RuntimeError("A pretrained BatchNorm layer is trainable")
exported_model_path = SOURCE_DIR / "model_package/deployment_model.keras"
if exported_model_path.is_file():
    exported_model = keras.models.load_model(exported_model_path, compile=False)
    if not np.allclose(probe_prob, exported_model(probe_input, training=False).numpy(), atol=1e-5):
        raise RuntimeError("Reconstructed checkpoint differs from EXP-014 exported deployment model")
    del exported_model
del probe_model, probe_prob
keras.backend.clear_session(); gc.collect()
print("EXP-014 checkpoint model reload smoke PASS")
''')

holdout_helper = "".join(SOURCE["cells"][20]["source"]).split(
    "# 19. Guarded, exposed real-world comparison", 1)[0]
holdout_helper = holdout_helper.replace(
    "Historical external classification evaluation embedded in EXP-014.",
    "Historical external classification evaluation embedded in EXP-015.", 1)
add("markdown", """## Đánh giá real holdout đã biết trước

Đối chiếu SHA của holdout, kiểm tra ảnh trùng byte với manifest V08 rồi loại
đúng một ảnh Test trùng. Chấm cả năm checkpoint 014; seed triển khai đã khóa
trước khi chấm. Macro metrics chỉ tính bốn lớp có ảnh thật trong holdout.
""")
add("code", holdout_helper)
add("code", r'''# 07. Run real holdout (enabled)
if not RUN_REAL_HOLDOUT:
    raise RuntimeError("Real holdout must be enabled in EXP-015")
holdout_archive = resolve_holdout_archive(REAL_HOLDOUT_ARCHIVE_NAME,
                                           REAL_HOLDOUT_ARCHIVE_OVERRIDE)
holdout_extract_dir = CACHE_ROOT / "real_holdout" / REAL_HOLDOUT_ARCHIVE_SHA256[:12]
holdout_dir = OUT_DIR / "real_holdout"
reference_rows = manifest_df[["path", "relative_path", "split", "sha256"]].to_dict("records")
holdout_manifest, holdout_audit = prepare_real_holdout(
    holdout_archive, holdout_extract_dir, CLASS_NAMES,
    REAL_HOLDOUT_ARCHIVE_SHA256, REAL_HOLDOUT_EXPECTED_COUNTS,
    reference_rows, holdout_dir)
if len(holdout_manifest) != 48 or holdout_audit["excluded_overlap_images"] != 1:
    raise ValueError("Expected exactly 48 scored images after excluding one Test duplicate")

def make_holdout_dataset(manifest):
    dataset = tf.data.Dataset.from_tensor_slices((
        manifest["path"].tolist(), manifest["label"].to_numpy(dtype=np.int32)))
    return dataset.map(decode_image, num_parallel_calls=tf.data.AUTOTUNE).batch(
        BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

def clear_holdout_session():
    keras.backend.clear_session(); gc.collect()

checkpoint_paths = {seed: SOURCE_DIR / f"seed_{seed}/best.weights.h5" for seed in SEEDS}
holdout_results_df = evaluate_real_holdout(
    holdout_manifest, holdout_audit, CLASS_NAMES, SEEDS, validation_results_df,
    checkpoint_paths, load_selected_model, make_holdout_dataset,
    clear_holdout_session, holdout_dir, outputs_are_logits=False)
for seed in SEEDS:
    seed_dir = holdout_dir / f"seed_{seed}"
    predictions = pd.read_csv(seed_dir / "predictions.csv")
    predictions[~predictions["correct"]].to_csv(seed_dir / "error_cases.csv", index=False)
for normalized, suffix in ((False, ""), (True, "_normalized")):
    figure, axes = plt.subplots(1, len(SEEDS), figsize=(28, 5.5), facecolor="white")
    for column, seed in enumerate(SEEDS):
        matrix = pd.read_csv(holdout_dir / f"seed_{seed}" /
                             f"confusion_matrix{suffix}.csv", index_col=0).to_numpy()
        axis = axes[column]
        axis.imshow(matrix, cmap=REAL_HOLDOUT_CMAP, vmin=0,
                    vmax=1 if normalized else max(1, matrix.max()))
        axis.set_title(f"Seed {seed}")
        axis.set_xticks(range(NUM_CLASSES), CLASS_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(NUM_CLASSES))
        axis.set_yticklabels(CLASS_NAMES if column == 0 else [])
        for row in range(NUM_CLASSES):
            for col in range(NUM_CLASSES):
                label = ("N/A" if normalized and matrix[row].sum() == 0 and row == 4
                         else f"{matrix[row, col]:.2f}" if normalized else str(int(matrix[row, col])))
                axis.text(col, row, label, ha="center", va="center", color="black", fontsize=8)
        axis.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    figure.tight_layout()
    figure.savefig(holdout_dir / f"holdout_confusion_matrices{suffix}_5seeds_panel.png",
                   dpi=190, bbox_inches="tight")
    plt.close(figure)
save_json(holdout_dir / "interpretation.json", {
    "status": "HISTORICAL_PREVIOUSLY_EXPOSED", "source_training_run": SOURCE_EXPERIMENT_ID,
    "scored_images": 48, "excluded_exact_test_duplicate": 1,
    "class_count_with_true_support": 4,
    "selection_source": "EXP-014 Validation lock only",
    "warning": "This is a descriptive comparison, not a fresh independent field test."
})
display(holdout_results_df)
''')

add("code", r'''# 08. Write model report with observed EXP-014 training and EXP-015 holdout results
def mean_sd(values, percent=False, digits=2):
    values = pd.Series(values, dtype=float)
    scale = 100 if percent else 1
    return f"{scale*values.mean():.{digits}f} ± {scale*values.std(ddof=1):.{digits}f}"

real_validation_df = pd.read_csv(SOURCE_DIR / "aggregate/real_validation_by_seed.csv")
real_validation_df = real_validation_df[real_validation_df["group"] == "real_filename"]
per_seed_lines = []
for seed in SEEDS:
    val = validation_results_df.loc[validation_results_df["seed"] == seed].iloc[0]
    hold = holdout_results_df.loc[holdout_results_df["seed"] == seed].iloc[0]
    train = training_summary_df.loc[training_summary_df["seed"] == seed].iloc[0]
    real = real_validation_df.loc[real_validation_df["seed"] == seed].iloc[0]
    per_seed_lines.append(
        f"| {seed} | {val['selected_stage']} / {int(val['selected_epoch'])} | "
        f"{100*val['val_accuracy']:.2f}% | {100*val['val_macro_f1']:.2f}% | "
        f"{int(real['correct'])}/{int(real['total'])} | "
        f"{100*hold['accuracy']:.2f}% | {100*hold['macro_f1_present_classes']:.2f}% | "
        f"{train['training_time_seconds']:.1f} s |")
deployment_holdout = holdout_results_df.loc[
    holdout_results_df["seed"] == deployment_seed].iloc[0]
report = f"""# DenseNet121 EXP-014 model and EXP-015 real-holdout evaluation

## Identity and scope

- Training run: `{SOURCE_EXPERIMENT_ID}`; five seeds completed in the linked source run.
- Evaluation run: `{EXPERIMENT_ID}`; **no retraining or weight changes**.
- Source config SHA256: `{source_lock['config_sha256']}`.
- Source Validation lock: `{file_sha256(lock_path)}`; locked at `{source_lock.get('locked_at', 'unknown')}`.
- Deployment seed: **{deployment_seed}**, selected by lowest Validation loss before holdout.
- Holdout status: **historical, previously exposed**; one exact Test duplicate excluded.

## Architecture and preprocessing

`RGB float32 224×224 (0–255)` → Train-only horizontal flip / rotation 0.08 /
zoom 0.10 / contrast 0.10 → `/255` → ImageNet mean/std normalization →
DenseNet121 ImageNet backbone (`include_top=False`, global average pooling) →
Dropout 0.40 → Dense(5, softmax). Validation and inference disable augmentation.
Class order: {', '.join(CLASS_NAMES)}. Total parameters: **{int(training_summary_df.iloc[0]['total_parameters']):,}**.

## Dataset and training configuration

- V08 Split (8): Train **4,790**, Validation **1,430**, Test **686**; ZIP SHA256 `{EXPECTED_DATASET_SHA256}`; canonical manifest SHA256 `{EXPECTED_MANIFEST_SHA256}`.
- Seeds: {SEEDS}; batch 16; sparse categorical crossentropy; AdamW weight decay 1e-4.
- Stage 1: frozen backbone (0 trainable backbone parameters), LR 3e-4, max 12 epochs; EarlyStopping patience 4; ReduceLROnPlateau factor 0.5, patience 2, min LR 1e-6.
- Stage 2: only `conv5_*` non-BatchNorm plus head trainable (**{int(training_summary_df.iloc[0]['stage2_backbone_trainable_params']):,}** backbone parameters), fresh AdamW LR 1e-5, max 30 epochs; EarlyStopping patience 8; ReduceLROnPlateau factor 0.5, patience 4, min LR 1e-7. Backbone always called with `training=False`.
- Stage best epoch: minimum Validation loss; stage tie: higher Validation macro F1, then Stage 1. Deployment seed tie: higher macro F1, then lower seed. Test and holdout did not select weights.
- Environment recorded by EXP-014: Python {source_environment.get('python', 'unknown')}, TensorFlow {source_environment.get('tensorflow', 'unknown')}, Keras {source_environment.get('keras', 'unknown')}, {source_environment.get('gpu_count', 'unknown')} GPU. EXP-015 runtime TensorFlow: {tf.__version__}.

## Measured results

| Seed | Selected stage / epoch | Val accuracy | Val macro F1 | Val `real_` correct | Holdout accuracy | Holdout macro F1 (4 classes) | Train time |
|---:|:---:|---:|---:|:---:|---:|---:|---:|
{chr(10).join(per_seed_lines)}

- Validation accuracy: **{mean_sd(validation_results_df['val_accuracy'], percent=True)}%** (mean ± sample SD, five seeds).
- Validation macro F1: **{mean_sd(validation_results_df['val_macro_f1'], percent=True)}%**.
- Validation `real_` filename group: **{mean_sd(real_validation_df['accuracy'], percent=True)}%** over 35 images per seed; the filename is only a proxy for field origin.
- Anthracnose recall: **{mean_sd(validation_results_df['anthracnose_recall'], percent=True)}%**; leaf miner recall: **{mean_sd(validation_results_df['leaf_miner_recall'], percent=True)}%**.
- False-healthy rate on diseased Validation leaves: **{mean_sd(validation_results_df['false_healthy_rate'], percent=True)}%**, denominator 900 per seed.
- Stage 1 training: **{mean_sd(training_summary_df['stage1_training_seconds'])} s**; Stage 2: **{mean_sd(training_summary_df['stage2_training_seconds'])} s**; total: **{mean_sd(training_summary_df['training_time_seconds'])} s** per seed.
- Validation inference including decode/resize: **{mean_sd(validation_results_df['inference_time_ms_per_image'])} ms/image**. Selected checkpoint size: **{mean_sd(training_summary_df['selected_checkpoint_size_mb'])} MiB**.
- Historical holdout accuracy: **{mean_sd(holdout_results_df['accuracy'], percent=True)}%**; macro F1 on four supported true classes: **{mean_sd(holdout_results_df['macro_f1_present_classes'], percent=True)}%** (five seeds, sample SD).
- Validation-selected deployment seed {deployment_seed}: holdout **{int(round(deployment_holdout['accuracy']*48))}/48 = {100*deployment_holdout['accuracy']:.2f}%** accuracy.

## Interpretation limits

The 49-image archive has one exact Test duplicate, which was removed before scoring. The remaining 48 images cover four true classes and contain no true `not_cashew_leaf` examples. The archive was previously inspected during model design, so this result is **descriptive**, not a new blind estimate of field performance. Seed SD is variability across training runs, not a confidence interval over future photos. A new independently collected field set is needed for a generalization claim.

## Files

- Per-seed predictions, confusion matrices, classification reports and error cases: `real_holdout/seed_*/`.
- Aggregate five-seed CSV and mean/SD: `real_holdout/aggregate/`.
- Overlap audit and frozen evaluation lock: `real_holdout/dataset_audit.json` and `real_holdout/evaluation_lock.json`.
"""
report_path = OUT_DIR / f"{EXPERIMENT_ID}_MODEL_REPORT.md"
report_path.write_text(report, encoding="utf-8")
save_json(OUT_DIR / "execution_status.json", {
    "experiment_id": EXPERIMENT_ID, "source_training_run": SOURCE_EXPERIMENT_ID,
    "training_in_015": "NOT_RUN_REUSED_014_CHECKPOINTS",
    "real_holdout": "COMPLETED_HISTORICAL_EXPOSED",
    "final_test": "NOT_RUN", "deployment_seed": deployment_seed,
    "holdout_images_scored": 48,
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
})
analysis_zip = RESULTS_ROOT / f"{EXPERIMENT_ID}_ANALYSIS.zip"
with zipfile.ZipFile(analysis_zip.with_suffix(".zip.tmp"), "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in OUT_DIR.rglob("*"):
        if path.is_file():
            archive.write(path, arcname=f"{EXPERIMENT_ID}/{path.relative_to(OUT_DIR).as_posix()}")
with zipfile.ZipFile(analysis_zip.with_suffix(".zip.tmp")) as archive:
    if archive.testzip() is not None or f"{EXPERIMENT_ID}/{report_path.name}" not in archive.namelist():
        raise RuntimeError("EXP-015 ANALYSIS ZIP integrity check failed")
analysis_zip.with_suffix(".zip.tmp").replace(analysis_zip)
print("Report:", report_path)
print("Analysis ZIP:", analysis_zip)
display(holdout_results_df)
''')

add("markdown", """## Hoàn tất

`015` chỉ hoàn tất khi kết quả holdout, overlap audit, báo cáo và ANALYSIS ZIP
được tạo sau khi nạp đủ năm checkpoint `014`. Không dùng số holdout này để
đổi seed hoặc train lại trong cùng thí nghiệm.
""")

notebook = {"cells": CELLS, "metadata": copy.deepcopy(SOURCE["metadata"]),
            "nbformat": SOURCE["nbformat"], "nbformat_minor": SOURCE["nbformat_minor"]}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Created", TARGET)
