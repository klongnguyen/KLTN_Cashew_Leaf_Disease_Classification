"""Build the standalone five-seed DeiT-tiny transfer notebook for V08.

The 006 DeiT implementation supplies the tested KerasHub model and evaluator.
This generator changes the dataset, protocol and output namespace without
editing any previous experiment or fabricating measured results.
"""

import base64
import copy
import json
import re
import sys
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "deit_tiny_finetune_5seed.ipynb"
TARGET = HERE / "vit_transfer_5seed_007.ipynb"
if TARGET.exists():
    if "--replace-existing-007" not in sys.argv:
        raise FileExistsError(f"Review existing 007 before replacing: {TARGET}")
    existing = json.loads(TARGET.read_text(encoding="utf-8"))
    if ("EXP-VIT-TRANSFER-5SEEDS-007" not in "".join(existing["cells"][0]["source"])
            or any(cell.get("outputs") or cell.get("execution_count") is not None
                   for cell in existing["cells"][7:] if cell["cell_type"] == "code")):
        raise ValueError("Refusing to replace a different or trained notebook")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]


def get(index):
    return "".join(cells[index]["source"])


def put(index, source):
    cells[index]["source"] = source.splitlines(keepends=True)


def replace(index, old, new, count=1):
    source = get(index)
    if source.count(old) != count:
        raise ValueError(f"Cell {index}: expected {count} occurrences of {old!r}, found {source.count(old)}")
    put(index, source.replace(old, new))


for index, cell in enumerate(cells):
    cell["id"] = f"vit007_cell_{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
    source = "".join(cell["source"])
    source = source.replace("EXP-DEIT-TINY-FINETUNE-5SEEDS-006", "EXP-VIT-TRANSFER-5SEEDS-007")
    source = source.replace("Cashew_dataV05", "Cashew_dataV08_real_val")
    source = source.replace("CashewData_Split (5).zip", "CashewData_Split (8).zip")
    source = source.replace("V05", "V08").replace("v05", "v08")
    cell["source"] = source.splitlines(keepends=True)

put(0, """# EXP-VIT-TRANSFER-5SEEDS-007

**NOT EXECUTED.** This independent DeiT-tiny ImageNet transfer experiment uses
the frozen CashewData_Split (8).zip, five fixed seeds, a frozen-backbone head
stage followed by low-rate fine-tuning, and Validation-only checkpoint and
deployment-seed selection. Final Test and the historical real holdout run only
after the five Validation checkpoints are locked. One exact Test duplicate is
excluded from holdout scoring after an explicit byte audit.

The KerasHub preset preprocessor is included exactly once in the saved model:
input is RGB float32 in [0, 255]. Train-only geometric and contrast
augmentation is disabled for Validation, Test and holdout inference.
Training/Validation curves include a measured pre-fit Validation point; no
accuracy is lowered or smoothed for presentation. Confusion matrices use a
white-to-blue scale. Both ZIP exports contain this exact notebook.
""")
put(1, '''# In Colab, run once before imports. Drive API supports shared ZIP fallback.
%pip install -q "keras-hub==0.32.0" scikit-learn pandas matplotlib pillow google-api-python-client
''')
put(3, """## 1. Frozen configuration

Version 007 is separate from the earlier V05 DeiT 006 and V08 scratch runs.
The V08 archive SHA-256, five class counts and five seeds are fixed before
training. Change any protocol value only in a new experiment ID.
""")
put(4, '''NOTEBOOK_VERSION = "007"
NOTEBOOK_FILENAME = "vit_transfer_5seed_007.ipynb"
EXPERIMENT_ID = "EXP-VIT-TRANSFER-5SEEDS-007"
DATASET_VERSION = "Cashew_dataV08_real_val"
DATASET_ARCHIVE_NAME = "CashewData_Split (8).zip"
DATASET_ARCHIVE_SHA256 = "75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7"
DATASET_ARCHIVE_SIZE = 1058695554
DATASET_DRIVE_FILE_ID = "1vaaOwAi72byaxFb7w_ixd_Y2HboKn2-G"
DATASET_ARCHIVE_OVERRIDE = None  # Optional exact mounted ZIP path, e.g. /content/drive/MyDrive/...zip
PRESET = "kaggle://keras/deit/keras/deit_tiny_distilled_patch16_224_imagenet/1"
CLASS_NAMES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
EXPECTED_COUNTS = {
    "train": [939, 803, 875, 1072, 1101],
    "val": [308, 216, 264, 328, 314],
    "test": [122, 118, 132, 158, 156],
}
EXPECTED_TOTALS = {"train": 4790, "val": 1430, "test": 686, "all": 6906}
REQUIRED_ABSENT_IMAGES = [
    "val/leaf_miner/leaf_miner_1056.jpg",
    "val/leaf_miner/leaf_miner_1091.jpg",
    "test/not_cashew_leaf/not_cashew_leaf_1514.JPG",
]
SEEDS = [42, 123, 2026, 3407, 7777]
IMG_SIZE = 224
BATCH_SIZE = 16
HEAD_EPOCHS = 8
FINETUNE_EPOCHS = 12
HEAD_LR = 1e-4
BACKBONE_LR = 5e-6
MIN_LR = 5e-7
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 5
RUN_PIPELINE_SMOKE = True
RUN_TRAINING = True
RUN_FINAL_TEST = True
RUN_REAL_HOLDOUT = True  # Historical, previously inspected 48-image set.
REAL_HOLDOUT_ARCHIVE_NAME = "RL_Cashew_holdout.zip"
REAL_HOLDOUT_ARCHIVE_SHA256 = "BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165"
REAL_HOLDOUT_ARCHIVE_SIZE = 150455582
REAL_HOLDOUT_DRIVE_FILE_ID = "1PBry_y2bMsvd5_YdmEUy9erSFarS3O52"
REAL_HOLDOUT_EXPECTED_COUNTS = {
    "anthracnose": 13, "healthy": 5, "leaf_miner": 24, "red_rust": 7
}
REAL_HOLDOUT_ARCHIVE_OVERRIDE = None
''')
put(5, """## 2. Archive and split audit

The notebook pins the V08 ZIP hash, exact class counts, and absent removed
files. Test labels are audited for integrity but never used during fitting or
Validation selection. Historical holdout overlap is checked before training.
If the ZIPs are elsewhere in mounted Drive, the notebook searches by exact
filename. If the shared folder is not mounted, it downloads by the pinned
Drive file IDs from the supplied Dataset folder. Both paths require SHA-256.
An exact path can be set with the two `*_OVERRIDE` settings above.
""")

# Dataset paths and class-order handling. V08 deliberately uses the same class
# mapping as ResNet50/DenseNet121, which is not alphabetical.
replace(6, 'results_root = project_root / "training_results" / "cashew_datav08"',
        'results_root = project_root / "training_results" / "current_dataset"', count=2)
replace(6, '''if not archive_path.is_file():
    raise FileNotFoundError(archive_path)

''', '')
replace(6, 'actual_archive_hash = sha256_file(archive_path)', '''def download_pinned_drive_zip(file_id, destination, expected_name,
                              expected_sha256, expected_size):
    """Fetch a shared Drive ZIP by ID, stream to disk, then verify pinned bytes."""
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload

    destination = Path(destination)
    if destination.is_file():
        if (destination.stat().st_size == expected_size
                and sha256_file(destination) == expected_sha256):
            return destination
        raise ValueError(f"Existing Drive download has unexpected bytes: {destination}")

    auth.authenticate_user()  # Select the account that can open the shared folder.
    credentials, _ = google.auth.default()
    service = build("drive", "v3", credentials=credentials, cache_discovery=False)
    metadata = service.files().get(
        fileId=file_id, fields="id,name,size,capabilities(canDownload)",
        supportsAllDrives=True).execute()
    if (metadata.get("name") != expected_name
            or int(metadata.get("size", -1)) != expected_size
            or metadata.get("capabilities", {}).get("canDownload") is False):
        raise ValueError(f"Unexpected or inaccessible Drive file metadata: {metadata}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".partial")
    try:
        with partial.open("wb") as stream:
            request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
            downloader = MediaIoBaseDownload(stream, request, chunksize=16 * 1024 * 1024)
            done = False
            while not done:
                _, done = downloader.next_chunk(num_retries=3)
        if partial.stat().st_size != expected_size:
            raise ValueError(f"Incomplete Drive download: {partial.stat().st_size} bytes")
        digest = sha256_file(partial)
        if digest != expected_sha256:
            raise ValueError(f"Drive download SHA-256 mismatch: {digest}")
        partial.replace(destination)
        return destination
    finally:
        partial.unlink(missing_ok=True)


def resolve_dataset_archive(preferred_path, search_roots=None):
    """Locate V08 by exact path or filename, then accept only its pinned bytes."""
    checked = set()
    wrong_hashes = []

    def verify(candidate):
        candidate = Path(candidate).expanduser()
        if not candidate.is_file():
            return None
        identity = str(candidate.resolve())
        if identity in checked:
            return None
        checked.add(identity)
        digest = sha256_file(candidate)
        if digest == DATASET_ARCHIVE_SHA256:
            return candidate, digest
        wrong_hashes.append(f"{candidate} (SHA-256 {digest})")
        return None

    if DATASET_ARCHIVE_OVERRIDE is not None:
        override = Path(DATASET_ARCHIVE_OVERRIDE).expanduser()
        if not override.is_file():
            raise FileNotFoundError(
                f"DATASET_ARCHIVE_OVERRIDE does not exist: {override}. "
                "Use the exact ZIP path shown in the Colab Files panel after Drive mount."
            )
        match = verify(override)
        if match:
            return match
    else:
        match = verify(preferred_path)
        if match:
            return match
        drive_roots = ([Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]
                       if search_roots is None else [Path(root) for root in search_roots])
        for candidate in (
            Path("/content") / DATASET_ARCHIVE_NAME,
            *(root / DATASET_ARCHIVE_NAME for root in drive_roots),
            *(root / "Dataset" / DATASET_ARCHIVE_NAME for root in drive_roots),
        ):
            match = verify(candidate)
            if match:
                return match
        for root in drive_roots:
            if root.is_dir():
                print(f"Searching {root} for {DATASET_ARCHIVE_NAME} ...", flush=True)
                for candidate in root.rglob(DATASET_ARCHIVE_NAME):
                    match = verify(candidate)
                    if match:
                        return match
        if search_roots is None and Path("/content").is_dir():
            print("V08 ZIP is not mounted; downloading the pinned Drive file ID.", flush=True)
            downloaded = download_pinned_drive_zip(
                DATASET_DRIVE_FILE_ID,
                Path("/content/verified_drive_archives") / DATASET_ARCHIVE_NAME,
                DATASET_ARCHIVE_NAME, DATASET_ARCHIVE_SHA256, DATASET_ARCHIVE_SIZE)
            match = verify(downloaded)
            if match:
                return match

    if wrong_hashes:
        raise ValueError(
            "Found ZIP(s) named like V08, but none match the pinned SHA-256 "
            f"{DATASET_ARCHIVE_SHA256}: " + "; ".join(wrong_hashes)
        )
    raise FileNotFoundError(
        f"Cannot find {DATASET_ARCHIVE_NAME}. Mount the Drive account containing "
        "the ZIP, or set DATASET_ARCHIVE_OVERRIDE in the configuration cell "
        "to its exact /content/drive/MyDrive/... path."
    )


archive_path, actual_archive_hash = resolve_dataset_archive(archive_path)
print("Verified V08 dataset ZIP:", archive_path)
''')
replace(6, 'if found != CLASS_NAMES:', 'if set(found) != set(CLASS_NAMES):')
replace(6, 'for folder in (out_dir, aggregate_dir, package_dir):',
        'for folder in (out_dir, aggregate_dir, package_dir):')
replace(6, 'if (split_overlap > 1).any():',
        '''if len(manifest) != EXPECTED_TOTALS["all"]:
    raise ValueError(f"Unexpected total image count: {len(manifest)}")
for relative in REQUIRED_ABSENT_IMAGES:
    if (dataset_root / relative).exists():
        raise ValueError(f"Removed V08 image reappeared: {relative}")
real_counts = {
    split: int(manifest.loc[manifest["split"] == split, "relative_path"]
               .map(lambda p: Path(p).name.startswith("real_")).sum())
    for split in ("train", "val", "test")
}
if real_counts != {"train": 135, "val": 35, "test": 28}:
    raise ValueError(f"Unexpected real_ file counts: {real_counts}")
if (split_overlap > 1).any():''')
replace(6, '"test_locked_by_default": True,', '"test_after_validation_lock": True,')
replace(6, '"training_mode": "ImageNet pretrained, head then full fine-tune",',
        '"training_mode": "ImageNet pretrained, frozen head then low-rate full fine-tune",')
replace(6, '"separate_from_v08_test": True,', '"separate_from_v08_test": True,')
replace(6, '"macro_scope": "true classes with holdout support; currently four",',
        '"macro_scope": "four classes with true holdout support; one exact Test duplicate excluded",')
replace(6, '"dataset_archive_sha256": DATASET_ARCHIVE_SHA256,',
        '"dataset_archive_sha256": DATASET_ARCHIVE_SHA256,\n    "dataset_drive_file_id": DATASET_DRIVE_FILE_ID,')
replace(6, '"archive_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,',
        '"archive_sha256": REAL_HOLDOUT_ARCHIVE_SHA256,\n        "drive_file_id": REAL_HOLDOUT_DRIVE_FILE_ID,')

# Reuse the 006 evaluator while explicitly excluding only the established
# holdout/Test byte duplicate. Any new overlap stops the run.
helper = get(8)
helper = helper.replace('REAL_HOLDOUT_HELPER_VERSION = "006.1"',
                        'REAL_HOLDOUT_HELPER_VERSION = "007.1"')
resolver_start = helper.index('def resolve_holdout_archive(')
resolver_end = helper.index('\n\ndef prepare_real_holdout(', resolver_start)
helper = helper[:resolver_start] + '''def resolve_holdout_archive(archive_name, override=None):
    if override is not None:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise FileNotFoundError(f"REAL_HOLDOUT_ARCHIVE_OVERRIDE does not exist: {candidate}")
        return candidate  # prepare_real_holdout verifies its pinned SHA-256.

    drive_roots = [Path("/content/drive/MyDrive"), Path("/content/drive/Shareddrives")]
    common = [Path("/content") / archive_name]
    for root in drive_roots:
        common.extend([root / "Cashew_Leaf_model_result" / "Dataset" / archive_name,
                       root / "Dataset" / archive_name, root / archive_name])
    if os.name == "nt":
        common.append(Path(r"D:\\Study\\KhoaLuanTotNghiep\\Dataset") / archive_name)
    if Path("/kaggle/input").is_dir():
        common.extend(sorted(Path("/kaggle/input").rglob(archive_name)))

    checked = set()
    wrong_hashes = []
    def verify(candidate):
        if not candidate.is_file():
            return None
        identity = str(candidate.resolve())
        if identity in checked:
            return None
        checked.add(identity)
        digest = holdout_sha256(candidate)
        if digest == REAL_HOLDOUT_ARCHIVE_SHA256:
            return candidate
        wrong_hashes.append(f"{candidate} (SHA-256 {digest})")
        return None

    for candidate in common:
        match = verify(candidate)
        if match:
            return match
    for root in drive_roots:
        if root.is_dir():
            print(f"Searching {root} for {archive_name} ...", flush=True)
            for candidate in root.rglob(archive_name):
                match = verify(candidate)
                if match:
                    return match
    if Path("/content").is_dir():
        print("Holdout ZIP is not mounted; downloading the pinned Drive file ID.", flush=True)
        downloaded = download_pinned_drive_zip(
            REAL_HOLDOUT_DRIVE_FILE_ID,
            Path("/content/verified_drive_archives") / archive_name,
            archive_name, REAL_HOLDOUT_ARCHIVE_SHA256, REAL_HOLDOUT_ARCHIVE_SIZE)
        match = verify(downloaded)
        if match:
            return match
    if wrong_hashes:
        raise ValueError("Holdout ZIP name found but SHA-256 differs from pinned archive: "
                         + "; ".join(wrong_hashes))
    raise FileNotFoundError(
        f"Cannot find {archive_name}. Set REAL_HOLDOUT_ARCHIVE_OVERRIDE in the "
        "configuration cell to its exact mounted ZIP path."
    )
''' + helper[resolver_end:]
helper = helper.replace('["#ffffff", "#dbeafe", "#93c5fd"]',
                        '["#ffffff", "#eaf4ff", "#bfdbfe", "#60a5fa", "#1d4ed8"]')
helper = helper.replace(
    '\n\ndef holdout_sha256(path):',
    '''\n\ndef holdout_matrix_text_color(value, upper):
    return "#ffffff" if float(value) / max(float(upper), 1.0) >= 0.68 else "#0f172a"

def holdout_sha256(path):''', 1)
start = helper.index('    holdout_hashes = set(manifest["sha256"])')
end = helper.index('\n\ndef save_real_holdout_seed(', start)
helper = helper[:start] + '''    holdout_by_hash = dict(zip(manifest["sha256"], manifest["relative_path"]))
    overlaps = []
    reference_digest = hashlib.sha256()
    for row in references:
        digest = (row.get("sha256") or holdout_sha256(row["path"])).upper()
        identity = row.get("relative_path", str(row["path"]))
        reference_digest.update(f"{row['split']}\\\\0{identity}\\\\0{digest}\\\\n".encode("utf-8"))
        if digest in holdout_by_hash:
            overlaps.append({
                "reference_path": str(row["path"]), "split": row["split"],
                "holdout_relative_path": holdout_by_hash[digest], "sha256": digest
            })
    expected_overlap = {("test", "red_rust/IMG_3244.jpg")}
    observed_overlap = {(row["split"], row["holdout_relative_path"]) for row in overlaps}
    if observed_overlap != expected_overlap or len(overlaps) != 1:
        raise ValueError(f"Unexpected holdout/V08 byte overlap: {overlaps}")
    excluded_paths = {row["holdout_relative_path"] for row in overlaps}
    scored = manifest.loc[~manifest["relative_path"].isin(excluded_paths)].reset_index(drop=True)
    if len(scored) != 48:
        raise ValueError(f"Expected 48 non-overlapping holdout images, found {len(scored)}")
    pd.DataFrame(overlaps).to_csv(audit_dir / "excluded_overlap_images.csv", index=False)
    present = [name for name in class_names if (scored["class"] == name).any()]
    audit = {
        "archive_sha256": actual_hash, "archive_name": archive_path.name,
        "manifest_sha256": hashlib.sha256(
            manifest[["relative_path", "class", "label", "sha256"]].to_csv(index=False).encode()
        ).hexdigest(),
        "source_images": len(manifest), "evaluated_images": len(scored),
        "counts": {name: expected_counts.get(name, 0) for name in class_names},
        "present_classes": present,
        "unsupported_classes": [name for name in class_names if name not in present],
        "reference_images_checked": len(references),
        "reference_fingerprint": reference_digest.hexdigest(),
        "byte_overlap_count": len(overlaps), "byte_overlaps": overlaps,
        "limitation": "Exact bytes only; source-group and near-duplicate independence are not established.",
    }
    (audit_dir / "dataset_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("Real holdout:", len(scored), "scored images; one exact Test duplicate excluded")
    return scored, audit
''' + helper[end:]
helper = helper.replace(
    'ax.text(col, row, text, ha="center", va="center", color="black")',
    'ax.text(col, row, text, ha="center", va="center", '
    'color=holdout_matrix_text_color(values[row, col], '
    '1 if normalized else matrix.max()))')
helper = helper.replace('real_holdout_006', 'real_holdout_007')
helper = helper.replace('cashew_real_holdout_006', 'cashew_real_holdout_007')
put(8, helper)
put(7, """## Historical real holdout preflight

The pinned 49-image archive contains one image byte-identical to V08 Test.
The preflight records and excludes precisely that image, then evaluates the
remaining 48. Any additional overlap aborts the experiment. This previously
inspected set is diagnostic evidence, not a new blind field test.
""")
put(9, """## 3. DeiT-tiny input and model

Images remain RGB float32 [0,255]. The KerasHub preset preprocessor is
called once inside the saved model. Augmentation runs in training mode only.
""")

setup = get(2).replace(
    '["#ffffff", "#dbeafe", "#93c5fd"]',
    '["#ffffff", "#eaf4ff", "#bfdbfe", "#60a5fa", "#1d4ed8"]')
setup += '''

def matrix_text_color(value, upper):
    return "#ffffff" if float(value) / max(float(upper), 1.0) >= 0.68 else "#0f172a"
'''
put(2, setup)

model = get(10)
model += '''

# Measures image read, decode, resize and tf.data delivery, without model
# inference. This single-pass timing is never used for model selection.
pipeline_started = time.perf_counter()
pipeline_images = 0
for batch_images, _ in make_dataset("val", SEEDS[0], shuffle=False):
    pipeline_images += int(tf.shape(batch_images)[0])
pipeline_seconds = time.perf_counter() - pipeline_started
if pipeline_images != EXPECTED_TOTALS["val"]:
    raise AssertionError("Input pipeline image count changed")
(out_dir / "image_processing_time.json").write_text(json.dumps({
    "images": pipeline_images,
    "seconds": pipeline_seconds,
    "ms_per_image": 1000 * pipeline_seconds / pipeline_images,
    "scope": "single-pass tf.data read, decode, resize and batch delivery; no model inference",
}, indent=2), encoding="utf-8")

if RUN_PIPELINE_SMOKE:
    import tempfile
    set_seed(SEEDS[0])
    smoke_model, smoke_classifier = build_model(SEEDS[0])
    probe = tf.random.stateless_uniform((2, IMG_SIZE, IMG_SIZE, 3),
                                        seed=(SEEDS[0], 7), minval=0, maxval=255)
    direct = np.asarray(smoke_model(probe, training=False))
    manual = np.asarray(smoke_classifier(
        smoke_classifier.preprocessor(probe), training=False))
    if direct.shape != (2, len(CLASS_NAMES)) or not np.isfinite(direct).all():
        raise AssertionError("DeiT pipeline output is invalid")
    if not np.allclose(direct, manual, rtol=1e-5, atol=1e-5):
        raise AssertionError("Nested DeiT preprocessing differs from one explicit pass")
    smoke_classifier.backbone.trainable = False
    compile_model(smoke_model, HEAD_LR)
    if smoke_classifier.backbone.trainable_weights:
        raise AssertionError("Stage 1 backbone is not frozen")
    smoke_classifier.backbone.trainable = True
    compile_model(smoke_model, BACKBONE_LR)
    if not smoke_classifier.backbone.trainable_weights:
        raise AssertionError("Stage 2 backbone is not trainable")
    with tempfile.TemporaryDirectory() as tmp:
        package = Path(tmp) / "pipeline.keras"
        smoke_model.save(package)
        loaded = keras.models.load_model(package, compile=False, safe_mode=True)
        if not np.allclose(direct, np.asarray(loaded(probe, training=False)),
                           rtol=1e-4, atol=1e-4):
            raise AssertionError("Saved DeiT package changed inference")
    (out_dir / "pipeline_smoke.json").write_text(json.dumps({
        "status": "PASS", "preprocessing": "single preset pass",
        "input": "RGB float32 0..255", "stage1_frozen": True,
        "stage2_trainable": True, "saved_model_reload": True,
    }, indent=2), encoding="utf-8")
    del smoke_model, smoke_classifier, loaded
    keras.backend.clear_session()
    gc.collect()
'''
put(10, model)
put(11, """## 4. Five-seed transfer training

Stage 1 trains the new head with the ImageNet backbone frozen. Stage 2
fine-tunes at a much lower learning rate. Both stage checkpoints are ranked
only by Validation loss. The deployment seed is chosen only from Validation
macro F1, then Validation loss. Training and preprocessing times are saved.
""")

training = get(12)
training = training.replace(
    '        classifier.backbone.trainable = False\n        compile_model(model, HEAD_LR)',
    '''        # True pre-fit score: random five-class head, before any update.
        pre_fit_logits = model.predict(val_ds, verbose=0)
        pre_fit_metrics, _ = prediction_metrics(labels["val"], pre_fit_logits)
        (seed_dir / "pre_fit_validation.json").write_text(json.dumps(
            {"seed": seed, **pre_fit_metrics, "images": len(labels["val"])},
            indent=2), encoding="utf-8")
        classifier.backbone.trainable = False
        compile_model(model, HEAD_LR)''')
training = training.replace(
    '        val_logits = model.predict(val_ds, verbose=0)',
    '''        inference_started = time.perf_counter()
        val_logits = model.predict(val_ds, verbose=0)
        val_inference_ms_per_image = 1000 * (time.perf_counter() - inference_started) / len(labels["val"])''')
training = training.replace(
    '        model.save(model_path)',
    '''        model.save(model_path)
        reloaded = keras.models.load_model(model_path, compile=False, safe_mode=True)
        sample_images = next(iter(val_ds))[0]
        if not np.allclose(np.asarray(model(sample_images, training=False)),
                           np.asarray(reloaded(sample_images, training=False)),
                           rtol=1e-4, atol=1e-4):
            raise AssertionError("Saved model changed Validation inference")
        del reloaded''')
training = training.replace(
    '            "training_seconds": float(head_training_seconds + finetune_training_seconds),',
    '''            "training_seconds": float(head_training_seconds + finetune_training_seconds),
            "val_inference_ms_per_image": float(val_inference_ms_per_image),''')
training = training.replace(
    '    shutil.copy2(source_model, deployment_model)',
    '    shutil.copy2(source_model, deployment_model)\n'
    'elif sha256_file(source_model) != sha256_file(deployment_model):\n'
    '    raise ValueError("Existing deployment model differs from Validation-selected checkpoint")')
training += '''

# Immutable Validation lock is written before Final Test or holdout inference.
selection_lock = {
    "experiment_id": EXPERIMENT_ID, "dataset_archive_sha256": DATASET_ARCHIVE_SHA256,
    "dataset_manifest_sha256": manifest_hash, "seeds": SEEDS,
    "deployment_seed": deployment_seed,
    "checkpoint_sha256": {
        str(seed): sha256_file(out_dir / f"seed_{seed}" / "best_model.keras")
        for seed in SEEDS
    },
    "validation_sha256": hashlib.sha256(
        validation_df.to_csv(index=False, float_format="%.12g").encode()
    ).hexdigest(),
    "selection_rule": "Validation macro F1, Validation loss tie-break",
}
lock_path = out_dir / "validation_selection_lock.json"
if lock_path.exists():
    if json.loads(lock_path.read_text(encoding="utf-8")) != selection_lock:
        raise ValueError("Validation checkpoint lock changed")
else:
    lock_path.write_text(json.dumps(selection_lock, indent=2), encoding="utf-8")
'''
put(12, training)

figures = get(14)
figures = figures.replace(
    '    hist = history_df.loc[history_df["seed"] == seed]',
    '''    hist = history_df.loc[history_df["seed"] == seed]
    pre_fit = json.loads((out_dir / f"seed_{seed}" / "pre_fit_validation.json").read_text())
    axes[0, col].plot([0, 1], [pre_fit["accuracy"], hist["val_accuracy"].iloc[0]],
                      color="#ea580c", marker="o", label="full Val from pre-fit")
    axes[0, col].axvline(1, color="#64748b", linestyle=":", alpha=0.4)''')
figures = figures.replace(
    'axes[col].text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", fontsize=7, color="black")',
    'axes[col].text(j, i, f"{normalized[i, j]:.2f}", ha="center", va="center", '
    'fontsize=7, color=matrix_text_color(normalized[i, j], 1))')
figures += '''

# Per-seed raw and normalized white-blue matrices.
for seed in SEEDS:
    cm = pd.read_csv(out_dir / f"seed_{seed}" / "validation_confusion_matrix.csv",
                     index_col=0).to_numpy()
    for normalized, suffix in ((False, ""), (True, "_normalized")):
        values = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalized else cm
        fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
        ax.imshow(values, cmap=MATRIX_CMAP, vmin=0,
                  vmax=1 if normalized else max(1, int(cm.max())))
        ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"Validation seed {seed}")
        for row in range(len(CLASS_NAMES)):
            for col in range(len(CLASS_NAMES)):
                ax.text(col, row, f"{values[row,col]:.2f}" if normalized else str(int(values[row,col])),
                        ha="center", va="center",
                        color=matrix_text_color(values[row,col], 1 if normalized else cm.max()))
        fig.tight_layout()
        fig.savefig(out_dir / f"seed_{seed}" / f"validation_confusion_matrix{suffix}.png", dpi=180)
        plt.close(fig)
'''
put(14, figures)
put(15, """## 5. Locked Final Test

The five Validation-selected model files and deployment seed are frozen in
validation_selection_lock.json before any Test prediction. Test results cannot
select or change a checkpoint, seed or hyperparameter.
""")

test = get(16)
test = test.replace(
    'if RUN_FINAL_TEST:\n    test_rows = []',
    '''if RUN_FINAL_TEST:
    if json.loads(lock_path.read_text(encoding="utf-8")) != selection_lock:
        raise ValueError("Validation lock changed before Final Test")
    if (aggregate_dir / "test_results_5seeds.csv").exists():
        raise FileExistsError("Final Test already recorded; refusing to overwrite")
    test_rows = []''')
test = test.replace(
    '        logits = model.predict(test_ds, verbose=0)',
    '''        inference_started = time.perf_counter()
        logits = model.predict(test_ds, verbose=0)
        test_inference_ms_per_image = 1000 * (time.perf_counter() - inference_started) / len(labels["test"])''')
test = test.replace(
    '        test_rows.append({"seed": seed, **{f"test_{k}": v for k, v in metrics.items()}})',
    '        test_rows.append({"seed": seed, "inference_ms_per_image": test_inference_ms_per_image, '
    '**{f"test_{k}": v for k, v in metrics.items()}})')
test = test.replace(
    '        pd.DataFrame(confusion_matrix(\n            labels["test"], predictions, labels=range(len(CLASS_NAMES))\n        ), index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(seed_dir / "test_confusion_matrix.csv")',
    '''        cm = confusion_matrix(labels["test"], predictions, labels=range(len(CLASS_NAMES)))
        pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
            seed_dir / "test_confusion_matrix.csv")
        for normalized, suffix in ((False, ""), (True, "_normalized")):
            values = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) if normalized else cm
            pd.DataFrame(values, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
                seed_dir / f"test_confusion_matrix{suffix}.csv")
            fig, ax = plt.subplots(figsize=(8, 7), facecolor="white")
            ax.imshow(values, cmap=MATRIX_CMAP, vmin=0,
                      vmax=1 if normalized else max(1, int(cm.max())))
            ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
            ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
            ax.set_xlabel("Predicted"); ax.set_ylabel("True")
            ax.set_title(f"Final Test seed {seed}")
            for row in range(len(CLASS_NAMES)):
                for col in range(len(CLASS_NAMES)):
                    ax.text(col, row,
                            f"{values[row,col]:.2f}" if normalized else str(int(values[row,col])),
                            ha="center", va="center",
                            color=matrix_text_color(values[row,col], 1 if normalized else cm.max()))
            fig.tight_layout()
            fig.savefig(seed_dir / f"test_confusion_matrix{suffix}.png", dpi=180)
            plt.close(fig)''')
put(16, test)
put(17, """## 6. Historical real holdout

Only the 48 images remaining after the exact Test duplicate is excluded are
scored. Four true classes have support; the fifth is N/A. The 5-seed evaluator
uses the immutable Validation-selected checkpoints.
""")
test += '''

if RUN_FINAL_TEST:
    for normalized, suffix in ((False, ""), (True, "_normalized")):
        fig, axes = plt.subplots(1, len(SEEDS), figsize=(27, 5.8), facecolor="white")
        for column, seed in enumerate(SEEDS):
            values = pd.read_csv(out_dir / f"seed_{seed}" /
                                 f"test_confusion_matrix{suffix}.csv", index_col=0).to_numpy()
            ax = axes[column]
            ax.imshow(values, cmap=MATRIX_CMAP, vmin=0,
                      vmax=1 if normalized else max(1, values.max()))
            ax.set_title(f"Final Test seed {seed}")
            ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_yticklabels(CLASS_NAMES if column == 0 else [])
            for row in range(len(CLASS_NAMES)):
                for col in range(len(CLASS_NAMES)):
                    ax.text(col, row,
                            f"{values[row,col]:.2f}" if normalized else str(int(values[row,col])),
                            ha="center", va="center", fontsize=7,
                            color=matrix_text_color(values[row,col],
                                                    1 if normalized else values.max()))
        fig.tight_layout()
        fig.savefig(aggregate_dir / f"test_confusion_matrices{suffix}_5seeds.png", dpi=180)
        plt.close(fig)
'''
put(16, test)

holdout_plot = get(18)
holdout_plot += '''

if RUN_FINAL_TEST and RUN_REAL_HOLDOUT:
    for normalized, suffix in ((False, ""), (True, "_normalized")):
        fig, axes = plt.subplots(1, len(SEEDS), figsize=(27, 5.8), facecolor="white")
        for column, seed in enumerate(SEEDS):
            values = pd.read_csv(out_dir / "real_holdout" / f"seed_{seed}" /
                                 f"confusion_matrix{suffix}.csv", index_col=0).to_numpy()
            ax = axes[column]
            ax.imshow(values, cmap=REAL_HOLDOUT_CMAP, vmin=0,
                      vmax=1 if normalized else max(1, values.max()))
            ax.set_title(f"Historical holdout seed {seed}")
            ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
            ax.set_yticks(range(len(CLASS_NAMES)))
            ax.set_yticklabels(CLASS_NAMES if column == 0 else [])
            for row in range(len(CLASS_NAMES)):
                for col in range(len(CLASS_NAMES)):
                    text = f"{values[row,col]:.2f}" if normalized else str(int(values[row,col]))
                    if normalized and not (real_holdout_manifest["label"] == row).any():
                        text = "N/A"
                    ax.text(col, row, text, ha="center", va="center", fontsize=7,
                            color=holdout_matrix_text_color(
                                values[row,col], 1 if normalized else values.max()))
        fig.tight_layout()
        fig.savefig(out_dir / "real_holdout" / "aggregate" /
                    f"holdout_confusion_matrices{suffix}_5seeds.png", dpi=180)
        plt.close(fig)
'''
put(18, holdout_plot)

# Add a self-contained export cell. Both ZIPs include the generated notebook.
cells.append({
    "cell_type": "markdown", "id": "vit007_cell_19", "metadata": {},
    "source": ["## 7. Result ZIPs and exact notebook source\n",
               "\n", "Run after all five seeds and gated evaluations finish.\n"],
})
export_code = '''# Reconstruct the exact output-free notebook even in Colab, where the uploaded
# .ipynb path is not guaranteed to be mounted in the runtime filesystem.
import base64
import zlib
NOTEBOOK_SOURCE_ZLIB_B64 = ""
source_notebook = json.loads(zlib.decompress(base64.b64decode(NOTEBOOK_SOURCE_ZLIB_B64)))
source_cell = "".join(source_notebook["cells"][-1]["source"])
source_notebook["cells"][-1]["source"] = source_cell.replace(
    'NOTEBOOK_SOURCE_ZLIB_B64 = ""',
    f'NOTEBOOK_SOURCE_ZLIB_B64 = "{NOTEBOOK_SOURCE_ZLIB_B64}"', 1).splitlines(keepends=True)
notebook_bytes = (json.dumps(source_notebook, ensure_ascii=False, indent=1) + "\\n").encode("utf-8")
if len(validation_df) != len(SEEDS) or not lock_path.is_file():
    raise RuntimeError("Five Validation seeds and selection lock are required for export")
if RUN_FINAL_TEST and not (aggregate_dir / "test_results_5seeds.csv").is_file():
    raise RuntimeError("Final Test was enabled but has not completed")
if RUN_FINAL_TEST and RUN_REAL_HOLDOUT and not (out_dir / "real_holdout" / "evaluation_completed.json").is_file():
    raise RuntimeError("Real holdout was enabled but has not completed")
for seed in SEEDS:
    if not (out_dir / f"seed_{seed}" / "best_model.keras").is_file():
        raise FileNotFoundError(f"Missing selected model for seed {seed}")
notebook_name = f"{EXPERIMENT_ID}/notebook/{NOTEBOOK_FILENAME}"
zip_paths = {}
for kind in ("FULL", "ANALYSIS"):
    destination = results_root / f"{EXPERIMENT_ID}_{kind}.zip"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing export: {destination}")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=3) as archive:
        archive.writestr(notebook_name, notebook_bytes)
        for artifact in sorted(out_dir.rglob("*")):
            if not artifact.is_file():
                continue
            if kind == "ANALYSIS" and artifact.suffix.lower() in {".keras", ".h5", ".weights"}:
                continue
            archive.write(artifact, f"{EXPERIMENT_ID}/{artifact.relative_to(out_dir).as_posix()}")
    with zipfile.ZipFile(destination) as archive:
        if archive.read(notebook_name) != notebook_bytes:
            raise AssertionError("Exported notebook differs from 007 source")
    zip_paths[kind] = str(destination)
    print(kind, destination, "bytes", destination.stat().st_size)
(out_dir / "export_locations.json").write_text(json.dumps(zip_paths, indent=2), encoding="utf-8")
'''
cells.append({
    "cell_type": "code", "id": "vit007_cell_20", "metadata": {},
    "execution_count": None, "outputs": [], "source": export_code.splitlines(keepends=True),
})

notebook["metadata"].setdefault("colab", {})["name"] = TARGET.name
snapshot = copy.deepcopy(notebook)
payload = base64.b64encode(zlib.compress(
    json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))).decode("ascii")
put(len(cells) - 1, get(len(cells) - 1).replace(
    'NOTEBOOK_SOURCE_ZLIB_B64 = ""', f'NOTEBOOK_SOURCE_ZLIB_B64 = "{payload}"', 1))
TARGET.write_bytes((json.dumps(notebook, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
print("Created", TARGET, "with", len(cells), "output-free cells")
