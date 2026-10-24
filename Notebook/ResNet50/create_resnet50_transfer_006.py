"""Build the standalone EXP-RESNET50-TRANSFER-5SEEDS-006 Colab notebook.

The proven V08 audit/evaluation machinery comes from DenseNet121-014. This
generator changes the architecture and protocol without touching older runs.
"""

import base64
import copy
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "DenseNet121" / "densenet121_transfer_5seed_014.ipynb"
TARGET = HERE / "resnet50_transfer_5seed_006.ipynb"
if TARGET.exists():
    raise FileExistsError(f"Review existing notebook before replacing: {TARGET}")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]
for cell in cells:
    cell["outputs"] = [] if cell["cell_type"] == "code" else cell.get("outputs", [])
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
    cell.pop("id", None)


def source(index):
    return "".join(cells[index]["source"])


def put(index, value):
    cells[index]["source"] = value.splitlines(keepends=True)


def change(index, old, new, count=1):
    value = source(index)
    actual = value.count(old)
    if actual != count:
        raise RuntimeError(f"cell {index}: expected {count} instances of {old!r}, got {actual}")
    put(index, value.replace(old, new))


# Overall framing: a new ResNet experiment, never a continuation of EXP-014.
put(0, """# EXP-RESNET50-TRANSFER-5SEEDS-006

**NOT_EXECUTED.** Five independent ImageNet ResNet50 runs on frozen V08.
Stage 1 trains the classification head; Stage 2 trains the head and non-BN
`conv5_block*` layers. The model accepts raw RGB 0–255 and contains Keras
ResNet50 Caffe preprocessing. Validation alone selects stage and seed.

Final Test and the previously exposed 48-image historical real holdout are
enabled **after** the five-seed Validation lock in this run. Neither split
is used to tune, select checkpoints, or choose a seed. The holdout has four
true classes and is not a new blind field evaluation.

This is a same-V08 architecture/method comparison with DenseNet121-014.
ResNet50 Scratch-005 used V05 and is historical context only.
""")

config = source(4)
config = config.replace("EXP-014", "EXP-006").replace('NOTEBOOK_VERSION = "014"', 'NOTEBOOK_VERSION = "006"')
config = config.replace('"EXP-DENSENET121-TRANSFER-5SEEDS-014"', '"EXP-RESNET50-TRANSFER-5SEEDS-006"')
config = config.replace('MODEL_NAME = "DenseNet121"', 'MODEL_NAME = "ResNet50"')
config = config.replace('RUN_FINAL_TEST = False', 'RUN_FINAL_TEST = True')
config = config.replace('RUN_REAL_HOLDOUT = False', 'RUN_REAL_HOLDOUT = True')
config = config.replace(
    'if RUN_FULL_TRAINING and (RUN_FINAL_TEST or RUN_REAL_HOLDOUT):\n'
    '    raise RuntimeError("Finish and lock all five Validation runs before enabling Test or holdout.")\n',
    '# The evaluation cells assert the immutable Validation lock after all five seeds.\n')
config = config.replace('NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 014 .ipynb on Drive.',
                        'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 006 .ipynb on Drive.')
config = config.replace('DRIVE_TARGET_URL = (\n    "https://drive.google.com/drive/folders/"\n    "1-Mrqs8Y9dwLexSKIVKBvyYvvQ8X8fXti?usp=sharing"\n)\nDRIVE_TARGET_FOLDER_ID = "1-Mrqs8Y9dwLexSKIVKBvyYvvQ8X8fXti"\n', '')
config = config.replace('REAL_HOLDOUT_EXPECTED_COUNTS = {"anthracnose": 13, "healthy": 5, "leaf_miner": 24, "red_rust": 7}',
                        'REAL_HOLDOUT_EXPECTED_COUNTS = {"anthracnose": 13, "healthy": 5, "leaf_miner": 24, "red_rust": 7}  # 49 source; one Test duplicate excluded')
config = config.replace('# Separate result namespace; EXP-013 is never modified.',
                        '# Separate result namespace; earlier experiments are never modified.')
config = config.replace('raise FileNotFoundError("Evaluation requires a completed EXP-006 Validation lock.")',
                        'raise FileNotFoundError("Evaluation-only mode requires a completed EXP-006 Validation lock.")')
put(4, config)

# The ZIP is verified before extract, and extraction never edits the source.
change(6, '        print("Extracting Split (8) into:", staging_root)',
       '        corrupt_member = dataset_archive.testzip()\n'
       '        if corrupt_member is not None:\n'
       '            raise ValueError(f"Dataset ZIP CRC failed: {corrupt_member}")\n'
       '        print("Extracting Split (8) into:", staging_root)')
change(6, 'Path(r"D:\\Study\\KhoaLuanTotNghiep\\Dataset")',
       'PROJECT_ROOT.parent / "Dataset"')

# The V08 canonical manifest logic, counts, pHash audit and class mapping
# remain exactly as the source notebook. A different algorithm must not be
# claimed to match the published V08 manifest hash.
put(10, """# 08. Explicit class/file order and EXIF-aware RGB tf.data pipeline
CLASS_NAMES = EXPECTED_CLASSES
NUM_CLASSES = len(CLASS_NAMES)
AUTOTUNE = tf.data.AUTOTUNE

def split_arrays(split_name):
    df = valid_df[valid_df["split"] == split_name].copy().reset_index(drop=True)
    assert (df["label"].to_numpy() == df["class"].map(
        {name: i for i, name in enumerate(CLASS_NAMES)}).to_numpy()).all()
    return df["path"].to_numpy(), df["label"].to_numpy(dtype=np.int32), df

train_paths, train_labels, train_meta = split_arrays("train")
val_paths, val_labels, val_meta = split_arrays("val")
test_paths, test_labels, test_meta = split_arrays("test")

def _pil_rgb(path):
    with Image.open(os.fsdecode(path.numpy())) as handle:
        rgb = ImageOps.exif_transpose(handle).convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)

def decode_image(path, label):
    image = tf.py_function(_pil_rgb, [path], Tout=tf.uint8)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method="bilinear")
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)

def make_dataset(paths, labels, seed, training=False):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    options = tf.data.Options()
    options.experimental_deterministic = not training
    ds = ds.with_options(options)
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE, deterministic=not training)
    return ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

print("Train/Val/Test:", len(train_paths), len(val_paths), len(test_paths))
print("Explicit class order:", list(enumerate(CLASS_NAMES)))
""")

put(11, """## ResNet50 Caffe preprocessing and model smoke

The loader returns raw RGB float32 pixels after EXIF transpose and bilinear
resize. The model alone converts RGB to BGR and subtracts Caffe channel means.
Augmentation layers run during training only. The smoke cell checks parity
against Keras, ImageNet initialization, trainability, BN state, checkpoints,
and a safe `.keras` round trip before the five-seed loop.
""")
put(12, """# 09. Model builder and mandatory pipeline smoke
import tempfile
import subprocess
from PIL import ImageOps

AUGMENTATION_CONFIG = {"horizontal_flip": True, "rotation_factor": 0.08,
                       "zoom_factor": 0.10, "contrast_factor": 0.10,
                       "brightness": False}

def resnet50_preprocessing(rgb):
    bgr = keras.ops.flip(rgb, axis=-1)
    return bgr - keras.ops.convert_to_tensor(
        [103.939, 116.779, 123.68], dtype="float32")

def build_transfer_model(seed, weights="imagenet"):
    set_global_seed(seed)
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="raw_rgb_0_255")
    augmented = keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.08, seed=seed),
        layers.RandomZoom(0.10, seed=seed),
        layers.RandomContrast(0.10, seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")(inputs)
    x = resnet50_preprocessing(augmented)
    backbone = keras.applications.ResNet50(
        include_top=False, weights=weights,
        input_shape=(IMG_SIZE, IMG_SIZE, 3), pooling="avg")
    backbone.trainable = False
    features = backbone(x, training=False)
    x = layers.Dropout(HEAD_DROPOUT, seed=seed)(features)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32",
                           name="disease_output")(x)
    return keras.Model(inputs, outputs, name=f"ResNet50_Transfer_seed_{seed}"), backbone

def set_backbone_stage(backbone, stage):
    if stage == "stage1":
        backbone.trainable = False
    elif stage == "stage2":
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = (layer.name.startswith("conv5_block") and
                               not isinstance(layer, layers.BatchNormalization))
        if not any(layer.trainable and layer.weights for layer in backbone.layers):
            raise RuntimeError("No weighted conv5_block* layer was unfrozen")
        if any(layer.trainable for layer in backbone.layers
               if isinstance(layer, layers.BatchNormalization)):
            raise RuntimeError("A backbone BatchNorm layer is trainable")
        if any(layer.trainable and not layer.name.startswith("conv5_block")
               for layer in backbone.layers):
            raise RuntimeError("A non-conv5 backbone layer is trainable")
    else:
        raise ValueError(stage)
    return int(sum(np.prod(v.shape) for v in backbone.trainable_weights))

def compile_stage(model, lr):
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr,
                                                  weight_decay=WEIGHT_DECAY),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"], steps_per_execution=1, jit_compile=False)

def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")

def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

if RUN_PIPELINE_SMOKE:
    set_global_seed(SEEDS[0])  # Required when TensorFlow op determinism is enabled.
    rgb = np.zeros((2, IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    rgb[0, 0, 0] = [255.0, 0.0, 0.0]
    rgb[0, 0, 1] = [0.0, 128.0, 255.0]
    rgb[1] = np.random.default_rng(SEEDS[0]).uniform(0, 255, rgb[1].shape)
    p = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    preprocess_model = keras.Model(p, resnet50_preprocessing(p))
    expected = keras.applications.resnet50.preprocess_input(rgb.copy())
    actual = preprocess_model(rgb, training=False).numpy()
    parity_error = float(np.max(np.abs(expected - actual)))
    if parity_error >= 1e-4 or not np.allclose(actual[0, 0, 0],
                                              [-103.939, -116.779, 131.32], atol=1e-4):
        raise AssertionError(f"ResNet50 Caffe preprocessing failed: {parity_error}")
    smoke_train = next(iter(make_dataset(train_paths, train_labels, SEEDS[0], True)))
    smoke_val = next(iter(make_dataset(val_paths, val_labels, SEEDS[0], False)))
    try:
        model, backbone = build_transfer_model(SEEDS[0], weights="imagenet")
    except Exception as exc:
        message = str(exc)
        blocked = any(token in message.lower() for token in
                      ("url fetch failure", "no such file", "download", "connection"))
        save_json(OUT_DIR / "pipeline_smoke.json", {
            "status": "BLOCKED" if blocked else "FAILED",
            "reason": "BLOCKED_PRETRAINED_WEIGHTS" if blocked else "MODEL_BUILD_FAILED",
            "error": message})
        raise RuntimeError("ImageNet ResNet50 initialization failed; see pipeline_smoke.json") from exc
    if model.output_shape != (None, NUM_CLASSES):
        raise AssertionError("Wrong model output shape")
    initial_backbone = [w.numpy().copy() for w in backbone.weights[:2]]
    stage1_params = set_backbone_stage(backbone, "stage1")
    if stage1_params != 0 or not model.trainable_weights:
        raise AssertionError("Stage 1 trainability is wrong")
    compile_stage(model, STAGE1_LR)
    model.train_on_batch(smoke_train[0][:2], smoke_train[1][:2])
    if any(not np.array_equal(a, w.numpy()) for a, w in zip(initial_backbone, backbone.weights[:2])):
        raise AssertionError("Frozen backbone changed in Stage 1")
    stage1_prob = model(smoke_val[0][:2], training=False).numpy()
    for batch in (smoke_val[0][:1], smoke_val[0][:16]):
        scores = model(batch, training=False).numpy()
        if scores.shape != (len(batch), NUM_CLASSES) or not np.isfinite(scores).all():
            raise AssertionError("Invalid softmax output")
        if not np.allclose(scores.sum(axis=1), 1.0, atol=1e-5):
            raise AssertionError("Softmax probabilities do not sum to one")
    if not np.allclose(stage1_prob, model(smoke_val[0][:2], training=False).numpy(), atol=1e-6):
        raise AssertionError("Inference path is stochastic")
    bn_layers = [layer for layer in backbone.layers if isinstance(layer, layers.BatchNormalization)]
    bn_before = [(layer.moving_mean.numpy().copy(), layer.moving_variance.numpy().copy())
                 for layer in bn_layers]
    stage2_params = set_backbone_stage(backbone, "stage2")
    if not 0 < stage2_params < backbone.count_params():
        raise AssertionError("Stage 2 parameter count is implausible")
    old_optimizer = model.optimizer
    compile_stage(model, STAGE2_LR)
    if model.optimizer is old_optimizer:
        raise AssertionError("Stage 2 reused the Stage 1 optimizer")
    model.train_on_batch(smoke_train[0][:2], smoke_train[1][:2])
    if any(not (np.array_equal(a, layer.moving_mean.numpy()) and
                np.array_equal(b, layer.moving_variance.numpy()))
           for (a, b), layer in zip(bn_before, bn_layers)):
        raise AssertionError("Backbone BatchNorm moving statistics changed")
    with tempfile.TemporaryDirectory() as temporary:
        weights_path = Path(temporary) / "smoke.weights.h5"
        model_path = Path(temporary) / "smoke.keras"
        expected_scores = model(smoke_val[0][:3], training=False).numpy()
        model.save_weights(weights_path)
        model.save(model_path)
        model.layers[-1].bias.assign_add(tf.one_hot(
            0, NUM_CLASSES, on_value=20.0, off_value=0.0,
            dtype=model.layers[-1].bias.dtype))
        if np.allclose(expected_scores, model(smoke_val[0][:3], training=False).numpy(), atol=1e-5):
            raise AssertionError("Smoke mutation did not change predictions")
        model.load_weights(weights_path)
        if not np.allclose(expected_scores, model(smoke_val[0][:3], training=False).numpy(), atol=1e-5):
            raise AssertionError("Checkpoint failed reload parity")
        restored = keras.models.load_model(model_path, compile=False, safe_mode=True)
        if not np.allclose(expected_scores, restored(smoke_val[0][:3], training=False).numpy(), atol=1e-5):
            raise AssertionError("Safe .keras reload changed predictions")
        input_path = Path(temporary) / "input.npy"
        output_path = Path(temporary) / "clean_process_output.npy"
        np.save(input_path, smoke_val[0][:3].numpy())
        clean_code = (
            "import sys,numpy as np,tensorflow as tf; from tensorflow import keras; "
            "keras.utils.set_random_seed(int(sys.argv[4])); "
            "tf.config.experimental.enable_op_determinism(); "
            "m=keras.models.load_model(sys.argv[1],compile=False,safe_mode=True); "
            "np.save(sys.argv[3],m(np.load(sys.argv[2]),training=False).numpy())"
        )
        subprocess.run([sys.executable, "-c", clean_code, str(model_path),
                        str(input_path), str(output_path), str(SEEDS[0])], check=True)
        clean_scores = np.load(output_path)
        if clean_scores.shape != expected_scores.shape or not np.isfinite(clean_scores).all():
            raise AssertionError("Clean-process .keras output has invalid shape or values")
        clean_max_abs_error = float(np.max(np.abs(expected_scores - clean_scores)))
        clean_same_top1 = bool(np.array_equal(expected_scores.argmax(axis=1),
                                              clean_scores.argmax(axis=1)))
        print("Clean-process reload max abs error:", clean_max_abs_error,
              "| same top-1:", clean_same_top1)
        # Independent TensorFlow processes can choose different GPU kernels.
        # Keep this a meaningful numerical check instead of bitwise parity.
        if not np.allclose(expected_scores, clean_scores, rtol=1e-3, atol=5e-4):
            raise AssertionError(
                f"Clean-process .keras reload differs: max_abs={clean_max_abs_error:.6g}, "
                f"same_top1={clean_same_top1}")
    save_json(OUT_DIR / "pipeline_smoke.json", {
        "status": "PASS", "preprocess_max_abs_error": parity_error,
        "stage1_backbone_trainable_params": stage1_params,
        "stage2_backbone_trainable_params": stage2_params,
        "backbone_batchnorm_layers": len(bn_layers),
        "weights_and_model_reload": "PASS", "imagenet_initialization": "PASS",
        "clean_process_max_abs_error": clean_max_abs_error,
        "clean_process_same_top1": clean_same_top1,
        "clean_process_atol": 5e-4, "clean_process_rtol": 1e-3})
    print("Pipeline smoke PASS: ImageNet, Caffe preprocessing, stages, BN, reload")
    del model, backbone, restored
    keras.backend.clear_session(); gc.collect()
else:
    print("Pipeline smoke NOT_RUN")
""")

recipe = source(14).replace("EXP-014", "EXP-006")
recipe = recipe.replace('"baseline_source": "EXP-DENSENET121-SCRATCH-5SEEDS-013 source notebook",',
                        '"baseline_source": "DenseNet121-014 on the same frozen V08; ResNet50 Scratch-005 on V05 is context only",')
recipe = recipe.replace('"comparison_scope": "method comparison; ImageNet + two stages versus scratch",',
                        '"comparison_scope": "same-V08 ImageNet two-stage method/architecture comparison with DenseNet121-014",')
recipe = recipe.replace('"backbone": "DenseNet121"', '"backbone": "ResNet50"')
recipe = recipe.replace('"preprocessing": "inside model: /255 then ImageNet mean/std"',
                        '"preprocessing": "inside model: RGB to BGR, subtract Caffe BGR means; no /255"')
recipe = recipe.replace('"conv5_* non-BatchNorm only"', '"conv5_block* non-BatchNorm only"')
recipe = recipe.replace('"test_protocol": "disabled by default; evaluate only after Validation lock",',
                        '"test_protocol": "enabled; evaluate only after five-seed Validation lock",')
recipe = recipe.replace('"holdout_protocol": "previously exposed 48-image historical comparison only",',
                        '"holdout_protocol": "enabled after lock; previously exposed 48-image four-true-class historical comparison only",')
recipe = recipe.replace('"selection_rule": "lowest Validation loss, higher Validation macro-F1, Stage 1 on tie",',
                        '"selection_rule": "val_loss tolerance 1e-6; higher macro-F1; Stage 1 on tie",')
recipe = recipe.replace('"deployment_rule": "lowest Validation loss, higher macro-F1, lower seed on tie",',
                        '"deployment_rule": "val_loss tolerance 1e-6; higher macro-F1; lower seed on tie",')
recipe = recipe.replace('    "optimizer": "AdamW",',
                        '    "selection_tolerance": 1e-6,\n    "run_final_test": RUN_FINAL_TEST,\n'
                        '    "run_real_holdout": RUN_REAL_HOLDOUT,\n    "optimizer": "AdamW",')
put(14, recipe)

train = source(15)
train = train.replace('    history.to_csv(stage_dir / "history.csv", index=False)',
                      '    history.to_csv(stage_dir / "history.csv", index=False)')
train = train.replace('    return audit, callbacks, checkpoint',
                      '    save_json(stage_dir / "callbacks.json", {\n'
                      '        "monitor": "val_loss", "mode": "min",\n'
                      '        "early_stopping_patience": early_patience,\n'
                      '        "reduce_lr_factor": LR_REDUCTION_FACTOR,\n'
                      '        "reduce_lr_patience": reduce_patience, "min_lr": min_lr,\n'
                      '        "restore_best_weights": False, "checkpoint": str(checkpoint)})\n'
                      '    return audit, callbacks, checkpoint')
train = train.replace('"filename": val_meta["path"], "is_real_filename": val_meta["is_real_filename"],',
                      '"relative_path": val_meta["relative_path"], "split": "val",\n'
                      '        "is_real_filename": val_meta["is_real_filename"],\n'
                      '        "true_class_idx": y_true, "predicted_class_idx": y_pred,')
train = train.replace('    selected = min((stage1, stage2),\n'
                      '                   key=lambda row: (row["val_loss"], -row["val_macro_f1"],\n'
                      '                                    0 if row["stage"] == "stage1" else 1))',
                      '    selected = choose_stage(stage1, stage2)')
anchor = 'training_rows, validation_rows = [], []'
train = train.replace(anchor, '''def choose_stage(stage1, stage2, tolerance=1e-6):
    delta = stage1["val_loss"] - stage2["val_loss"]
    if abs(delta) > tolerance:
        return stage1 if delta < 0 else stage2
    if abs(stage1["val_macro_f1"] - stage2["val_macro_f1"]) > 1e-12:
        return max((stage1, stage2), key=lambda row: row["val_macro_f1"])
    return stage1


''' + anchor)
train = train.replace('ranked = validation_results_df.sort_values(["val_loss", "val_macro_f1", "seed"],\n'
                      '                                            ascending=[True, False, True])\n'
                      'deployment_seed = int(ranked.iloc[0]["seed"])',
                      '''def choose_deployment_seed(rows, tolerance=1e-6):
    candidates = rows.to_dict("records")
    lowest = min(float(row["val_loss"]) for row in candidates)
    near = [row for row in candidates if float(row["val_loss"]) <= lowest + tolerance]
    return int(sorted(near, key=lambda row: (-float(row["val_macro_f1"]), int(row["seed"])))[0]["seed"])

deployment_seed = choose_deployment_seed(validation_results_df)''')
train = train.replace('    "selected_checkpoint_sha256": {',
                      '    "per_seed_selection": {str(seed): json.loads(\n'
                      '        (OUT_DIR / f"seed_{seed}" / "stage_selection.json").read_text(encoding="utf-8"))\n'
                      '        for seed in SEEDS},\n    "selected_checkpoint_sha256": {')
put(15, train)

analysis = source(16)
analysis += '''
# The 35 real_ filenames are a proxy subgroup, with per-class diagnostics.
real_class_rows = []
for seed in SEEDS:
    predictions = pd.read_csv(OUT_DIR / f"seed_{seed}" / "validation" / "predictions.csv")
    real_only = predictions[predictions["is_real_filename"]]
    for class_name, group in real_only.groupby("true_label"):
        real_class_rows.append({"seed": seed, "class": class_name,
                                "correct": int(group["correct"].sum()),
                                "total": len(group), "accuracy": float(group["correct"].mean())})
pd.DataFrame(real_class_rows).to_csv(AGGREGATE_DIR / "real_validation_by_class_seed.csv", index=False)

# Repeated confusion and confidence diagnostics; descriptive, never tuning.
all_errors = []
for seed in SEEDS:
    predictions = pd.read_csv(OUT_DIR / f"seed_{seed}" / "validation" / "predictions.csv")
    for _, row in predictions[~predictions["correct"]].iterrows():
        all_errors.append({"seed": seed, "relative_path": row["relative_path"],
                           "true_label": row["true_label"],
                           "predicted_label": row["predicted_label"],
                           "confidence": row["confidence"]})
pd.DataFrame(all_errors).to_csv(AGGREGATE_DIR / "validation_errors_all_seeds.csv", index=False)
if all_errors:
    recurrent = pd.DataFrame(all_errors).groupby("relative_path").agg(
        error_seed_count=("seed", "nunique"), true_label=("true_label", "first"),
        max_wrong_confidence=("confidence", "max")).reset_index().sort_values(
            ["error_seed_count", "max_wrong_confidence"], ascending=False)
    recurrent.to_csv(AGGREGATE_DIR / "recurrent_validation_errors.csv", index=False)
    pd.DataFrame(all_errors).sort_values("confidence", ascending=False).head(50).to_csv(
        AGGREGATE_DIR / "top_confident_wrong_validation.csv", index=False)
low_correct = []
for seed in SEEDS:
    predictions = pd.read_csv(OUT_DIR / f"seed_{seed}" / "validation" / "predictions.csv")
    correct = predictions[predictions["correct"]].nsmallest(20, "confidence")
    low_correct.append(correct.assign(seed=seed))
pd.concat(low_correct).to_csv(AGGREGATE_DIR / "low_confident_correct_validation.csv", index=False)
'''
put(16, analysis)

test = source(18).replace("EXP-014", "EXP-006")
test = test.replace('"filename": test_meta["path"],',
                    '"relative_path": test_meta["relative_path"], "split": "test",\n'
                    '                                    "true_class_idx": y_true, "predicted_class_idx": y_pred,')
put(18, test)

holdout = source(20).replace("EXP-014", "EXP-006")
holdout = holdout.replace('real_holdout_014', 'real_holdout_006')
holdout = holdout.replace(
    '    ranked = validation_results.sort_values(["val_loss", "val_macro_f1", "seed"], ascending=[True, False, True])\n'
    '    deployment_seed = int(ranked.iloc[0]["seed"])',
    '    deployment_seed = int(selection_lock["deployment_seed"])\n'
    '    if deployment_seed not in seeds:\n'
    '        raise ValueError("Holdout deployment seed differs from the Validation lock")')
put(20, holdout)

package = source(22).replace('densenet121', 'resnet50').replace('DenseNet121', 'ResNet50')
package = package.replace('densenet_preprocessing', 'resnet50_preprocessing')
package = package.replace('"conv5_"', '"conv5_block"').replace('EXP-014', 'EXP-006')
package = package.replace('keras.models.load_model(deployment_path, compile=False)',
                          'keras.models.load_model(deployment_path, compile=False, safe_mode=True)')
package = package.replace('"historical_holdout_exposed": True,',
                          '"historical_holdout_exposed": True, "holdout_true_classes": 4,')
package = package.replace('classifier = keras.models.load_model(MODEL_DIR / "deployment_model.keras", compile=False)',
                          'classifier = keras.models.load_model(MODEL_DIR / "deployment_model.keras", compile=False, safe_mode=True)')
package = package.replace('    if isinstance(crop, np.ndarray):\n        crop = Image.fromarray(crop.astype(np.uint8), mode="RGB")',
                          '    if isinstance(crop, np.ndarray):\n'
                          '        if crop.ndim != 3 or crop.shape[-1] != 3:\n'
                          '            raise ValueError("Expected an RGB HxWx3 array")\n'
                          '        crop = Image.fromarray(crop.astype(np.uint8), mode="RGB")')
package = package.replace('"Pillow\\nscikit-learn\\nmatplotlib\\nscipy\\n"',
                          '"Pillow\\nscikit-learn\\nmatplotlib\\nscipy\\n"')
package = package.replace('"elapsed_ms_including_preprocessing": 1000 * (time.perf_counter() - started)}\n"""',
                          '"elapsed_ms_including_preprocessing": 1000 * (time.perf_counter() - started)}\n\n'
                          'def classify_bbox(image, box, origin=None):\n'
                          '    if isinstance(image, np.ndarray):\n'
                          '        if image.ndim != 3 or image.shape[-1] != 3:\n'
                          '            raise ValueError("Expected an RGB HxWx3 array")\n'
                          '        image = Image.fromarray(image.astype(np.uint8), mode="RGB")\n'
                          '    image = ImageOps.exif_transpose(image).convert("RGB")\n'
                          '    left, top, right, bottom = map(float, box)\n'
                          '    left, top = max(0, int(left)), max(0, int(top))\n'
                          '    right, bottom = min(image.width, int(right)), min(image.height, int(bottom))\n'
                          '    if right <= left or bottom <= top:\n'
                          '        raise ValueError("Empty YOLO crop after clamping")\n'
                          '    result = classify_crop(image.crop((left, top, right, bottom)))\n'
                          '    result.update(bbox=[left, top, right, bottom], origin=origin)\n'
                          '    return result\n"""')
package = package.replace('continue_code = """', '''preprocessing_code = """from tensorflow import keras

def resnet50_preprocessing(rgb):
    bgr = keras.ops.flip(rgb, axis=-1)
    return bgr - keras.ops.convert_to_tensor([103.939, 116.779, 123.68], dtype="float32")
"""
(CODE_PACKAGE_DIR / "resnet50_preprocessing.py").write_text(preprocessing_code, encoding="utf-8")
builder_code = """from tensorflow import keras
from tensorflow.keras import layers
from resnet50_preprocessing import resnet50_preprocessing

def build_model(seed, weights="imagenet", img_size=224, num_classes=5):
    keras.utils.set_random_seed(seed)
    inputs = keras.Input((img_size, img_size, 3), name="raw_rgb_0_255")
    augmentation = keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.08, seed=seed),
        layers.RandomZoom(0.10, seed=seed),
        layers.RandomContrast(0.10, seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")
    x = resnet50_preprocessing(augmentation(inputs))
    backbone = keras.applications.ResNet50(include_top=False, weights=weights,
                                           pooling="avg", input_shape=(img_size, img_size, 3))
    backbone.trainable = False
    x = layers.Dropout(0.40, seed=seed)(backbone(x, training=False))
    output = layers.Dense(num_classes, activation="softmax", dtype="float32",
                          name="disease_output")(x)
    return keras.Model(inputs, output), backbone

def set_stage(backbone, stage):
    backbone.trainable = stage == "stage2"
    if stage == "stage2":
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("conv5_block") and not isinstance(
                layer, layers.BatchNormalization)
    elif stage != "stage1":
        raise ValueError(stage)
"""
(CODE_PACKAGE_DIR / "resnet50_model_builder.py").write_text(builder_code, encoding="utf-8")
save_json(MODEL_PACKAGE_DIR / "model_load_test.json", {
    "status": "PASS", "safe_mode": True, "prediction_parity_atol": 1e-5,
    "input_count": len(sample_x), "checkpoint_selection": "Validation lock"})
(MODEL_PACKAGE_DIR / "model_sha256.txt").write_text(file_sha256(deployment_path) + "\\n", encoding="utf-8")
continue_code = """''')
put(22, package)

archive = source(24).replace("DenseNet121", "ResNet50").replace("densenet121", "resnet50")
archive = archive.replace("EXP-014", "EXP-006")
archive = archive.replace('resnet50_transfer_5seed_014.ipynb',
                          'resnet50_transfer_5seed_006.ipynb')
archive = archive.replace('"COMPLETED_RETROSPECTIVE"', '"COMPLETED"')
archive = archive.replace('"COMPLETED_HISTORICAL_EXPOSED"', '"COMPLETED"')
archive = archive.replace('else "PENDING"', 'else "NOT_RUN"')
archive = archive.replace(
    '- EXP-013 comparator: same frozen dataset, class order and augmentation. EXP-013 used scratch weights and one training stage. This is a method comparison, not a pretrained-weight-only ablation.',
    '- DenseNet121-014 comparator: same frozen V08, seeds and two-stage recipe; ResNet50 differs in architecture and required Caffe preprocessing.\n'
    '- ResNet50 Scratch-005 is V05 historical context, not a controlled pretrained ablation.')
archive = archive.replace('Earlier EXP-013 Test results make any comparison retrospective.',
                          'This split is opened only after the Validation lock.')
archive = re.sub(r'^NOTEBOOK_SOURCE_ZLIB_B64 = ".*"$',
                 'NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"', archive,
                 count=1, flags=re.MULTILINE)
archive = archive.replace('    "five_seed_training": "COMPLETED", "final_test": test_status,',
                          '    "five_seed_training": "COMPLETED", "validation": "COMPLETED",\n'
                          '    "final_test": test_status,')
archive = archive.replace('"real_holdout": holdout_status, "deployment_seed": deployment_seed,',
                          '"historical_holdout": holdout_status, "fresh_real_holdout": "NOT_RUN",\n'
                          '    "deployment_seed": deployment_seed,\n'
                          '    "historical_holdout_protocol": "previously exposed, four true classes",')
archive = archive.replace('    "experiment_id": EXPERIMENT_ID, "pipeline_smoke": "PASS" if (',
                          '    "experiment_id": EXPERIMENT_ID, "pipeline_smoke": "PASS" if (')
archive = archive.replace('  trainability, BatchNorm state and checkpoint reload before the full run.',
                          '  trainability, BatchNorm state and checkpoint reload before the full run.')
archive = archive.replace('inside model: /255 then ImageNet mean/std',
                          'inside model: RGB to BGR, subtract Caffe means')
archive = archive.replace('only conv5_* non-BN', 'only conv5_block* non-BN')
archive = archive.replace('Test and historical holdout remain `PENDING` until their flags are explicitly',
                          'Test and historical holdout run after the Validation lock because their flags are')
archive = archive.replace('  enabled in a later, evaluation-only run.',
                          '  enabled in this notebook.')
archive = archive.replace('The historical holdout lacks true `not_cashew_leaf` images;',
                          'The historical holdout lacks true `not_cashew_leaf` images;')
archive = archive.replace('readme = f"""# {EXPERIMENT_ID}',
    '''selected_deployment = json.loads((OUT_DIR / f"seed_{deployment_seed}" /
    "stage_selection.json").read_text(encoding="utf-8"))
val_accuracy_mean = validation_results_df["val_accuracy"].mean()
val_accuracy_sd = validation_results_df["val_accuracy"].std(ddof=1)
val_f1_mean = validation_results_df["val_macro_f1"].mean()
val_f1_sd = validation_results_df["val_macro_f1"].std(ddof=1)
readme = f"""# {EXPERIMENT_ID}''')
archive = archive.replace('- Five-seed training: COMPLETED. Per-stage durations and epochs:',
                          '- Validation Accuracy: {100*val_accuracy_mean:.2f} ± {100*val_accuracy_sd:.2f}%; Macro F1: {100*val_f1_mean:.2f} ± {100*val_f1_sd:.2f}% (sample SD, five seeds).\n'
                          '- DenseNet121-014 same-V08 reference: 97.10 ± 0.20% Validation Accuracy, 96.96 ± 0.21% Macro F1; interpret as an architecture/method comparison.\n'
                          '- Five-seed training: COMPLETED. Per-stage durations and epochs:')
archive = archive.replace('- Deployment seed: {deployment_seed}, chosen before optional final evaluations.',
                          '- Deployment seed: {deployment_seed}, stage {selected_deployment["selected_stage"]}, epoch {selected_deployment["selected_epoch"]}, Validation loss {selected_deployment["val_loss"]:.6f}; chosen before final evaluations.')
archive = archive.replace('## Model and reuse\n',
                          '## Model and reuse\n- FULL ZIP: `{EXPERIMENT_ID}_FULL.zip`; ANALYSIS ZIP: `{EXPERIMENT_ID}_ANALYSIS.zip` in the results root.\n')
archive = archive.replace('        names = archive.namelist()\n',
    '''        names = archive.namelist()
        expected_names = {
            (Path(EXPERIMENT_ID) / file_path.relative_to(OUT_DIR)).as_posix()
            for file_path in OUT_DIR.rglob("*") if file_path.is_file()
            and (not analysis_only or file_path.suffix.lower() in analysis_suffixes)
        }
        if set(names) != expected_names:
            raise RuntimeError(f"ZIP file manifest mismatch: {temp}")
''')
archive = archive.replace('    print("Verified archive:", target, f"{target.stat().st_size / (1024**2):.2f} MB")',
                          '    print("Verified archive:", target, f"{target.stat().st_size / (1024**2):.2f} MB",\n'
                          '          "SHA256:", file_sha256(target))')
archive = archive.replace('            if analysis_only and ("model_package" in relative.parts or\n'
                          '                                  file_path.suffix.lower() not in analysis_suffixes):',
                          '            if analysis_only and file_path.suffix.lower() not in analysis_suffixes:')
archive += '''\n(RESULTS_ROOT / f"{EXPERIMENT_ID}_ARCHIVES.sha256").write_text(
    f"{file_sha256(full_zip)}  {full_zip.name}\\n"
    f"{file_sha256(analysis_zip)}  {analysis_zip.name}\\n", encoding="utf-8")
'''
put(24, archive)

put(17, """## Final Test after the Validation lock

`RUN_FINAL_TEST=True` is enabled. This section asserts the saved config,
all five selected checkpoint hashes, and the fixed Validation selection lock
before opening Test. Test outcomes cannot change the selected seed or stage.
""")
put(19, """## Historical real holdout after the Validation lock

`RUN_REAL_HOLDOUT=True` is enabled. The source archive contains 49 images;
one byte-identical V08 Test image is excluded, leaving 48 scored images from
four true classes. This set has been examined previously and is suitable only
for historical comparison. A fresh independent field set remains `NOT_RUN`.
""")
put(25, """## Completion and handoff

The notebook starts with `NOT_EXECUTED` outputs. Running all cells must pass
ZIP/manifest and image checks, ImageNet/preprocessing/BN/save-reload smoke,
and five independent stage1+stage2 runs. It writes the Validation lock before
the enabled Final Test and historical holdout cells. Artifacts and metrics are
reported only after they exist; no previous experiment is overwritten.
""")

notebook["metadata"].setdefault("colab", {})["name"] = TARGET.name
notebook["metadata"].setdefault("kernelspec", {"name": "python3", "display_name": "Python 3"})

# Embed an output-free copy so Colab exports the exact notebook source even
# when the runtime cannot read the editor's .ipynb file from its filesystem.
snapshot = copy.deepcopy(notebook)
put(24, source(24).replace('NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
                          'NOTEBOOK_SOURCE_ZLIB_B64 = ""'))
snapshot["cells"][24]["source"] = copy.deepcopy(cells[24]["source"])
payload = base64.b64encode(zlib.compress(
    json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))).decode("ascii")
put(24, source(24).replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                          f'NOTEBOOK_SOURCE_ZLIB_B64 = "{payload}"'))

TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Created {TARGET} ({len(cells)} cells)")
