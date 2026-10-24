"""Create independent ResNet50 transfer 009 from the completed 008 recipe.

009 uses a slower head fit, a small low-LR backbone update, and matched
inference probes. It preserves 008 and never fabricates training results.
"""

import base64
import copy
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "resnet50_transfer_5seed_008.ipynb"
TARGET = HERE / "resnet50_transfer_5seed_009.ipynb"
if TARGET.exists():
    raise FileExistsError(f"Review existing 009 notebook before replacing: {TARGET}")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]


def get(index):
    return "".join(cells[index]["source"])


def put(index, value):
    cells[index]["source"] = value.splitlines(keepends=True)


def swap(index, old, new, expected=1):
    value = get(index)
    if value.count(old) != expected:
        raise ValueError(f"cell {index}: expected {expected} occurrences of {old!r}, got {value.count(old)}")
    put(index, value.replace(old, new))


for index, cell in enumerate(cells):
    cell["id"] = f"resnet009_cell_{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

source, count = re.subn(
    r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]*"$',
    'NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
    get(24), count=1, flags=re.MULTILINE)
if count != 1:
    raise ValueError("Expected one embedded source assignment")
put(24, source)

for cell in cells:
    value = "".join(cell["source"])
    value = value.replace("EXP-RESNET50-TRANSFER-5SEEDS-008",
                          "EXP-RESNET50-TRANSFER-5SEEDS-009")
    value = value.replace("resnet50_transfer_5seed_008.ipynb",
                          "resnet50_transfer_5seed_009.ipynb")
    value = value.replace("real_holdout_008", "real_holdout_009")
    value = value.replace("EXP-008", "EXP-009")
    cell["source"] = value.splitlines(keepends=True)

put(0, """# EXP-RESNET50-TRANSFER-5SEEDS-009: cautious partial fine-tuning

**NOT_EXECUTED.** ResNet50 starts from ImageNet weights on the unchanged V08
dataset. Stage 1 trains the classification head with a lower learning rate
than 008. Stage 2 opens only `conv5_block3_*` non-BatchNorm weights at 3e-6;
all other backbone layers and BatchNorm statistics stay frozen. Validation
alone selects checkpoints and the deployment seed.

The completed 008 run started at 9-27% pre-fit Validation accuracy, then
reached about 89-92% after 300 batches in epoch 1. That jump does not prove
leakage. Its five-seed clean Train probe was 97.08% and Validation 95.09%,
while the previously exposed 48-image holdout was 77.08%. The field-image
gap is the central generalization concern, not a target curve shape.

009 logs matched Train and Validation inference probes at step 0, several
steps within epoch 1, and every epoch. Both curves use the same inference
mode and are diagnostic only; their shape is neither forced nor used for
selection. Final Test and historical holdout remain enabled after the
five-seed Validation lock. Both result ZIPs include this exact notebook.
No 009 performance is claimed before training.
""")

swap(4, 'NOTEBOOK_VERSION = "008"', 'NOTEBOOK_VERSION = "009"')
swap(4, 'STAGE1_LR, STAGE1_EPOCHS = 3e-4, 12',
        'STAGE1_LR, STAGE1_EPOCHS = 1e-4, 18')
swap(4, 'STAGE1_EARLY_PATIENCE, STAGE1_REDUCE_PATIENCE, STAGE1_MIN_LR = 4, 2, 1e-6',
        'STAGE1_EARLY_PATIENCE, STAGE1_REDUCE_PATIENCE, STAGE1_MIN_LR = 5, 2, 1e-6')
swap(4, 'STAGE2_LR, STAGE2_EPOCHS = 3e-5, 6',
        'STAGE2_LR, STAGE2_EPOCHS = 3e-6, 8')
swap(4, 'STAGE2_EARLY_PATIENCE, STAGE2_REDUCE_PATIENCE, STAGE2_MIN_LR = 2, 1, 1e-7',
        'STAGE2_EARLY_PATIENCE, STAGE2_REDUCE_PATIENCE, STAGE2_MIN_LR = 3, 2, 1e-7')
swap(4, 'MODEL_NAME = "ResNet50"',
        'MODEL_NAME = "ResNet50"\nSTAGE2_TRAINABLE_PREFIX = "conv5_block3_"\n'
        'MATCHED_PROBE_PER_CLASS = 20\nMATCHED_PROBE_EVERY_BATCHES = 50')

model = get(12)
start, end = model.index('def set_backbone_stage('), model.index('def compile_stage(')
model = model[:start] + '''def set_backbone_stage(backbone, stage):
    if stage == "stage1":
        backbone.trainable = False
    elif stage == "stage2":
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = (layer.name.startswith(STAGE2_TRAINABLE_PREFIX) and
                               not isinstance(layer, layers.BatchNormalization))
        if not any(layer.trainable and layer.weights for layer in backbone.layers):
            raise RuntimeError("No weighted conv5_block3_* layer was unfrozen")
        if any(layer.trainable for layer in backbone.layers
               if isinstance(layer, layers.BatchNormalization)):
            raise RuntimeError("A backbone BatchNorm layer is trainable")
        if any(layer.trainable and not layer.name.startswith(STAGE2_TRAINABLE_PREFIX)
               for layer in backbone.layers):
            raise RuntimeError("Unexpected backbone layer is trainable")
    else:
        raise ValueError(stage)
    return int(sum(np.prod(v.shape) for v in backbone.trainable_weights))

''' + model[end:]
old = 'if stage2_params != 0 or not model.trainable_weights:\n        raise AssertionError("Stage 2 must train only the classification head")'
new = 'if not 0 < stage2_params < backbone.count_params():\n        raise AssertionError("Stage 2 partial fine-tuning count is implausible")'
if model.count(old) != 1:
    raise ValueError("Stage 2 smoke assertion changed")
put(12, model.replace(old, new))

swap(14,
     '"comparison_scope": "frozen backbone versus ResNet50 Transfer-007 on the same V08; no claim until 008 is run",',
     '"comparison_scope": "cautious partial fine-tuning versus completed frozen-backbone 008 on the same V08; no claim until 009 is run",')
swap(14, '"trainable_backbone": "none; classification head only"',
         '"trainable_backbone": "conv5_block3_* non-BatchNorm only"')
swap(14, '"diagnostics_do_not_select_model": True},',
         '"diagnostics_do_not_select_model": True,\n'
         '        "matched_inference_probe_per_class": MATCHED_PROBE_PER_CLASS,\n'
         '        "stage1_first_epoch_probe_every_batches": MATCHED_PROBE_EVERY_BATCHES},')

training = get(15)
probe_class = '''class MatchedInferenceProbe(tf.keras.callbacks.Callback):
    """Same fixed, balanced Train/Validation images in inference mode.

    The scores are descriptive; they never select a checkpoint or seed.
    """
    def __init__(self, stage, train_data, val_data, steps_per_epoch):
        super().__init__()
        self.stage = stage
        self.train_batches = list(train_data)
        self.val_batches = list(val_data)
        self.steps_per_epoch = steps_per_epoch
        self.rows = []
        self.epoch = 0

    def _accuracy(self, batches):
        correct = total = 0
        for images, labels in batches:
            predicted = self.model(images, training=False).numpy().argmax(axis=1)
            true = labels.numpy()
            correct += int(np.sum(predicted == true))
            total += len(true)
        return correct / total

    def _record(self, point, epoch, step):
        train_acc = self._accuracy(self.train_batches)
        val_acc = self._accuracy(self.val_batches)
        self.rows.append({"stage": self.stage, "point": point, "epoch": epoch,
                          "step": step, "train_probe_accuracy": train_acc,
                          "val_probe_accuracy": val_acc,
                          "train_minus_val_accuracy": train_acc - val_acc})

    def on_train_begin(self, logs=None):
        self._record("stage_start", 0, 0)

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch

    def on_train_batch_end(self, batch, logs=None):
        if (self.stage == "stage1" and self.epoch == 0 and
                0 < batch + 1 < self.steps_per_epoch and
                (batch + 1) % MATCHED_PROBE_EVERY_BATCHES == 0):
            self._record("first_epoch_step", 1, batch + 1)

    def on_epoch_end(self, epoch, logs=None):
        self._record("epoch_end", epoch + 1, (epoch + 1) * self.steps_per_epoch)

'''
if training.count('def stage_callbacks(') != 1:
    raise ValueError("Stage callback definition changed")
training = training.replace('def stage_callbacks(', probe_class + 'def stage_callbacks(')
needle = 'assert len(TRAIN_PROBE_INDICES) == 500\n'
addition = '''assert len(TRAIN_PROBE_INDICES) == 500
matched_rng = np.random.default_rng(20261020)
MATCHED_TRAIN_INDICES = np.sort(np.concatenate([
    matched_rng.choice(np.flatnonzero(train_labels == class_id),
                       size=MATCHED_PROBE_PER_CLASS, replace=False)
    for class_id in range(NUM_CLASSES)]))
MATCHED_VAL_INDICES = np.sort(np.concatenate([
    matched_rng.choice(np.flatnonzero(val_labels == class_id),
                       size=MATCHED_PROBE_PER_CLASS, replace=False)
    for class_id in range(NUM_CLASSES)]))
assert len(MATCHED_TRAIN_INDICES) == len(MATCHED_VAL_INDICES) == NUM_CLASSES * MATCHED_PROBE_PER_CLASS
save_json(OUT_DIR / "matched_probe_definition.json", {
    "per_class_per_split": MATCHED_PROBE_PER_CLASS,
    "within_first_epoch_every_batches": MATCHED_PROBE_EVERY_BATCHES,
    "train_relative_paths": train_meta.iloc[MATCHED_TRAIN_INDICES]["relative_path"].tolist(),
    "val_relative_paths": val_meta.iloc[MATCHED_VAL_INDICES]["relative_path"].tolist(),
    "selection_use": "none; diagnostic only"})
'''
if training.count(needle) != 1:
    raise ValueError("Train probe location changed")
training = training.replace(needle, addition)
needle = '''    if all(path.is_file() for path in (selected_path, summary_path, validation_path,
                                        stage1_dir / "best.weights.h5", stage2_dir / "best.weights.h5")):'''
replacement = '''    if all(path.is_file() for path in (selected_path, summary_path, validation_path,
                                        stage1_dir / "best.weights.h5", stage2_dir / "best.weights.h5",
                                        stage1_dir / "matched_inference_curve.csv",
                                        stage2_dir / "matched_inference_curve.csv")):'''
if training.count(needle) != 1:
    raise ValueError("Completed-seed guard changed")
training = training.replace(needle, replacement)
needle = '    model, backbone = build_transfer_model(seed, weights="imagenet")\n'
addition = '''    model, backbone = build_transfer_model(seed, weights="imagenet")
    matched_train_ds = make_dataset(train_paths[MATCHED_TRAIN_INDICES],
                                    train_labels[MATCHED_TRAIN_INDICES], seed, training=False)
    matched_val_ds = make_dataset(val_paths[MATCHED_VAL_INDICES],
                                  val_labels[MATCHED_VAL_INDICES], seed, training=False)
    steps_per_epoch = (len(train_labels) + BATCH_SIZE - 1) // BATCH_SIZE
'''
if training.count(needle) != 1:
    raise ValueError("Model construction location changed")
training = training.replace(needle, addition)
needle = '''    started = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=STAGE1_EPOCHS,
              callbacks=callbacks1, verbose=1)
    stage1_seconds = time.perf_counter() - started'''
replacement = '''    matched1 = MatchedInferenceProbe("stage1", matched_train_ds, matched_val_ds,
                                     steps_per_epoch)
    callbacks1.insert(1, matched1)
    started = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=STAGE1_EPOCHS,
              callbacks=callbacks1, verbose=1)
    stage1_seconds = time.perf_counter() - started
    pd.DataFrame(matched1.rows).to_csv(stage1_dir / "matched_inference_curve.csv", index=False)'''
if training.count(needle) != 1:
    raise ValueError("Stage 1 fit location changed")
training = training.replace(needle, replacement)
needle = '''    started = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS,
              callbacks=callbacks2, verbose=1)
    stage2_seconds = time.perf_counter() - started'''
replacement = '''    matched2 = MatchedInferenceProbe("stage2", matched_train_ds, matched_val_ds,
                                     steps_per_epoch)
    callbacks2.insert(1, matched2)
    started = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS,
              callbacks=callbacks2, verbose=1)
    stage2_seconds = time.perf_counter() - started
    pd.DataFrame(matched2.rows).to_csv(stage2_dir / "matched_inference_curve.csv", index=False)'''
if training.count(needle) != 1:
    raise ValueError("Stage 2 fit location changed")
put(15, training.replace(needle, replacement))

analysis = get(16)
analysis += '''
# Matched inference curves show the actual learning trajectory. They are
# sampled on fixed balanced probes and never affect selection or training.
matched_rows = []
fig, axes = plt.subplots(1, len(SEEDS), figsize=(28, 5.5), facecolor="white")
for column, seed in enumerate(SEEDS):
    first = pd.read_csv(OUT_DIR / f"seed_{seed}" / "stage1" / "matched_inference_curve.csv")
    second = pd.read_csv(OUT_DIR / f"seed_{seed}" / "stage2" / "matched_inference_curve.csv")
    steps_per_epoch = (len(train_labels) + BATCH_SIZE - 1) // BATCH_SIZE
    boundary = int(first["epoch"].max()) * steps_per_epoch
    first["global_step"] = first["step"]
    second["global_step"] = second["step"] + boundary
    joined = pd.concat([first, second], ignore_index=True)
    joined.insert(0, "seed", seed)
    matched_rows.append(joined)
    ax = axes[column]
    x = joined["global_step"] / steps_per_epoch
    ax.plot(x, joined["train_probe_accuracy"], marker=".", label="Train inference probe")
    ax.plot(x, joined["val_probe_accuracy"], marker=".", label="Val inference probe")
    ax.axvline(boundary / steps_per_epoch, color="black", linestyle="--", alpha=0.6)
    ax.set_title(f"Seed {seed}")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Equivalent epochs from optimizer steps")
    ax.grid(alpha=0.25)
axes[0].set_ylabel("Accuracy on fixed balanced probes")
axes[0].legend(fontsize=8)
fig.suptitle("Actual matched inference curves; diagnostic only")
fig.tight_layout()
fig.savefig(PANEL_DIR / "matched_inference_curves_5seeds_panel.png", dpi=190,
            bbox_inches="tight")
plt.close(fig)
pd.concat(matched_rows, ignore_index=True).to_csv(
    AGGREGATE_DIR / "matched_inference_curves_5seeds.csv", index=False)
'''
put(16, analysis)

swap(20, 'REAL_HOLDOUT_HELPER_VERSION = "008.1"',
         'REAL_HOLDOUT_HELPER_VERSION = "009.1"')

package = get(22)
old_builder = '''def set_stage(backbone, stage):
    if stage not in ("stage1", "stage2"):
        raise ValueError(stage)
    backbone.trainable = False'''
new_builder = '''def set_stage(backbone, stage):
    if stage == "stage1":
        backbone.trainable = False
    elif stage == "stage2":
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("conv5_block3_") and not isinstance(
                layer, layers.BatchNormalization)
    else:
        raise ValueError(stage)'''
if package.count(old_builder) != 1:
    raise ValueError("Model package builder changed")
package = package.replace(old_builder, new_builder)
old_continue = '''backbone.trainable = False
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=3e-5, weight_decay=1e-4),'''
new_continue = '''backbone.trainable = True
for layer in backbone.layers:
    layer.trainable = layer.name.startswith("conv5_block3_") and not isinstance(
        layer, keras.layers.BatchNormalization)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=3e-6, weight_decay=1e-4),'''
if package.count(old_continue) != 1:
    raise ValueError("Continue-training example changed")
put(22, package.replace(old_continue, new_continue))

report = get(24)
report = report.replace(
    '- Stage 2: backbone frozen, head-only refinement, fresh AdamW LR {STAGE2_LR}, up to {STAGE2_EPOCHS} epochs.',
    '- Stage 2: only conv5_block3_* non-BN + head trainable, fresh AdamW LR {STAGE2_LR}, up to {STAGE2_EPOCHS} epochs.')
report = report.replace(
    '- EXP-007 is the same-V08 transfer reference. EXP-009 freezes the backbone in both stages and shortens Stage 2.',
    '- EXP-008 is the completed same-V08 frozen-feature reference. EXP-009 opens only the final ResNet block at a very low Stage 2 LR.')
report = report.replace(
    '- Diagnostics: pre-fit random-head scores, first-batch history, fixed 500-image Train probe, Stage 2 loss rebound, and pHash candidate Val subgroup. These never select a checkpoint.',
    '- Diagnostics: pre-fit and first-batch scores, fixed 500-image Train probe, matched Train/Val inference curves from step 0, Stage 2 loss rebound, and pHash candidate Val subgroup. These never select a checkpoint.')
report = report.replace(
    '- Training curves and stage boundaries: `figures/panels/training_curves_5seeds_panel.png`.',
    '- Training curves and stage boundaries: `figures/panels/training_curves_5seeds_panel.png`.\n'
    '- Matched inference curves, including within epoch 1: `figures/panels/matched_inference_curves_5seeds_panel.png` and `aggregate/matched_inference_curves_5seeds.csv`.')
put(24, report)

put(25, """## Interpretation and completion

- The 008 first-epoch Validation jump followed 300 updates; 009 records
  matched Train/Val inference accuracy within that epoch. The original
  online Train metric and full Validation metric remain in the histories.
- The probe curves are sampled diagnostics. Their relationship cannot be
  prescribed; checkpoint and seed selection still use the full Validation
  set's loss and predefined tie-breaks.
- Stage 2 fine-tunes only `conv5_block3_*` non-BN weights at 3e-6. An
  improved V08 score does not by itself establish field generalization.
- Keep Test and historical holdout out of tuning. Review pHash candidates
  manually and collect a fresh independent field set for a blind estimate.
- Report 009 only after five seeds, selection lock, Final Test, historical
  holdout, exported models and both verified ZIPs finish.
""")

notebook["metadata"].setdefault("colab", {})["name"] = TARGET.name
snapshot = copy.deepcopy(notebook)
snapshot_source = get(24).replace('NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
                                  'NOTEBOOK_SOURCE_ZLIB_B64 = ""')
snapshot["cells"][24]["source"] = snapshot_source.splitlines(keepends=True)
payload = base64.b64encode(zlib.compress(
    json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))).decode("ascii")
put(24, get(24).replace('NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
                        f'NOTEBOOK_SOURCE_ZLIB_B64 = "{payload}"'))
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Created", TARGET, "with", len(cells), "output-free cells")
