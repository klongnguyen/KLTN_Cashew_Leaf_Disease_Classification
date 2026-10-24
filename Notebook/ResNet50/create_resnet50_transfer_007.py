"""Create EXP-RESNET50-TRANSFER-5SEEDS-007 from the audited 006 notebook.

007 changes only the fine-tuning capacity/schedule and adds diagnostics. It
never edits V08 or the partial 006 run and does not fabricate outputs.
"""

import base64
import copy
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "resnet50_transfer_5seed_006.ipynb"
TARGET = HERE / "resnet50_transfer_5seed_007.ipynb"
if TARGET.exists():
    raise FileExistsError(f"Review existing 007 notebook before replacing: {TARGET}")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]


def get(index):
    return "".join(cells[index]["source"])


def put(index, text):
    cells[index]["source"] = text.splitlines(keepends=True)


def swap(index, old, new, expected=1):
    text = get(index)
    if text.count(old) != expected:
        raise ValueError(f"cell {index}: {old!r} appears {text.count(old)} times, expected {expected}")
    put(index, text.replace(old, new))


# The copied notebook is source only; historical execution remains in 006.
for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
        cell["outputs"] = []
        cell["execution_count"] = None
    cell["id"] = f"resnet007_cell_{index:02d}"

# Remove the old embedded source before textual changes; rebuild it at end.
archive_source = get(24)
archive_source, payload_count = re.subn(
    r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]*"$',
    'NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
    archive_source, count=1, flags=re.MULTILINE)
if payload_count != 1:
    raise ValueError("006 source has no unique embedded notebook line")
put(24, archive_source)

for cell in cells:
    text = "".join(cell["source"])
    text = text.replace("EXP-RESNET50-TRANSFER-5SEEDS-006",
                        "EXP-RESNET50-TRANSFER-5SEEDS-007")
    text = text.replace("resnet50_transfer_5seed_006.ipynb",
                        "resnet50_transfer_5seed_007.ipynb")
    text = text.replace("real_holdout_006", "real_holdout_007")
    cell["source"] = text.splitlines(keepends=True)

put(0, """# EXP-RESNET50-TRANSFER-5SEEDS-007 — controlled overfitting diagnostic

**NOT_EXECUTED.** This independent ImageNet ResNet50 experiment uses the same
frozen V08 archive and class mapping as 006. It keeps Stage 1 unchanged and
reduces Stage 2 trainable capacity to `conv5_block3_*` non-BatchNorm layers
plus the head. Stage 2 has a shorter fixed schedule. This tests the overfit
signal seen in the partial 006 logs; it does not target a cosmetically lower
training accuracy.

Epoch 1 accuracy is measured **after 300 training batches**, so 70–80% then is
not an initialization score. The notebook records a pre-fit random-head
baseline, first-batch metrics, a fixed unaugmented Train probe, and separate
Validation near-candidate diagnostics to make that distinction explicit.

Validation alone selects checkpoints. Final Test and the previously exposed
48-image historical holdout remain enabled only after the five-seed lock.
Neither split informs training or model selection. Existing 006 outputs and
the V08 dataset are untouched; no 007 performance is claimed yet.
""")

swap(4, 'NOTEBOOK_VERSION = "006"', 'NOTEBOOK_VERSION = "007"')
swap(4, 'STAGE2_LR, STAGE2_EPOCHS = 1e-5, 30',
        'STAGE2_LR, STAGE2_EPOCHS = 1e-5, 12')
swap(4, 'STAGE2_EARLY_PATIENCE, STAGE2_REDUCE_PATIENCE, STAGE2_MIN_LR = 8, 4, 1e-7',
        'STAGE2_EARLY_PATIENCE, STAGE2_REDUCE_PATIENCE, STAGE2_MIN_LR = 4, 2, 1e-7')
swap(4, 'MODEL_NAME = "ResNet50"',
        'MODEL_NAME = "ResNet50"\nSTAGE2_TRAINABLE_PREFIX = "conv5_block3_"')
swap(4, 'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 006 .ipynb on Drive.',
        'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 007 .ipynb on Drive.')

model = get(12)
model = model.replace('layer.name.startswith("conv5_block")',
                      'layer.name.startswith(STAGE2_TRAINABLE_PREFIX)')
model = model.replace('"No weighted conv5_block* layer was unfrozen"',
                      '"No weighted conv5_block3_* layer was unfrozen"')
model = model.replace('not layer.name.startswith("conv5_block")',
                      'not layer.name.startswith(STAGE2_TRAINABLE_PREFIX)')
put(12, model)

recipe = get(14)
recipe = recipe.replace(
    '"baseline_source": "DenseNet121-014 on the same frozen V08; ResNet50 Scratch-005 on V05 is context only",',
    '"baseline_source": "partial ResNet50 Transfer-006 logs; DenseNet121-014 is a same-V08 external comparator",')
recipe = recipe.replace(
    '"comparison_scope": "same-V08 ImageNet two-stage method/architecture comparison with DenseNet121-014",',
    '"comparison_scope": "reduced Stage 2 capacity versus ResNet50 Transfer-006; no claim until 007 is run",')
recipe = recipe.replace('"conv5_block* non-BatchNorm only"',
                        '"conv5_block3_* non-BatchNorm only"')
recipe = recipe.replace('    "optimizer": "AdamW",',
    '    "diagnostics": {"pre_fit_random_head": True, "first_batch_metrics": True,\n'
    '        "train_probe_per_class": 100, "val_train_near_candidate_report": True,\n'
    '        "diagnostics_do_not_select_model": True},\n'
    '    "optimizer": "AdamW",')
put(14, recipe)

training = get(15)
training = training.replace(
    '    def on_epoch_begin(self, epoch, logs=None):\n'
    '        self.current_lr = float(self.model.optimizer.learning_rate.numpy())',
    '    def on_epoch_begin(self, epoch, logs=None):\n'
    '        self.current_lr = float(self.model.optimizer.learning_rate.numpy())\n'
    '        self.first_batch_accuracy = None\n'
    '        self.first_batch_loss = None\n\n'
    '    def on_train_batch_end(self, batch, logs=None):\n'
    '        if batch == 0:\n'
    '            self.first_batch_accuracy = float((logs or {}).get("accuracy", float("nan")))\n'
    '            self.first_batch_loss = float((logs or {}).get("loss", float("nan")))')
training = training.replace(
    '"learning_rate": self.current_lr})',
    '"learning_rate": self.current_lr,\n'
    '                          "first_batch_accuracy": self.first_batch_accuracy,\n'
    '                          "first_batch_loss": self.first_batch_loss})')
training = training.replace(
    'training_rows, validation_rows = [], []',
    '''# Fixed, balanced Train probe measures inference without augmentation/dropout.
# It is diagnostic only and never participates in checkpoint/seed selection.
probe_rng = np.random.default_rng(20261019)
TRAIN_PROBE_INDICES = np.sort(np.concatenate([
    probe_rng.choice(np.flatnonzero(train_labels == class_id), size=100, replace=False)
    for class_id in range(NUM_CLASSES)
]))
assert len(TRAIN_PROBE_INDICES) == 500

training_rows, validation_rows = [], []''')
training = training.replace(
    '    save_json(output / "metrics.json", metrics)',
    '''    probe_ds = make_dataset(train_paths[TRAIN_PROBE_INDICES],
                            train_labels[TRAIN_PROBE_INDICES], seed, training=False)
    probe_probabilities = np.asarray(model.predict(probe_ds, verbose=0))
    probe_predicted = probe_probabilities.argmax(axis=1)
    probe_true = train_labels[TRAIN_PROBE_INDICES]
    metrics.update(train_probe_size=int(len(probe_true)),
                   train_probe_accuracy=float(accuracy_score(probe_true, probe_predicted)),
                   train_probe_macro_f1=float(f1_score(probe_true, probe_predicted,
                                                      average="macro", zero_division=0)),
                   train_probe_minus_val_accuracy=float(
                       accuracy_score(probe_true, probe_predicted) - metrics["val_accuracy"]))
    save_json(output / "metrics.json", metrics)''')
training = training.replace(
    '    compile_stage(model, STAGE1_LR)\n'
    '    audit1, callbacks1, _ = stage_callbacks(',
    '''    compile_stage(model, STAGE1_LR)
    # Epoch-zero reference: random classification head, ImageNet backbone,
    # before any EXP-007 gradient update. Never used to choose hyperparameters.
    pre_fit_probability = np.asarray(model.predict(val_ds, verbose=0))
    pre_fit_predicted = pre_fit_probability.argmax(axis=1)
    pre_fit = {"seed": seed, "images": len(val_labels),
               "accuracy": float(accuracy_score(val_labels, pre_fit_predicted)),
               "macro_f1": float(f1_score(val_labels, pre_fit_predicted,
                                           average="macro", zero_division=0)),
               "loss": float(-np.log(np.clip(pre_fit_probability[
                   np.arange(len(val_labels)), val_labels], 1e-7, 1)).mean()),
               "description": "ImageNet backbone plus random head before Stage 1"}
    save_json(seed_dir / "pre_fit_validation.json", pre_fit)
    print("Pre-fit Validation (diagnostic only):", pre_fit)
    audit1, callbacks1, _ = stage_callbacks(''')
training = training.replace(
    '        "stage1_epochs": stage1["actual_epochs"], "stage2_epochs": stage2["actual_epochs"],',
    '        "stage1_epochs": stage1["actual_epochs"], "stage2_epochs": stage2["actual_epochs"],\n'
    '        "pre_fit_val_accuracy": pre_fit["accuracy"],')
put(15, training)

analysis = get(16)
analysis = analysis.replace(
    'summary_columns = ["val_accuracy",',
    'summary_columns = ["val_accuracy", "train_probe_accuracy", "train_probe_minus_val_accuracy",')
analysis += '''
# Potential train-to-Validation near matches are descriptive candidates,
# not confirmed duplicates; never remove images or use this to rank seeds.
candidate_val_paths = set()
for _, pair in near_df.iterrows():
    if {pair["a_split"], pair["b_split"]} == {"train", "val"}:
        candidate_val_paths.add(pair["a"] if pair["a_split"] == "val" else pair["b"])
candidate_rows = []
for seed in SEEDS:
    predictions = pd.read_csv(OUT_DIR / f"seed_{seed}" / "validation" / "predictions.csv")
    mask = predictions["relative_path"].isin(candidate_val_paths)
    for group, selected in (("phash_candidate", mask), ("other_val", ~mask)):
        part = predictions[selected]
        candidate_rows.append({"seed": seed, "group": group, "correct": int(part["correct"].sum()),
                               "total": len(part), "accuracy": float(part["correct"].mean())})
pd.DataFrame(candidate_rows).to_csv(
    AGGREGATE_DIR / "near_candidate_validation_by_seed.csv", index=False)
save_json(AGGREGATE_DIR / "near_candidate_scope.json", {
    "val_train_candidate_paths": sorted(candidate_val_paths),
    "candidate_count": len(candidate_val_paths),
    "meaning": "pHash screening candidates only; no automatic leakage claim or split change"})

overfit_rows = []
for seed in SEEDS:
    stage1, stage2 = histories[seed]
    best = stage2.loc[stage2["val_loss"].idxmin()]
    last = stage2.iloc[-1]
    overfit_rows.append({"seed": seed, "stage2_best_epoch": int(best["epoch"]),
                         "stage2_best_val_loss": float(best["val_loss"]),
                         "stage2_last_val_loss": float(last["val_loss"]),
                         "stage2_val_loss_rebound": float(last["val_loss"] - best["val_loss"]),
                         "stage2_last_online_train_accuracy": float(last["accuracy"]),
                         "stage2_last_val_accuracy": float(last["val_accuracy"]),
                         "online_train_minus_val_accuracy": float(
                             last["accuracy"] - last["val_accuracy"])})
pd.DataFrame(overfit_rows).to_csv(AGGREGATE_DIR / "overfit_diagnostics_by_seed.csv", index=False)
'''
put(16, analysis)

holdout = get(20).replace('REAL_HOLDOUT_HELPER_VERSION = "006.1"',
                          'REAL_HOLDOUT_HELPER_VERSION = "007.1"')
put(20, holdout)

package = get(22).replace('layer.name.startswith("conv5_block")',
                           'layer.name.startswith("conv5_block3_")')
put(22, package)

report = get(24)
report = report.replace('only conv5_block* non-BN', 'only conv5_block3_* non-BN')
report = report.replace(
    '- DenseNet121-014 comparator: same frozen V08, seeds and two-stage recipe; ResNet50 differs in architecture and required Caffe preprocessing.',
    '- EXP-006 is the same-V08 transfer reference. EXP-007 changes Stage 2 to the final ResNet block only and a shorter fixed schedule.\n'
    '- DenseNet121-014 remains a same-V08 architecture comparator.')
report = report.replace(
    '- Validation `real_` filename subgroup:',
    '- Diagnostics: pre-fit random-head scores, first-batch history, fixed 500-image Train probe, Stage 2 loss rebound, and pHash candidate Val subgroup. These never select a checkpoint.\n'
    '- Validation `real_` filename subgroup:')
# The embedded copy has an empty payload placeholder. Match that exact line
# when restoring the current 007 payload before the ZIP manifest is built.
report, template_count = re.subn(
    r'^template_line = \'NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]*"\'$',
    'template_line = \'NOTEBOOK_SOURCE_ZLIB_B64 = ""\'',
    report, count=1, flags=re.MULTILINE)
if template_count != 1:
    raise ValueError("Cannot locate the unique notebook-copy payload template")
put(24, report)

put(25, """## Completion and interpretation

- A 70–80% first **epoch** score follows 300 gradient updates; compare it with
  the recorded pre-fit score and first-batch score before calling it suspicious.
- The changed Stage 2 trains only `conv5_block3_*` non-BN weights; the rest of
  the 006 recipe and V08 snapshot stay fixed. The notebook measures the
  Train/Validation gap and Validation loss rebound but does not tune on Test
  or the historical real holdout.
- The 51 pHash pairs are screening candidates, not confirmed leaks. Their
  Validation subgroup is reported separately, without changing V08.
- All five seeds, checkpoint hashes, the Validation lock, Final Test and
  historical holdout must finish before reporting a 007 performance result.
""")

# After the experiment-id replacement, update operational text that used the
# short ID. Keep the explicit historical 006 comparator in cell 24.
for index in (4, 14, 18, 20, 22):
    put(index, get(index).replace("EXP-006", "EXP-007"))

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
