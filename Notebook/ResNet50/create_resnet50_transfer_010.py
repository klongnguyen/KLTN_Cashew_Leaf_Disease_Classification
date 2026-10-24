"""Create ResNet50 transfer 010: domain augmentation and honest step-zero plots.

Uses the fixed 009 architecture and optimizer schedule. 010 does not edit
V08, old experiments, Test/holdout selection rules, or measured results.
"""

import base64
import copy
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "resnet50_transfer_5seed_009.ipynb"
TARGET = HERE / "resnet50_transfer_5seed_010.ipynb"
if TARGET.exists():
    raise FileExistsError(f"Review existing 010 notebook before replacing: {TARGET}")

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
    cell["id"] = f"resnet010_cell_{index:02d}"
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
    value = value.replace("EXP-RESNET50-TRANSFER-5SEEDS-009",
                          "EXP-RESNET50-TRANSFER-5SEEDS-010")
    value = value.replace("resnet50_transfer_5seed_009.ipynb",
                          "resnet50_transfer_5seed_010.ipynb")
    value = value.replace("real_holdout_009", "real_holdout_010")
    value = value.replace("EXP-009", "EXP-010")
    cell["source"] = value.splitlines(keepends=True)

put(0, """# EXP-RESNET50-TRANSFER-5SEEDS-010: domain robustness audit

**NOT_EXECUTED.** This independent ImageNet ResNet50 transfer experiment keeps
the 009 V08 archive, class mapping, five seeds, architecture, learning rates,
Stage 2 partial fine-tuning, Validation-only selection, Final Test and
historical real holdout. Train-only brightness and translation augmentation
are added to address photographic variation. No result is claimed yet.

The 009 CSV shows that the model did not begin with high Validation accuracy:
its balanced Validation probe was 7-26% at step 0, rose through intermediate
steps, and reached 75-86% only after about 300 updates in epoch 1. Its final
full Validation accuracy was 95.23%, but the 35 `real_` Validation images
averaged 73.71% and the exposed 48-image holdout averaged 75.83%. A high
full-split score is therefore an incomplete account of field performance.

The primary learning-curve panel now starts at step 0 and pairs Train and
Validation probes measured in the same inference mode. It also shows the
small `real_` Validation subgroup separately; its size makes it noisy. The
original online Train/full Validation epoch plot is retained separately.
The curves are measurements, not targets to shape or selection criteria.
The 51 pHash candidates are exported as contact sheets for manual review.
Both result ZIPs include this exact 010 notebook.
""")

swap(4, 'NOTEBOOK_VERSION = "009"', 'NOTEBOOK_VERSION = "010"')
swap(4, 'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 007 .ipynb on Drive.',
        'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 010 .ipynb on Drive.')

audit = get(8)
audit += '''
# Visual screening only: these pHash pairs are not confirmed duplicates.
contact_dir = OUT_DIR / "dataset_audit" / "phash_contact_sheets"
contact_dir.mkdir(parents=True, exist_ok=True)
path_by_relative = dict(zip(valid_df["relative_path"], valid_df["path"]))
for page_start in range(0, len(near_df), 12):
    page = near_df.iloc[page_start:page_start + 12]
    fig, axes = plt.subplots(len(page), 2, figsize=(11, 2.4 * len(page)),
                             squeeze=False, facecolor="white")
    for row_index, (_, pair) in enumerate(page.iterrows()):
        for column, side in enumerate(("a", "b")):
            with Image.open(path_by_relative[pair[side]]) as image:
                preview = ImageOps.exif_transpose(image).convert("RGB")
                preview.thumbnail((320, 210))
                axes[row_index, column].imshow(preview)
            axes[row_index, column].set_title(
                f"{pair[f'{side}_split']} / {pair[f'{side}_class']} / {pair[side]} "
                f"| pHash={pair['phash_hamming']}, MAD={pair['mad_64_gray']:.1f}",
                fontsize=7)
            axes[row_index, column].axis("off")
    fig.tight_layout()
    fig.savefig(contact_dir / f"pairs_{page_start + 1:03d}_{page_start + len(page):03d}.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)
(contact_dir / "REVIEW_REQUIRED.json").write_text(json.dumps({
    "candidate_pairs": int(len(near_df)),
    "meaning": "pHash screen only; visually review same leaf/plant/capture before leakage claims",
    "dataset_changed": False}, indent=2), encoding="utf-8")
'''
put(8, audit)

model = get(12)
old = '''AUGMENTATION_CONFIG = {"horizontal_flip": True, "rotation_factor": 0.08,
                       "zoom_factor": 0.10, "contrast_factor": 0.10,
                       "brightness": False}'''
new = '''AUGMENTATION_CONFIG = {"horizontal_flip": True, "rotation_factor": 0.08,
                       "zoom_factor": 0.10, "contrast_factor": 0.10,
                       "brightness_factor": 0.12, "translation_factor": 0.05}'''
if model.count(old) != 1:
    raise ValueError("Augmentation config changed")
model = model.replace(old, new)
old = '''        layers.RandomContrast(0.10, seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")(inputs)'''
new = '''        layers.RandomContrast(0.10, seed=seed),
        layers.RandomBrightness(0.12, value_range=(0, 255), seed=seed),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")(inputs)'''
if model.count(old) != 1:
    raise ValueError("Model augmentation stack changed")
put(12, model.replace(old, new))

swap(14, '"stage1_first_epoch_probe_every_batches": MATCHED_PROBE_EVERY_BATCHES},',
         '"stage1_first_epoch_probe_every_batches": MATCHED_PROBE_EVERY_BATCHES,\n'
         '        "real_val_filename_subgroup_during_training": True,\n'
         '        "matched_probe_inference_loss": True,\n'
         '        "phash_contact_sheets_manual_review": True},')

training = get(15)
start, end = training.index('class MatchedInferenceProbe('), training.index('def stage_callbacks(')
probe_class = '''class MatchedInferenceProbe(tf.keras.callbacks.Callback):
    """Fixed image probes, all scored with augmentation and dropout disabled.

    The 35-image real_ subgroup is a noisy filename proxy. None of these
    probe scores participates in checkpoint or seed selection.
    """
    def __init__(self, stage, train_data, val_data, real_val_data, steps_per_epoch):
        super().__init__()
        self.stage = stage
        self.train_batches = list(train_data)
        self.val_batches = list(val_data)
        self.real_val_batches = list(real_val_data)
        self.steps_per_epoch = steps_per_epoch
        self.rows = []
        self.epoch = 0

    def _metrics(self, batches):
        correct = total = 0
        nll = 0.0
        for images, labels in batches:
            probability = self.model(images, training=False).numpy()
            true = labels.numpy()
            correct += int(np.sum(probability.argmax(axis=1) == true))
            nll += float(-np.log(np.clip(probability[np.arange(len(true)), true],
                                         1e-7, 1)).sum())
            total += len(true)
        return correct / total, nll / total

    def _record(self, point, epoch, step):
        train_acc, train_loss = self._metrics(self.train_batches)
        val_acc, val_loss = self._metrics(self.val_batches)
        real_acc, real_loss = self._metrics(self.real_val_batches)
        self.rows.append({"stage": self.stage, "point": point, "epoch": epoch,
                          "step": step, "train_probe_accuracy": train_acc,
                          "val_probe_accuracy": val_acc,
                          "real_val_accuracy": real_acc,
                          "train_probe_loss": train_loss,
                          "val_probe_loss": val_loss,
                          "real_val_loss": real_loss,
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
training = training[:start] + probe_class + training[end:]
needle = 'assert len(MATCHED_TRAIN_INDICES) == len(MATCHED_VAL_INDICES) == NUM_CLASSES * MATCHED_PROBE_PER_CLASS\n'
replacement = '''assert len(MATCHED_TRAIN_INDICES) == len(MATCHED_VAL_INDICES) == NUM_CLASSES * MATCHED_PROBE_PER_CLASS
REAL_VAL_INDICES = np.flatnonzero(val_meta["is_real_filename"].to_numpy(dtype=bool))
assert len(REAL_VAL_INDICES) == 35
'''
if training.count(needle) != 1:
    raise ValueError("Matched probe definition changed")
training = training.replace(needle, replacement)
swap_text = '    "val_relative_paths": val_meta.iloc[MATCHED_VAL_INDICES]["relative_path"].tolist(),\n'
if training.count(swap_text) != 1:
    raise ValueError("Probe path report changed")
training = training.replace(swap_text,
    swap_text + '    "real_val_relative_paths": val_meta.iloc[REAL_VAL_INDICES]["relative_path"].tolist(),\n')
needle = '''    matched_val_ds = make_dataset(val_paths[MATCHED_VAL_INDICES],
                                  val_labels[MATCHED_VAL_INDICES], seed, training=False)
    steps_per_epoch'''
replacement = '''    matched_val_ds = make_dataset(val_paths[MATCHED_VAL_INDICES],
                                  val_labels[MATCHED_VAL_INDICES], seed, training=False)
    real_val_ds = make_dataset(val_paths[REAL_VAL_INDICES],
                               val_labels[REAL_VAL_INDICES], seed, training=False)
    steps_per_epoch'''
if training.count(needle) != 1:
    raise ValueError("Probe dataset construction changed")
training = training.replace(needle, replacement)
for stage in ("stage1", "stage2"):
    needle = (f'MatchedInferenceProbe("{stage}", matched_train_ds, matched_val_ds,\n'
              '                                     steps_per_epoch)')
    replacement = (f'MatchedInferenceProbe("{stage}", matched_train_ds, matched_val_ds,\n'
                   '                                     real_val_ds, steps_per_epoch)')
    if training.count(needle) != 1:
        raise ValueError(f"{stage} callback construction changed")
    training = training.replace(needle, replacement)
put(15, training)

analysis = get(16)
needle = 'fig.savefig(PANEL_DIR / "training_curves_5seeds_panel.png", dpi=190, bbox_inches="tight")'
if analysis.count(needle) != 1:
    raise ValueError("Online curve figure changed")
analysis = analysis.replace(needle,
    'fig.savefig(PANEL_DIR / "online_epoch_curves_5seeds_panel.png", dpi=190, bbox_inches="tight")')
needle = '    ax.plot(x, joined["val_probe_accuracy"], marker=".", label="Val inference probe")'
if analysis.count(needle) != 1:
    raise ValueError("Matched curve plot changed")
analysis = analysis.replace(needle, needle + '\n'
    '    ax.plot(x, joined["real_val_accuracy"], marker=".", linestyle=":",\n'
    '            label="real_ Val, n=35")')
analysis += '''
# Primary chart: paired inference-mode metrics from step zero. Online epoch
# means are saved separately; neither visualization controls selection.
all_matched = pd.concat(matched_rows, ignore_index=True)
fig, axes = plt.subplots(2, len(SEEDS), figsize=(28, 10), sharex="col",
                         facecolor="white")
steps_per_epoch = (len(train_labels) + BATCH_SIZE - 1) // BATCH_SIZE
for column, seed in enumerate(SEEDS):
    part = all_matched[all_matched["seed"] == seed].sort_values("global_step")
    x = part["global_step"] / steps_per_epoch
    boundary = int(part[part["stage"] == "stage1"]["epoch"].max())
    for row, metric in enumerate(("accuracy", "loss")):
        ax = axes[row, column]
        ax.plot(x, part[f"train_probe_{metric}"], marker=".",
                label="Train, inference mode")
        ax.plot(x, part[f"val_probe_{metric}"], marker=".",
                label="Val, inference mode")
        if metric == "accuracy":
            ax.plot(x, part["real_val_accuracy"], marker=".", linestyle=":",
                    label="real_ Val (35 images)")
            ax.set_ylim(0, 1)
        ax.axvline(boundary, color="black", linestyle="--", alpha=0.6)
        ax.grid(alpha=0.25)
        if column == 0:
            ax.set_ylabel(metric.title())
        if row == 1:
            ax.set_xlabel("Equivalent epochs from optimizer steps")
    axes[0, column].set_title(f"Seed {seed}")
axes[0, 0].legend(fontsize=7)
fig.suptitle("Matched inference probes from step 0; Stage 2 boundary dashed")
fig.tight_layout()
fig.savefig(PANEL_DIR / "training_curves_5seeds_panel.png", dpi=190,
            bbox_inches="tight")
plt.close(fig)
'''
put(16, analysis)

swap(20, 'REAL_HOLDOUT_HELPER_VERSION = "009.1"',
         'REAL_HOLDOUT_HELPER_VERSION = "010.2"')

package = get(22)
needle = '''        layers.RandomContrast(0.10, seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")'''
replacement = '''        layers.RandomContrast(0.10, seed=seed),
        layers.RandomBrightness(0.12, value_range=(0, 255), seed=seed),
        layers.RandomTranslation(0.05, 0.05, fill_mode="reflect", seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")'''
if package.count(needle) != 1:
    raise ValueError("Package builder augmentation changed")
put(22, package.replace(needle, replacement))

report = get(24)
report = report.replace(
    '- EXP-008 is the completed same-V08 frozen-feature reference. EXP-010 opens only the final ResNet block at a very low Stage 2 LR.',
    '- EXP-009 is the completed same-V08 partial-fine-tuning reference. EXP-010 keeps its learning rates and trainable layers while adding train-only lighting/framing augmentation.')
report = report.replace(
    '- Dropout {HEAD_DROPOUT}, weight decay {WEIGHT_DECAY}, batch {BATCH_SIZE}.',
    '- Dropout {HEAD_DROPOUT}, weight decay {WEIGHT_DECAY}, batch {BATCH_SIZE}. Train-only brightness 0.12 and translation 0.05 supplement the original augmentation.')
report = report.replace(
    '- Training curves and stage boundaries: `figures/panels/training_curves_5seeds_panel.png`.',
    '- Primary matched inference curves from step 0: `figures/panels/training_curves_5seeds_panel.png`; original online epoch curves: `figures/panels/online_epoch_curves_5seeds_panel.png`.')
report = report.replace(
    '- Matched inference curves, including within epoch 1: `figures/panels/matched_inference_curves_5seeds_panel.png` and `aggregate/matched_inference_curves_5seeds.csv`.',
    '- Matched Train/Val and real_ subgroup curves within epoch 1: `figures/panels/matched_inference_curves_5seeds_panel.png` and `aggregate/matched_inference_curves_5seeds.csv`. The real_ group has only 35 images.')
report = report.replace(
    '- Diagnostics: pre-fit and first-batch scores, fixed 500-image Train probe, matched Train/Val inference curves from step 0, Stage 2 loss rebound, and pHash candidate Val subgroup. These never select a checkpoint.',
    '- Diagnostics: pre-fit and first-batch scores, fixed 500-image Train probe, matched Train/Val and real_ inference curves from step 0, Stage 2 loss rebound, and pHash candidate contact sheets. These never select a checkpoint.')
put(24, report)

put(25, """## Interpretation and completion

- The main chart starts at optimizer step 0 and uses identical inference
  conditions for Train and Validation probes. The 009 evidence already showed
  a gradual rise within epoch 1. An epoch-end score is not an initial score.
- The 35 `real_` Validation files are a small filename-defined subgroup.
  Track their accuracy separately, but do not select hyperparameters, a
  checkpoint or a seed from this subgroup or historical holdout.
- The only training-recipe change from 009 is train-only lighting/framing
  augmentation. It addresses plausible camera/domain variation but may not
  improve independent field performance. Report the measured result.
- Review the 51 pHash contact-sheet pairs manually for shared leaf/plant/
  capture sessions. Exact-byte checks alone do not rule out group leakage.
- Five seeds, Validation lock, Final Test, historical holdout, model package
  and both ZIPs must complete before claiming a 010 result. A fresh blind
  field dataset remains necessary for generalization evidence.
""")

# Apply the same white-to-blue matrix style to Validation, Final Test and
# historical holdout, including the independently exported holdout helper.
matrix_colors = '["#ffffff", "#eaf4ff", "#bfdbfe", "#60a5fa", "#1d4ed8"]'
swap(2, '["#ffffff", "#dbeafe", "#93c5fd"]', matrix_colors)
put(2, get(2).replace(
    'plt.rcParams.update({',
    '''def matrix_text_color(value, upper):
    """Readable annotations on white and deep-blue cells."""
    return "#ffffff" if float(value) / max(float(upper), 1.0) >= 0.68 else "#0f172a"

plt.rcParams.update({''', 1))

chart = get(15)
old = 'ha="center", va="center", color="black")'
new = ('ha="center", va="center",\n'
       '                         color=matrix_text_color(matrix[row, column],\n'
       '                             1 if suffix else matrix.max()))')
if chart.count(old) != 1:
    raise ValueError("Per-seed Validation matrix annotation changed")
put(15, chart.replace(old, new))

for index in (16, 18):
    chart = get(index)
    old = 'ha="center", va="center", color="black", fontsize=8)'
    new = ('ha="center", va="center",\n'
           '                         color=matrix_text_color(matrix[row, col],\n'
           '                             1 if normalized else matrix.max()), fontsize=8)')
    if chart.count(old) != 1:
        raise ValueError(f"Cell {index} aggregate matrix annotation changed")
    put(index, chart.replace(old, new))

holdout = get(20)
old_colors = '["#ffffff", "#dbeafe", "#93c5fd"]'
if holdout.count(old_colors) != 1:
    raise ValueError("Historical holdout palette changed")
holdout = holdout.replace(old_colors, matrix_colors)
holdout = holdout.replace(
    '\n\ndef holdout_sha256(path):',
    '''\n\ndef holdout_matrix_text_color(value, upper):
    return "#ffffff" if float(value) / max(float(upper), 1.0) >= 0.68 else "#0f172a"

def holdout_sha256(path):''', 1)
old = 'ax.text(col, row, text, ha="center", va="center", color="black")'
new = ('ax.text(col, row, text, ha="center", va="center",\n'
       '                        color=holdout_matrix_text_color(values[row, col],\n'
       '                            1 if normalized else matrix.max()))')
if holdout.count(old) != 1:
    raise ValueError("Per-seed holdout matrix annotation changed")
holdout = holdout.replace(old, new)
old = 'ha="center", va="center", color="black", fontsize=8)'
new = ('ha="center", va="center",\n'
       '                         color=holdout_matrix_text_color(matrix[row, col],\n'
       '                             1 if normalized else matrix.max()), fontsize=8)')
if holdout.count(old) != 1:
    raise ValueError("Aggregate holdout matrix annotation changed")
put(20, holdout.replace(old, new))

# Full Validation is measured once before the first update. Its actual value
# belongs at x=0 on the epoch plot; epoch 1 follows all optimizer steps.
analysis = get(16)
old = '''    stage1, stage2 = histories[seed]
    boundary = len(stage1)
    for row, metric in enumerate(("accuracy", "loss")):
        ax = axes[row, column]
'''
new = '''    stage1, stage2 = histories[seed]
    pre_fit = json.loads((OUT_DIR / f"seed_{seed}" / "pre_fit_validation.json").read_text())
    boundary = len(stage1)
    for row, metric in enumerate(("accuracy", "loss")):
        ax = axes[row, column]
        ax.scatter([0], [pre_fit[metric]], marker="o", s=46, color="#ea580c",
                   zorder=5, label="Full Val before fit" if row == 0 else None)
        ax.plot([0, stage1["epoch"].iloc[0]],
                [pre_fit[metric], stage1[f"val_{metric}"].iloc[0]],
                color="#ea580c", alpha=0.9)
        ax.axvline(1, color="#64748b", linestyle=":", alpha=0.45)
'''
if analysis.count(old) != 1:
    raise ValueError("Full Validation epoch plot changed")
analysis = analysis.replace(old, new)
analysis = analysis.replace(
    'fig.suptitle("Stage boundary: dashed; learning-rate reductions: dotted green")',
    'fig.suptitle("Measured full Validation at step 0 and epoch end; epoch 1 after all batches")')
old = '''    x = part["global_step"] / steps_per_epoch
    boundary = int(part[part["stage"] == "stage1"]["epoch"].max())
'''
new = '''    x = part["global_step"] / steps_per_epoch
    pre_fit = json.loads((OUT_DIR / f"seed_{seed}" / "pre_fit_validation.json").read_text())
    boundary = int(part[part["stage"] == "stage1"]["epoch"].max())
'''
if analysis.count(old) != 1:
    raise ValueError("Primary curve seed loop changed")
analysis = analysis.replace(old, new)
old = '''        if metric == "accuracy":
            ax.plot(x, part["real_val_accuracy"], marker=".", linestyle=":",
                    label="real_ Val (35 images)")
            ax.set_ylim(0, 1)
'''
new = '''        if metric == "accuracy":
            ax.plot(x, part["real_val_accuracy"], marker=".", linestyle=":",
                    label="real_ Val (35 images)")
            ax.set_ylim(0, 1)
        ax.scatter([0], [pre_fit[metric]], marker="x", s=65, color="#ea580c",
                   zorder=6, label="Full Val before fit" if metric == "accuracy" else None)
        ax.axvline(1, color="#64748b", linestyle=":", alpha=0.45)
'''
if analysis.count(old) != 1:
    raise ValueError("Primary matched curve changed")
analysis = analysis.replace(old, new)
analysis = analysis.replace(
    'fig.suptitle("Matched inference probes from step 0; Stage 2 boundary dashed")',
    'fig.suptitle("Measured pre-fit and matched probes; epoch 1 is after all batches")')
put(16, analysis)

put(0, get(0).replace(
    'The primary learning-curve panel now starts at step 0 and pairs Train and\nValidation probes measured in the same inference mode.',
    'The primary learning-curve panel starts at step 0, marks the measured\nfull Validation score before any update, and pairs Train and Validation\nprobes measured in the same inference mode. Epoch 1 is measured after all\ntraining batches.'))
put(25, get(25).replace(
    '- The main chart starts at optimizer step 0 and uses identical inference\n  conditions for Train and Validation probes.',
    '- The main chart starts at optimizer step 0, marks actual full Validation\n  before fit, and uses identical inference conditions for Train and Validation\n  probes. Confusion matrices use a consistent white-to-blue palette.'))
put(24, get(24).replace(
    'Primary matched inference curves from step 0:',
    'Primary matched inference curves from step 0 with measured full Validation pre-fit point:').replace(
    '- Diagnostics: pre-fit and first-batch scores,',
    '- White-to-blue confusion matrices with readable annotations on Validation, Final Test and historical holdout. Diagnostics: pre-fit and first-batch scores,'))

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
