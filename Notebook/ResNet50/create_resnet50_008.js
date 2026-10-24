// Create a ResNet50 five-seed notebook with the EXP-011 DenseNet reporting style.
// The original ResNet50 notebooks remain available for comparison.
const fs = require('node:fs');
const path = require('node:path');
const zlib = require('node:zlib');

const dir = __dirname;
const activePath = path.join(dir, 'resnet50_5seed.ipynb');
const preservedPath = path.join(dir, 'resnet50_5seed_006.ipynb');
const versionedPath = path.join(dir, 'resnet50_5seed_008.ipynb');
const referencePath = path.join(dir, 'resnet50_5seed_reference_recipe.ipynb');
const densePath = path.join(dir, '..', 'DenseNet121', 'densenet121_5seed_011.ipynb');
if (!fs.existsSync(preservedPath)) fs.copyFileSync(activePath, preservedPath, fs.constants.COPYFILE_EXCL);

const nb = JSON.parse(fs.readFileSync(referencePath, 'utf8'));
const dense = JSON.parse(fs.readFileSync(densePath, 'utf8'));
const get = (book, index) => book.cells[index].source.join('');
const set = (index, value) => {
  nb.cells[index].source = value.match(/.*(?:\n|$)/g).filter(Boolean);
};
const replace = (index, from, to) => {
  const original = get(nb, index);
  if (!original.includes(from)) throw new Error(`Missing anchor in cell ${index}`);
  set(index, original.replace(from, to));
};

for (const cell of nb.cells) {
  const source = cell.source.join('')
    .replaceAll('EXP-RESNET50-SCRATCH-5SEEDS-007', 'EXP-RESNET50-SCRATCH-5SEEDS-008')
    .replaceAll('EXP-007', 'EXP-008')
    .replace('NOTEBOOK_VERSION = "007"', 'NOTEBOOK_VERSION = "008"');
  cell.source = source.match(/.*(?:\n|$)/g).filter(Boolean);
  if (cell.cell_type === 'code') {
    cell.execution_count = null;
    cell.outputs = [];
  }
}

set(0, `# EXP-RESNET50-SCRATCH-5SEEDS-008

Five independent ResNet50 scratch runs on the frozen Cashew_dataV07_clean_val
split: seeds [42, 123, 2026, 3407, 7777]. This experiment tries the
training techniques used in DenseNet121 EXP-011: raw RGB 224x224 input with
model-contained preprocessing, horizontal flip, rotation 0.04, zoom 0.05,
Dropout 0.30, batch 16, Adam 1e-4, ReduceLROnPlateau, EarlyStopping, and
minimum-validation-loss checkpoints. Only the architecture and its required
preprocessing differ. ResNet50 uses its BGR/mean-subtraction preprocessing.

The notebook saves accuracy/loss, Validation macro-F1 and learning-rate curves,
class distribution, per-class F1, confusion matrices, reports, predictions,
error analysis, and later locked-Test results. Raw epoch curves remain visible;
a 3-epoch rolling mean is overlaid for readability only. Confusion matrices
use black text on a white-to-blue scale.

RUN_FINAL_TEST=False for the training run. Enable Test only in a separate run
after the five Validation results are frozen. The FULL and ANALYSIS ZIP files
include a versioned copy of this notebook.
`);

// Use the same display defaults and per-seed trend plots as DenseNet EXP-011.
replace(2, 'print("Imports completed.")',
  `plt.rcParams.update({
    "text.color": "black", "axes.labelcolor": "black",
    "axes.titlecolor": "black", "xtick.color": "black", "ytick.color": "black",
})

print("Imports completed.")`);
replace(4, 'NOTEBOOK_VERSION = "008"',
  'NOTEBOOK_VERSION = "008"\nNOTEBOOK_SOURCE_PATH = None  # Optional path to this exact EXP-008 notebook on Drive.');
replace(4,
  `if RUN_REAL_HOLDOUT:
    raise RuntimeError("Real Holdout is disabled for this EXP-008 training experiment.")`,
  `if RUN_REAL_HOLDOUT and (RUN_TRAINING or not RUN_FINAL_TEST):
    raise RuntimeError("Real holdout requires RUN_TRAINING=False and RUN_FINAL_TEST=True after Validation is frozen.")`);

const denseTrain = get(dense, 19);
const plotStart = '    for metric, filename, ylabel in [';
const plotEnd = '    model.load_weights(best_weights_path)';
const densePlot = denseTrain.slice(denseTrain.indexOf(plotStart), denseTrain.indexOf(plotEnd))
  .replaceAll('DenseNet121', 'ResNet50');
const resTrain = get(nb, 19);
const resStart = resTrain.indexOf(plotStart);
const resEnd = resTrain.indexOf(plotEnd);
if (resStart < 0 || resEnd < resStart) throw new Error('ResNet plot anchors missing');
set(19, resTrain.slice(0, resStart) + densePlot + resTrain.slice(resEnd));

// Validation comparison panels are architecture-neutral.
set(21, get(dense, 21).replaceAll('DenseNet121', 'ResNet50'));

const validationExtras = `
# ------------------------------------------------------------
# C. Validation macro-F1 and learning rate, all five seeds
# ------------------------------------------------------------
fig, axes = plt.subplots(2, len(SEEDS), figsize=(27, 8), sharey="row")
for col, seed in enumerate(SEEDS):
    hist = histories[seed]
    best_epoch = int(hist.sort_values(["val_loss", "epoch"]).iloc[0]["epoch"])
    raw = hist["val_macro_f1"]
    axes[0, col].plot(hist["epoch"], raw, color="#ea580c", alpha=0.28, linewidth=1)
    axes[0, col].plot(hist["epoch"], raw.rolling(3, center=True, min_periods=1).mean(),
                      color="#ea580c", linewidth=2)
    axes[0, col].axvline(best_epoch, color="#2563eb", linestyle="--")
    axes[0, col].set_title(f"Seed {seed}")
    axes[0, col].set_ylim(0, 1)
    axes[0, col].grid(alpha=0.25)
    axes[1, col].plot(hist["epoch"], hist["learning_rate"], color="#2563eb", linewidth=1.8)
    axes[1, col].set_yscale("log")
    axes[1, col].axvline(best_epoch, color="#2563eb", linestyle="--")
    axes[1, col].set_xlabel("Epoch")
    axes[1, col].grid(alpha=0.25)
axes[0, 0].set_ylabel("Validation macro-F1")
axes[1, 0].set_ylabel("Learning rate (log scale)")
fig.suptitle("ResNet50 Scratch - Validation Macro-F1 and Learning Rate", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.96))
diagnostic_panel_path = PANEL_DIR / "validation_f1_lr_5seeds_panel.png"
fig.savefig(diagnostic_panel_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)

# D. Per-class Validation F1 (rows=classes, columns=seeds)
val_class_f1 = np.column_stack([
    pd.read_csv(OUT_DIR / f"seed_{seed}" / "validation" / "classification_report.csv",
                index_col=0).loc[CLASS_NAMES, "f1-score"].to_numpy(dtype=float)
    for seed in SEEDS
])
fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.imshow(val_class_f1, cmap=MATRIX_CMAP, vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(SEEDS)), [str(seed) for seed in SEEDS])
ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
ax.set_xlabel("Seed", color="black")
ax.set_ylabel("True class", color="black")
ax.set_title("ResNet50 - Validation F1 by Class and Seed", color="black")
for row in range(NUM_CLASSES):
    for col in range(len(SEEDS)):
        ax.text(col, row, f"{val_class_f1[row, col]:.2f}", ha="center", va="center",
                color="black", fontweight="bold")
fig.tight_layout()
val_class_panel_path = PANEL_DIR / "validation_per_class_f1_5seeds_panel.png"
fig.savefig(val_class_panel_path, dpi=200, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("Created:", diagnostic_panel_path)
print("Created:", val_class_panel_path)
display(Image.open(diagnostic_panel_path))
display(Image.open(val_class_panel_path))
`;
set(21, get(nb, 21) + validationExtras);

const testExtras = `
# Per-class Test F1 panel uses only the locked final-Test reports.
if RUN_FINAL_TEST:
    test_class_f1 = np.column_stack([
        pd.read_csv(OUT_DIR / f"seed_{seed}" / "test_metrics" / "classification_report.csv",
                    index_col=0).loc[CLASS_NAMES, "f1-score"].to_numpy(dtype=float)
        for seed in SEEDS
    ])
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.imshow(test_class_f1, cmap=MATRIX_CMAP, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(SEEDS)), [str(seed) for seed in SEEDS])
    ax.set_yticks(range(NUM_CLASSES), CLASS_NAMES)
    ax.set_xlabel("Seed", color="black")
    ax.set_ylabel("True class", color="black")
    ax.set_title("ResNet50 - Test F1 by Class and Seed", color="black")
    for row in range(NUM_CLASSES):
        for col in range(len(SEEDS)):
            ax.text(col, row, f"{test_class_f1[row, col]:.2f}", ha="center", va="center",
                    color="black", fontweight="bold")
    fig.tight_layout()
    test_class_panel_path = PANEL_DIR / "test_per_class_f1_5seeds_panel.png"
    fig.savefig(test_class_panel_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Created:", test_class_panel_path)
    display(Image.open(test_class_panel_path))
`;
set(26, get(nb, 26) + testExtras);

replace(34, '- README, manifest and SHA256 checksums.',
  '- README, manifest and SHA256 checksums;\n- notebook/resnet50_5seed_008.ipynb.');
replace(34, '    "## Validation confusion matrices',
  `    "## Validation macro-F1 and learning rate", "",
    "![Validation F1 and LR](./figures/panels/validation_f1_lr_5seeds_panel.png)", "",
    "## Per-class Validation F1", "",
    "![Validation F1 by class](./figures/panels/validation_per_class_f1_5seeds_panel.png)", "",
    "## Validation confusion matrices`);
replace(34, '        "![Normalized Test CM](./figures/panels/test_confusion_matrices_normalized_5seeds_panel.png)", "",',
  `        "![Normalized Test CM](./figures/panels/test_confusion_matrices_normalized_5seeds_panel.png)", "",
        "## Per-class Test F1", "",
        "![Test F1 by class](./figures/panels/test_per_class_f1_5seeds_panel.png)", "",`);
replace(34, '    "## Google Drive archive", "",',
  `    "## Notebook source", "",
    "Both ZIP archives contain notebook/resnet50_5seed_008.ipynb.",
    "Raw epoch values remain visible; a 3-epoch mean is display-only.",
    "Confusion matrices use black text on white-to-blue cells.", "",
    "## Google Drive archive", "",`);
replace(34, '    f"- Initial LR: `{INITIAL_LR}`",',
  `    f"- Initial LR: \u0060{INITIAL_LR}\u0060",
    f"- Head Dropout: \u0060{HEAD_DROPOUT}\u0060",
    f"- Rotation/zoom: \u0060{AUGMENTATION_CONFIG['rotation_factor']}\u0060 / \u0060{AUGMENTATION_CONFIG['zoom_factor']}\u0060",`);

// Reuse the tested notebook-copy and ZIP mechanism from DenseNet EXP-011.
let archive = get(dense, 35)
  .replaceAll('densenet121_5seed_011.ipynb', 'resnet50_5seed_008.ipynb')
  .replaceAll('densenet121_5seed.ipynb', 'resnet50_5seed.ipynb')
  .replaceAll('DenseNet121', 'ResNet50')
  .replaceAll('EXP-011', 'EXP-008')
  .replaceAll('EXP-DENSENET121-SCRATCH-5SEEDS-011', 'EXP-RESNET50-SCRATCH-5SEEDS-008')
  .replaceAll('NOTEBOOK_VERSION = \\"011\\"', 'NOTEBOOK_VERSION = \\"008\\"');
archive = archive.replace(/NOTEBOOK_SOURCE_ZLIB_B64 = "[A-Za-z0-9+/=]+"/,
  'NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_NOTEBOOK_SOURCE__"');
if (!archive.includes('__EMBEDDED_NOTEBOOK_SOURCE__')) throw new Error('Archive payload anchor missing');
set(35, archive);
set(36, get(nb, 36).replace('- [ ] README.',
  '- [ ] README.\n- [ ] notebook/resnet50_5seed_008.ipynb in both ZIP archives.'));

// The portable fallback is an output-free, runnable source snapshot. The
// payload marker is empty there to avoid recursively embedding the notebook.
const snapshot = JSON.parse(JSON.stringify(nb));
snapshot.cells[35].source = get(nb, 35).replace('__EMBEDDED_NOTEBOOK_SOURCE__', '')
  .match(/.*(?:\n|$)/g).filter(Boolean);
const payload = zlib.deflateSync(Buffer.from(JSON.stringify(snapshot), 'utf8')).toString('base64');
set(35, get(nb, 35).replace('__EMBEDDED_NOTEBOOK_SOURCE__', payload));

fs.writeFileSync(activePath, JSON.stringify(nb, null, 1) + '\n', 'utf8');
fs.copyFileSync(activePath, versionedPath);
console.log(`Preserved ResNet50 previous notebook: ${preservedPath}`);
console.log(`Created ResNet50 EXP-008: ${activePath}`);
console.log(`Versioned copy: ${versionedPath}`);
console.log(`Embedded source snapshot: ${payload.length} base64 characters`);
