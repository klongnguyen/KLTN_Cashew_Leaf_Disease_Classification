"""Generate the standalone Colab notebook from the checked-in EXP-014 implementation."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts/vit_deit_small_014/experiment.py"
TARGET = Path(__file__).with_name("vit_deit_small_transfer_5seeds_014_256px.ipynb")
cells = []


def add(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)


add("markdown", """# EXP-014 · DeiT-Small distilled · V11 · 256×256 · 5 seeds

Fresh ImageNet supervised transfer, two stages per seed. Stage 1 trains two new 5-class heads; Stage 2 opens only the last two transformer blocks and the heads. The class and distillation logits are averaged by the pretrained model implementation. Selection and all plots use **Validation only**. Test is read solely for source ZIP integrity; Holdout is never opened.

**Protocol revision from EXP-011:** KerasHub's documented classifier uses the CLS token only. This notebook uses the official Facebook DeiT-Small distilled two-head checkpoint through Hugging Face Transformers so that 256×256 inputs can use the documented `interpolate_pos_encoding=True` path. The source artifact revision and SHA256 are recorded at runtime. Results are pending until all gates and GPU training run.

Run cells 01–16 in order on Colab GPU. Cell 17 reopens saved results without training. Cell 18 is optional Train/Validation image inference.
""")
add("markdown", "## 01 · Frozen configuration")
add("code", '''from pathlib import Path
import os, json, hashlib, subprocess
EXPERIMENT_ID = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-014-256PX"
SEEDS = [42, 123, 2026, 3407, 7777]
CLASS_NAMES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
DATASET_ZIP_OVERRIDE = None  # Set a verified Drive path if necessary.
RESULT_ROOT_OVERRIDE = None  # Set a Drive result root if necessary.
SOURCE_GIT_COMMIT = None     # Optional SHA of the notebook source commit.
CONFIG = {
    "experiment_id": EXPERIMENT_ID, "dataset_version": "Cashew_dataV11",
    "dataset_zip_sha256": "5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49",
    "manifest_sha256": "91d8279239eaad2007b652f1859667359d0cf01f94af917dd27b0a934a7b7e75",
    "class_names": CLASS_NAMES, "split_counts": {"train": 4916, "val": 1447, "test": 685},
    "seeds": SEEDS, "image_size": 256, "patch_size": 16, "batch_size": 16,
    "pretrained_source": "facebook/deit-small-distilled-patch16-224",
    "framework": "PyTorch + Hugging Face Transformers; dual-head distilled protocol revision",
    "interpolation": "Transformers interpolate_pos_encoding=True; patch grid bicubic; special tokens retained",
    "stage1": {"max_epochs": 30, "lr": 1e-4, "patience": 7, "trainable": "two 5-class heads"},
    "stage2": {"max_epochs": 30, "lr": 1e-5, "minimum_lr": 1e-6, "warmup_epochs": 3,
               "patience": 7, "clipnorm": 1.0, "trainable": "last two encoder blocks and two heads"},
    "optimizer": "AdamW", "weight_decay": 1e-4,
    "augmentation": {"horizontal_flip": True, "rotation_factor": 0.08, "zoom_factor": 0.10,
                     "contrast_factor": 0.10, "brightness_factor": 0.08, "translation": False},
    "test_inference": False, "holdout_inference": False,
    "source_git_commit": SOURCE_GIT_COMMIT,
}
print(json.dumps(CONFIG, indent=2, ensure_ascii=False))
''')
add("markdown", "## 02 · Mount Drive and resolve paths")
add("code", '''from google.colab import drive
drive.mount("/content/drive")
drive_root = Path("/content/drive/MyDrive/Cashew_Leaf_model_result")
DATASET_ZIP = Path(DATASET_ZIP_OVERRIDE) if DATASET_ZIP_OVERRIDE else drive_root / "Dataset/CashewData_Split (11).zip"
RESULT_ROOT = Path(RESULT_ROOT_OVERRIDE) if RESULT_ROOT_OVERRIDE else drive_root / "training_results/current_dataset"
OUT_DIR = RESULT_ROOT / EXPERIMENT_ID
if not DATASET_ZIP.is_file():
    raise FileNotFoundError(f"Verify Dataset path after mounting Drive: {DATASET_ZIP}")
if OUT_DIR.exists() and (OUT_DIR / "experiment_config.json").is_file():
    existing = json.loads((OUT_DIR / "experiment_config.json").read_text(encoding="utf-8"))
    if existing != CONFIG:
        raise RuntimeError(f"Existing experiment config differs; no overwrite: {OUT_DIR}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "experiment_config.json").write_text(json.dumps(CONFIG, indent=2, ensure_ascii=False) + "\\n", encoding="utf-8")
(OUT_DIR / "source_git_commit.txt").write_text((SOURCE_GIT_COMMIT or "UNSET") + "\\n", encoding="utf-8")
CONFIG_SHA = hashlib.sha256((OUT_DIR / "experiment_config.json").read_bytes()).hexdigest()
print("Dataset:", DATASET_ZIP, "\\nResults:", OUT_DIR, "\\nConfig SHA256:", CONFIG_SHA)
''')
add("markdown", "## 03 · Runtime and pinned Python dependencies")
add("code", '''%pip -q install "transformers==4.57.1" "huggingface_hub==0.36.0" "safetensors==0.6.2" "scipy==1.16.2" "scikit-learn==1.7.2" "pandas==2.2.3" "matplotlib==3.10.0"
import torch, transformers, torchvision, pandas, scipy, sklearn, safetensors, platform
if not torch.cuda.is_available():
    raise RuntimeError("EXP-014 requires a Colab CUDA GPU. Training is blocked.")
torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda")
environment = {"python": platform.python_version(), "torch": torch.__version__,
               "torchvision": torchvision.__version__, "transformers": transformers.__version__,
               "pandas": pandas.__version__, "scipy": scipy.__version__, "sklearn": sklearn.__version__,
               "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
               "determinism": "warn_only; GPU kernels may remain nondeterministic"}
(OUT_DIR / "environment.json").write_text(json.dumps(environment, indent=2) + "\\n")
with open(OUT_DIR / "requirements.lock.txt", "w") as stream:
    subprocess.run(["python", "-m", "pip", "freeze"], stdout=stream, check=True)
print(environment)
''')
add("markdown", "## 04 · Implementation and Gate 0 dataset integrity")
add("code", SOURCE.read_text(encoding="utf-8") + "\n")
add("code", '''MANIFEST, GATE0 = audit_archive(DATASET_ZIP, OUT_DIR)
print(GATE0)
DATA_ROOT = extract_train_val(DATASET_ZIP, MANIFEST, Path("/content/exp014_train_val"))
print("Train/Validation extracted:", DATA_ROOT)
''')
add("markdown", "## 05 · V11 dataset distribution and pHash review queue")
add("code", '''from IPython.display import display
distribution = MANIFEST.groupby(["class", "split"]).size().unstack(fill_value=0).reindex(CLASS_NAMES)
display(distribution)
print("real_ filename counts (proxy only):", MANIFEST.groupby("split").is_real_filename.sum().to_dict())
print("pHash candidates for manual review:", GATE0["phash_cross_split_candidates"])
ax = distribution[["train", "val", "test"]].plot.bar(figsize=(9, 4), title="V11 image counts by class and split")
ax.set(ylabel="Images", xlabel="Class"); ax.figure.tight_layout()
(OUT_DIR / "figures").mkdir(exist_ok=True)
ax.figure.savefig(OUT_DIR / "figures/dataset_distribution.png", dpi=160)
display(ax.figure)
''')
add("markdown", "## 06 · Train augmentation and Validation preprocessing")
add("code", '''from PIL import Image
import matplotlib.pyplot as plt
train_example = MANIFEST[MANIFEST.split.eq("train")].iloc[0]
with Image.open(DATA_ROOT / train_example.relative_path) as image:
    image = image.convert("RGB")
    raw = transform(False)(image)
    aug = transform(True)(image)
print("Model input tensors:", tuple(raw.shape), tuple(aug.shape), "float32 normalized once")
assert tuple(raw.shape) == (3, 256, 256) and raw.dtype == torch.float32
fig, axes = plt.subplots(1, 2, figsize=(8, 4))
for ax, tensor, title in zip(axes, (raw, aug), ("Validation preprocessing", "Train augmentation")):
    pixels = tensor * torch.tensor(STD)[:, None, None] + torch.tensor(MEAN)[:, None, None]
    ax.imshow(pixels.permute(1, 2, 0).clamp(0, 1)); ax.set_title(title); ax.axis("off")
display(fig); plt.close(fig)
''')
add("markdown", "## 07 · Download exact pretrained source and inspect 256-token path")
add("code", '''SNAPSHOT, SOURCE_EVIDENCE = source_snapshot(OUT_DIR)
print(SOURCE_EVIDENCE)
probe = new_model(SNAPSHOT, SEEDS[0])
print("Position embedding:", tuple(probe.deit.embeddings.position_embeddings.shape))
print("Encoder blocks:", len(probe.deit.encoder.layer), "hidden size:", probe.config.hidden_size)
del probe
''')
add("markdown", "## 08 · Gate 1 forward, gradients and save/reload smoke")
add("code", '''TRAIN_LOADER, VAL_LOADER, VAL_FRAME = loaders(MANIFEST, DATA_ROOT, SEEDS[0])
GATE1 = smoke(SNAPSHOT, TRAIN_LOADER, OUT_DIR, DEVICE)
if GATE1["status"] != "PASS":
    raise RuntimeError(f"Gate 1 failed: {GATE1}")
display(GATE1)
''')
add("markdown", "## 09 · Five independent seeds, two stages")
add("code", '''import contextlib, traceback, sys
class Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for stream in self.streams: stream.write(data); stream.flush()
    def flush(self):
        for stream in self.streams: stream.flush()
(OUT_DIR / "logs").mkdir(exist_ok=True)
with open(OUT_DIR / "logs/full_run.log", "a", encoding="utf-8") as log:
    try:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)):
            if not (GATE0["status"].startswith("PASS") and GATE1["status"] == "PASS"):
                raise RuntimeError("Gate 0/1 did not pass")
            for seed in SEEDS:
                run_seed(seed, SNAPSHOT, MANIFEST, DATA_ROOT, OUT_DIR, DEVICE, CONFIG_SHA)
    except Exception:
        traceback.print_exc(file=log)
        raise
''')
add("markdown", "## 10–11 · Validation metrics, five-seed aggregation and lock")
add("code", '''SELECTED, SUMMARY, LOCK = aggregate(OUT_DIR)
display(SELECTED[["seed", "stage", "val_loss", "accuracy", "macro_f1", "healthy_precision", "healthy_recall", "healthy_f1", "disease_to_healthy_count", "best_epoch"]])
display(SUMMARY)
print("Validation-only selection:", LOCK)
''')
add("markdown", "## 12–13 · Learning curves, confusion matrices, Healthy errors")
add("code", '''figures(OUT_DIR, SELECTED)
healthy = SELECTED[["seed", "healthy_precision", "healthy_recall", "healthy_f1", "disease_to_healthy_count", "disease_to_healthy_rate", "ood_to_healthy_count"]]
healthy.to_csv(OUT_DIR / "aggregate/healthy_by_seed.csv", index=False)
display(healthy)
error_rows = []
for _, chosen in SELECTED.iterrows():
    pred = pd.read_csv(OUT_DIR / f"seed_{int(chosen.seed)}" / f"predictions_val_{chosen.stage}.csv")
    error_rows.append(pred.loc[~pred.correct, ["image_id", "true_label", "predicted_label", "confidence"]].assign(seed=int(chosen.seed)))
errors = pd.concat(error_rows, ignore_index=True)
consensus = errors.groupby("image_id").agg(error_seeds=("seed", "nunique"), true_label=("true_label", "first"), max_confidence=("confidence", "max")).reset_index()
consensus.to_csv(OUT_DIR / "aggregate/validation_error_consensus.csv", index=False)
(OUT_DIR / "analysis").mkdir(exist_ok=True)
consensus[consensus.error_seeds >= 4].assign(status="HUMAN_LABEL_REVIEW_ONLY").to_csv(OUT_DIR / "analysis/LABEL_AUDIT_CANDIDATES.csv", index=False)
display(consensus[consensus.error_seeds >= 4].sort_values(["error_seeds", "max_confidence"], ascending=False).head(20))
''')
add("markdown", "## 14 · Comparison scope")
add("code", '''print("Compare only machine-readable Validation V11/256 artifacts after verifying dataset manifest and class mapping.")
print("DenseNet121 EXP-024 and ResNet50 EXP-025: supervised ImageNet CNN references.")
print("DINOv2 EXP-013: self-supervised pretraining; a broader reference, not matched pretraining.")
print("No baseline metrics are inserted from README text without machine-readable verification.")
''')
add("markdown", "## 15 · Reports and gate status")
add("code", '''(OUT_DIR / "analysis").mkdir(exist_ok=True)
report = f"""# EXP-014 scientific report

Dataset: V11 Split (11), ZIP SHA256 `{ZIP_SHA}`, manifest SHA256 `{MANIFEST_SHA}`. Train {COUNTS['train']}, Validation {COUNTS['val']}; Test integrity only. Holdout not opened.

Architecture: DeiT-Small distilled two-head, Facebook ImageNet checkpoint at revision `{SOURCE_EVIDENCE['revision']}`. 256×256 inputs, 16×16 patch grid, 256 patch tokens plus CLS and DIST. Positional interpolation verified in Gate 1.

Five-seed Validation mean Macro F1: {SUMMARY.set_index('metric').loc['macro_f1', 'mean']:.5f}; sample SD: {SUMMARY.set_index('metric').loc['macro_f1', 'sample_sd']:.5f}. Selected seed {LOCK['seed']} {LOCK['stage']} using Validation loss and tie rules. This is not a WebApp promotion.

Limitations: One fixed split, possible cross-split pHash candidates await visual/source review. `real_` is a filename proxy. Different augmentation, framework and pretrained artifact from EXP-011; not a one-factor ablation. GPU determinism is best effort. No Test/Holdout performance is claimed.
"""
(OUT_DIR / "analysis/SCIENTIFIC_REPORT.md").write_text(report, encoding="utf-8")
(OUT_DIR / "analysis/FAILURE_ANALYSIS.md").write_text("# Validation errors\\n\\nSee aggregate/validation_error_consensus.csv and per-seed predictions. Images with errors in at least four seeds require manual review; no labels were changed.\\n", encoding="utf-8")
(OUT_DIR / "analysis/METHOD_AND_LEAKAGE_REVIEW.md").write_text("# Method and leakage review\\n\\nTest ZIP members were read only for hash/CRC during Gate 0; no Test loader or inference was created. Holdout was never opened. pHash candidates are not proof of same leaf/tree/session.\\n", encoding="utf-8")
write_json(OUT_DIR / "gate_status.json", {"gate0": GATE0["status"], "gate1": GATE1["status"], "gate2": "PASS", "gate3": "PASS", "gate4": "PENDING_EXPORT", "test": "NOT_RUN", "holdout": "NOT_RUN"})
print(report)
''')
add("markdown", "## 16 · Analysis ZIP and checksums")
add("code", '''ZIP_PATH, ZIP_DIGEST = export_analysis(OUT_DIR)
status = json.loads((OUT_DIR / "gate_status.json").read_text())
status["gate4"] = "PASS"
write_json(OUT_DIR / "gate_status.json", status)
print("Analysis ZIP:", ZIP_PATH, "SHA256:", ZIP_DIGEST)
''')
add("markdown", "## 17 · RESULTS ONLY (run after setup and Drive mount; no training)")
add("code", '''from IPython.display import display, Image as DisplayImage
import pandas as pd
result_root = Path(RESULT_ROOT_OVERRIDE) if RESULT_ROOT_OVERRIDE else Path("/content/drive/MyDrive/Cashew_Leaf_model_result/training_results/current_dataset")
saved = result_root / EXPERIMENT_ID
summary_path = saved / "aggregate/validation_mean_std.csv"
if not summary_path.is_file():
    raise FileNotFoundError("No completed five-seed result to display")
display(pd.read_csv(summary_path))
for name in ("learning_curves_5seeds.png", "confusion_matrices_5seeds.png"):
    path = saved / "figures" / name
    if path.is_file(): display(DisplayImage(filename=str(path)))
''')
add("markdown", "## 18 · Optional single image from Train or Validation")
add("code", '''# Execute only after cells 01–11. Set a relative path listed in the audited Train/Validation manifest.
MANUAL_RELATIVE_PATH = None
if MANUAL_RELATIVE_PATH:
    row = MANIFEST[MANIFEST.relative_path.eq(MANUAL_RELATIVE_PATH)]
    if len(row) != 1 or row.iloc[0]["split"] not in ("train", "val"):
        raise ValueError("Manual inference permits only audited Train/Validation images")
    model = new_model(SNAPSHOT, int(LOCK["seed"]))
    checkpoint = OUT_DIR / f"seed_{LOCK['seed']}/checkpoints/{LOCK['stage']}_best.safetensors"
    model.load_state_dict(load_file(checkpoint)); model.to(DEVICE).eval()
    with Image.open(DATA_ROOT / MANUAL_RELATIVE_PATH) as image:
        x = transform(False)(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): p = torch.softmax(logits(model, x), dim=1)[0].cpu().numpy()
    display(pd.DataFrame({"class": CLASSES, "probability": p}).sort_values("probability", ascending=False))
''')

notebook = {"cells": cells, "metadata": {"colab": {"name": TARGET.name},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET)
