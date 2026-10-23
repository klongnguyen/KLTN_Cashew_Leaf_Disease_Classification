"""Generate the output-free EXP-011 Google Colab notebook."""

import json
import textwrap
from pathlib import Path


def markdown(source):
    return {"cell_type": "markdown", "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": textwrap.dedent(source).strip() + "\n"}


cells = [
    markdown("""
    # EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011 — DeiT-Small 30/30

    This notebook prepares and runs a **new** experiment, separate from EXP-010.
    Stage 1 trains the head with a frozen backbone; Stage 2 fine-tunes only
    transformer blocks 11–12 and the head from the best same-seed Stage 1
    checkpoint. The five fixed seeds are 42, 123, 2026, 3407, 7777.
    Both stages have a 30-epoch cap and patience 7. Validation loss selects
    each stage checkpoint; the EXP-010 Validation guardrails select stage/seed.

    Only Train V08 updates weights. Only Validation V08 selects and evaluates.
    Test V08 is read only during ZIP integrity audit; it is never extracted
    for a model or used for inference. Historical Holdout is not read.
    No EXP-011 result is claimed before this notebook actually runs.

    Choose a Colab GPU before running Cell 01. The dataset and result path
    are printed below. If your Google account does not contain the ZIP,
    mount the correct Drive account; this notebook does not download or
    copy the whole source ZIP. Run each cell in order through Cell 13.
    Cell 14 can be run alone later to view finished results.
    """),
    code("""
    # Cell 01 — Frozen protocol and independent path overrides.
    EXPERIMENT_ID = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011"
    SEEDS = [42, 123, 2026, 3407, 7777]
    STAGE_MAX_EPOCHS = (30, 30)
    STAGE_PATIENCE = (7, 7)
    SOURCE_BRANCH = "exp/vit-deitsmall-extended-epochs-011"
    SOURCE_COMMIT = "fbe35d0b26820b2acf55f9d95ac42a302eaef4b2"
    SOURCE_URL = "https://github.com/klongnguyen/KLTN_Cashew_Leaf_Disease_Classification.git"
    ZIP_SHA256 = "75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7"
    DATASET_OVERRIDE = None  # existing V08 ZIP on the mounted Drive
    RESULT_ROOT_OVERRIDE = None  # parent folder of EXP-011
    SOURCE_REPO_OVERRIDE = None  # local EXP-011 source clone, if already present
    assert len(SEEDS) == len(set(SEEDS)) == 5
    """),
    code("""
    # Cell 02 — Mount the account that owns the V08 ZIP and show real paths.
    from google.colab import drive
    from pathlib import Path
    drive.mount("/content/drive")
    drive_root = Path("/content/drive/MyDrive/Cashew_Leaf_model_result")
    dataset_zip = Path(DATASET_OVERRIDE) if DATASET_OVERRIDE else (
        drive_root / "Dataset" / "CashewData_Split (8).zip")
    result_root = Path(RESULT_ROOT_OVERRIDE) if RESULT_ROOT_OVERRIDE else (
        drive_root / "training_results/current_dataset")
    output = result_root / EXPERIMENT_ID
    if not dataset_zip.is_file():
        raise FileNotFoundError(
            f"V08 ZIP missing: {dataset_zip}. Check the mounted Google account, "
            "the Dataset folder, and Drive read permission. Do not use a shortcut "
            "or download a guessed replacement ZIP.")
    print("Dataset ZIP:", dataset_zip)
    print("Result root:", result_root)
    print("EXP-011 output:", output)
    """),
    code("""
    # Cell 03 — Pin source and dependencies; inspect GPU and versions.
    import os, subprocess, sys
    source = Path(SOURCE_REPO_OVERRIDE) if SOURCE_REPO_OVERRIDE else Path("/content/exp011_source")
    if not source.exists():
        subprocess.run(["git", "clone", "--single-branch", "--branch", SOURCE_BRANCH,
                        SOURCE_URL, str(source)], check=True)
    current = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if current != SOURCE_COMMIT:
        subprocess.run(["git", "-C", str(source), "fetch", "origin", SOURCE_BRANCH], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", SOURCE_COMMIT], check=True)
    current = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    assert current == SOURCE_COMMIT, (current, SOURCE_COMMIT)
    frozen = source / "training_results/current_dataset" / EXPERIMENT_ID
    scripts = source / "scripts/vit_deit_small_011"
    assert (scripts / "runner.py").is_file() and (frozen / "experiment_config.json").is_file()
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir",
                    "-r", str(frozen / "requirements.lock.txt")], check=True)
    import tensorflow as tf, keras, keras_hub
    gpu = tf.config.list_physical_devices("GPU")
    if not gpu:
        raise RuntimeError("EXP-011 requires a TensorFlow GPU. Enable Colab GPU.")
    print("Source commit:", current)
    print("GPU:", gpu)
    print("TensorFlow:", tf.__version__, "Keras:", keras.__version__,
          "KerasHub:", keras_hub.__version__)
    """),
    code("""
    # Cell 04 — Freeze inputs on Drive, check V08 ZIP SHA, and stream all later logs.
    import hashlib, json, shutil
    if output.exists() and any(output.iterdir()) and not (output / "experiment_config.json").is_file():
        raise FileExistsError(f"Existing unrecognized EXP-011 folder: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("experiment_config.json", "requirements.lock.txt",
                 "baseline_tiny_validation_008.csv", "baseline_tiny_008_per_seed.json",
                 "CONFIG_DIFF.md", "README.md"):
        src, dst = frozen / name, output / name
        if dst.exists():
            if hashlib.sha256(dst.read_bytes()).digest() != hashlib.sha256(src.read_bytes()).digest():
                raise ValueError(f"Frozen input differs: {dst}")
        else:
            shutil.copyfile(src, dst)
    commit_file = output / "source_git_commit.txt"
    if commit_file.exists() and commit_file.read_text().strip() != SOURCE_COMMIT:
        raise ValueError("Source commit changed between EXP-011 resume attempts")
    commit_file.write_text(SOURCE_COMMIT + "\\n")
    digest = hashlib.sha256()
    with dataset_zip.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest().upper() == ZIP_SHA256, "Wrong V08 ZIP"
    subprocess.run([sys.executable, str(source / "scripts/vit_deit_small/capture_environment.py"),
                    str(output / "environment.json")], check=True)
    print("V08 ZIP SHA256 PASS:", digest.hexdigest().upper())

    def run_logged(script, *arguments):
        log = output / "logs/full_run.log"
        log.parent.mkdir(exist_ok=True)
        env = os.environ.copy()
        env["KAGGLEHUB_CACHE"] = str(output / "_preset_cache")
        command = [sys.executable, str(script), *map(str, arguments)]
        with log.open("a", encoding="utf-8") as sink:
            sink.write("\\nCOMMAND: " + " ".join(command) + "\\n")
            sink.flush()
            process = subprocess.Popen(command, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1, env=env)
            for line in process.stdout:
                print(line, end="", flush=True)
                sink.write(line)
                sink.flush()
            result = process.wait()
        if result:
            raise RuntimeError(f"Process exited {result}; root traceback is above and in {log}")

    def run_exp011(mode, seed=None, with_zip=False):
        args = [mode, "--output", output, "--source-root", source]
        if with_zip:
            args += ["--zip", dataset_zip]
        if seed is not None:
            args += ["--seed", seed]
        run_logged(scripts / "runner.py", *args)
    """),
    markdown("""
    ## Gate 0 — V08 integrity audit

    The ZIP audit checks SHA, CRC, 6906 images, exact duplicates, and pHash
    candidates. Test members are read for integrity only. The training
    dataloader later extracts only Train and Validation.
    """),
    code("""
    # Cell 05 — Audit the existing ZIP; record V08 identity.
    if not (output / "dataset_audit.json").exists():
        run_logged(scripts / "audit_011.py", dataset_zip, output)
    audit = json.loads((output / "dataset_audit.json").read_text(encoding="utf-8"))
    assert audit["experiment_id"] == EXPERIMENT_ID
    assert audit["total_images"] == 6906 and audit["decode_errors"] == 0
    assert audit["cross_split_exact_duplicates"] == 0
    print("Gate 0 PASS:", audit["counts"])
    print("pHash candidates pending source/group review:",
          audit["cross_split_phash_le_6_candidates"])
    """),
    markdown("""
    ## Gate 1 — Pretrained architecture, gradients, and checkpoint smoke

    The separate smoke uses two real Train images. It verifies 5 logits,
    one preprocessing pass, Stage 1 and Stage 2 trainable parameters,
    gradient updates and save/reload parity. It is not a benchmark run.
    """),
    code("""
    # Cell 06 — Extract two Train images to temporary disk and run the shared smoke.
    import pandas as pd, zipfile, tempfile
    frame = pd.read_csv(output / "data_manifest.csv")
    smoke_root = Path(tempfile.gettempdir()) / "exp011_smoke" / "CashewData_Split"
    with zipfile.ZipFile(dataset_zip) as archive:
        for label in (0, 1):
            row = frame[(frame.split == "train") & (frame.label == label)].iloc[0]
            target = smoke_root / row.relative_path
            target.parent.mkdir(parents=True, exist_ok=True)
            raw = archive.read("CashewData_Split/" + row.relative_path)
            assert hashlib.sha256(raw).hexdigest().upper() == row.sha256
            target.write_bytes(raw)
    run_logged(source / "scripts/vit_deit_small/smoke_test.py", smoke_root,
               output / "data_manifest.csv", output, output / "_preset_cache")
    smoke = json.loads((output / "gate1_smoke.json").read_text(encoding="utf-8"))
    assert smoke["status"] == "PASS" and smoke["model_parameters"] == 21668357
    print("Gate 1 PASS; logits=5")
    print("Stage 1 trainable parameters:", smoke["stage1_trainable_parameters"])
    print("Stage 2 trainable parameters:", smoke["stage2_trainable_parameters"])
    """),
    code("""
    # Cell 07 — Check frozen config and simulate interrupted Stage 2 resume.
    run_exp011("verify", with_zip=True)
    run_logged(scripts / "runner.py", "selftest-resume", "--output", output)
    config = json.loads((output / "experiment_config.json").read_text(encoding="utf-8"))
    assert config["seeds"] == SEEDS
    assert (config["stage1"]["max_epochs"], config["stage2"]["max_epochs"]) == STAGE_MAX_EPOCHS
    assert (config["stage1"]["early_stopping_patience"],
            config["stage2"]["early_stopping_patience"]) == STAGE_PATIENCE
    print("EXP-011 30/30, patience 7, five-seed protocol frozen")
    """),
    markdown("""
    ## Gate 2 — Five seeds, two stages

    Run one seed cell at a time. Each cell streams epoch, Validation loss,
    Macro F1, disease-to-healthy count, learning rate, and checkpoint path
    to the notebook and Drive log. On rerun, complete stages are SHA256
    verified and reused. Partial stages are preserved before stage restart;
    a stage restart is not an exact epoch/optimizer-state resume.
    """),
]

for seed in (42, 123, 2026, 3407, 7777):
    cells.append(code(f"""
    # Cell 08 — Seed {seed}: Stage 1 then Stage 2.
    run_exp011("train", seed={seed}, with_zip=True)
    """))

cells += [
    code("""
    # Cell 09 — Read back stage markers, checkpoint SHA256 and per-seed metrics.
    from IPython.display import display
    sys.path.insert(0, str(scripts))
    import runner as exp011_runner
    config_sha = exp011_runner.sha_file(output / "experiment_config.json")
    rows = []
    for seed in SEEDS:
        seed_dir = output / f"seed_{seed}"
        assert (seed_dir / "seed_complete.json").is_file()
        for stage in ("stage1", "stage2"):
            assert exp011_runner.validate_stage(seed_dir, stage, config_sha)
            metric = json.loads((seed_dir / f"metrics_{stage}.json").read_text())
            rows.append({"seed": seed, "stage": stage, "best_epoch": metric["best_epoch"],
                         "epochs_ran": metric["epochs_ran"],
                         "val_macro_f1": metric["macro_f1"],
                         "D2H": metric["disease_to_healthy_count"]})
    display(pd.DataFrame(rows))
    """),
    markdown("""
    ## Gate 3 — Validation selection, comparisons, figures and reports

    The locked EXP-010 guardrails and tie rules determine the research
    checkpoint. Tiny A EXP-008 and EXP-010 comparisons read their existing
    Validation results only. A NOT PROMOTED result is valid.
    """),
    code("""
    # Cell 10 — Aggregate 5 seeds and lock Validation-only selection.
    run_exp011("aggregate")
    from IPython.display import display, Markdown, Image
    display(pd.read_csv(output / "aggregate/validation_mean_std.csv"))
    display(pd.read_csv(output / "aggregate/comparison_exp008_exp010_exp011.csv"))
    lock = json.loads((output / "selection/validation_selection_lock.json").read_text())
    print("Validation selection:", lock["status"], lock["selected"])
    """),
    code("""
    # Cell 11 — Generate and display all ten figures from real Validation data.
    run_exp011("plots", with_zip=True)
    chart_names = (
        "accuracy_loss_curves_5seeds.png", "macro_f1_curves_5seeds.png",
        "confusion_matrices_5seeds.png", "confusion_matrices_normalized_5seeds.png",
        "per_class_precision_recall_f1.png", "macro_f1_by_seed.png",
        "disease_to_healthy_by_seed.png", "comparison_exp008_exp010_exp011.png",
        "best_epoch_by_seed.png", "validation_error_examples_seed123.png")
    for name in chart_names:
        path = output / "figures" / name
        if not path.is_file():
            raise FileNotFoundError(f"Expected figure missing: {path}")
        print(name)
        display(Image(filename=str(path)))
    """),
    code("""
    # Cell 12 — Write and display scientific and failure analysis.
    run_exp011("report")
    display(Markdown((output / "analysis/SCIENTIFIC_REPORT.md").read_text(encoding="utf-8")))
    display(Markdown((output / "analysis/FAILURE_ANALYSIS.md").read_text(encoding="utf-8")))
    """),
    markdown("""
    ## Gate 4 — Verified analysis package

    The small ANALYSIS ZIP contains source metadata, metrics, predictions,
    errors, figures and reports. It excludes Keras checkpoints, the source
    ZIP and extracted images. The selected checkpoint is never deployed.
    """),
    code("""
    # Cell 13 — Package, CRC-check and hash the analysis artifact.
    run_exp011("export")
    archive = output / "packages" / f"{EXPERIMENT_ID}_ANALYSIS.zip"
    checksum = output / "packages" / f"{EXPERIMENT_ID}_ANALYSIS.sha256"
    assert archive.is_file() and checksum.is_file()
    digest = hashlib.sha256(archive.read_bytes()).hexdigest().upper()
    assert checksum.read_text().split()[0] == digest
    print("ANALYSIS ZIP:", archive)
    print("SHA256:", digest)
    print("Gate status:", (output / "gate_status.json").read_text())
    """),
    markdown("""
    ## Cell 14 — VIEW RESULTS ONLY

    This cell can run by itself in a clean Colab runtime. It mounts Drive
    if needed, reads existing CSV/JSON/PNG files, and never clones source,
    initializes a model or calls training.
    """),
    code("""
    # Cell 14 — Independent result viewer; no training dependencies.
    from pathlib import Path
    import json
    import pandas as pd
    from IPython.display import display, Markdown, Image
    from google.colab import drive
    mount = Path("/content/drive/MyDrive")
    if not mount.is_dir():
        drive.mount("/content/drive")
    view_root = mount / "Cashew_Leaf_model_result/training_results/current_dataset"
    view_output = view_root / "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011"
    if not (view_output / "selection/validation_selection_lock.json").is_file():
        raise FileNotFoundError(f"No completed EXP-011 Validation results at {view_output}")
    display(pd.read_csv(view_output / "aggregate/validation_mean_std.csv"))
    display(pd.read_csv(view_output / "aggregate/comparison_exp008_exp010_exp011.csv"))
    display(Markdown((view_output / "analysis/SCIENTIFIC_REPORT.md").read_text(encoding="utf-8")))
    for image_path in sorted((view_output / "figures").glob("*.png")):
        print(image_path.name)
        display(Image(filename=str(image_path)))
    print("Selection:", json.loads(
        (view_output / "selection/validation_selection_lock.json").read_text())["status"])
    """),
]

for index, cell in enumerate(cells):
    cell["id"] = f"exp011-{index:02d}"

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": "vit_deit_small_transfer_5seeds_011.ipynb"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

target = Path(__file__).with_name("vit_deit_small_transfer_5seeds_011.ipynb")
target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(target)
