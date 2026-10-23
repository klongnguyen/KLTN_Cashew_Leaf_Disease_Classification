"""Validation-only aggregation, figures, reports, and analysis package."""

import json
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataset_audit import sha_file
from modeling import CLASSES
from report import error_montage
from train import aggregate, extract_train_val, figures

SEEDS = [42, 123, 2026, 3407, 7777]
FIGURES = (
    "accuracy_loss_curves_5seeds.png", "macro_f1_curves_5seeds.png",
    "confusion_matrices_5seeds.png", "confusion_matrices_normalized_5seeds.png",
    "per_class_precision_recall_f1.png", "macro_f1_by_seed.png",
    "disease_to_healthy_by_seed.png", "comparison_exp008_exp010_exp011.png",
    "best_epoch_by_seed.png", "validation_error_examples_seed123.png",
)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, data):
    from runner import write_json as atomic_write
    atomic_write(path, data)


def comparison(output):
    old_dir = output.parent / "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010"
    old_gate = read_json(old_dir / "gate_status.json")
    if old_gate["gate2"] != "PASS" or old_gate["completed_seeds"] != SEEDS:
        raise RuntimeError("EXP-010 five-seed Validation comparison is incomplete")
    ours = pd.read_csv(output / "aggregate/validation_5seeds.csv")
    old = pd.read_csv(old_dir / "aggregate/validation_5seeds.csv")
    tiny = pd.read_csv(output / "baseline_tiny_validation_008.csv")
    tiny = tiny[tiny.classifier == "A"]
    if len(ours) != 10 or len(old) != 10 or len(tiny) != 5:
        raise ValueError("Expected 5 seeds x 2 stages and 5 Tiny A rows")
    metrics = ("accuracy", "macro_f1", "disease_to_healthy_count",
               "healthy_recall", "not_cashew_leaf_recall",
               "anthracnose_to_leaf_miner", "leaf_miner_to_anthracnose",
               "best_epoch", "epochs_ran", "training_seconds")
    joined = ours.merge(old, on=["seed", "stage"], suffixes=("_011", "_010"),
                        validate="one_to_one")
    paired = []
    for row in joined.itertuples(index=False):
        item = {"seed": row.seed, "stage": row.stage}
        for metric in metrics:
            a, b = getattr(row, metric + "_011"), getattr(row, metric + "_010")
            item.update({metric + "_011": a, metric + "_010": b, metric + "_delta": a - b})
        paired.append(item)
    pd.DataFrame(paired).to_csv(output / "aggregate/paired_exp011_minus_exp010.csv", index=False)
    rows = []
    for label, frame in (
        ("Tiny A EXP-008", tiny),
        ("Small EXP-010 S1", old[old.stage == "stage1"]),
        ("Small EXP-010 S2", old[old.stage == "stage2"]),
        ("Small EXP-011 S1", ours[ours.stage == "stage1"]),
        ("Small EXP-011 S2", ours[ours.stage == "stage2"]),
    ):
        for metric in ("accuracy", "macro_f1", "disease_to_healthy_count"):
            rows.append({"model": label, "metric": metric, "mean": float(frame[metric].mean()),
                         "sample_sd": float(frame[metric].std(ddof=1)), "n_seeds": len(frame)})
    pd.DataFrame(rows).to_csv(output / "aggregate/comparison_exp008_exp010_exp011.csv", index=False)
    return ours, old, tiny


def aggregate_results(output, config, config_sha):
    stage_metrics = {seed: {
        stage: read_json(output / f"seed_{seed}/metrics_{stage}.json")
        for stage in ("stage1", "stage2")} for seed in SEEDS}
    lock = aggregate(output, config, stage_metrics, config_sha)
    lock["guardrail_thresholds"] = config["validation_rules"]
    lock["test_used"] = False
    lock["historical_holdout_used"] = False
    write_json(output / "selection/validation_selection_lock.json", lock)
    comparison(output)
    print(f"Validation aggregate: {lock['status']}; selected={lock['selected']}", flush=True)


def plot_results(output, archive):
    if archive is None:
        raise ValueError("--zip is required for the Validation error-image figure")
    ours, old, tiny = comparison(output)
    figdir = output / "figures"
    figdir.mkdir(exist_ok=True)
    figures(output, SEEDS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, stage in zip(axes, ("stage1", "stage2")):
        for seed in SEEDS:
            history = pd.read_csv(output / f"seed_{seed}/history_{stage}.csv")
            ax.plot(history.epoch, history.val_macro_f1, label=str(seed))
        ax.set(title=stage + " Validation Macro F1", xlabel="epoch", ylabel="Macro F1")
        ax.legend(title="seed", fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "macro_f1_curves_5seeds.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 5, figsize=(19, 8))
    for col, seed in enumerate(SEEDS):
        for row, stage in enumerate(("stage1", "stage2")):
            matrix = pd.read_csv(output / f"seed_{seed}/confusion_matrix_val_{stage}.csv",
                                 index_col=0).to_numpy(dtype=float)
            sums = matrix.sum(axis=1, keepdims=True)
            normalized = np.divide(matrix, sums, out=np.zeros_like(matrix), where=sums != 0)
            ax = axes[row, col]
            ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
            ax.set_title(f"{seed} {stage}")
            ax.set_xticks(range(5), CLASSES, rotation=90, fontsize=7)
            ax.set_yticks(range(5), CLASSES, fontsize=7)
    fig.tight_layout()
    fig.savefig(figdir / "confusion_matrices_normalized_5seeds.png", dpi=160)
    plt.close(fig)

    classes = pd.read_csv(output / "aggregate/per_class_validation.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    x = np.arange(5)
    for ax, metric in zip(axes, ("precision", "recall", "f1")):
        for offset, stage in ((-.18, "stage1"), (.18, "stage2")):
            part = classes[classes.stage == stage].groupby("class")[metric].agg(
                ["mean", "std"]).reindex(CLASSES)
            ax.bar(x + offset, part["mean"], .36, yerr=part["std"], capsize=3, label=stage)
        ax.set_xticks(x, CLASSES, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_title(metric + " mean +/- sample SD")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "per_class_precision_recall_f1.png", dpi=160)
    plt.close(fig)

    tiny = tiny.set_index("seed").loc[SEEDS]
    def by_seed(frame, stage, metric):
        return frame[frame.stage == stage].set_index("seed").loc[SEEDS, metric].to_numpy()
    for metric, name in (("macro_f1", "macro_f1_by_seed.png"),
                         ("disease_to_healthy_count", "disease_to_healthy_by_seed.png")):
        fig, ax = plt.subplots(figsize=(11, 5))
        for offset, label, values in (
            (-.27, "Tiny A 008", tiny[metric].to_numpy()),
            (-.09, "Small 010 S2", by_seed(old, "stage2", metric)),
            (.09, "Small 011 S1", by_seed(ours, "stage1", metric)),
            (.27, "Small 011 S2", by_seed(ours, "stage2", metric)),
        ):
            ax.bar(np.arange(5) + offset, values, .18, label=label)
        ax.set_xticks(np.arange(5), SEEDS)
        ax.set(xlabel="seed", ylabel=metric)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / name, dpi=160)
        plt.close(fig)

    compare = pd.read_csv(output / "aggregate/comparison_exp008_exp010_exp011.csv")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, metric in zip(axes, ("accuracy", "macro_f1", "disease_to_healthy_count")):
        part = compare[compare.metric == metric]
        ax.bar(range(5), part["mean"], yerr=part.sample_sd, capsize=3)
        ax.set_xticks(range(5), part.model, rotation=40, ha="right", fontsize=8)
        ax.set_title(metric + " mean +/- sample SD")
    fig.tight_layout()
    fig.savefig(figdir / "comparison_exp008_exp010_exp011.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for ax, stage in zip(axes, ("stage1", "stage2")):
        part = ours[ours.stage == stage].set_index("seed").loc[SEEDS]
        ax.bar(x - .17, part.best_epoch, .34, label="best epoch")
        ax.bar(x + .17, part.epochs_ran, .34, label="epochs ran")
        ax.set_xticks(x, SEEDS)
        ax.set(title=stage, xlabel="seed", ylabel="epochs")
        ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "best_epoch_by_seed.png", dpi=160)
    plt.close(fig)

    manifest = pd.read_csv(output / "data_manifest.csv")
    root = extract_train_val(archive, Path(__import__("tempfile").gettempdir()) /
                             "exp011_train_val/CashewData_Split", manifest)
    (output / "analysis").mkdir(exist_ok=True)
    error_montage(output, root, read_json(output / "selection/validation_selection_lock.json"))
    missing = [name for name in FIGURES if not (figdir / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing required figures: {missing}")
    print(f"Saved {len(FIGURES)} Validation figures", flush=True)


def report_results(output):
    ours, old, tiny = comparison(output)
    lock = read_json(output / "selection/validation_selection_lock.json")
    analysis = output / "analysis"
    analysis.mkdir(exist_ok=True)
    paired = pd.read_csv(output / "aggregate/paired_exp011_minus_exp010.csv")
    compare = pd.read_csv(output / "aggregate/comparison_exp008_exp010_exp011.csv")
    lines = [
        "# EXP-011 — Validation V08 scientific report", "",
        "## Method", "",
        "DeiT-Small distilled ImageNet pretrained, 5 seeds. Stage 1 trains only the head; "
        "Stage 2 fine-tunes the last two encoder blocks and head from the same-seed EXP-011 "
        "best Stage 1 checkpoint. Epoch caps are 30/30 and patience is 7.",
        "Only Train updates weights. Only Validation selects epochs, stages and seeds. "
        "Test V08 is read solely for ZIP integrity. Historical Holdout is not used.", "",
        "The epoch cap, patience and Stage 2 cosine horizon change together. "
        "This paired comparison does not isolate the effect of epoch cap.", "",
        "## Five-seed results", "",
        "| Model | Accuracy mean +/- sample SD | Macro F1 mean +/- sample SD | D2H / 900 mean +/- sample SD |",
        "|---|---:|---:|---:|",
    ]
    for model in compare.model.unique():
        frame = compare[compare.model == model].set_index("metric")
        values = [f"{frame.loc[m, 'mean']:.4f} +/- {frame.loc[m, 'sample_sd']:.4f}"
                  for m in ("accuracy", "macro_f1", "disease_to_healthy_count")]
        lines.append("| " + model + " | " + " | ".join(values) + " |")
    lines += ["", "## Paired seed deltas, EXP-011 minus EXP-010", "",
              "| Stage | Accuracy delta | Macro F1 delta | D2H delta |",
              "|---|---:|---:|---:|"]
    for stage in ("stage1", "stage2"):
        frame = paired[paired.stage == stage]
        values = [f"{frame[m + '_delta'].mean():+.4f} +/- {frame[m + '_delta'].std(ddof=1):.4f}"
                  for m in ("accuracy", "macro_f1", "disease_to_healthy_count")]
        lines.append("| " + stage + " | " + " | ".join(values) + " |")
    lines += ["", "## Validation selection", "",
              f"Status: {lock['status']}. Selected research checkpoint: "
              f"{lock['selected'] if lock['selected'] else 'none met guardrails'}.",
              "The EXP-010 guardrails and tie rules were frozen before this run.", "",
              "## Limits", "",
              "Five seeds share one Validation split; sample SD is seed variation rather than "
              "independent dataset uncertainty. The V08 audit has 226 cross-split pHash "
              "candidates requiring source/group review. Only 35 Validation images have "
              "the real_ prefix. Tiny and Small differ in capacity. No Test/Holdout "
              "inference was run.", ""]
    (analysis / "SCIENTIFIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    failure = ["# EXP-011 Validation failure analysis", "",
               "Use seed error CSVs to inspect individual Validation predictions. "
               "Do not change labels or splits from these observations.", "",
               "| Seed | Stage | Anthracnose to Leaf Miner | Leaf Miner to Anthracnose | D2H / 900 |",
               "|---:|---|---:|---:|---:|"]
    for row in ours.itertuples(index=False):
        failure.append(f"| {row.seed} | {row.stage} | {int(row.anthracnose_to_leaf_miner)} | "
                       f"{int(row.leaf_miner_to_anthracnose)} | {int(row.disease_to_healthy_count)} |")
    (analysis / "FAILURE_ANALYSIS.md").write_text("\n".join(failure) + "\n", encoding="utf-8")
    (analysis / "EXP010_VS_EXP011.md").write_text(
        "# EXP-010 versus EXP-011\n\nSee aggregate/paired_exp011_minus_exp010.csv "
        "for same-seed deltas. Epoch cap, patience and schedule horizon change jointly; "
        "no isolated causal effect is claimed.\n", encoding="utf-8")
    (analysis / "LIMITATIONS.md").write_text(
        "# Limitations\n\n- Five seeds share one Validation split.\n"
        "- Cross-split pHash candidates need source/group review.\n"
        "- Test V08 and Historical Holdout were not used for inference.\n"
        "- The epoch cap effect is not isolated.\n", encoding="utf-8")
    print(f"Reports saved: {analysis}", flush=True)


def export_analysis(output):
    from runner import validate_stage
    config_sha = sha_file(output / "experiment_config.json")
    for seed in SEEDS:
        path = output / f"seed_{seed}"
        if not (validate_stage(path, "stage1", config_sha) and
                validate_stage(path, "stage2", config_sha)):
            raise RuntimeError(f"Seed {seed} is incomplete")
    required = [output / "figures" / name for name in FIGURES]
    required += [output / "analysis" / name for name in
                 ("SCIENTIFIC_REPORT.md", "FAILURE_ANALYSIS.md", "EXP010_VS_EXP011.md",
                  "LIMITATIONS.md", "error_examples_seed123.csv")]
    required += [output / "aggregate" / name for name in
                 ("validation_5seeds.csv", "validation_mean_std.csv",
                  "paired_exp011_minus_exp010.csv", "comparison_exp008_exp010_exp011.csv")]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        raise RuntimeError(f"Analysis package is incomplete: {missing}")
    members = []
    for name in ("README.md", "CONFIG_DIFF.md", "experiment_config.json",
                 "requirements.lock.txt", "dataset_audit.json", "environment.json",
                 "gate1_smoke.json", "source_git_commit.txt"):
        path = output / name
        if path.is_file():
            members.append(path)
    for folder in ("aggregate", "analysis", "figures", "selection"):
        members += [p for p in (output / folder).rglob("*") if p.is_file()]
    for seed in SEEDS:
        members += [p for p in (output / f"seed_{seed}").iterdir()
                    if p.is_file() and p.suffix in (".csv", ".json")]
    members = sorted(set(members))
    if any(p.suffix == ".keras" for p in members):
        raise RuntimeError("Checkpoint included in analysis package")
    package_dir = output / "packages"
    package_dir.mkdir(exist_ok=True)
    stem = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011_ANALYSIS"
    manifest = pd.DataFrame([{"path": p.relative_to(output).as_posix(),
                              "bytes": p.stat().st_size, "sha256": sha_file(p)} for p in members])
    manifest_file = package_dir / "PACKAGE_MANIFEST.csv"
    manifest.to_csv(manifest_file, index=False)
    final = package_dir / (stem + ".zip")
    part = package_dir / (stem + ".part.zip")
    with zipfile.ZipFile(part, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for path in members + [manifest_file]:
            z.write(path, path.relative_to(output).as_posix())
    with zipfile.ZipFile(part) as z:
        if z.testzip() is not None:
            raise RuntimeError("Analysis package CRC check failed")
        expected = {p.relative_to(output).as_posix() for p in members + [manifest_file]}
        if set(z.namelist()) != expected:
            raise RuntimeError("Analysis package member list mismatch")
    part.replace(final)
    digest = sha_file(final)
    (package_dir / (stem + ".sha256")).write_text(
        f"{digest}  {final.name}\n", encoding="utf-8")
    status = read_json(output / "gate_status.json")
    status["gate3"] = "PASS"
    status["gate4"] = "PASS_ANALYSIS_PACKAGE"
    status["validation_status"] = read_json(output / "selection/validation_selection_lock.json")["status"]
    write_json(output / "gate_status.json", status)
    print(f"Verified {final}; SHA256={digest}", flush=True)


def run_results(mode, output, config, config_sha, archive=None):
    output = Path(output)
    if mode == "aggregate":
        aggregate_results(output, config, config_sha)
    elif mode == "plots":
        plot_results(output, archive)
    elif mode == "report":
        report_results(output)
    elif mode == "export":
        export_analysis(output)
    else:
        raise ValueError(mode)
