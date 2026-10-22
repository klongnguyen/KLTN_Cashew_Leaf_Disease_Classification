"""Generate Validation-only plots, error examples, and scientific summary."""

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from modeling import CLASSES


def plot_summary(output):
    output = Path(output)
    df = pd.read_csv(output / "aggregate" / "per_class_validation.csv")
    recall = df.groupby(["stage", "class"]).recall.agg(["mean", "std"])
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(CLASSES))
    for offset, stage in ((-.17, "stage1"), (.17, "stage2")):
        part = recall.loc[stage].reindex(CLASSES)
        ax.bar(x + offset, part["mean"], .34, yerr=part["std"], capsize=3, label=stage)
    ax.set_xticks(x, CLASSES, rotation=20)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation recall (mean ± sample SD; 5 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "figures" / "per_class_recall.png", dpi=160)
    plt.close(fig)

    compare = pd.read_csv(output / "aggregate" / "baseline_tiny_vs_small.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric, title in ((axes[0], "macro_f1", "Macro F1"),
                              (axes[1], "disease_to_healthy_count", "Disease → Healthy / 900")):
        part = compare[compare.metric == metric]
        values = [part.tiny_mean.iloc[0], *part.small_mean.tolist()]
        errors = [part.tiny_sample_sd.iloc[0], *part.small_sample_sd.tolist()]
        ax.bar(["Tiny A", "Small S1", "Small S2"], values, yerr=errors, capsize=3)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output / "figures" / "comparison_macro_f1_d2h.png", dpi=160)
    plt.close(fig)


def error_montage(output, dataset_root, lock):
    output, dataset_root = Path(output), Path(dataset_root)
    choice = next(x for x in lock["per_seed_stage_choices"] if x["seed"] == 123)
    errors = pd.read_csv(output / "seed_123" / f"errors_val_{choice['stage']}.csv")
    categories = {
        "disease_to_healthy": errors[errors.disease_to_healthy],
        "anthracnose_leaf_miner": errors[errors.anthracnose_leaf_miner],
        "high_confidence_wrong": errors,
    }
    rows = []
    for category, part in categories.items():
        for item in part.sort_values("confidence", ascending=False).head(8).itertuples(index=False):
            rows.append({"category": category, "relative_path": item.relative_path,
                         "true_class": item.true_class, "predicted_class": item.predicted_class,
                         "confidence": item.confidence})
    pd.DataFrame(rows).to_csv(output / "analysis" / "error_examples_seed123.csv", index=False)
    if not rows:
        return
    fig, axes = plt.subplots(3, 8, figsize=(20, 9))
    for ax in axes.flat:
        ax.axis("off")
    for row_index, (category, part) in enumerate(categories.items()):
        for col_index, item in enumerate(part.sort_values("confidence", ascending=False).head(8).itertuples(index=False)):
            with Image.open(dataset_root / item.relative_path) as image:
                axes[row_index, col_index].imshow(image.convert("RGB"))
            axes[row_index, col_index].set_title(
                f"{item.true_class} → {item.predicted_class}\np={item.confidence:.2f}", fontsize=7)
        axes[row_index, 0].set_ylabel(category, fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "figures" / "validation_error_examples_seed123.png", dpi=130)
    plt.close(fig)


def write_report(output, dataset_root):
    output = Path(output)
    lock = json.loads((output / "selection" / "validation_selection_lock.json").read_text(encoding="utf-8"))
    rows = pd.read_csv(output / "aggregate" / "validation_5seeds.csv")
    compare = pd.read_csv(output / "aggregate" / "baseline_tiny_vs_small.csv")
    plot_summary(output)
    (output / "analysis").mkdir(exist_ok=True)
    error_montage(output, dataset_root, lock)
    lines = ["# EXP-010 — Báo cáo Validation V08", "",
             "## Phương pháp", "",
             "DeiT-Small distilled 224 pretrained ImageNet, 5 logits. Năm seed cố định: 42, 123, 2026, 3407, 7777. "
             "Stage 1 huấn luyện head với backbone đóng băng; Stage 2 bắt đầu từ checkpoint tốt nhất cùng seed, "
             "chỉ mở hai transformer block cuối và head. Cả hai stage chọn epoch bằng Validation loss.", "",
             "Không sử dụng Test V08 hoặc Historical Holdout để chọn kiến trúc, epoch, seed, threshold hay checkpoint. "
             "Các độ lệch chuẩn bên dưới là sample SD giữa 5 seed, không phản ánh 5 bộ dữ liệu độc lập.", "",
             "## Kết quả", "",
             "| Mô hình | Accuracy mean ± SD | Macro F1 mean ± SD | Disease → Healthy mean ± SD |", "|---|---:|---:|---:|"]
    tiny = pd.read_csv(output / "baseline_tiny_validation_008.csv")
    tiny = tiny[tiny.classifier == "A"]
    for label, frame in (("DeiT-Tiny A EXP-008", tiny),
                         ("DeiT-Small Stage 1", rows[rows.stage == "stage1"]),
                         ("DeiT-Small Stage 2", rows[rows.stage == "stage2"])):
        lines.append(f"| {label} | {frame.accuracy.mean():.4f} ± {frame.accuracy.std(ddof=1):.4f} | "
                     f"{frame.macro_f1.mean():.4f} ± {frame.macro_f1.std(ddof=1):.4f} | "
                     f"{frame.disease_to_healthy_count.mean():.2f} ± {frame.disease_to_healthy_count.std(ddof=1):.2f} / 900 |")
    lines += ["", "## Ablation và lỗi ưu tiên", ""]
    lines += ["| Stage | Balanced Acc | Macro Precision | Macro Recall | Log Loss | Healthy Recall | Not-Cashew Recall |",
              "|---|---:|---:|---:|---:|---:|---:|"]
    for stage in ("stage1", "stage2"):
        frame = rows[rows.stage == stage]
        cells = [f"{frame[name].mean():.4f} ± {frame[name].std(ddof=1):.4f}" for name in
                 ("balanced_accuracy", "macro_precision", "macro_recall", "log_loss",
                  "healthy_recall", "not_cashew_leaf_recall")]
        lines.append("| " + stage + " | " + " | ".join(cells) + " |")
    lines += ["", "Mean ± sample SD theo từng lớp:", "",
              "| Stage | Class | Precision | Recall | F1 |", "|---|---|---:|---:|---:|"]
    class_df = pd.read_csv(output / "aggregate" / "per_class_validation.csv")
    for stage in ("stage1", "stage2"):
        for klass in CLASSES:
            part = class_df[(class_df.stage == stage) & (class_df["class"] == klass)]
            vals = [f"{part[name].mean():.4f} ± {part[name].std(ddof=1):.4f}" for name in
                    ("precision", "recall", "f1")]
            lines.append("| " + stage + " | " + klass + " | " + " | ".join(vals) + " |")
    lines += ["", "Thời gian huấn luyện mỗi seed/stage, kích thước checkpoint và latency inference "
              "được ghi trong `aggregate/validation_5seeds.csv` và từng `metrics_stage*.json`. "
              "Latency đo trên runtime thực hiện thí nghiệm, không đại diện cho WebApp.", ""]
    paired = rows.pivot(index="seed", columns="stage", values="macro_f1")
    delta = paired.stage2 - paired.stage1
    lines.append(f"Stage 2 − Stage 1 Macro F1 theo cặp seed: mean {delta.mean():+.4f}, sample SD {delta.std(ddof=1):.4f}. "
                 "Sự khác biệt có thể gồm tác động fine-tuning và biến thiên tối ưu hóa; thí nghiệm này không nhận diện nhân quả riêng lẻ.")
    for seed in (42, 123, 2026, 3407, 7777):
        for stage in ("stage1", "stage2"):
            row = rows[(rows.seed == seed) & (rows.stage == stage)].iloc[0]
            lines.append(f"- Seed {seed}/{stage}: Anthracnose→Leaf Miner {int(row.anthracnose_to_leaf_miner)}, "
                         f"Leaf Miner→Anthracnose {int(row.leaf_miner_to_anthracnose)}, "
                         f"Disease→Healthy {int(row.disease_to_healthy_count)}/900; Macro F1 {row.macro_f1:.4f}.")
    lines += ["", "## Quyết định theo Validation", "",
              f"Trạng thái: **{lock['status']}**. Seed/checkpoint nghiên cứu được chọn: "
              f"{lock['selected'] if lock['selected'] else 'không có seed đạt guardrails'}. "
              "Quyết định này không thay đổi seed 123/A đã khóa của EXP-008.", "",
              "## Hạn chế", "",
              "V08 Validation có 35 ảnh `real_`, chỉ là proxy và không chứng minh độc lập theo lá/cây/buổi chụp. "
              "Audit pHash tìm 226 cặp cross-split có khoảng cách ≤6 cần kiểm tra nguồn/nhóm; chưa xác minh group leakage. "
              "Baseline Tiny lấy từ EXP-008 đã có kết quả Validation; sự khác biệt về capacity và fine-tuning đều hiện diện, "
              "nên so sánh Tiny/Small không cô lập hoàn toàn một yếu tố. Test và Holdout chưa được chạy cho EXP-010.", ""]
    (output / "analysis" / "SCIENTIFIC_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    failure = ["# Validation failure analysis — EXP-010", "",
               "Tất cả ví dụ bên dưới thuộc Validation V08. Không sửa nhãn, ảnh hoặc split dựa trên lỗi.", "",
               "| Seed | Stage | Anthracnose → Leaf Miner | Leaf Miner → Anthracnose | Disease → Healthy / 900 |",
               "|---:|---|---:|---:|---:|"]
    for row in rows.itertuples(index=False):
        failure.append(f"| {row.seed} | {row.stage} | {int(row.anthracnose_to_leaf_miner)} | "
                       f"{int(row.leaf_miner_to_anthracnose)} | {int(row.disease_to_healthy_count)} |")
    failure += ["", "Xem `seed_*/errors_val_stage*.csv` để kiểm tra từng ảnh sai và mức tự tin; "
                "`analysis/error_examples_seed123.csv` cùng `figures/validation_error_examples_seed123.png` "
                "nêu ví dụ seed 123 theo stage đã chọn bằng Validation. "
                "Phân tích này mô tả lỗi, không chứng minh nguyên nhân bệnh học hoặc domain shift.", ""]
    (output / "analysis" / "FAILURE_ANALYSIS.md").write_text("\n".join(failure), encoding="utf-8")
    package = output / "deployment_package"
    package.mkdir(exist_ok=True)
    (package / "label_map.json").write_text(json.dumps({str(i): x for i, x in enumerate(CLASSES)}, indent=2), encoding="utf-8")
    (package / "preprocessing.json").write_text(json.dumps({"input": "RGB float32 0..255",
        "resize": "tf.image.resize bilinear 224x224", "preset_preprocessor_passes": 1,
        "model_output": "5 logits", "probability": "softmax(logits)"}, indent=2), encoding="utf-8")
    if lock["selected"]:
        chosen = lock["selected"]
        source = output / f"seed_{chosen['seed']}" / "checkpoints" / f"{chosen['stage']}_best.keras"
        target = package / "deit_small_selected.keras"
        shutil.copyfile(source, target)
        from dataset_audit import sha_file
        if sha_file(target) != chosen["checkpoint_sha256"]:
            raise RuntimeError("Staged checkpoint SHA mismatch")
        (package / "metadata.json").write_text(json.dumps({"research_only": True,
            "validation_status": lock["status"], "seed": chosen["seed"], "stage": chosen["stage"],
            "checkpoint_sha256": chosen["checkpoint_sha256"], "preset": "deit_small_distilled_patch16_224_imagenet",
            "selection_lock": "selection/validation_selection_lock.json"}, indent=2), encoding="utf-8")
    return lock["status"]
