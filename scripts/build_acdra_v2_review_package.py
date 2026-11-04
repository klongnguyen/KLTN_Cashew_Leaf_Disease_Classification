"""Audit saved EXP-006 Validation artifacts and package them for visual review."""

import csv
import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

import yaml
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cashew_detection.postprocess.adaptive_regions_v2 import DEFAULT_CONFIG
from src.cashew_detection.postprocess.region_metrics import summarize_tp_composition

EXP = ROOT / "experiments" / "EXP-Y26S-NEG-V03-006-ACDRA-V2"
REPORT = ROOT / "reports" / "EXP-006"
RESULTS = EXP / "results"
POINTS = ("baseline_A", "conservative", "balanced", "aggressive")
REPORT_NAME = "VALIDATION_REVIEW_2026-11-03.md"
PACKAGE_NAME = "EXP006_VALIDATION_REVIEW_2026-11-03.zip"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pct(value):
    return f"{100 * value:.2f}%"


def render_pareto_review(rows, operating):
    """Plot three two-objective projections, marking all operating points."""
    panels = (("region_excess_area_ratio", "Excess area ratio", False),
              ("region_precision", "Region precision", True),
              ("region_display_occupancy", "Display occupancy", False))
    selected = {value["trial_id"]: name for name, value in operating.items()}
    colors = {"A": "#374151", "B": "#2563eb", "C": "#d97706"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), sharey=True)
    for ax, (field, label, maximize_x) in zip(axes, panels):
        for variant in ("A", "B", "C"):
            subset = [row for row in rows if row["variant"] == variant]
            ax.scatter([float(row[field]) for row in subset],
                       [float(row["dense_ge100.reduction"]) for row in subset],
                       c=colors[variant], alpha=.55, s=38, label=variant)
        frontier = []
        for row in rows:
            x, y = float(row[field]), float(row["dense_ge100.reduction"])
            dominated = any(
                float(other["dense_ge100.reduction"]) >= y and
                (float(other[field]) >= x if maximize_x else float(other[field]) <= x) and
                (float(other["dense_ge100.reduction"]) > y or
                 float(other[field]) != x)
                for other in rows if other is not row)
            if not dominated:
                frontier.append(row)
        frontier.sort(key=lambda row: float(row[field]))
        ax.plot([float(row[field]) for row in frontier],
                [float(row["dense_ge100.reduction"]) for row in frontier],
                color="#111827", linewidth=1.2, alpha=.8, label="Pareto 2D")
        for row in rows:
            if row["trial_id"] not in selected:
                continue
            x, y = float(row[field]), float(row["dense_ge100.reduction"])
            ax.scatter([x], [y], s=105, facecolors="none", edgecolors="#111827",
                       linewidths=1.8, zorder=5)
            ax.annotate(selected[row["trial_id"]], (x, y), xytext=(4, 7),
                        textcoords="offset points", fontsize=8)
        ax.set_xlabel(label)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Reduction on images with ≥100 raw boxes")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("EXP-006 Validation: 19 valid trials and selected operating points")
    fig.tight_layout()
    path = REPORT / "pareto_review_2026-11-03.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def audit_point(point):
    directory = RESULTS / point
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    per_image = read_csv(directory / "per_image.csv")
    region_rows = read_csv(directory / "region_statistics.csv")
    by_image = {row["image_id"]: row for row in per_image}
    assert len(by_image) == len(per_image) == summary["images"] == 951
    count = Counter(row["image_id"] for row in region_rows)
    tp = Counter()
    for row in region_rows:
        tp[row["image_id"]] += int(row["tp_members"])
    for image_id, row in by_image.items():
        assert count[image_id] == int(row["region_count"])
        assert tp[image_id] == int(row["raw_tp"])
    assert len(region_rows) == summary["regions"]
    raw_fp = sum(int(row["raw_fp"]) for row in per_image)
    assert sum(int(row["n_detections"]) for row in region_rows) == summary["raw_boxes"]
    assert sum(int(row["n_detections"]) - int(row["tp_members"])
               for row in region_rows) == raw_fp
    composition = summarize_tp_composition(region_rows, raw_fp)
    assert composition["regions"] == summary["regions"]
    dense_ids = {image_id for image_id, row in by_image.items()
                 if int(row["raw_box_count"]) >= 100}
    mixed_ids = {image_id for image_id, row in by_image.items()
                 if row["mixed_disease"].lower() == "true"}
    def subset(ids):
        selected = [row for row in region_rows if row["image_id"] in ids]
        fp_count = sum(int(by_image[image_id]["raw_fp"]) for image_id in ids)
        return summarize_tp_composition(selected, fp_count)
    by_class = {}
    for class_name in ("anthracnose", "leaf_miner", "red_rust"):
        selected = [row for row in region_rows if row["class_name"] == class_name]
        fp_count = sum(int(row["n_detections"]) - int(row["tp_members"])
                       for row in selected)
        by_class[class_name] = summarize_tp_composition(selected, fp_count)
    return {"summary": summary, "composition": composition,
            "dense_composition": subset(dense_ids),
            "mixed_composition": subset(mixed_ids), "by_class": by_class}


def main():
    locked_path = EXP / "config_locked.yaml"
    locked = yaml.safe_load(locked_path.read_text(encoding="utf-8"))
    assert locked["approved_point"] == "balanced"
    assert sha256(locked_path) == (EXP / "config_locked.sha256").read_text().strip()
    assert sha256(REPORT / "validation_tuning.csv") == locked["validation_tuning_sha256"]
    assert sha256(REPORT / "visual_gallery_manifest.csv") == locked["gallery_manifest_sha256"]
    tuning = read_csv(REPORT / "validation_tuning.csv")
    assert len(tuning) == len({row["trial_id"] for row in tuning})
    valid = [row for row in tuning if row["valid_for_selection"].lower() == "true"]
    operating = yaml.safe_load((EXP / "operating_points.yaml").read_text(encoding="utf-8"))
    assert operating["balanced"]["trial_id"] == locked["trial_id"]
    pareto_review = render_pareto_review(valid, operating)
    points = {point: audit_point(point) for point in POINTS}
    confidence = read_csv(REPORT / "confidence_statistics.csv")

    lines = ["# Hồ sơ đánh giá EXP-006 trên Validation", "",
             "Các kết quả dưới đây lấy từ artifact Validation đã lưu. Việc bổ sung chỉ số không mở lại Test, không đổi trial, cấu hình khóa hoặc `validation_tuning.csv`.", "",
             "## Cấu hình đã khóa", "",
             f"Điểm **balanced**, trial `{locked['trial_id']}`. SHA-256 `config_locked.yaml`: `{sha256(locked_path)}`.", "",
             "| Tham số đang dùng trong B | Giá trị |", "|---|---:|"]
    effective = {**DEFAULT_CONFIG, **locked["candidate_config"]}
    for key in ("k", "cluster_gap_ratio", "overlap_iou", "containment_ratio",
                "min_fill_ratio", "min_group_members", "max_region_area_ratio",
                "max_region_span_ratio"):
        lines.append(f"| `{key}` | {effective[key]} |")
    lines += ["", "`τ_fill` không tham gia biến thể B; biến thể C mạnh tay dùng `τ_fill=0.15`. Khoảng cách B dùng `max(0.02 × cạnh ngắn ảnh, 0.5 × kích thước lesion nhỏ hơn trong cặp)`.", "",
              "## Các trial và ba điểm vận hành", "",
              f"`validation_tuning.csv` chứa **{len(tuning)} trial**: " +
              ", ".join(f"{variant}={sum(row['variant'] == variant for row in tuning)}"
                        for variant in ("A", "B", "C")) +
              f". {len(valid)} trial hợp lệ để chọn; {len(tuning)-len(valid)} trial C từ phiên bản chia cụm cũ được giữ lại nhưng đánh dấu không hợp lệ.", "",
              "| Điểm | Trial | Giảm ảnh ≥100 box | Excess area | Precision vùng | Nhiễm GT khác class | Giảm ảnh ≤10 box |",
              "|---|---|---:|---:|---:|---:|---:|"]
    for point in POINTS:
        s = points[point]["summary"]
        trial = "A_v005" if point == "baseline_A" else operating[point]["trial_id"]
        lines.append(f"| {point} | `{trial}` | {pct(s['dense_ge100']['reduction'])} | "
                     f"{pct(s['region_excess_area_ratio'])} | {pct(s['region_precision'])} | "
                     f"{pct(s['other_class_contamination'])} | {pct(s['sparse_le10']['reduction'])} |")
    lines += ["", "Điểm mạnh tay đạt mức giảm 50% trên ảnh dày ở Validation, nhưng excess area là 46,47%, so với 28,97% của balanced. Ba đồ thị Pareto dùng 19 trial hợp lệ:", "",
              "- [Ba mặt cắt Pareto, đánh dấu ba điểm vận hành](pareto_review_2026-11-03.png)",
              "- [Giảm vùng và excess area](pareto_excess_area.png)",
              "- [Giảm vùng và precision](pareto_region_precision.png)",
              "- [Giảm vùng và display occupancy](pareto_occupancy.png)", "",
              "## Hai chỉ số bổ sung", "",
              "TP raw được ghép một-một theo confidence giảm dần, cùng class và IoU ≥0,5. "
              "Tỷ lệ TP đa số = số region có `2 × TP members ≥ n_detections` chia tổng region. "
              "FP bị hấp thụ = tổng số thành viên FP nằm trong region có ít nhất một TP. "
              "Tỷ lệ FP bị hấp thụ chia cho tổng raw FP. Region đơn lẻ được tính trong tỷ lệ toàn bộ; "
              "cột `n≥2` tách riêng các region thực sự gộp.", "",
              "| Điểm | Region TP đa số | TP đa số trong region n≥2 | Raw FP bị hấp thụ | Tỷ lệ raw FP | Region trộn TP+FP |",
              "|---|---:|---:|---:|---:|---:|"]
    for point in POINTS:
        c = points[point]["composition"]
        lines.append(f"| {point} | {c['tp_majority_regions']:,}/{c['regions']:,} ({pct(c['tp_majority_region_share'])}) | "
                     f"{c['aggregated_tp_majority_regions']:,}/{c['aggregated_regions']:,} "
                     f"({pct(c['aggregated_tp_majority_share'])}) | {c['absorbed_raw_fp_boxes']:,} | "
                     f"{pct(c['absorbed_raw_fp_share'])} | {c['regions_mixing_tp_fp']:,} |")
    lines += ["", "FP bị hấp thụ không biến thành TP của detector; chỉ không còn khung riêng trên màn hình. "
              "Metric này phản ánh khả năng che lỗi khi hiển thị và cần đọc cùng precision, excess area và ảnh.", "",
              "### Ảnh dày và ảnh nhiều bệnh", "",
              "| Điểm | Nhóm | Region TP đa số | Raw FP bị hấp thụ / tổng raw FP |",
              "|---|---|---:|---:|"]
    for point in POINTS:
        for label, key in (("≥100 raw box", "dense_composition"),
                           ("GT nhiều bệnh", "mixed_composition")):
            c = points[point][key]
            lines.append(f"| {point} | {label} | {pct(c['tp_majority_region_share'])} | "
                         f"{c['absorbed_raw_fp_boxes']:,}/{c['raw_fp_boxes']:,} "
                         f"({pct(c['absorbed_raw_fp_share'])}) |")
    lines += ["", "## Theo predicted class và ảnh nhiều bệnh", "",
              "| Điểm | Class | Giảm số vùng | TP đa số | FP bị hấp thụ / raw FP |",
              "|---|---|---:|---:|---:|"]
    for point in POINTS:
        s = points[point]["summary"]
        for class_name in ("anthracnose", "leaf_miner", "red_rust"):
            c = points[point]["by_class"][class_name]
            reduction = s["predicted_class_reduction"][class_name]["reduction"]
            lines.append(f"| {point} | {class_name} | {pct(reduction)} | "
                         f"{pct(c['tp_majority_region_share'])} | "
                         f"{c['absorbed_raw_fp_boxes']:,}/{c['raw_fp_boxes']:,} "
                         f"({pct(c['absorbed_raw_fp_share'])}) |")
    lines += ["", "Nhóm GT nhiều bệnh có 503 ảnh. Số liệu reduction và excess area theo nhóm/class chính "
              "nằm trong `EXP006_FINAL_REPORT.md`; các cột class trên đây phân theo **predicted class của region**.", "",
              "## Thống kê confidence trên region n≥2", "",
              "Các score là thống kê confidence thành viên, không phải xác suất region. "
              "Bảng lấy thống kê PR-AUC tốt nhất cho từng target trong `confidence_statistics.csv`.", "",
              "| Điểm | Target | Score | ROC-AUC | PR-AUC |",
              "|---|---|---|---:|---:|"]
    for point in POINTS[1:]:
        for target in ("has_tp", "tp_member_share_ge_half"):
            subset = [row for row in confidence if row["operating_point"] == point and row["target"] == target]
            assert subset
            winner = max(subset, key=lambda row: float(row["pr_auc"]))
            lines.append(f"| {point} | {target} | {winner['statistic']} | "
                         f"{float(winner['roc_auc']):.3f} | {float(winner['pr_auc']):.3f} |")
    lines += ["", "## Ảnh Original → Raw → ACDRA", "",
              "Gallery có 20 ảnh cố định cho mỗi tổ hợp của 4 nhóm (sparse, moderate, dense, mixed) "
              "và 3 điểm vận hành, tổng cộng 240 ảnh. Mỗi JPG đã ghép ba khung để đối chiếu trực tiếp.", ""]
    gallery_files = []
    for group in ("sparse", "moderate", "dense", "mixed"):
        for point in POINTS[1:]:
            files = sorted((REPORT / "visual_gallery" / group / point).glob("*.jpg"))
            assert len(files) == 20, (group, point, len(files))
            gallery_files.extend(files)
            sample = files[0].relative_to(REPORT).as_posix()
            lines.append(f"- {group} / {point}: [ảnh mẫu]({sample}), 20 ảnh trong thư mục cùng tên.")
    lines += ["", "Các ảnh cho thấy một số vùng của điểm mạnh tay bao phủ cả khoảng nền giữa các đốm. "
              "Vì vậy mức giảm khung 52,35% cần được đánh giá cùng excess area và ảnh từng trường hợp. "
              "Balanced đã được khóa và Test đã đánh giá một lần; việc đổi điểm sau khi xem Test "
              "sẽ là một thí nghiệm mới, không phải điều chỉnh EXP-006 hiện tại.", ""]
    report_path = REPORT / REPORT_NAME
    report_path.write_text("\n".join(lines), encoding="utf-8")
    package_path = REPORT / PACKAGE_NAME
    inputs = [REPORT / "validation_tuning.csv", REPORT / "pareto_excess_area.png",
              REPORT / "pareto_region_precision.png", REPORT / "pareto_occupancy.png",
              pareto_review,
              REPORT / "visual_gallery_manifest.csv", REPORT / "confidence_statistics.csv",
              REPORT / "EXP006_FINAL_REPORT.md", report_path,
              EXP / "config_locked.yaml", EXP / "operating_points.yaml", *gallery_files]
    with zipfile.ZipFile(package_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for source in inputs:
            arcname = (source.relative_to(REPORT).as_posix() if source.is_relative_to(REPORT)
                       else source.name)
            archive.write(source, arcname)
    with zipfile.ZipFile(package_path) as archive:
        assert archive.testzip() is None
        assert len(archive.namelist()) == len(inputs)
    print(json.dumps({"package": str(package_path), "files": len(inputs),
                      "images": len(gallery_files), "trials": len(tuning),
                      "valid_trials": len(valid), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
