"""Validation-only coarse ACDRA v2 tuning, Pareto plots and candidate points."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

from run_acdra_v2 import EXP, REPORT, evaluate, load_split


def trial_grid():
    # Coarse coverage of every specified value, then two refinements.
    b = [(.5, .10, .35, 2), (1, .15, .45, 3), (1.5, .25, .55, 5),
         (2, .35, .35, 8), (3, .10, .45, 5), (4, .35, .55, 2),
         (2, .25, .55, 5),
         (.5, .10, .35, 3), (.5, .10, .35, 5),
         (1, .10, .35, 5), (.5, .15, .45, 5)]
    c = [(.15, .35, .55, 2), (.25, .25, .45, 3),
         (.35, .15, .35, 5), (.5, .10, .55, 8),
         (.25, .35, .55, 5),
         (.20, .15, .35, 5), (.15, .15, .35, 5)]
    for k, area, span, members in b:
        yield {"variant": "B", "k": k, "max_region_area_ratio": area,
               "max_region_span_ratio": span, "min_group_members": members,
               "min_fill_ratio": .15}
    for fill, area, span, members in c:
        yield {"variant": "C", "tau_fill": fill, "min_fill_ratio": fill,
               "max_region_area_ratio": area, "max_region_span_ratio": span,
               "min_group_members": members}


def flatten(summary, trial_id, config):
    row = {"trial_id": trial_id, "variant": config["variant"],
           "config_json": json.dumps(config, sort_keys=True),
           "valid_for_selection": True}
    for key, value in summary.items():
        if isinstance(value, (int, float, str)) and key not in {"variant"}:
            row[key] = value
    for prefix, data in (("dense_ge100", summary["dense_ge100"]),
                         ("sparse_le10", summary["sparse_le10"])):
        for key, value in data.items():
            row[f"{prefix}.{key}"] = value
    for prefix, slices in (("density", summary["slices"]),
                           ("class", summary["class_slices"])):
        for label, data in slices.items():
            for key, value in data.items():
                row[f"{prefix}.{label}.{key}"] = value
    for label, data in summary["predicted_class_reduction"].items():
        for key, value in data.items():
            row[f"predicted_class.{label}.{key}"] = value
    return row


def write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader(); writer.writerows(rows)


def feasible(row, raw_precision):
    return (int(row["lost_tp"]) == 0 and
            float(row["region_precision"]) >= raw_precision - .02 and
            float(row["sparse_le10.reduction"]) <= .10)


def pareto(rows):
    objectives = lambda row: (float(row["dense_ge100.reduction"]),
                              -float(row["region_excess_area_ratio"]),
                              float(row["region_precision"]),
                              -float(row["region_display_occupancy"]))
    result = []
    for row in rows:
        current = objectives(row)
        if not any(all(a >= b for a, b in zip(objectives(other), current)) and
                   any(a > b for a, b in zip(objectives(other), current))
                   for other in rows if other is not row):
            result.append(row["trial_id"])
    return set(result)


def plot_pareto(rows, frontier):
    plots = (("region_excess_area_ratio", "Excess area ratio", "pareto_excess_area.png"),
             ("region_precision", "Region precision", "pareto_region_precision.png"),
             ("region_display_occupancy", "Display occupancy", "pareto_occupancy.png"))
    for field, label, name in plots:
        fig, ax = plt.subplots(figsize=(8, 6))
        for variant, color in (("A", "black"), ("B", "#2563eb"), ("C", "#e06b1c")):
            subset = [row for row in rows if row["variant"] == variant]
            ax.scatter([float(row[field]) for row in subset],
                       [float(row["dense_ge100.reduction"]) for row in subset],
                       c=color, label=variant, alpha=.8)
        for row in rows:
            if row["trial_id"] in frontier:
                ax.annotate(row["trial_id"], (float(row[field]),
                            float(row["dense_ge100.reduction"])), fontsize=7)
        ax.set(xlabel=label, ylabel="Reduction on images with >=100 raw boxes",
               title="Validation Pareto trade-off (A is frozen V005 baseline)")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(REPORT / name, dpi=150); plt.close(fig)


def select_points(rows, baseline):
    candidates = [row for row in rows if row["variant"] != "A" and
                  feasible(row, float(baseline["raw_precision"]))]
    if len(candidates) < 3:
        raise RuntimeError(f"Only {len(candidates)} feasible candidates; cannot claim three operating points")
    safe = [row for row in candidates
            if float(row["dense_ge100.reduction"]) >=
               float(baseline["dense_ge100"]["reduction"]) - .02 and
               float(row["region_excess_area_ratio"]) <=
               float(baseline["region_excess_area_ratio"]) + .02]
    if not safe:
        raise RuntimeError("No candidate preserves baseline dense reduction with low excess area")
    conservative = min(safe, key=lambda row: (
        float(row["region_excess_area_ratio"]),
        float(row["region_display_occupancy"]),
        -float(row["region_precision"])))
    remaining = [row for row in candidates if row is not conservative]
    aggressive = max(remaining, key=lambda row: float(row["dense_ge100.reduction"]))
    remaining = [row for row in remaining if row is not aggressive]
    balanced = max(remaining, key=lambda row: (
        float(row["dense_ge100.reduction"]) -
        1.0 * max(0, float(row["region_excess_area_ratio"]) -
                    float(baseline["region_excess_area_ratio"])) -
        .5 * max(0, float(row["region_display_occupancy"]) -
                    float(baseline["region_display_occupancy"]))))
    return {"conservative": conservative, "balanced": balanced, "aggressive": aggressive}


def main():
    if (EXP / "config_locked.yaml").exists() or (REPORT / "test_evaluation_completed.json").exists():
        raise RuntimeError("Validation tuning is frozen after owner approval and Test evaluation")
    REPORT.mkdir(parents=True, exist_ok=True)
    baseline_path = EXP / "results" / "baseline_A" / "summary.json"
    if not baseline_path.is_file():
        raise FileNotFoundError("Run the V005 baseline reproduction first")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if (baseline["raw_boxes"], baseline["regions"]) != (56237, 42602):
        raise RuntimeError("Baseline mismatch: stop before tuning")
    csv_path = REPORT / "validation_tuning.csv"
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8"))) if csv_path.exists() else []
    for row in rows:
        if row["trial_id"].startswith("C_"):
            row["valid_for_selection"] = False  # Earlier C revision did not split oversized candidates.
        else:
            row["valid_for_selection"] = True
    if not any(row["trial_id"] == "A_v005" for row in rows):
        rows.insert(0, flatten(baseline, "A_v005", {"variant": "A"}))
        write_csv(csv_path, rows)
    completed = {row["trial_id"] for row in rows}
    records, class_names, _ = load_split("val")
    # The dataset uses not_cashew_* filenames for the not_cashew_leaf slice.
    # Repair only this resume-time slice; aggregate metrics and selections are unchanged.
    negative_leaf = [record for record in records
                     if record["main_class"] == "not_cashew_leaf"]
    if len(negative_leaf) != 20:
        raise RuntimeError(f"Expected 20 not_cashew_leaf Validation images, got {len(negative_leaf)}")
    repaired = False
    for row in rows:
        if int(float(row.get("class.not_cashew_leaf.images", 0) or 0)) == 20:
            continue
        config = json.loads(row["config_json"])
        subset, _, _ = evaluate(negative_leaf, class_names, config)
        for key, value in subset["class_slices"]["not_cashew_leaf"].items():
            row[f"class.not_cashew_leaf.{key}"] = value
        repaired = True
    if repaired:
        write_csv(csv_path, rows)
    for config in trial_grid():
        prefix = "C2" if config["variant"] == "C" else config["variant"]
        trial_id = prefix + "_" + hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()).hexdigest()[:8]
        if trial_id in completed:
            continue
        print("RUN", trial_id, config, flush=True)
        summary, _, _ = evaluate(records, class_names, config)
        row = flatten(summary, trial_id, config)
        rows.append(row)
        write_csv(csv_path, rows)
        completed.add(trial_id)
        print("DONE", trial_id, "dense reduction", row["dense_ge100.reduction"],
              "excess", row["region_excess_area_ratio"],
              "precision", row["region_precision"], flush=True)
    active_rows = [row for row in rows if row["valid_for_selection"] is True or
                   str(row["valid_for_selection"]).lower() == "true"]
    frontier = pareto(active_rows)
    plot_pareto(active_rows, frontier)
    points = select_points(active_rows, baseline)
    frozen = {label: {"trial_id": row["trial_id"],
                       "config": json.loads(row["config_json"])} for label, row in points.items()}
    (EXP / "operating_points.yaml").write_text(yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8")
    lines = ["# Ba điểm vận hành trên Validation (chờ chủ dự án chọn)", "",
             "Điều kiện: lost_tp=0, precision vùng ≥ raw precision − 2 điểm %, giảm ảnh thưa ≤10%.", "",
             "| Điểm | Trial | Giảm ảnh ≥100 box | Excess area | Region precision | Giảm ảnh ≤10 box | Vùng cạnh ngắn <32 px |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for label, row in points.items():
        lines.append(f"| {label} | {row['trial_id']} | {float(row['dense_ge100.reduction']):.3f} | "
                     f"{float(row['region_excess_area_ratio']):.3f} | {float(row['region_precision']):.3f} | "
                     f"{float(row['sparse_le10.reduction']):.3f} | {float(row['region_short_side_lt32_share']):.3f} |")
    lines += ["", "Các điểm Pareto được đánh dấu trên ba biểu đồ trong thư mục báo cáo.",
              "Test chưa được mở; chủ dự án chọn điểm sau khi xem ảnh trước/sau."]
    (REPORT / "operating_points.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Selected candidate trial IDs:", {label: row["trial_id"] for label, row in points.items()})


if __name__ == "__main__":
    main()
