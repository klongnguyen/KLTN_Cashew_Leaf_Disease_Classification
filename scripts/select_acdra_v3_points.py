"""Validation-only four-objective Pareto review and proposed operating points."""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_acdra_v3 import EXP, REPORT


def objectives(row):
    return (float(row["dense_ge100.reduction"]),
            -float(row["region_excess_area_ratio"]),
            float(row["region_precision_strict"]),
            -float(row["same_class_overlap_area_ratio"]))


def pareto(rows):
    result = []
    for row in rows:
        current = objectives(row)
        if not any(all(a >= b for a, b in zip(objectives(other), current)) and
                   any(a > b for a, b in zip(objectives(other), current))
                   for other in rows if other is not row):
            result.append(row)
    return result


def select_points(frontier):
    # The 45–70% range is a review focus, not an acceptance constraint.
    candidates = [row for row in frontier if row["variant"] != "A" and
                  .45 <= float(row["dense_ge100.reduction"]) <= .70]
    if len(candidates) < 3:
        raise RuntimeError("Fewer than three nondominated points in the 45–70% review range")
    conservative = min(candidates, key=lambda row: (
        float(row["region_excess_area_ratio"]),
        -float(row["region_precision_strict"])))
    aggressive = max((row for row in candidates if row is not conservative),
                     key=lambda row: float(row["dense_ge100.reduction"]))
    remaining = [row for row in candidates if row not in (conservative, aggressive)]
    # Review point: achieve at least 60% dense reduction at the lowest excess.
    # This is a presentation preference, not a feasibility constraint on trials.
    middle = [row for row in remaining
              if float(row["dense_ge100.reduction"]) >= .60]
    if not middle:
        raise RuntimeError("No nondominated candidate reaches 60% dense reduction")
    balanced = min(middle, key=lambda row: (
        float(row["region_excess_area_ratio"]),
        -float(row["region_precision_strict"])))
    return {"conservative": conservative, "balanced": balanced,
            "aggressive": aggressive}


def plot(rows, frontier, selected):
    panels = (("region_excess_area_ratio", "Excess area ratio", "pareto_excess_area.png"),
              ("region_precision_strict", "Strict region precision", "pareto_strict_precision.png"),
              ("same_class_overlap_area_ratio", "Same-class overlap area ratio",
               "pareto_same_class_overlap.png"))
    colors = {"A": "#374151", "B": "#2563eb", "C": "#d97706", "D": "#059669"}
    selected_ids = {row["trial_id"]: name for name, row in selected.items()}
    frontier_ids = {row["trial_id"] for row in frontier}
    for field, label, filename in panels:
        fig, ax = plt.subplots(figsize=(9, 6))
        for variant in ("A", "B", "C", "D"):
            subset = [row for row in rows if row["variant"] == variant]
            ax.scatter([float(row[field]) for row in subset],
                       [float(row["dense_ge100.reduction"]) for row in subset],
                       c=colors[variant], alpha=.65, s=48, label=variant)
        for row in rows:
            if row["trial_id"] in frontier_ids:
                ax.scatter([float(row[field])],
                           [float(row["dense_ge100.reduction"])],
                           facecolors="none", edgecolors="#111827", s=95,
                           linewidths=1, zorder=4)
            if row["trial_id"] in selected_ids or row["trial_id"] == "B_e6e7d65f":
                name = selected_ids.get(row["trial_id"], "V006 balanced")
                x, y = float(row[field]), float(row["dense_ge100.reduction"])
                ax.annotate(name, (x, y), xytext=(5, 7),
                            textcoords="offset points", fontsize=8)
        ax.set(xlabel=label, ylabel="Reduction on images ≥100 raw boxes",
               title="EXP-007 Validation: four-objective Pareto points (outlined)")
        ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
        fig.savefig(REPORT / filename, dpi=160)
        plt.close(fig)


def main():
    if (EXP / "config_locked_v007.yaml").exists():
        raise RuntimeError("V007 config locked; selection is frozen")
    with (REPORT / "validation_tuning_v007.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) < 10:
        raise RuntimeError("V007 sweep incomplete")
    if any(row["split"] != "val" or int(float(row["lost_tp"])) != 0 for row in rows):
        raise RuntimeError("Trial split or lost_tp violation")
    frontier = pareto(rows)
    selected = select_points(frontier)
    output = {name: {"trial_id": row["trial_id"],
                     "config": json.loads(row["config_json"]),
                     "selection_status": "proposed; owner review pending"}
              for name, row in selected.items()}
    (REPORT / "operating_points_v007.yaml").write_text(
        yaml.safe_dump(output, sort_keys=False), encoding="utf-8")
    with (REPORT / "pareto_frontier_v007.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["trial_id", "variant",
            "dense_ge100.reduction", "region_excess_area_ratio",
            "region_precision_strict", "same_class_overlap_area_ratio",
            "config_sha256"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames}
                         for row in frontier)
    plot(rows, frontier, selected)
    print("frontier", len(frontier), "selected",
          {name: row["trial_id"] for name, row in selected.items()})


if __name__ == "__main__":
    main()
