"""Validation-only ACDRA v3 evaluation on pinned V005/NEG-V03 sources."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import run_acdra_v2 as previous
from src.cashew_detection.postprocess.adaptive_regions_v3 import aggregate_detections
from src.cashew_detection.postprocess.region_metrics_v3 import (
    per_image_metrics_v3, summarize_v3)

EXP = ROOT / "experiments" / "EXP-Y26S-NEG-V03-007-ACDRA-V3"
REPORT = ROOT / "reports" / "EXP-007"


def load_validation():
    return previous.load_split("val")


def evaluate(records, class_names, config, output_dir=None, save_regions=False,
             split_name="val"):
    """Reuse V006's GT calculations, adding V007 metrics without editing V006."""
    old_aggregate = previous.aggregate_detections
    old_metrics = previous.per_image_metrics
    previous.aggregate_detections = aggregate_detections
    previous.per_image_metrics = per_image_metrics_v3
    try:
        summary, rows, region_rows = previous.evaluate(
            records, class_names, config, output_dir=output_dir,
            save_regions=save_regions, split_name=split_name)
    finally:
        previous.aggregate_detections = old_aggregate
        previous.per_image_metrics = old_metrics
    summary.update(summarize_v3(rows))
    summary["dense_ge100"].update(summarize_v3(
        row for row in rows if row["raw_box_count"] >= 100))
    summary["sparse_le10"].update(summarize_v3(
        row for row in rows if row["raw_box_count"] <= 10))
    for name, part in summary["slices"].items():
        part.update(summarize_v3(row for row in rows if row["density_slice"] == name))
    for name, part in summary["class_slices"].items():
        if name == "mixed_disease":
            selected = (row for row in rows if row["mixed_disease"])
        elif name == "negative":
            selected = (row for row in rows if row["is_negative"])
        else:
            selected = (row for row in rows if row["main_class"] == name)
        part.update(summarize_v3(selected))
    if output_dir:
        (Path(output_dir) / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, rows, region_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--config-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("V007 tuning may read Validation only; Test requires owner approval")
    records, names, _ = load_validation()
    config = json.loads(args.config_json.read_text(encoding="utf-8"))
    summary, _, _ = evaluate(records, names, config, args.output_dir)
    print(json.dumps({key: summary[key] for key in (
        "images", "raw_boxes", "regions", "reduction", "region_excess_area_ratio",
        "region_precision_strict", "same_class_overlap_area_ratio", "lost_tp")}, indent=2))


if __name__ == "__main__":
    main()
