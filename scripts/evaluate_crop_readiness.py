"""Measure Validation crop feasibility for V007 candidates without creating crops."""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_acdra_v3 import EXP, REPORT, evaluate, load_validation
from src.cashew_detection.postprocess.geometry import area, intersection_area, iou
from src.cashew_detection.postprocess.region_metrics import match_raw


def crop_window(box, image_size, minimum):
    width, height = image_size
    side = max(box[2] - box[0], box[3] - box[1], minimum)
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    proposed = [cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2]
    clipped = (proposed[0] < 0 or proposed[1] < 0 or
               proposed[2] > width or proposed[3] > height)
    return [max(0, proposed[0]), max(0, proposed[1]),
            min(width, proposed[2]), min(height, proposed[3])], clipped


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def confidence_rows(point, region_rows):
    aggregated = [row for row in region_rows if row["n_detections"] >= 2]
    features = {
        "max": lambda row: row["yolo_conf_max"],
        "mean": lambda row: row["yolo_conf_mean"],
        "top3_mean": lambda row: row["yolo_conf_topk_mean"],
        "n_detections": lambda row: row["n_detections"],
        "fill_ratio": lambda row: row["fill_ratio"],
        "max_times_fill": lambda row: row["yolo_conf_max"] * row["fill_ratio"],
    }
    for target in ("has_tp", "tp_member_share_ge_half"):
        labels = [int(row[target]) for row in aggregated]
        base_rate = sum(labels) / len(labels) if labels else 0.0
        for name, feature in features.items():
            scores = [feature(row) for row in aggregated]
            valid = len(set(labels)) == 2
            yield {"point": point, "target": target, "statistic": name,
                   "regions_n_ge2": len(labels), "base_rate": base_rate,
                   "roc_auc": roc_auc_score(labels, scores) if valid else "N/A",
                   "pr_auc": average_precision_score(labels, scores) if valid else "N/A"}


def measure_point(point, records, names, config):
    output = EXP / "results" / point
    summary, _, region_rows = evaluate(records, names, config, output, save_regions=True)
    tp_by_image = {}
    for record in records:
        tp_by_image[record["image_id"]] = match_raw(record["detections"], record["gt"])[0]
    buckets = defaultdict(lambda: {"short": [], "long": [], "regions": 0,
                                  "raw_boxes": 0, "raw_tp": 0, "gt": 0,
                                  "verifiable_regions": 0, "verifiable_raw_boxes": 0,
                                  "verifiable_raw_tp": 0, "verifiable_gt": 0,
                                  "short_lt32": 0, "short_lt64": 0})
    crops = defaultdict(lambda: Counter())
    region_path = output / "val_regions.jsonl"
    with region_path.open(encoding="utf-8") as stream:
        for record, line in zip(records, stream, strict=True):
            content = json.loads(line)
            assert content["image_id"] == record["image_id"]
            regions = content["regions"]
            raw = record["detections"]
            gt = record["gt"]
            tp_ids = tp_by_image[record["image_id"]]
            for class_id, class_name in names.items():
                class_regions = [region for region in regions
                                 if int(region["class_id"]) == class_id]
                class_raw = [item for item in raw if int(item["class_id"]) == class_id]
                class_gt = [item for item in gt if int(item["class_id"]) == class_id]
                for threshold in (32, 48, 64):
                    bucket = buckets[(class_name, threshold)]
                    bucket["regions"] += len(class_regions)
                    bucket["raw_boxes"] += len(class_raw)
                    bucket["raw_tp"] += sum(str(item["id"]) in tp_ids for item in class_raw)
                    bucket["gt"] += len(class_gt)
                    qualified = []
                    for region in class_regions:
                        box = region["bbox_xyxy"]
                        short, long = sorted((box[2] - box[0], box[3] - box[1]))
                        bucket["short"].append(short)
                        bucket["long"].append(long)
                        bucket["short_lt32"] += short < 32
                        bucket["short_lt64"] += short < 64
                        if region["n_detections"] >= 2 and short >= threshold:
                            qualified.append(region)
                            bucket["verifiable_raw_boxes"] += region["n_detections"]
                            bucket["verifiable_raw_tp"] += sum(
                                str(member) in tp_ids for member in region["member_detection_ids"])
                    bucket["verifiable_regions"] += len(qualified)
                    bucket["verifiable_gt"] += sum(any(
                        intersection_area(target["bbox_xyxy"], region["bbox_xyxy"]) /
                        area(target["bbox_xyxy"]) >= .5 for region in qualified)
                        for target in class_gt)
                for minimum in (64, 96, 128):
                    key = (class_name, minimum)
                    bucket = crops[key]
                    windows = []
                    for region in class_regions:
                        window, clipped = crop_window(region["bbox_xyxy"],
                                                      record["image_size"], minimum)
                        bucket["regions"] += 1
                        bucket["boundary_clipped"] += clipped
                        bucket["other_class_region_inside"] += any(
                            other["class_id"] != class_id and
                            intersection_area(window, other["bbox_xyxy"]) > 0
                            for other in regions)
                        windows.append(window)
                    bucket["same_class_crop_overlap_pairs_iou_gt05"] += sum(
                        iou(a, b) > .5 for position, a in enumerate(windows)
                        for b in windows[position + 1:])
    readiness = []
    for (class_name, threshold), bucket in sorted(buckets.items()):
        shorts = sorted(bucket["short"])
        longs = sorted(bucket["long"])
        n = bucket["regions"]
        readiness.append({"point": point, "class": class_name, "s_min_px": threshold,
                          "regions": n, "raw_boxes": bucket["raw_boxes"],
                          "raw_tp": bucket["raw_tp"], "gt": bucket["gt"],
                          "short_side_p50_px": shorts[n // 2] if n else "N/A",
                          "short_side_p95_px": shorts[int(.95 * (n - 1))] if n else "N/A",
                          "long_side_p50_px": longs[n // 2] if n else "N/A",
                          "short_lt32_share": bucket["short_lt32"] / n if n else 0,
                          "short_lt64_share": bucket["short_lt64"] / n if n else 0,
                          "verifiable_regions": bucket["verifiable_regions"],
                          "verifiable_region_share": bucket["verifiable_regions"] / n if n else 0,
                          "verifiable_raw_boxes": bucket["verifiable_raw_boxes"],
                          "verifiable_raw_box_share": bucket["verifiable_raw_boxes"] /
                              bucket["raw_boxes"] if bucket["raw_boxes"] else 0,
                          "verifiable_raw_tp": bucket["verifiable_raw_tp"],
                          "verifiable_raw_tp_share": bucket["verifiable_raw_tp"] /
                              bucket["raw_tp"] if bucket["raw_tp"] else 0,
                          "verifiable_gt": bucket["verifiable_gt"],
                          "verifiable_gt_share": bucket["verifiable_gt"] /
                              bucket["gt"] if bucket["gt"] else 0})
    crop_rows = []
    for (class_name, minimum), bucket in sorted(crops.items()):
        n = bucket["regions"]
        crop_rows.append({"point": point, "class": class_name, "W_min_px": minimum,
                          "regions": n, "boundary_clipped": bucket["boundary_clipped"],
                          "boundary_clipped_share": bucket["boundary_clipped"] / n if n else 0,
                          "other_class_region_inside": bucket["other_class_region_inside"],
                          "other_class_region_inside_share": bucket["other_class_region_inside"] /
                              n if n else 0,
                          "same_class_crop_overlap_pairs_iou_gt05":
                              bucket["same_class_crop_overlap_pairs_iou_gt05"]})
    return summary, readiness, crop_rows, list(confidence_rows(point, region_rows))


def main():
    if (EXP / "config_locked_v007.yaml").exists():
        raise RuntimeError("V007 config already locked; crop evaluation is frozen")
    points = yaml.safe_load((REPORT / "operating_points_v007.yaml").read_text(encoding="utf-8"))
    records, names, _ = load_validation()
    counts = Counter(tuple(record["image_size"]) for record in records)
    write_csv(REPORT / "image_size_distribution.csv", [
        {"width": width, "height": height, "images": number,
         "share": number / len(records)}
        for (width, height), number in sorted(counts.items())])
    readiness, crops, confidence = [], [], []
    for point, entry in points.items():
        summary, part, crop_part, conf_part = measure_point(
            point, records, names, entry["config"])
        readiness.extend(part); crops.extend(crop_part); confidence.extend(conf_part)
        print(point, "regions", summary["regions"], "lost_tp", summary["lost_tp"], flush=True)
    write_csv(REPORT / "crop_readiness.csv", readiness)
    write_csv(REPORT / "crop_window_readiness.csv", crops)
    write_csv(REPORT / "confidence_statistics_v007.csv", confidence)


if __name__ == "__main__":
    main()
