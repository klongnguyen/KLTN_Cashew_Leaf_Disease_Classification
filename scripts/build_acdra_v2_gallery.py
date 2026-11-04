"""Render fixed Validation galleries and diagnostic tables for three candidates."""

import csv
import io
import json
import random
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw
from sklearn.metrics import average_precision_score, roc_auc_score
import yaml

from run_acdra_v2 import DATA, DATA_ROOT, EXP, REPORT, evaluate, load_split
from src.cashew_detection.postprocess.adaptive_regions_v2 import aggregate_detections

COLORS = {0: "#dd6b20", 1: "#2563eb", 2: "#c026d3"}


def selected_images(records):
    groups = {
        "sparse": [r for r in records if len(r["detections"]) <= 10],
        "moderate": [r for r in records if 21 <= len(r["detections"]) <= 80],
        "dense": [r for r in records if len(r["detections"]) >= 100],
        "mixed": [r for r in records if len({g["class_id"] for g in r["gt"]}) > 1],
    }
    chosen = {}
    for index, (group, candidates) in enumerate(groups.items()):
        if len(candidates) < 20:
            raise RuntimeError(f"Not enough Validation images for {group}: {len(candidates)}")
        chosen[group] = sorted(random.Random(42 + index).sample(candidates, 20),
                               key=lambda r: r["image_id"])
    return chosen


def draw_comparison(image, raw, regions, names):
    original = image.convert("RGB")
    left, right = original.copy(), original.copy()
    for canvas, entries, is_region in ((left, raw, False), (right, regions, True)):
        draw = ImageDraw.Draw(canvas)
        for item in entries:
            box = tuple(item["bbox_xyxy"])
            color = COLORS[int(item["class_id"])]
            draw.rectangle(box, outline=color, width=2)
            if is_region and item["n_detections"] >= 2:
                label = f"{names[int(item['class_id'])]} ×{item['n_detections']}"
                draw.text((box[0], max(0, box[1] - 12)), label, fill=color)
    header_height = 24
    result = Image.new("RGB", (original.width * 3, original.height + header_height), "white")
    header = ImageDraw.Draw(result)
    for index, frame in enumerate((original, left, right)):
        result.paste(frame, (index * original.width, header_height))
        header.text((index * original.width + 8, 5),
                    ("Original", "Raw YOLO", "ACDRA v2")[index], fill="black")
    return result


def confidence_tables(label, region_rows, writer):
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
        truth = [row[target] for row in aggregated]
        for name, fn in features.items():
            scores = [fn(row) for row in aggregated]
            roc = roc_auc_score(truth, scores) if len(set(truth)) == 2 else None
            pr = average_precision_score(truth, scores) if len(set(truth)) == 2 else None
            writer.writerow({"operating_point": label, "target": target, "statistic": name,
                             "regions_n_ge2": len(aggregated), "roc_auc": roc,
                             "pr_auc": pr})


def size_table(label, region_rows, writer):
    by_class = defaultdict(list)
    for row in region_rows:
        by_class[row["class_name"]].append(row)
    for name, rows in by_class.items():
        short = sorted(row["short_side_px"] for row in rows)
        long = sorted(row["long_side_px"] for row in rows)
        writer.writerow({"operating_point": label, "class": name, "regions": len(rows),
                         "short_side_p50": short[len(short) // 2],
                         "short_side_p95": short[int(.95 * (len(short) - 1))],
                         "long_side_p50": long[len(long) // 2],
                         "short_side_lt32_share": sum(x < 32 for x in short) / len(short),
                         "other_class_overlap_ratio_mean": sum(r["other_class_overlap_ratio"] for r in rows) / len(rows)})


def main():
    if (EXP / "config_locked.yaml").exists() or (REPORT / "test_evaluation_completed.json").exists():
        raise RuntimeError("Validation gallery is frozen after owner approval and Test evaluation")
    point_path = EXP / "operating_points.yaml"
    if not point_path.is_file():
        raise FileNotFoundError("Run Validation tuning before galleries")
    points = yaml.safe_load(point_path.read_text(encoding="utf-8"))
    records, names, _ = load_split("val")
    chosen = selected_images(records)
    gallery_root = REPORT / "visual_gallery"
    gallery_root.mkdir(parents=True, exist_ok=True)
    manifest = []
    with (REPORT / "confidence_statistics.csv").open("w", newline="", encoding="utf-8") as cf, (
            REPORT / "region_size_statistics.csv").open("w", newline="", encoding="utf-8") as sf, zipfile.ZipFile(DATA) as archive:
        conf = csv.DictWriter(cf, fieldnames=["operating_point", "target", "statistic", "regions_n_ge2", "roc_auc", "pr_auc"])
        size = csv.DictWriter(sf, fieldnames=["operating_point", "class", "regions", "short_side_p50", "short_side_p95", "long_side_p50", "short_side_lt32_share", "other_class_overlap_ratio_mean"])
        conf.writeheader(); size.writeheader()
        for label, point in points.items():
            config = point["config"]
            output = EXP / "results" / label
            summary, _, region_rows = evaluate(records, names, config, output, save_regions=True)
            confidence_tables(label, region_rows, conf)
            size_table(label, region_rows, size)
            print(label, "regions", summary["regions"], flush=True)
            for group, entries in chosen.items():
                folder = gallery_root / group / label
                folder.mkdir(parents=True, exist_ok=True)
                for record in entries:
                    image_id = record["image_id"]
                    with Image.open(io.BytesIO(archive.read(
                            f"{DATA_ROOT}/images/val/{image_id}"))) as image:
                        regions = aggregate_detections(record["detections"], record["image_size"],
                                                       image_id, names, config)
                        canvas = draw_comparison(image, record["detections"], regions, names)
                    destination = folder / (Path(image_id).stem + ".jpg")
                    canvas.save(destination, quality=82)
                    manifest.append({"group": group, "operating_point": label,
                                     "image_id": image_id,
                                     "relative_path": destination.relative_to(REPORT).as_posix()})
    with (REPORT / "visual_gallery_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["group", "operating_point", "image_id", "relative_path"])
        writer.writeheader(); writer.writerows(manifest)
    print("Gallery images:", len(manifest))


if __name__ == "__main__":
    main()
