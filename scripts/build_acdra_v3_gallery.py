"""Render fixed V007 Validation comparisons and targeted diagnostic galleries."""

import csv
import io
import json
import random
import sys
import zipfile
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_acdra_v3 import EXP, REPORT, load_validation
from scripts.run_acdra_v2 import DATA, DATA_ROOT

PALETTE = ("#dd6b20", "#2563eb", "#c026d3", "#059669", "#b91c1c")


def draw_comparison(image, raw, regions, names):
    original = image.convert("RGB")
    raw_canvas, region_canvas = original.copy(), original.copy()
    for canvas, items, is_region in ((raw_canvas, raw, False),
                                     (region_canvas, regions, True)):
        draw = ImageDraw.Draw(canvas)
        for item in items:
            box = tuple(item["bbox_xyxy"])
            class_id = int(item["class_id"])
            color = PALETTE[class_id % len(PALETTE)]
            draw.rectangle(box, outline=color, width=2)
            if is_region and item["n_detections"] >= 2:
                label = f"{names[class_id]} ×{item['n_detections']}"
                draw.text((box[0], max(0, box[1] - 12)), label, fill=color)
    header = 24
    output = Image.new("RGB", (3 * original.width, original.height + header), "white")
    painter = ImageDraw.Draw(output)
    for index, frame in enumerate((original, raw_canvas, region_canvas)):
        output.paste(frame, (index * original.width, header))
        painter.text((index * original.width + 8, 5),
                     ("Original", "Raw YOLO", "ACDRA v3")[index], fill="black")
    return output


def main():
    if (EXP / "config_locked_v007.yaml").exists():
        raise RuntimeError("V007 config locked; gallery selection is frozen")
    points = yaml.safe_load((REPORT / "operating_points_v007.yaml").read_text(encoding="utf-8"))
    records, names, _ = load_validation()
    lookup = {record["image_id"]: record for record in records}
    base_groups = {
        "sparse": [r for r in records if len(r["detections"]) <= 10],
        "moderate": [r for r in records if 21 <= len(r["detections"]) <= 80],
        "dense": [r for r in records if len(r["detections"]) >= 100],
        "mixed": [r for r in records if len({g["class_id"] for g in r["gt"]}) > 1],
    }
    fixed = {}
    for index, (group, candidates) in enumerate(base_groups.items()):
        if len(candidates) < 20:
            raise RuntimeError(f"Not enough Validation images in {group}")
        fixed[group] = sorted(random.Random(42 + index).sample(candidates, 20),
                              key=lambda record: record["image_id"])
    manifest = []
    with zipfile.ZipFile(DATA) as archive:
        for point in points:
            output = EXP / "results" / point
            with (output / "val_regions.jsonl").open(encoding="utf-8") as stream:
                region_lookup = {row["image_id"]: row["regions"]
                                 for row in map(json.loads, stream)}
            with (output / "per_image.csv").open(newline="", encoding="utf-8") as stream:
                per_image = list(csv.DictReader(stream))
            changed_ids = [row["image_id"] for row in per_image
                           if row["sparse_changed"] == "True"]
            changed = sorted(random.Random(713).sample(
                changed_ids, min(20, len(changed_ids))))
            worst = [row["image_id"] for row in sorted(
                per_image, key=lambda row: (-float(row["same_class_overlap_area_ratio"]),
                                            row["image_id"]))[:10]]
            groups = {**fixed,
                      "sparse_changed": [lookup[image_id] for image_id in changed],
                      "overlap_worst": [lookup[image_id] for image_id in worst]}
            for group, entries in groups.items():
                folder = REPORT / "visual_gallery" / group / point
                folder.mkdir(parents=True, exist_ok=True)
                for record in entries:
                    image_id = record["image_id"]
                    with Image.open(io.BytesIO(archive.read(
                            f"{DATA_ROOT}/images/val/{image_id}"))) as image:
                        output_image = draw_comparison(image, record["detections"],
                                                       region_lookup[image_id], names)
                    destination = folder / f"{Path(image_id).stem}.jpg"
                    output_image.save(destination, quality=82)
                    manifest.append({"group": group, "point": point,
                                     "image_id": image_id,
                                     "relative_path": destination.relative_to(REPORT).as_posix()})
            print(point, "gallery images", sum(row["point"] == point for row in manifest), flush=True)
    with (REPORT / "visual_gallery_manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(manifest[0]))
        writer.writeheader(); writer.writerows(manifest)


if __name__ == "__main__":
    main()
