"""Reproduce V005 and score ACDRA v2 on the pinned YOLO ZIP artifacts."""

import argparse
import csv
import hashlib
import io
import json
import statistics
import sys
import time
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cashew_detection.postprocess.adaptive_regions_v2 import aggregate_detections
from src.cashew_detection.postprocess.region_metrics import per_image_metrics, summarize

EXP = ROOT / "experiments" / "EXP-Y26S-NEG-V03-006-ACDRA-V2"
REPORT = ROOT / "reports" / "EXP-006"
FULL = EXP / "source" / "EXP-Y26S-NEG-V03-005_FULL.zip"
ANALYSIS = EXP / "source" / "EXP-Y26S-NEG-V03-005_ANALYSIS.zip"
DATA = EXP / "source" / "CASHEW_YOLO_MASTER_FINAL_NEG_V03.zip"
FULL_SHA256 = "990df19b094cc14d6e5706183616245dda83d21fac3b52c20f9a2a3f5488c845"
ANALYSIS_SHA256 = "38ed98c637fd822f0ec534aa28a0954efce301e6d50a8802de3d80ef9ef96859"
DATA_SHA256 = "312870474cfbab8c3fc77db7b19e821c1edf34a0787cb45592f00bde841b4dbe"
CHECKPOINT_SHA256 = "12f5717ff7e97c6ffcc3b2ab8c66b869bc8065995507dac3a5dbd95ce7714d19"
DATA_ROOT = "CASHEW_YOLO_MASTER_FINAL_NEG_V03"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sources():
    for path, expected in ((FULL, FULL_SHA256), (ANALYSIS, ANALYSIS_SHA256),
                           (DATA, DATA_SHA256)):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Pinned source missing or SHA256 mismatch: {path}")
    with zipfile.ZipFile(FULL) as archive, zipfile.ZipFile(ANALYSIS) as analysis:
        if hashlib.sha256(archive.read("train/weights/best.pt")).hexdigest() != CHECKPOINT_SHA256:
            raise RuntimeError("V005 checkpoint SHA256 mismatch")
        for member in ("region_aggregation/validation_per_image.csv",
                       "region_aggregation/validation_summary.json"):
            if archive.read(member) != analysis.read(member):
                raise RuntimeError(f"V005 FULL/ANALYSIS mismatch: {member}")


def load_class_names(full):
    config = yaml.safe_load(full.read("data_fixed.yaml"))
    names = config["names"]
    return {i: name for i, name in enumerate(names)} if isinstance(names, list) else {int(k): v for k, v in names.items()}


def _main_class(image_id, class_names):
    name = Path(image_id).stem
    if name.startswith("not_cashew_"):
        return "not_cashew_leaf"
    for label in sorted(list(class_names.values()) + ["healthy", "not_cashew_leaf"], key=len, reverse=True):
        if name.startswith(label + "_") or name.startswith("real_" + label + "_"):
            return label
    return "unknown"


def load_split(split="val"):
    if split != "val":
        raise ValueError("Tuning and baseline may read Validation only; Test needs a locked runner")
    verify_sources()
    records = []
    with zipfile.ZipFile(FULL) as full, zipfile.ZipFile(DATA) as data:
        class_names = load_class_names(full)
        original = json.loads(full.read("region_aggregation/validation_summary.json"))
        previous = {row["image_id"]: row for row in csv.DictReader(io.StringIO(
            full.read("region_aggregation/validation_per_image.csv").decode("utf-8")))}
        for line in full.open("region_aggregation/validation_raw_detections.jsonl"):
            row = json.loads(line)
            image_id = row["image_id"]
            image_member = f"{DATA_ROOT}/images/val/{image_id}"
            with Image.open(io.BytesIO(data.read(image_member))) as image:
                width, height = image.size
            label_member = f"{DATA_ROOT}/labels/val/{Path(image_id).stem}.txt"
            gt = []
            try:
                label_text = data.read(label_member).decode("utf-8")
            except KeyError:
                label_text = ""
            for line in label_text.splitlines():
                if not line.strip():
                    continue
                values = list(map(float, line.split()))
                if len(values) != 5 or int(values[0]) not in class_names:
                    raise ValueError(f"Invalid GT label: {label_member}: {line}")
                class_id, x, y, w, h = values
                gt.append({"class_id": int(class_id), "bbox_xyxy": [
                    (x - w / 2) * width, (y - h / 2) * height,
                    (x + w / 2) * width, (y + h / 2) * height]})
            records.append({"image_id": image_id, "detections": row["detections"],
                            "gt": gt, "image_size": (width, height),
                            "main_class": _main_class(image_id, class_names),
                            "baseline_row": previous[image_id]})
    if len(records) != 951 or sum(len(r["detections"]) for r in records) != 56237:
        raise RuntimeError("V005 Validation JSONL count differs from 951 / 56,237")
    return records, class_names, original


def evaluate(records, class_names, config, output_dir=None, baseline_check=False,
             save_regions=False, split_name="val"):
    output_dir = Path(output_dir) if output_dir else None
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
    rows, region_rows, latency = [], [], []
    raw_by_class, regions_by_class = Counter(), Counter()
    out_file = (output_dir / f"{split_name}_regions.jsonl").open("w", encoding="utf-8") if save_regions else None
    try:
        for record in records:
            raw = record["detections"]
            raw_by_class.update(int(item["class_id"]) for item in raw)
            t0 = time.perf_counter()
            regions = aggregate_detections(raw, record["image_size"], record["image_id"],
                                           class_names, config)
            regions_by_class.update(int(item["class_id"]) for item in regions)
            elapsed = 1000 * (time.perf_counter() - t0)
            latency.append((record["image_id"], elapsed))
            if baseline_check and len(regions) != int(record["baseline_row"]["display_region_count"]):
                raise RuntimeError(f"V005 per-image region count differs: {record['image_id']}")
            if out_file:
                out_file.write(json.dumps({"image_id": record["image_id"], "regions": regions}) + "\n")
            row, tp_ids = per_image_metrics(record["image_id"], raw, regions,
                                            record["gt"], record["image_size"],
                                            record["main_class"])
            row["aggregation_ms"] = elapsed
            rows.append(row)
            for region in regions:
                box = region["bbox_xyxy"]
                members = region["member_detection_ids"]
                tp_count = sum(member in tp_ids for member in members)
                region_rows.append({"image_id": record["image_id"],
                                    "region_id": region["region_id"],
                                    "class_id": region["class_id"],
                                    "class_name": region["class_name"],
                                    "n_detections": len(members), "tp_members": tp_count,
                                    "has_tp": int(tp_count > 0),
                                    "tp_member_share_ge_half": int(tp_count / len(members) >= .5),
                                    "short_side_px": min(box[2]-box[0], box[3]-box[1]),
                                    "long_side_px": max(box[2]-box[0], box[3]-box[1]),
                                    "fill_ratio": region["fill_ratio"],
                                    "other_class_overlap_ratio": region["other_class_overlap_ratio"],
                                    "yolo_conf_max": region["yolo_conf_max"],
                                    "yolo_conf_mean": region["yolo_conf_mean"],
                                    "yolo_conf_topk_mean": region["yolo_conf_topk_mean"]})
    finally:
        if out_file:
            out_file.close()
    summary = summarize(rows)
    latencies = sorted(value for _, value in latency)
    summary.update({"variant": config.get("variant", "B"), "config": config,
                    "latency_p50_ms": statistics.median(latencies),
                    "latency_p95_ms": latencies[int(.95 * (len(latencies) - 1))],
                    "latency_max_ms": latencies[-1],
                    "slowest_image": max(latency, key=lambda pair: pair[1])[0],
                    "dense_ge100": summarize(row for row in rows if row["raw_box_count"] >= 100),
                    "sparse_le10": summarize(row for row in rows if row["raw_box_count"] <= 10),
                    "slices": {key: summarize(row for row in rows if row["density_slice"] == key)
                               for key in ("le5", "6_10", "11_20", "21_40", "41_80", "81_150", "gt150")}})
    summary["class_slices"] = {
        name: summarize(row for row in rows if row["main_class"] == name)
        for name in list(class_names.values()) + ["healthy", "not_cashew_leaf"]}
    summary["class_slices"]["mixed_disease"] = summarize(row for row in rows if row["mixed_disease"])
    summary["class_slices"]["negative"] = summarize(row for row in rows if row["is_negative"])
    summary["predicted_class_reduction"] = {
        class_names[class_id]: {"raw": raw_by_class[class_id], "regions": regions_by_class[class_id],
                                "reduction": 1 - regions_by_class[class_id] / raw_by_class[class_id]
                                if raw_by_class[class_id] else 0.0}
        for class_id in class_names}
    if summary["lost_tp"]:
        raise AssertionError("lost_tp must be zero")
    if output_dir:
        with (output_dir / "per_image.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        with (output_dir / "region_statistics.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(region_rows[0]))
            writer.writeheader(); writer.writerows(region_rows)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary, rows, region_rows


def write_baseline_report(summary, original):
    REPORT.mkdir(parents=True, exist_ok=True)
    if (summary["raw_boxes"], summary["regions"]) != (56237, 42602):
        raise RuntimeError("Stop: V005 baseline is not reproduced")
    report = f"""# Tái hiện baseline V005 trên Validation

Nguồn: ZIP FULL V005 SHA256 `{FULL_SHA256}`; dataset NEG-V03 SHA256 `{DATA_SHA256}`;
checkpoint `best.pt` SHA256 `{CHECKPOINT_SHA256}`. Mapping class đọc từ `data_fixed.yaml`.

Biến thể A chạy lại trực tiếp từ `validation_raw_detections.jsonl`: **{summary['raw_boxes']:,} raw box → {summary['regions']:,} region**, giảm **{100*summary['reduction']:.4f}%**.
Số region khớp `validation_per_image.csv` của V005 tại **951/951 ảnh**.
Định nghĩa TP/FP mới: ghép tham lam theo confidence giảm dần, cùng class, IoU ≥ 0,5;
mỗi GT chỉ được ghép một lần. Region có TP nếu ít nhất một box thành viên là TP.

| Metric | Raw | Region A |
|---|---:|---:|
| GT coverage IoA ≥ 0,5 | {summary['raw_gt_coverage']:.4f} | {summary['region_gt_coverage']:.4f} |
| GT center coverage | {summary['raw_gt_center_coverage']:.4f} | {summary['region_gt_center_coverage']:.4f} |
| Excess area ratio (GT cùng class) | {summary['raw_excess_area_ratio']:.4f} | {summary['region_excess_area_ratio']:.4f} |
| Display occupancy | {summary['raw_display_occupancy']:.4f} | {summary['region_display_occupancy']:.4f} |
| FP/ảnh | {summary['raw_fp_per_image']:.4f} | {summary['fp_regions_per_image']:.4f} |
| Precision | {summary['raw_precision']:.4f} | {summary['region_precision']:.4f} |
| Lost TP | N/A | {summary['lost_tp']} |

Các metric vùng là chẩn đoán hiển thị, không phải AP/mAP detector. Coverage tăng cần đọc cùng excess area và occupancy.
"""
    (REPORT / "baseline_reproduction.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["baseline", "trial"])
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--config-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("Test is forbidden in baseline/tuning; use the locked evaluator")
    records, class_names, original = load_split(args.split)
    config = {"variant": "A"} if args.command == "baseline" else json.loads(args.config_json.read_text())
    output = args.output_dir or (EXP / "results" / "baseline_A" if args.command == "baseline" else None)
    summary, _, _ = evaluate(records, class_names, config, output,
                             baseline_check=args.command == "baseline",
                             save_regions=args.command == "baseline")
    if args.command == "baseline":
        write_baseline_report(summary, original)
        raw_copy = EXP / "results" / "validation_raw_detections.jsonl"
        with zipfile.ZipFile(FULL) as archive:
            raw_bytes = archive.read("region_aggregation/validation_raw_detections.jsonl")
        if raw_copy.exists() and raw_copy.read_bytes() != raw_bytes:
            raise RuntimeError("Standalone raw detections differ from frozen V005 source")
        raw_copy.write_bytes(raw_bytes)
    print(json.dumps({k: summary[k] for k in ("images", "raw_boxes", "regions", "reduction",
                                              "raw_fp_per_image", "fp_regions_per_image",
                                              "region_precision", "lost_tp")}, indent=2))


if __name__ == "__main__":
    main()
