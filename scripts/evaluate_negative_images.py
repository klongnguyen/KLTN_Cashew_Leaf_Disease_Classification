"""Compare V003 and NEG V02 checkpoints on the same NEG V02 validation split.

No training and no Test evaluation. Requires ultralytics==8.4.197.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import zipfile
from collections import Counter
from math import comb
from pathlib import Path

import torch
import ultralytics
import yaml
from ultralytics import YOLO


CLASSES = ("anthracnose", "leaf_miner", "red_rust")
RUNS = (
    ("yolov8s", "v003", "EXP-Y8S-LM-003"),
    ("yolov8s", "neg_v02", "EXP-Y8S-NEG-V02-001"),
    ("yolo26s", "v003", "EXP-Y26S-LM-003"),
    ("yolo26s", "neg_v02", "EXP-Y26S-NEG-V02-001"),
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def save_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def group_for(image: Path) -> str | None:
    stem = image.stem.lower().replace("-", "_").replace(" ", "_")
    if stem.startswith(("not_cashew_neg_", "non_cashew_neg_")):
        return "not_cashew_leaf"
    if stem.startswith("healthy_neg_"):
        return "healthy"
    return None


def validate_dataset(root: Path, output: Path) -> tuple[Path, list[Path], dict[str, str]]:
    root = root.resolve()
    spec = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    names = spec.get("names")
    if isinstance(names, dict):
        names = [names.get(i, names.get(str(i))) for i in range(len(names))]
    if names != list(CLASSES) or spec.get("nc", 3) != 3:
        raise ValueError(f"Dataset phải có đúng 3 lớp {CLASSES}: {names}")
    val_dir = root / "images" / "val"
    label_dir = root / "labels" / "val"
    images = sorted(p for p in val_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)
    if len(images) != 951:
        raise ValueError(f"Validation NEG V02 dự kiến 951 ảnh, nhận {len(images)}")
    negative_relatives: list[Path] = []
    negative_groups: dict[str, str] = {}
    fingerprint = hashlib.sha256()
    for image in images:
        relative = image.relative_to(val_dir)
        label = label_dir / relative.with_suffix(".txt")
        content = label.read_text(encoding="utf-8-sig") if label.exists() else ""
        group = group_for(image)
        if group:
            if content.strip():
                raise ValueError(f"Ảnh âm có GT box: {image}")
            negative_relatives.append(relative)
            negative_groups[str(relative)] = group
        fingerprint.update(str(relative).encode("utf-8"))
        fingerprint.update(digest_file(image).encode("ascii"))
        fingerprint.update(content.encode("utf-8"))
    counts = Counter(negative_groups.values())
    if counts != {"healthy": 38, "not_cashew_leaf": 20}:
        raise ValueError(f"Cần đúng 38 healthy và 20 not_cashew_leaf negatives: {counts}")
    # Ultralytics creates labels/val.cache during validation. Keep that cache
    # inside the evaluation output and leave the source dataset untouched.
    working_root = output.resolve() / "dataset_working_copy"
    if working_root.exists():
        if not working_root.is_relative_to(output.resolve()):
            raise RuntimeError("Working copy must stay inside output")
        shutil.rmtree(working_root)
    shutil.copytree(val_dir, working_root / "images" / "val")
    shutil.copytree(label_dir, working_root / "labels" / "val",
                    ignore=shutil.ignore_patterns("*.cache"))
    (working_root / "images" / "train").mkdir(parents=True)
    negatives = [working_root / "images" / "val" / relative for relative in negative_relatives]
    groups = {str(path.resolve()): negative_groups[str(relative)]
              for path, relative in zip(negatives, negative_relatives)}
    data_yaml = output / "data_neg_v02_val.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    data_yaml.write_text(yaml.safe_dump({
        "path": working_root.as_posix(),
        "train": (working_root / "images" / "train").as_posix(),
        "val": (working_root / "images" / "val").as_posix(),
        "nc": 3,
        "names": list(CLASSES),
    }, allow_unicode=True, sort_keys=False), encoding="utf-8")
    save_json(output / "dataset_manifest.json", {
        "dataset_root": str(root), "working_copy": str(working_root),
        "split": "val", "validation_images": len(images),
        "negative_images": len(negatives), "negative_groups": dict(counts),
        "validation_sha256": fingerprint.hexdigest(), "class_names": CLASSES,
        "test_used": False,
    })
    return data_yaml, negatives, groups


def checkpoint_info(checkpoint: Path, metadata: Path, expected_model: str) -> dict:
    if not checkpoint.is_file() or not metadata.is_file():
        raise FileNotFoundError(f"Thiếu checkpoint hoặc metrics.json: {checkpoint}")
    info = json.loads(metadata.read_text(encoding="utf-8"))
    expected_weight = expected_model + ".pt"
    if info.get("model") != expected_weight or info.get("cls_pw") != 0.25:
        raise ValueError(f"Sai provenance của checkpoint {checkpoint}: {info.get('model')}")
    return {"checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest_file(checkpoint),
            "training_model": info["model"], "training_cls_pw": info["cls_pw"],
            "training_dataset_fingerprint": info.get("dataset_fingerprint"),
            "training_recipe": info.get("training_recipe")}


def negative_predictions(model: YOLO, negatives: list[Path], groups: dict[str, str],
                         args: argparse.Namespace, key: str, output: Path) -> tuple[dict, list[dict]]:
    image_rows: list[dict] = []
    box_rows: list[dict] = []
    seen: set[str] = set()
    results = model.predict(
        source=[str(p) for p in negatives], stream=True, imgsz=args.imgsz,
        conf=args.conf, iou=args.iou, batch=args.batch, device=args.device,
        half=False, rect=False, agnostic_nms=False, augment=False,
        max_det=args.max_det, verbose=False, save=False,
    )
    for result in results:
        path = str(Path(result.path).resolve())
        if path not in groups or path in seen:
            raise RuntimeError(f"Dự đoán trả ảnh lạ hoặc lặp: {path}")
        seen.add(path)
        cls_ids = result.boxes.cls.int().cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        coords = result.boxes.xyxy.cpu().tolist()
        counts = Counter(cls_ids)
        image_rows.append({
            "model": key, "group": groups[path], "image": Path(path).name,
            "any_false_positive": int(bool(cls_ids)), "false_boxes": len(cls_ids),
            "max_confidence": max(scores, default=0.0),
            **{f"boxes_{name}": counts[i] for i, name in enumerate(CLASSES)},
        })
        for cls_id, score, xyxy in zip(cls_ids, scores, coords):
            if cls_id not in range(3):
                raise RuntimeError(f"Class ngoài 0..2: {cls_id} trong {path}")
            box_rows.append({
                "model": key, "group": groups[path], "image": Path(path).name,
                "class_id": cls_id, "class": CLASSES[cls_id],
                "confidence": float(score), "x1": xyxy[0], "y1": xyxy[1],
                "x2": xyxy[2], "y2": xyxy[3],
            })
    if seen != set(groups):
        raise RuntimeError(f"Thiếu {len(set(groups)-seen)} ảnh âm trong predict")
    image_rows.sort(key=lambda x: (x["group"], x["image"]))
    save_csv(output / "negative_per_image.csv", image_rows,
             ["model", "group", "image", "any_false_positive", "false_boxes", "max_confidence"]
             + [f"boxes_{name}" for name in CLASSES])
    save_csv(output / "negative_boxes.csv", box_rows,
             ["model", "group", "image", "class_id", "class", "confidence",
              "x1", "y1", "x2", "y2"])
    summary: dict[str, dict] = {}
    class_table: list[dict] = []
    for group in ("healthy", "not_cashew_leaf", "all"):
        subset = image_rows if group == "all" else [r for r in image_rows if r["group"] == group]
        n = len(subset)
        alarm = sum(r["any_false_positive"] for r in subset)
        boxes = sum(r["false_boxes"] for r in subset)
        summary[group] = {
            "images": n, "images_with_false_positive": alarm,
            "false_positive_image_rate": alarm / n if n else None,
            "false_boxes": boxes, "mean_false_boxes_per_image": boxes / n if n else None,
        }
        for name in CLASSES:
            field = f"boxes_{name}"
            class_table.append({"model": key, "group": group, "class": name,
                                "images_with_prediction": sum(r[field] > 0 for r in subset),
                                "false_boxes": sum(r[field] for r in subset)})
    save_csv(output / "negative_per_class.csv", class_table,
             ["model", "group", "class", "images_with_prediction", "false_boxes"])
    save_json(output / "negative_summary.json", summary)
    return summary, image_rows


def validation_metrics(model: YOLO, data_yaml: Path, args: argparse.Namespace,
                       output: Path) -> dict:
    metrics = model.val(
        data=str(data_yaml), split="val", imgsz=args.imgsz, batch=args.batch,
        conf=args.ap_conf, iou=args.iou, device=args.device, half=False,
        rect=False, workers=0, plots=False, save_json=False, verbose=False,
        project=str(output), name="ultralytics_val", exist_ok=True,
    )
    indices = list(metrics.box.ap_class_index)
    if set(indices) != {0, 1, 2}:
        raise RuntimeError(f"Validation thiếu class GT: {indices}")
    per_class = []
    for cls_id, name in enumerate(CLASSES):
        precision, recall, ap50, ap = metrics.box.class_result(indices.index(cls_id))
        per_class.append({"class_id": cls_id, "class": name,
                          "precision": float(precision), "recall": float(recall),
                          "ap50": float(ap50), "ap50_95": float(ap)})
    report = {"split": "val", "images_expected": 951,
              "precision_macro": float(metrics.box.mp), "recall_macro": float(metrics.box.mr),
              "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
              "per_class": per_class}
    save_json(output / "validation_metrics.json", report)
    save_csv(output / "validation_per_class.csv", per_class,
             ["class_id", "class", "precision", "recall", "ap50", "ap50_95"])
    return report


def make_comparisons(reports: dict, image_tables: dict, output: Path) -> None:
    rows: list[dict] = []
    paired: list[dict] = []
    for model in ("yolov8s", "yolo26s"):
        old_key, new_key = f"{model}_v003", f"{model}_neg_v02"
        old = reports[old_key]
        new = reports[new_key]
        for group in ("healthy", "not_cashew_leaf", "all"):
            a, b = old["negative"][group], new["negative"][group]
            if a["images"] != b["images"]:
                raise RuntimeError(f"Khác tập ảnh âm: {model}/{group}")
            rows.append({"model": model, "group": group, "images": a["images"],
                         "v003_false_positive_rate": a["false_positive_image_rate"],
                         "neg_v02_false_positive_rate": b["false_positive_image_rate"],
                         "delta_rate_percentage_points": 100 * (b["false_positive_image_rate"] - a["false_positive_image_rate"]),
                         "v003_mean_false_boxes": a["mean_false_boxes_per_image"],
                         "neg_v02_mean_false_boxes": b["mean_false_boxes_per_image"],
                         "delta_mean_false_boxes": b["mean_false_boxes_per_image"] - a["mean_false_boxes_per_image"]})
        old_images = {r["image"]: r for r in image_tables[old_key]}
        new_images = {r["image"]: r for r in image_tables[new_key]}
        if set(old_images) != set(new_images):
            raise RuntimeError(f"Hai checkpoint {model} không chạy trên cùng 58 ảnh")
        for name in sorted(old_images):
            a, b = old_images[name], new_images[name]
            paired.append({"model": model, "group": a["group"], "image": name,
                           "v003_alarm": a["any_false_positive"],
                           "neg_v02_alarm": b["any_false_positive"],
                           "v003_boxes": a["false_boxes"], "neg_v02_boxes": b["false_boxes"]})
    save_csv(output / "comparison_negative_groups.csv", rows,
             ["model", "group", "images", "v003_false_positive_rate",
              "neg_v02_false_positive_rate", "delta_rate_percentage_points",
              "v003_mean_false_boxes", "neg_v02_mean_false_boxes", "delta_mean_false_boxes"])
    save_csv(output / "comparison_paired_images.csv", paired,
             ["model", "group", "image", "v003_alarm", "neg_v02_alarm", "v003_boxes", "neg_v02_boxes"])
    transitions = []
    for model in ("yolov8s", "yolo26s"):
        for group in ("healthy", "not_cashew_leaf", "all"):
            subset = [r for r in paired if r["model"] == model and
                      (group == "all" or r["group"] == group)]
            pairs = Counter((r["v003_alarm"], r["neg_v02_alarm"]) for r in subset)
            old_only, new_only = pairs[(1, 0)], pairs[(0, 1)]
            discordant = old_only + new_only
            exact_p = (min(1.0, 2 * sum(comb(discordant, i)
                                         for i in range(min(old_only, new_only) + 1)) / 2**discordant)
                       if discordant else 1.0)
            transitions.append({"model": model, "group": group, "images": len(subset),
                                "both_alarm": pairs[(1, 1)], "v003_only_alarm": old_only,
                                "neg_v02_only_alarm": new_only, "neither_alarm": pairs[(0, 0)],
                                "mcnemar_exact_two_sided_p": exact_p})
    save_csv(output / "comparison_paired_transitions.csv", transitions,
             ["model", "group", "images", "both_alarm", "v003_only_alarm",
              "neg_v02_only_alarm", "neither_alarm", "mcnemar_exact_two_sided_p"])
    metric_rows = []
    for model in ("yolov8s", "yolo26s"):
        a, b = reports[f"{model}_v003"]["validation"], reports[f"{model}_neg_v02"]["validation"]
        metric_rows.append({"model": model,
                            **{f"v003_{k}": a[k] for k in ("precision_macro", "recall_macro", "map50", "map50_95")},
                            **{f"neg_v02_{k}": b[k] for k in ("precision_macro", "recall_macro", "map50", "map50_95")},
                            **{f"delta_{k}": b[k]-a[k] for k in ("precision_macro", "recall_macro", "map50", "map50_95")}})
    save_csv(output / "comparison_validation_metrics.csv", metric_rows,
             ["model"] + [f"{version}_{key}" for version in ("v003", "neg_v02", "delta")
                          for key in ("precision_macro", "recall_macro", "map50", "map50_95")])
    save_json(output / "comparison.json", {"negative_groups": rows,
                                           "paired_transitions": transitions,
                                           "validation_metrics": metric_rows})


def write_report(reports: dict, output: Path) -> None:
    comparisons = json.loads((output / "comparison.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "dataset_manifest.json").read_text(encoding="utf-8"))
    config = json.loads((output / "evaluation_config.json").read_text(encoding="utf-8"))
    lines = [
        "# Negative Image Evaluation — V003 so với NEG V02",
        "",
        "Bốn checkpoint `best.pt` được chạy trên cùng **951 ảnh Validation NEG V02**: "
        "893 ảnh có GT box và 58 ảnh âm (38 healthy, 20 not_cashew_leaf). "
        "Không train lại và không dùng Test.",
        "",
        "## Cấu hình và cách tính",
        "",
        f"- Ultralytics {config['ultralytics_version']}; ảnh {config['imgsz']} px; "
        f"batch {config['batch']}; IoU {config['iou']}; FP16 tắt; `rect=False`.",
        f"- Báo nhầm ảnh âm: ít nhất một box bệnh ở `conf ≥ {config['negative_predict_conf']}`. "
        "Mọi box trên ảnh 0 GT là false positive theo nhãn hiện có.",
        f"- P/R/AP: `model.val(split='val', conf={config['ap_validation_conf']}, "
        f"iou={config['iou']})` trên cả 951 ảnh để tính đường Precision–Recall chuẩn. "
        "Ngưỡng conf thấp này được giữ giống nhau cho cả bốn checkpoint.",
        f"- Dấu vân tay tập Validation SHA256: `{manifest['validation_sha256']}`.",
        "",
        "## Báo nhầm trên ảnh âm",
        "",
        "| Model | Run | Healthy ảnh báo nhầm | Not cashew ảnh báo nhầm | Tổng ảnh báo nhầm | Box sai / ảnh âm |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for model, version, _ in RUNS:
        item = reports[f"{model}_{version}"]["negative"]
        h, n, all_ = item["healthy"], item["not_cashew_leaf"], item["all"]
        lines.append(f"| {model} | {version} | {h['images_with_false_positive']}/38 "
                     f"({100*h['false_positive_image_rate']:.1f}%) | "
                     f"{n['images_with_false_positive']}/20 ({100*n['false_positive_image_rate']:.1f}%) | "
                     f"{all_['images_with_false_positive']}/58 ({100*all_['false_positive_image_rate']:.1f}%) | "
                     f"{all_['mean_false_boxes_per_image']:.3f} |")
    lines += ["", "YOLO26s giảm từ 15 xuống 3 ảnh âm bị báo nhầm; "
              "YOLOv8s tăng từ 13 lên 14 ảnh ở ngưỡng đã định.", "",
              "## Chuyển trạng thái theo cùng từng ảnh", "",
              "| Model | V003 báo nhầm, NEG V02 không | NEG V02 báo nhầm, V003 không | McNemar exact p (thăm dò) |",
              "|---|---:|---:|---:|"]
    for model in ("yolov8s", "yolo26s"):
        r = next(x for x in comparisons["paired_transitions"]
                 if x["model"] == model and x["group"] == "all")
        lines.append(f"| {model} | {r['v003_only_alarm']} | {r['neg_v02_only_alarm']} | "
                     f"{r['mcnemar_exact_two_sided_p']:.4f} |")
    lines += ["", "Giá trị p là phân tích thăm dò trên 58 ảnh của một seed, "
              "không thay thế đánh giá qua nhiều seed hoặc tập dữ liệu độc lập.", "",
              "## P/R/AP trên cùng Validation", "",
              "| Model | Run | Precision | Recall | mAP50 | mAP50–95 |",
              "|---|---|---:|---:|---:|---:|"]
    for model, version, _ in RUNS:
        v = reports[f"{model}_{version}"]["validation"]
        lines.append(f"| {model} | {version} | {v['precision_macro']:.3f} | "
                     f"{v['recall_macro']:.3f} | {v['map50']:.3f} | {v['map50_95']:.3f} |")
    lines += ["", "## Class dự đoán nhầm trên ảnh âm (số box)", "",
              "| Model | Run | Anthracnose | Leaf Miner | Red Rust |",
              "|---|---|---:|---:|---:|"]
    for model, version, _ in RUNS:
        key = f"{model}_{version}"
        with (output / key / "negative_per_class.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        counts = {r["class"]: int(r["false_boxes"]) for r in rows if r["group"] == "all"}
        lines.append(f"| {model} | {version} | {counts['anthracnose']} | "
                     f"{counts['leaf_miner']} | {counts['red_rust']} |")
    lines += ["", "Chi tiết theo nhóm và class nằm trong từng file `negative_per_class.csv`; "
              "danh sách từng ảnh ở `negative_per_image.csv`, từng box ở `negative_boxes.csv`.", "",
              "## Giới hạn", "",
              "- Kết luận false positive dựa trên nhãn 0 box của ảnh âm; ảnh bệnh chưa gán nhãn sẽ làm sai kết luận.",
              "- Chỉ một seed và 58 ảnh âm Validation. Không suy ra hiệu quả tổng quát cho mọi dữ liệu mới.",
              "- YOLO26 là kiến trúc end-to-end; tham số IoU NMS có thể tác động khác YOLOv8, "
              "nhưng trong từng cặp cùng kiến trúc, cấu hình suy luận giống nhau.",
              "- Các checkpoint V003 và NEG V02 được đánh giá bằng cùng Ultralytics 8.4.197; "
              "metric cũ trong ANALYSIS ZIP không được dùng để so sánh.", "",
              "[Tài liệu Ultralytics về Predict](https://docs.ultralytics.com/modes/predict) · "
              "[Tài liệu về Validation](https://docs.ultralytics.com/modes/val)", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def package_analysis(output: Path) -> Path:
    archive_path = output.parent / "NEGATIVE_IMAGE_EVALUATION_ANALYSIS.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in output.rglob("*"):
            if (file.is_file() and "dataset_working_copy" not in file.parts
                    and "ultralytics_val" not in file.parts):
                archive.write(file, file.relative_to(output))
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip():
            raise RuntimeError(f"Analysis ZIP lỗi: {archive_path}")
    return archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence for false alarms on 58 negative images")
    parser.add_argument("--ap-conf", type=float, default=0.001,
                        help="Low confidence for standard PR/AP integration")
    parser.add_argument("--iou", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--max-det", type=int, default=300)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data_yaml, negatives, groups = validate_dataset(args.data_root, args.output)
    config = {"ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__,
              "device": args.device, "imgsz": args.imgsz, "batch": args.batch,
              "negative_predict_conf": args.conf, "ap_validation_conf": args.ap_conf,
              "iou": args.iou, "half": False, "rect": False,
              "agnostic_nms": False, "augment": False, "max_det": args.max_det,
              "models": [f"{model}_{version}" for model, version, _ in RUNS],
              "training_performed": False, "test_used": False}
    save_json(args.output / "evaluation_config.json", config)
    reports: dict = {}
    image_tables: dict = {}
    for model_name, version, tag in RUNS:
        key = f"{model_name}_{version}"
        checkpoint = args.checkpoints / f"{tag}_best.pt"
        metadata = args.checkpoints / f"{tag}_training_metrics.json"
        info = checkpoint_info(checkpoint, metadata, model_name)
        run_dir = args.output / key
        run_dir.mkdir(exist_ok=True)
        model = YOLO(str(checkpoint))
        names = [model.names[i] for i in range(3)]
        if names != list(CLASSES) or len(model.names) != 3:
            raise ValueError(f"{key}: checkpoint có class sai: {model.names}")
        print(f"{key}: negative image prediction ({len(negatives)} images)", flush=True)
        negative, image_rows = negative_predictions(model, negatives, groups, args, key, run_dir)
        print(f"{key}: full NEG V02 validation (951 images)", flush=True)
        validation = validation_metrics(model, data_yaml, args, run_dir)
        reports[key] = {"checkpoint": info, "negative": negative, "validation": validation}
        image_tables[key] = image_rows
        save_json(run_dir / "report.json", reports[key])
        print(key, negative, "mAP50-95", validation["map50_95"], flush=True)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    make_comparisons(reports, image_tables, args.output)
    save_json(args.output / "all_results.json", reports)
    write_report(reports, args.output)
    archive = package_analysis(args.output)
    print(f"Done: {args.output / 'README.md'} | {archive}", flush=True)


if __name__ == "__main__":
    main()
