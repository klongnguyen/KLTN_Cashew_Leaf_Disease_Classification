"""One-time Test inference after the owner approves a frozen Validation config."""

import csv
import hashlib
import io
import json
import os
import tempfile
import time
import zipfile
from pathlib import Path

from PIL import Image
import yaml

from run_acdra_v2 import (CHECKPOINT_SHA256, DATA, DATA_ROOT, EXP, FULL, REPORT,
                          _main_class, evaluate, load_class_names, sha256,
                          verify_sources)


def checked_lock():
    lock = EXP / "config_locked.yaml"
    checksum = EXP / "config_locked.sha256"
    if not lock.is_file() or not checksum.is_file():
        raise RuntimeError("Validation config is not locked and owner-approved")
    digest = sha256(lock)
    if checksum.read_text(encoding="utf-8").strip() != digest:
        raise RuntimeError("Locked config SHA256 mismatch")
    data = yaml.safe_load(lock.read_text(encoding="utf-8"))
    candidate_path = EXP / "operating_points.yaml"
    candidates = yaml.safe_load(candidate_path.read_text(encoding="utf-8"))
    if (data.get("approved_point") not in candidates or
            data.get("candidate_config") != candidates[data["approved_point"]]["config"] or
            data.get("trial_id") != candidates[data["approved_point"]]["trial_id"] or
            data.get("validation_tuning_sha256") != sha256(REPORT / "validation_tuning.csv") or
            data.get("gallery_manifest_sha256") != sha256(REPORT / "visual_gallery_manifest.csv")):
        raise RuntimeError("Locked config differs from approved Validation candidate")
    return data, digest


def infer_test_raw(destination):
    """Read only the pinned Test images and produce immutable raw YOLO JSONL."""
    os.environ.setdefault("YOLO_CONFIG_DIR", str(EXP / "ultralytics_config"))
    from ultralytics import YOLO
    with zipfile.ZipFile(FULL) as full, tempfile.TemporaryDirectory() as temp:
        checkpoint = Path(temp) / "best.pt"
        checkpoint.write_bytes(full.read("train/weights/best.pt"))
        if sha256(checkpoint) != CHECKPOINT_SHA256:
            raise RuntimeError("Checkpoint changed")
        model = YOLO(str(checkpoint))
        with zipfile.ZipFile(DATA) as archive, destination.open("w", encoding="utf-8") as stream:
            images = sorted(name for name in archive.namelist()
                            if name.startswith(f"{DATA_ROOT}/images/test/") and
                            name.lower().endswith((".jpg", ".jpeg", ".png")))
            for index, member in enumerate(images):
                image_id = Path(member).name
                image_path = Path(temp) / image_id
                image_path.write_bytes(archive.read(member))
                result = model.predict(source=str(image_path), imgsz=640, conf=.2, iou=.7,
                                       max_det=500, agnostic_nms=False, verbose=False)[0]
                image_path.unlink()
                detections = [{"id": f"test_{index:06d}_{j:04d}", "image_id": image_id,
                               "model_id": "EXP-Y26S-NEG-V03-005-ACDRA", "seed": 42,
                               "class_id": int(cls), "confidence": float(conf),
                               "bbox_xyxy": [float(x) for x in box]}
                              for j, (cls, conf, box) in enumerate(zip(
                                  result.boxes.cls.cpu().numpy(),
                                  result.boxes.conf.cpu().numpy(),
                                  result.boxes.xyxy.cpu().numpy()))]
                stream.write(json.dumps({"image_id": image_id, "detections": detections}) + "\n")
                if (index + 1) % 100 == 0:
                    print("Test inference", index + 1, "/", len(images), flush=True)
    return len(images)


def test_records(raw_path):
    with zipfile.ZipFile(FULL) as full, zipfile.ZipFile(DATA) as data:
        class_names = load_class_names(full)
        records = []
        for line in raw_path.open(encoding="utf-8"):
            row = json.loads(line)
            image_id = row["image_id"]
            with Image.open(io.BytesIO(data.read(f"{DATA_ROOT}/images/test/{image_id}"))) as image:
                width, height = image.size
            label_member = f"{DATA_ROOT}/labels/test/{Path(image_id).stem}.txt"
            try:
                label_text = data.read(label_member).decode("utf-8")
            except KeyError:
                label_text = ""
            gt = []
            for line in label_text.splitlines():
                if not line.strip():
                    continue
                class_id, x, y, w, h = map(float, line.split())
                gt.append({"class_id": int(class_id), "bbox_xyxy": [
                    (x - w / 2) * width, (y - h / 2) * height,
                    (x + w / 2) * width, (y + h / 2) * height]})
            records.append({"image_id": image_id, "detections": row["detections"],
                            "gt": gt, "image_size": (width, height),
                            "main_class": _main_class(image_id, class_names)})
    if any(not record["gt"] for record in records):
        raise RuntimeError("Expected zero negative Test images; source differs")
    return records, class_names


def main():
    locked, lock_sha = checked_lock()
    verify_sources()
    REPORT.mkdir(parents=True, exist_ok=True)
    completion = REPORT / "test_evaluation_completed.json"
    if completion.exists():
        raise RuntimeError("Test was already evaluated under the locked config")
    raw_path = EXP / "results" / "test_raw_detections.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    if not raw_path.exists():
        infer_test_raw(raw_path)
    records, class_names = test_records(raw_path)
    config = locked["candidate_config"]
    summary, _, _ = evaluate(records, class_names, config,
                             EXP / "results" / "test_locked", save_regions=True,
                             split_name="test")
    if summary["lost_tp"]:
        raise AssertionError("Test lost_tp is nonzero")
    row = {"approved_point": locked["approved_point"],
           "trial_id": locked["trial_id"], "images": len(records),
           "negative_images": 0,
           **{key: value for key, value in summary.items()
              if isinstance(value, (str, int, float))}}
    with (REPORT / "test_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)
    completion.write_text(json.dumps({"status": "COMPLETE", "lock_sha256": lock_sha,
                                      "raw_jsonl_sha256": sha256(raw_path),
                                      "dataset_sha256": sha256(DATA),
                                      "checkpoint_sha256": CHECKPOINT_SHA256,
                                      "elapsed_seconds": time.time() - started,
                                      "note": "One-time Test audit; zero negative images; no retuning"}, indent=2),
                          encoding="utf-8")
    print("Locked Test evaluation complete:", row)


if __name__ == "__main__":
    main()
