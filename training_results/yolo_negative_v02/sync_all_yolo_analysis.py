#!/usr/bin/env python3
"""Import all seven previously uploaded YOLO ANALYSIS archives, without model weights.

From a clean repository clone:
  python training_results/yolo_negative_v02/sync_all_yolo_analysis.py \
    --input-dir PATH_TO_SEVEN_ZIPS --repo-root . --push

Omit --push to only extract and verify, without committing.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

SOURCE = (
    ("EXP-Y8S-LM-003_ANALYSIS(1).zip", "5edc9cff6a76a7c50db23ee752749cf14b355785cdffdfc4a10b8209d0059497", "EXP-Y8S-LM-003-CLASS-WEIGHTED", "class_weight"),
    ("EXP-Y26S-LM-003_ANALYSIS(1).zip", "a987c0c65f059c9f4a5a432256f903dec1cbf63fa1b88cd79e700aa65c3dc131", "EXP-Y26S-LM-003-CLASS-WEIGHTED", "class_weight"),
    ("EXP-Y8S-LM-004-NO-CLASS-WEIGHT-20261101T124902Z-1-001.zip", "506586dcef2f1826fcc86dea4952f8ff1f82827bcf9b74651d002bd7517962ab", "EXP-Y8S-LM-004-NO-CLASS-WEIGHT", "class_weight"),
    ("EXP-Y26S-LM-004-NO-CLASS-WEIGHT-20261101T124909Z-1-001.zip", "3e317ad67a95c9d3c593000c437490ba6aa841df56da13e2a0d0e028079a0cc8", "EXP-Y26S-LM-004-NO-CLASS-WEIGHT", "class_weight"),
    ("EXP-Y8S-NEG-V02-001_ANALYSIS.zip", "b8af8ab5a38b346c942250df6307910583f6f4ba74e8c0c4e6c679ce5c60ed6c", "EXP-Y8S-NEG-V02-001-CLASS-WEIGHTED", "neg_v02"),
    ("EXP-Y26S-NEG-V02-001_ANALYSIS.zip", "2028def8b43046a3c4532252b02714291a750bab97d3cdfefd34cd9ee4e035fa", "EXP-Y26S-NEG-V02-001-CLASS-WEIGHTED", "neg_v02"),
    ("NEGATIVE_IMAGE_EVALUATION_ANALYSIS.zip", "6489e0c41bc19b804302f7c12edcad38e6d069670cae95b759c73119ec5cbd13", "NEGATIVE-IMAGE-EVALUATION", "fp_eval"),
)
DATASET_HASH = {
    "class_weight": "9cd75fd917a0edbce88e43d535c823ae464e348468d9a5e5ef2c98cf328349d7",
    "neg_v02": "55d0ae3678af8f79ecd1c146dce06054930205363b771d181b4be1c3991b0ede",
}
FORBIDDEN = {".pt", ".pth", ".keras", ".h5", ".onnx", ".zip", ".ckpt", ".bin"}

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def run(*args, cwd: Path):
    subprocess.run(args, cwd=cwd, check=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()
    root, input_dir = args.repo_root.resolve(), args.input_dir.resolve()

    if args.push:
        if not (root / ".git").is_dir():
            raise SystemExit("--repo-root must be a git clone when using --push")
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                               check=True, capture_output=True, text=True).stdout
        if dirty.strip():
            raise SystemExit("Git working tree not clean; commit or stash changes before --push")

    # Preflight checksums and experiment metadata before changing any files.
    verified = []
    for filename, expected_sha, eid, group in SOURCE:
        src = input_dir / filename
        if not src.is_file():
            raise SystemExit(f"Missing archive: {src}")
        raw = src.read_bytes()
        if sha256(raw) != expected_sha:
            raise SystemExit(f"Checksum mismatch: {src}")
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            sub = [n for n in z.namelist() if n.endswith("_ANALYSIS.zip")]
            if len(sub) > 1:
                raise SystemExit(f"Multiple nested ANALYSIS files: {filename}")
            payload = z.read(sub[0]) if sub else raw
        with zipfile.ZipFile(io.BytesIO(payload)) as analysis:
            names = analysis.namelist()
            if group != "fp_eval":
                metrics = json.loads(analysis.read("metrics.json"))
                if (metrics.get("experiment_id") != eid or metrics.get("split") != "val"
                        or metrics.get("dataset_fingerprint") != DATASET_HASH[group]):
                    raise SystemExit(f"Incorrect experiment / dataset: {filename}")
            elif "comparison.json" not in names:
                raise SystemExit(f"Missing negative comparison.json: {filename}")
            if any(PurePosixPath(x).is_absolute() or ".." in PurePosixPath(x).parts
                   or PurePosixPath(x).suffix.lower() in FORBIDDEN for x in names):
                raise SystemExit(f"Unsafe archive member: {filename}")
            if any(i.file_size > 20_000_000 for i in analysis.infolist()):
                raise SystemExit(f"Unexpected large file: {filename}")
        verified.append((filename, expected_sha, eid, group, payload))

    total = 0
    for filename, source_sha, eid, group, payload in verified:
        if group == "class_weight":
            target = root / "training_results/yolo_leaf_miner_class_weight/experiments" / eid / "source_analysis"
        elif group == "neg_v02":
            target = root / "training_results/yolo_negative_v02/experiments" / eid / "source_analysis"
        else:
            target = root / "training_results/yolo_negative_v02/negative_image_evaluation/source_analysis"
        files = []
        with zipfile.ZipFile(io.BytesIO(payload)) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                rel = PurePosixPath(info.filename)
                data = z.read(info)
                dest = target.joinpath(*rel.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.read_bytes() != data:
                    dest.write_bytes(data)
                files.append({"relative_path": str(rel), "size": len(data), "sha256": sha256(data)})
        manifest = {"experiment": eid, "archive": filename, "source_sha256": source_sha,
                    "analysis_sha256": sha256(payload), "file_count": len(files), "files": files}
        (target / "SOURCE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        total += len(files)
        print(f"Verified {eid}: {len(files)} original analysis files")

    print(f"DONE: {total} files from seven archives")
    if args.push:
        run("git", "add", "training_results/yolo_leaf_miner_class_weight/experiments",
            "training_results/yolo_negative_v02", cwd=root)
        diff = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root)
        if diff.returncode == 1:
            run("git", "commit", "-m", "results(yolo): import all 7 verified analysis archives",
                cwd=root)
            run("git", "push", "origin", "HEAD", cwd=root)
            print("GitHub push succeeded")
        elif diff.returncode == 0:
            print("Nothing new to commit")
        else:
            raise SystemExit("git diff --cached failed")

if __name__ == "__main__":
    main()
