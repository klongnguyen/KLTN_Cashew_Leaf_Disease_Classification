#!/usr/bin/env python3
"""Import the four YOLO V003/V004 ANALYSIS ZIP archives into this git repository.

Does not train, resplit, evaluate Test or modify original source archives.
"""
from __future__ import annotations
import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

SOURCES = {
    "EXP-Y8S-LM-003-CLASS-WEIGHTED": (
        "EXP-Y8S-LM-003_ANALYSIS(1).zip",
        "5edc9cff6a76a7c50db23ee752749cf14b355785cdffdfc4a10b8209d0059497",
        "5edc9cff6a76a7c50db23ee752749cf14b355785cdffdfc4a10b8209d0059497",
    ),
    "EXP-Y26S-LM-003-CLASS-WEIGHTED": (
        "EXP-Y26S-LM-003_ANALYSIS(1).zip",
        "a987c0c65f059c9f4a5a432256f903dec1cbf63fa1b88cd79e700aa65c3dc131",
        "a987c0c65f059c9f4a5a432256f903dec1cbf63fa1b88cd79e700aa65c3dc131",
    ),
    "EXP-Y8S-LM-004-NO-CLASS-WEIGHT": (
        "EXP-Y8S-LM-004-NO-CLASS-WEIGHT-20261101T124902Z-1-001.zip",
        "506586dcef2f1826fcc86dea4952f8ff1f82827bcf9b74651d002bd7517962ab",
        "e20c1cb3905df88b9bd289cad105f9827d61fd22a9509a5a3cdc4d0492699812",
    ),
    "EXP-Y26S-LM-004-NO-CLASS-WEIGHT": (
        "EXP-Y26S-LM-004-NO-CLASS-WEIGHT-20261101T124909Z-1-001.zip",
        "3e317ad67a95c9d3c593000c437490ba6aa841df56da13e2a0d0e028079a0cc8",
        "4a40e318de36503592b4178e62218e8428f0e953f9782a9b77970fa5d13488d9",
    ),
}
ROOT_REL = Path("training_results/yolo_leaf_miner_class_weight/experiments")
DENY = {".pt", ".pth", ".keras", ".onnx", ".h5", ".ckpt", ".bin", ".zip"}
DATASET_FINGERPRINT = "9cd75fd917a0edbce88e43d535c823ae464e348468d9a5e5ef2c98cf328349d7"

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--repo-root", type=Path, default=Path.cwd())
    p.add_argument("--push", action="store_true", help="Commit and push results to origin")
    args = p.parse_args()
    root = args.repo_root.resolve()
    input_dir = args.input_dir.resolve()
    if args.push:
        if not (root / ".git").exists():
            raise SystemExit("Not a git working tree")
        status = subprocess.run(["git", "status", "--porcelain"], cwd=root,
                                capture_output=True, text=True, check=True)
        if status.stdout.strip():
            raise SystemExit("Please commit/stash all changes before --push")

    total_files = 0
    for experiment_id, (filename, wrapper_hash, analysis_hash) in SOURCES.items():
        raw = (input_dir / filename).read_bytes()
        if sha256(raw) != wrapper_hash:
            raise SystemExit("Source SHA-256 mismatch: " + filename)
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            nested = [n for n in z.namelist() if n.endswith("_ANALYSIS.zip")]
            if len(nested) > 1:
                raise SystemExit("Ambiguous nested archives in " + filename)
            analysis_raw = z.read(nested[0]) if nested else raw
        if sha256(analysis_raw) != analysis_hash:
            raise SystemExit("ANALYSIS SHA-256 mismatch: " + filename)

        destination = root / ROOT_REL / experiment_id / "source_analysis"
        file_list = []
        with zipfile.ZipFile(io.BytesIO(analysis_raw)) as z:
            metrics = json.loads(z.read("metrics.json"))
            if metrics.get("experiment_id") != experiment_id or metrics.get("split") != "val":
                raise SystemExit("Experiment or evaluation split mismatch: " + filename)
            if metrics.get("dataset_fingerprint") != DATASET_FINGERPRINT:
                raise SystemExit("Dataset fingerprint mismatch: " + filename)
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = PurePosixPath(info.filename)
                if name.is_absolute() or ".." in name.parts or name.suffix.lower() in DENY:
                    raise SystemExit("Unsafe/unwanted file: " + info.filename)
                if info.file_size > 16_000_000:
                    raise SystemExit("Unexpected oversized artifact: " + info.filename)
                data = z.read(info)
                target = destination.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists() or target.read_bytes() != data:
                    target.write_bytes(data)
                file_list.append({"file": str(name), "size": len(data), "sha256": sha256(data)})
        manifest = {"experiment_id": experiment_id, "source_zip": filename,
                    "source_sha256": wrapper_hash, "analysis_sha256": analysis_hash,
                    "files": file_list}
        (destination / "SOURCE_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        total_files += len(file_list)
        print(f"OK {experiment_id}: {len(file_list)} files")

    print(f"Imported {total_files} files across 4 experiments")
    if args.push:
        subprocess.run(["git", "add", str(ROOT_REL)], cwd=root, check=True)
        changed = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=root)
        if changed.returncode == 0:
            print("No changes")
        else:
            subprocess.run(["git", "commit", "-m",
                            "results(yolo): add original V003 V004 analysis artifacts"],
                           cwd=root, check=True)
            subprocess.run(["git", "push", "origin", "HEAD"], cwd=root, check=True)
            print("PUSH OK")
if __name__ == "__main__":
    main()
