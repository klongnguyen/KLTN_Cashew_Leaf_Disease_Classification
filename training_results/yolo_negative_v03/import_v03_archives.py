#!/usr/bin/env python3
"""Import original NEG V03 ANALYSIS files from 3 verified source ZIPs.

Examples:
  python training_results/yolo_negative_v03/import_v03_archives.py \
    --input-dir "/path/to/three/zips" --repo-root . --push

Use --prepare-only to verify and extract without git operations.
No model weights, FULL.zip or locked Test images are imported.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from pathlib import Path, PurePosixPath

SOURCES = (
    (
        "EXP-Y8S-NEG-V03-001_ANALYSIS.zip",
        "c9e1375b1456dd70c16d8a6b9cb908724744cc0f871c9062b7036ea6a5b1b29a",
        "EXP-Y8S-NEG-V03-001-CLASS-WEIGHTED",
        "experiments/EXP-Y8S-NEG-V03-001-CLASS-WEIGHTED/source_analysis",
        37,
    ),
    (
        "EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED-20261102T043914Z-1-001.zip",
        "d65d59c4b3e410b19fbcb41b09d5620c6d3d4ac6b2c029b9dd03ea796f340c46",
        "EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED",
        "experiments/EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED/source_analysis",
        39,
    ),
    (
        "NEG_V03_NEGATIVE_IMAGE_EVALUATION_ANALYSIS.zip",
        "91435db7b9fbe9ac408417bc0290d9656c9c08829601401a8899754153c5dcbb",
        None,
        "negative_image_evaluation/source_analysis",
        52,
    ),
)
DATASET_FINGERPRINT = "1acc0edcf7f736f9754e4977b61133a7ec5582b44d8cf3eef27208f49a202add"
DISALLOWED = {".pt", ".pth", ".keras", ".h5", ".onnx", ".ckpt", ".zip"}
TARGET_BASE = Path("training_results/yolo_negative_v03")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(*command: str, cwd: Path) -> None:
    subprocess.run(command, check=True, cwd=cwd)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path.cwd(), type=Path)
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.push and args.prepare_only:
        parser.error("--push and --prepare-only cannot be combined")

    workspace = args.repo_root.resolve()
    input_dir = args.input_dir.resolve()
    if args.push:
        if not (workspace / ".git").is_dir():
            raise SystemExit(f"Not a Git checkout: {workspace}")
        git_status = subprocess.run(
            ("git", "status", "--porcelain"), cwd=workspace,
            capture_output=True, text=True, check=True
        ).stdout
        if git_status.strip():
            raise SystemExit("Working tree is not clean; commit/stash before --push")

    # Fail before writing if an archive is missing, changed, unsafe, or wrong EXP.
    verified = []
    for filename, expected_sha, experiment_id, relative_output, expected_count in SOURCES:
        source_file = input_dir / filename
        if not source_file.is_file():
            raise SystemExit(f"Missing file: {source_file}")
        outer_bytes = source_file.read_bytes()
        if sha256(outer_bytes) != expected_sha:
            raise SystemExit(f"Archive SHA256 mismatch: {filename}")
        if filename.startswith("EXP-Y26S"):
            with zipfile.ZipFile(io.BytesIO(outer_bytes)) as outer:
                nested = [name for name in outer.namelist() if name.endswith("_ANALYSIS.zip")]
                if len(nested) != 1:
                    raise SystemExit("YOLO26 wrapper ZIP must contain exactly one ANALYSIS ZIP")
                inner_bytes = outer.read(nested[0])
        else:
            inner_bytes = outer_bytes
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as archive:
            members = [m for m in archive.infolist() if not m.is_dir()]
            if len(members) != expected_count:
                raise SystemExit(f"Unexpected ANALYSIS file count for {filename}: {len(members)}")
            for member in members:
                rel = PurePosixPath(member.filename)
                if (rel.is_absolute() or ".." in rel.parts
                        or rel.suffix.lower() in DISALLOWED
                        or member.file_size > 25_000_000):
                    raise SystemExit(f"Unsafe entry: {member.filename}")
            if experiment_id:
                metrics = json.loads(archive.read("metrics.json"))
                if (metrics.get("experiment_id") != experiment_id
                    or metrics.get("dataset_fingerprint") != DATASET_FINGERPRINT
                    or metrics.get("split") != "val"
                    or metrics.get("cls_pw") != 0.25):
                    raise SystemExit(f"Experiment or dataset mismatch in {filename}")
            else:
                config = json.loads(archive.read("evaluation_config.json"))
                manifest = json.loads(archive.read("dataset_manifest.json"))
                if (config.get("test_used") is not False
                        or manifest.get("validation_images") != 951
                        or manifest.get("negative_images") != 58):
                    raise SystemExit("Unexpected negative-image evaluation protocol")
        verified.append(
            (filename, expected_sha, experiment_id, relative_output, inner_bytes)
        )

    total_files = 0
    for filename, source_sha, experiment_id, relative_output, inner_bytes in verified:
        target_dir = workspace / TARGET_BASE / relative_output
        file_manifest = []
        with zipfile.ZipFile(io.BytesIO(inner_bytes)) as archive:
            for member in archive.infolist():
                if member.is_dir():
                    continue
                relative = PurePosixPath(member.filename)
                content = archive.read(member)
                dest = target_dir.joinpath(*relative.parts)
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists() or dest.read_bytes() != content:
                    dest.write_bytes(content)
                file_manifest.append(
                    {"relative_path": str(relative), "sha256": sha256(content),
                     "bytes": len(content)}
                )
        out_manifest = {
            "source_archive": filename,
            "source_sha256": source_sha,
            "analysis_sha256": sha256(inner_bytes),
            "experiment": experiment_id or "NEGATIVE_IMAGE_EVALUATION_V03",
            "file_count": len(file_manifest),
            "files": file_manifest,
        }
        (target_dir / "SOURCE_MANIFEST.json").write_text(
            json.dumps(out_manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        total_files += len(file_manifest)
        print(f"OK: {filename} ({len(file_manifest)} source files)")
    print(f"Extracted {total_files} original ANALYSIS files + 3 manifests")

    if args.push:
        run("git", "add", "--", str(TARGET_BASE), cwd=workspace)
        has_diff = subprocess.run(
            ("git", "diff", "--cached", "--quiet"), cwd=workspace
        ).returncode
        if has_diff == 1:
            run("git", "commit", "-m",
                "results(yolo): import all verified NEG V03 analysis files",
                cwd=workspace)
            run("git", "push", "origin", "HEAD", cwd=workspace)
            print("GitHub push completed.")
        elif has_diff == 0:
            print("Nothing new to commit.")
        else:
            raise SystemExit("git diff --cached failed")


if __name__ == "__main__":
    main()
