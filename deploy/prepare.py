"""Build the two deploy folders from the repository.

    python deploy/prepare.py --api-url https://<user>-<space>.hf.space

Outputs (git-ignored, under deploy/_build):
    space/   -> push to the Hugging Face Space (Docker backend + model binaries)
    vercel/  -> deploy with `vercel --prod` (static frontend only)

Run it from a checkout that has the model binaries (*.keras, best.pt, ood/*.npz).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
BUILD = DEPLOY / "_build"
BUNDLE_ID = "EXP-DENSENET121-YOLO-COMBINED-TRAINING-035-NEGV04"
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", "tools")


def reset(path: Path) -> Path:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(bundle: Path) -> None:
    """Fail early if a binary is missing or differs from artifact_sha256.csv."""
    manifest = bundle / "artifact_sha256.csv"
    if not manifest.exists():
        sys.exit(f"Missing {manifest}")
    bad = []
    with manifest.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            p = bundle / row["path"]
            if not p.exists():
                bad.append(f"missing  {row['path']}")
            elif sha256(p) != row["sha256"]:
                bad.append(f"mismatch {row['path']}")
    if bad:
        sys.exit("Bundle check failed:\n  " + "\n  ".join(bad))
    print(f"bundle OK ({manifest.name})")


def build_space() -> Path:
    out = reset(BUILD / "space")
    bundle_src = ROOT / BUNDLE_ID / "pipeline_bundle"
    verify_bundle(bundle_src)
    for name in ("Dockerfile", "requirements.txt", "README.md", ".gitattributes"):
        shutil.copy2(DEPLOY / "space" / name, out / name)
    # static/ stays: /api/predict-sample reads the sample photos from it.
    shutil.copytree(ROOT / "Application", out / "Application", ignore=IGNORE)
    shutil.copytree(bundle_src, out / BUNDLE_ID / "pipeline_bundle")
    print(f"space  -> {out}")
    return out


def build_vercel(api_url: str) -> Path:
    out = reset(BUILD / "vercel")
    static = ROOT / "Application" / "static"
    shutil.copytree(static, out / "static")
    # Same-origin paths in index.html (/static/...) keep working: index at /, assets at /static.
    shutil.move(str(out / "static" / "index.html"), str(out / "index.html"))
    (out / "static" / "config.js").write_text(
        f'window.CASHEW_API_BASE = {json.dumps(api_url.rstrip("/"))};\n', encoding="utf-8"
    )
    (out / "vercel.json").write_text(
        json.dumps({"$schema": "https://openapi.vercel.sh/vercel.json", "cleanUrls": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"vercel -> {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", required=True, help="Public URL of the Hugging Face Space")
    ap.add_argument("--only", choices=["space", "vercel"], help="Build just one target")
    args = ap.parse_args()
    if not args.api_url.startswith("https://"):
        sys.exit("--api-url must start with https:// (browsers block http from an https page)")
    if args.only != "vercel":
        build_space()
    if args.only != "space":
        build_vercel(args.api_url)


if __name__ == "__main__":
    main()
