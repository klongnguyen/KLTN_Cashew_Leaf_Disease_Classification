"""Build the static frontend for Vercel.

    python deploy/prepare.py --api-url https://<workspace>--cashew-leaf-api-web.us-east.modal.run
    cd deploy/_build/vercel && vercel deploy --prod

Output (git-ignored): deploy/_build/vercel. The backend (Modal) is deployed separately,
see deploy/DEPLOY.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deploy" / "_build" / "vercel"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-url", required=True, help="Public https URL of the Modal backend")
    args = ap.parse_args()
    if not args.api_url.startswith("https://"):
        sys.exit("--api-url must start with https:// (browsers block http from an https page)")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(ROOT / "Application" / "static", OUT / "static")
    # index.html uses /static/... paths: index at /, assets under /static.
    shutil.move(str(OUT / "static" / "index.html"), str(OUT / "index.html"))
    (OUT / "static" / "config.js").write_text(
        f'window.CASHEW_API_BASE = {json.dumps(args.api_url.rstrip("/"))};\n', encoding="utf-8"
    )
    (OUT / "vercel.json").write_text(
        json.dumps({"$schema": "https://openapi.vercel.sh/vercel.json", "cleanUrls": True}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"vercel -> {OUT}")


if __name__ == "__main__":
    main()
