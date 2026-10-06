"""Audit ZIP identities/labels/decoding and byte overlap without running models."""
import argparse
import hashlib
import io
import json
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from PIL import Image

from real_holdout import holdout_sha256


def audit_archives(holdout_zip, v05_zip):
    expected_holdout = "BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165"
    expected_v05 = "37E76938B930025B88154D53EA56084E0828C57A0B0FB725387462C835EA8CDF"
    if holdout_sha256(holdout_zip) != expected_holdout or holdout_sha256(v05_zip) != expected_v05:
        raise ValueError("An archive does not match the frozen source identity.")
    counts, holdout_hashes, overlaps = Counter(), {}, []
    with zipfile.ZipFile(holdout_zip) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if len(parts) != 3 or parts[0] != "RL_Cashew_holdout":
                raise ValueError(f"Unexpected holdout path: {member.filename}")
            raw = archive.read(member)
            with Image.open(io.BytesIO(raw)) as image:
                image.verify()
            with Image.open(io.BytesIO(raw)) as image:
                image.convert("RGB").load()
            digest = hashlib.sha256(raw).hexdigest().upper()
            if digest in holdout_hashes:
                raise ValueError("Duplicate holdout image bytes.")
            holdout_hashes[digest] = member.filename
            counts[parts[1]] += 1
    if dict(counts) != {"anthracnose": 13, "healthy": 5, "leaf_miner": 24, "red_rust": 7}:
        raise ValueError(f"Holdout class counts changed: {counts}")
    reference_count = 0
    with zipfile.ZipFile(v05_zip) as archive:
        for member in archive.infolist():
            if member.is_dir():
                continue
            if PurePosixPath(member.filename).suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}:
                continue
            digest = hashlib.sha256()
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            reference_count += 1
            hex_digest = digest.hexdigest().upper()
            if hex_digest in holdout_hashes:
                overlaps.append({"holdout": holdout_hashes[hex_digest], "v05": member.filename})
    if reference_count != 6911:
        raise ValueError(f"Wrong V05 reference image count: {reference_count}")
    return {
        "notebook_version": "006", "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "holdout_archive_name": "RL_Cashew_holdout.zip", "holdout_archive_sha256": expected_holdout,
        "holdout_archive_size_bytes": Path(holdout_zip).stat().st_size,
        "drive_source": "https://drive.google.com/file/d/1PBry_y2bMsvd5_YdmEUy9erSFarS3O52/view",
        "source_verification": "Authenticated Drive download and local ZIP have identical SHA-256.",
        "counts": {**dict(counts), "not_cashew_leaf": 0}, "holdout_images": sum(counts.values()),
        "decode_errors": 0, "within_holdout_byte_duplicates": 0,
        "v05_archive_sha256": expected_v05, "reference_images_checked": reference_count,
        "cross_dataset_byte_overlaps": overlaps,
        "limitation": "Exact bytes only. Near-duplicates, shared leaves/trees and capture sessions are not audited.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--v05", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit_archives(args.holdout, args.v05)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if result["cross_dataset_byte_overlaps"]:
        sys.exit("Holdout overlaps V05; resolve before external evaluation.")
