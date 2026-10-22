"""Read-only V08 ZIP audit for EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010."""

import argparse
import hashlib
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn

EXPERIMENT = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-010"
CLASSES = ("anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf")
COUNTS = {"train": (939, 803, 875, 1072, 1101),
          "val": (308, 216, 264, 328, 314),
          "test": (122, 118, 132, 158, 156)}
ZIP_SHA = "75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7"
MANIFEST_SHA = "C699A9ACA707DA25F143DB1FC3C9B5133EA2858BCDC468A158E5644E500EC715"
ABSENT = ("val/leaf_miner/leaf_miner_1056.jpg", "val/leaf_miner/leaf_miner_1091.jpg",
          "test/not_cashew_leaf/not_cashew_leaf_1514.JPG")


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def phash(raw):
    with Image.open(io.BytesIO(raw)) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    values = dctn(gray, type=2, norm="ortho")[:8, :8]
    threshold = np.median(values.flatten()[1:])
    bits = (values > threshold).flatten()
    bits[0] = False
    return sum(int(bit) << i for i, bit in enumerate(bits))


def run(dataset_zip, output):
    dataset_zip, output = Path(dataset_zip), Path(output)
    if sha_file(dataset_zip) != ZIP_SHA:
        raise ValueError("V08 archive SHA256 mismatch")
    groups = defaultdict(list)
    with zipfile.ZipFile(dataset_zip) as archive:
        bad = archive.testzip()
        if bad:
            raise ValueError(f"ZIP CRC failure: {bad}")
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe member: {item.filename}")
            if item.is_dir():
                continue
            if len(path.parts) != 4 or path.parts[0] != "CashewData_Split":
                raise ValueError(f"Unexpected file: {item.filename}")
            _, split, klass, _ = path.parts
            if split not in COUNTS or klass not in CLASSES:
                raise ValueError(f"Unexpected split/class: {item.filename}")
            groups[(split, klass)].append(item.filename)
        for split, counts in COUNTS.items():
            for klass, expected in zip(CLASSES, counts):
                actual = len(groups[(split, klass)])
                if actual != expected:
                    raise ValueError(f"Count changed: {split}/{klass}: {actual} != {expected}")
        rows, hash_values, decode_errors = [], [], []
        for split in COUNTS:
            for label, klass in enumerate(CLASSES):
                for name in sorted(groups[(split, klass)]):
                    raw = archive.read(name)
                    relative = str(PurePosixPath(name).relative_to("CashewData_Split"))
                    try:
                        with Image.open(io.BytesIO(raw)) as image:
                            image.verify()
                        hash_value = phash(raw)
                    except Exception as exc:
                        decode_errors.append({"relative_path": relative, "error": repr(exc)})
                        continue
                    rows.append({"split": split, "class": klass, "label": label,
                                 "relative_path": relative,
                                 "sha256": hashlib.sha256(raw).hexdigest().upper()})
                    hash_values.append(hash_value)
    if decode_errors:
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(decode_errors).to_csv(output / "decode_errors.csv", index=False)
        raise ValueError(f"{len(decode_errors)} unreadable images; see decode_errors.csv")
    manifest = pd.DataFrame(rows)
    if len(manifest) != 6906 or manifest.relative_path.duplicated().any():
        raise ValueError("V08 image count or unique path failed")
    if set(ABSENT) & set(manifest.relative_path):
        raise ValueError("A known absent V08 image unexpectedly reappeared")
    payload = manifest[["split", "class", "relative_path", "sha256"]].to_csv(index=False, lineterminator="\n")
    manifest_hash = hashlib.sha256(payload.encode()).hexdigest().upper()
    if manifest_hash != MANIFEST_SHA:
        raise ValueError(f"EXP-007/008 manifest mismatch: {manifest_hash}")
    exact = manifest[manifest.sha256.duplicated(keep=False)].copy()
    cross_split_exact = [g for _, g in exact.groupby("sha256") if g.split.nunique() > 1]
    if cross_split_exact:
        raise ValueError(f"{len(cross_split_exact)} exact duplicates across splits")
    # Eight 8-bit bands guarantee a shared band for any 64-bit pHash pair at Hamming <= 6.
    buckets = defaultdict(list)
    for index, value in enumerate(hash_values):
        for band in range(8):
            buckets[(band, (value >> (8 * band)) & 255)].append(index)
    candidate_pairs = set()
    splits = manifest.split.to_numpy()
    for members in buckets.values():
        for i_pos, i in enumerate(members):
            for j in members[i_pos + 1:]:
                if splits[i] != splits[j] and (hash_values[i] ^ hash_values[j]).bit_count() <= 6:
                    candidate_pairs.add((min(i, j), max(i, j)))
    candidates = pd.DataFrame([{"path_a": manifest.relative_path.iloc[i],
        "path_b": manifest.relative_path.iloc[j], "split_a": splits[i], "split_b": splits[j],
        "phash_distance": (hash_values[i] ^ hash_values[j]).bit_count(),
        "status": "VISUAL_AND_SOURCE_GROUP_REVIEW_REQUIRED"} for i, j in sorted(candidate_pairs)],
        columns=["path_a", "path_b", "split_a", "split_b", "phash_distance", "status"])
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output / "data_manifest.csv", index=False)
    exact.to_csv(output / "exact_duplicates.csv", index=False)
    candidates.to_csv(output / "phash_cross_split_candidates.csv", index=False)
    result = {"experiment_id": EXPERIMENT, "dataset_zip_sha256": ZIP_SHA,
        "manifest_sha256": manifest_hash, "total_images": len(manifest),
        "counts": {split: dict(zip(CLASSES, values)) for split, values in COUNTS.items()},
        "decode_errors": 0, "cross_split_exact_duplicates": 0,
        "within_split_exact_duplicate_rows": len(exact),
        "cross_split_phash_le_6_candidates": len(candidates),
        "group_leakage_status": "GROUP_LEAKAGE_NOT_VERIFIED",
        "candidate_interpretation": "pHash is a visual shortlist, not proof of same leaf/tree/session.",
        "test_use": "read-only integrity audit; no inference, tuning, or selection"}
    (output / "dataset_audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_zip")
    parser.add_argument("output")
    args = parser.parse_args()
    print(json.dumps(run(args.dataset_zip, args.output), indent=2))
