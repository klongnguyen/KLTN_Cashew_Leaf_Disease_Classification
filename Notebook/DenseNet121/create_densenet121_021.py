"""Create EXP-021 holdout-only evaluation from EXP-020, using Holdout (4)."""

import base64
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "densenet121_transfer_5seed_020.ipynb"
TARGET = HERE / "densenet121_transfer_5seed_021.ipynb"
book = json.loads(SOURCE.read_text(encoding="utf-8"))

OLD_HASH = "134072def34a6d766068b7c2de68085e3a6876d8544450d51b9428d0dfa6f6c9"
NEW_HASH = "3b340ea830418638fa610fb2176dcc171739cc7fe0bd8e8430c8ed2a614717cc"

for index, cell in enumerate(book["cells"]):
    content = "".join(cell["source"])
    if index == 12:
        content, replacements = re.subn(
            r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]+"$',
            'NOTEBOOK_SOURCE_ZLIB_B64 = ""', content, count=1, flags=re.M)
        if replacements != 1:
            raise RuntimeError("EXP-020 embedded notebook source not found")
    substitutions = [
        ("EXP-DENSENET121-TRANSFER-5SEEDS-020", "EXP-DENSENET121-TRANSFER-5SEEDS-021"),
        ("EXP-020", "EXP-021"),
        ("exp020", "exp021"),
        ("holdout03", "holdout04"),
        ("RL_Cashew_holdout (3).zip", "RL_Cashew_holdout (4).zip"),
        (OLD_HASH, NEW_HASH),
        ("RL_Cashew_holdout_03", "RL_Cashew_holdout_04"),
        ("densenet121_transfer_5seed_020.ipynb", "densenet121_transfer_5seed_021.ipynb"),
        ("Holdout (3)", "Holdout (4)"),
        ("holdout (3)", "holdout (4)"),
        ("Holdout 3", "Holdout 4"),
        ("holdout 3", "holdout 4"),
        ("126 images", "120 images"),
        ('"holdout_images": 126', '"holdout_images": 120'),
        ("len(holdout_manifest) != 126", "len(holdout_manifest) != 120"),
        ('"healthy": 19', '"healthy": 13'),
    ]
    for old, new in substitutions:
        content = content.replace(old, new)
    cell["source"] = content.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

def edit(index, old, new, expected=1):
    current = "".join(book["cells"][index]["source"])
    if current.count(old) != expected:
        raise RuntimeError(f"Cell {index}: expected {expected} occurrence(s): {old[:80]}")
    book["cells"][index]["source"] = current.replace(old, new).splitlines(keepends=True)

edit(1, 'SPLIT_ZIP = "CashewData_Split (9).zip"',
     'SPLIT_ZIP = "CashewData_Split (10).zip"')
edit(1, 'SPLIT_SHA256 = "1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4"',
     'SOURCE_SPLIT_SHA256 = "1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4"\n'
     'SPLIT_SHA256 = "b24f0faa5e918fb43d7d16500c6b8234df1cd771cc4a27eb2354051bce84edac"')
edit(2, 'if not source_dir.is_dir():',
     '''existing_lock = out_dir / "evaluation_lock.json"
if existing_lock.is_file():
    previous = json.loads(existing_lock.read_text(encoding="utf-8"))
    previous_reference = previous.get("reference_split_zip_sha256", previous.get("split_zip_sha256"))
    if previous_reference != SPLIT_SHA256:
        raise RuntimeError(
            "Existing EXP-021 outputs use another reference split. Move that result "
            "directory aside or choose a new results root before running this version.")
if not source_dir.is_dir():''')
edit(4, 'config.get("dataset", {}).get("archive_sha256", "").lower() != SPLIT_SHA256',
     'config.get("dataset", {}).get("archive_sha256", "").lower() != SOURCE_SPLIT_SHA256')
edit(4, 'lock.get("dataset_sha256", "").lower() != SPLIT_SHA256',
     'lock.get("dataset_sha256", "").lower() != SOURCE_SPLIT_SHA256')
edit(4, 'Split 9 or Holdout 4 archive SHA-256 mismatch',
     'Split 10 or Holdout 4 archive SHA-256 mismatch')
edit(5, 'cache_root / "split09"', 'cache_root / "split10"')
edit(5, 'raise ValueError(f"Split 9 extracted image changed: {row.relative_path}")',
     'raise ValueError(f"EXP-018 image changed in Split 10: {row.relative_path}")')
edit(5, 'reference_hashes = set()\nfor row in source_manifest.itertuples(index=False):',
     'reference_hashes = set()\nsource_paths = set()\n'
     'for row in source_manifest.itertuples(index=False):')
edit(5, '    reference_hashes.add(row.sha256)\nval_manifest =',
     '''    reference_hashes.add(row.sha256)
    source_paths.add(relative.as_posix())
reference_files = {p.relative_to(split_root).as_posix(): p
                   for p in split_root.rglob("*") if p.is_file()}
if len(reference_files) != 7051:
    raise ValueError(f"Split 10 expected 7,051 images, found {len(reference_files)}")
expected_new = {f"train/healthy/real_healthy_{number:03d}.jpg"
                for number in (35, 37, 38, 39, 40, 43)}
new_paths = set(reference_files) - source_paths
if new_paths != expected_new or source_paths - set(reference_files):
    raise ValueError(f"Split 10 changes beyond the six expected Train/healthy images: {new_paths}")
for relative in sorted(new_paths):
    reference_hashes.add(sha256_file(reference_files[relative]))
if len(reference_hashes) != 7051:
    raise ValueError("Split 10 contains duplicate image bytes")
val_manifest =''')
edit(5, 'Holdout image overlaps Split 9', 'Holdout image overlaps Split 10')
edit(5, '    "split_zip_sha256": SPLIT_SHA256, "source_manifest_sha256": manifest_digest,',
     '    "source_training_split_sha256": SOURCE_SPLIT_SHA256,\n'
     '    "reference_split_zip_sha256": SPLIT_SHA256, "reference_images": 7051,\n'
     '    "source_manifest_sha256": manifest_digest,\n'
     '    "split10_added_train_healthy": sorted(new_paths),')
edit(8, '"split_zip_sha256": SPLIT_SHA256, "holdout_zip_sha256": HOLDOUT_SHA256,',
     '"source_training_split_sha256": SOURCE_SPLIT_SHA256,\n'
     '    "reference_split_zip_sha256": SPLIT_SHA256, "holdout_zip_sha256": HOLDOUT_SHA256,')
edit(12, 'Split 9 SHA-256 `{SPLIT_SHA256}`; holdout 4 SHA-256 `{HOLDOUT_SHA256}`.',
     'EXP-018 training Split 9 SHA-256 `{SOURCE_SPLIT_SHA256}`.\n'
     'Split 10 overlap reference SHA-256 `{SPLIT_SHA256}`.\n'
     'Holdout 4 SHA-256 `{HOLDOUT_SHA256}`.')
edit(12, 'The two macro metrics use different class support. `not_cashew_leaf` holdout',
     '`not_cashew_leaf` holdout')
edit(12, '"split_zip_sha256": SPLIT_SHA256, "holdout_zip_sha256": HOLDOUT_SHA256,',
     '"source_training_split_sha256": SOURCE_SPLIT_SHA256,\n'
     '    "reference_split_zip_sha256": SPLIT_SHA256, "holdout_zip_sha256": HOLDOUT_SHA256,')

intro = """# EXP-DENSENET121-TRANSFER-5SEEDS-021 — real holdout (4)

**Evaluation only.** Load the same five Validation-locked DenseNet121
checkpoints from EXP-018 and score `RL_Cashew_holdout (4).zip` once. There is
no training and no internal Final Test in this notebook. Output goes to the
new EXP-021 directory; prior experiments remain untouched.

EXP-018 was trained and selected on Split (9). This notebook uses
`CashewData_Split (10).zip` only as the reference for dataset and overlap
audits; it does not retrain on Split (10) or score its internal Test.

Holdout (4) has 120 images across four true classes: 46 anthracnose,
13 healthy, 36 leaf miner and 25 red rust. `not_cashew_leaf` remains a possible
prediction but has zero true support. Macro metrics use four supported true
classes. The ZIP SHA-256, image counts and exact-byte overlap with Split 10
are checked before inference.

Compared with Holdout (3), six healthy images were removed; the remaining
images have identical bytes. Keep the two holdout versions and their hashes
separate when reporting results.

Run all cells in order on Colab. The project shortcut is discovered in the
current account's My Drive. Set `FORCE_REMOUNT_DRIVE=True` if the runtime
still mounts the previous account. Use the path overrides when needed.
"""
book["cells"][0]["source"] = intro.splitlines(keepends=True)

archive = "".join(book["cells"][12]["source"])
archive = archive.replace(
    'Relative to Holdout (2), one healthy image was removed and eight\n'
    'healthy images changed bytes; compare versions with that provenance.',
    'Relative to Holdout (3), six healthy images were removed and the\n'
    'remaining image bytes are unchanged; retain separate dataset hashes.')
if "Relative to Holdout (2)" in archive:
    raise RuntimeError("Stale Holdout (2) comparison in archive report")
book["cells"][12]["source"] = archive.splitlines(keepends=True)

book["cells"][13]["source"] = """## Done

EXP-021 reads the EXP-018 Validation lock and five checkpoints. It writes
only Holdout (4) predictions, four-class summaries, plots, manifests and
FULL/ANALYSIS ZIPs. No training or Final Test is run. Holdout (3) and (4)
remain separate dataset versions.
Split (10) is an overlap reference; EXP-018 remains a Split (9) model.
""".splitlines(keepends=True)

embedded = base64.b64encode(zlib.compress(json.dumps(
    book, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
archive = "".join(book["cells"][12]["source"])
if archive.count('NOTEBOOK_SOURCE_ZLIB_B64 = ""') < 1:
    raise RuntimeError("Empty source placeholder missing")
archive = archive.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                          f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"', 1)
book["cells"][12]["source"] = archive.splitlines(keepends=True)
TARGET.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(book['cells'])} cells, {TARGET.stat().st_size} bytes")
