"""Build holdout-only EXP-020 from the verified EXP-019 evaluation notebook."""

import base64
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "densenet121_transfer_5seed_019.ipynb"
TARGET = HERE / "densenet121_transfer_5seed_020.ipynb"
book = json.loads(SOURCE.read_text(encoding="utf-8"))


def body(index):
    return "".join(book["cells"][index]["source"])


def put(index, value):
    book["cells"][index]["source"] = value.splitlines(keepends=True)


def replace(index, old, new, expected=1):
    current = body(index)
    found = current.count(old)
    if found != expected:
        raise RuntimeError(f"Cell {index}: expected {expected}, found {found}: {old[:90]}")
    put(index, current.replace(old, new))


put(0, """# EXP-DENSENET121-TRANSFER-5SEEDS-020 — real holdout (3)

**Evaluation only.** Load the same five Validation-locked DenseNet121
checkpoints from EXP-018 and score `RL_Cashew_holdout (3).zip` once. There is
no training and no internal Final Test in this notebook. Output goes to a
new EXP-020 directory; EXP-018 and EXP-019 remain untouched.

Holdout (3) has 126 images across four true classes: 46 anthracnose,
19 healthy, 36 leaf miner and 25 red rust. `not_cashew_leaf` remains a possible
prediction, but has zero true support. Macro metrics use four supported true
classes. The notebook checks the ZIP SHA-256 and exact byte overlap with all
of Split 9 before inference.

Compared with Holdout (2), this ZIP removes `healthy/real_healthy_028.jpg`
and changes the bytes of eight other healthy images. Scores from the two
versions must retain their separate dataset hashes.

Run all cells in order on Colab. The configuration finds the project shortcut
in the current account's My Drive. If this runtime still has the old account
mounted, set `FORCE_REMOUNT_DRIVE=True`. Use the path overrides if the shared
folder is elsewhere or read-only.
""")

replace(1, 'EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-019"',
        'EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-020"')
replace(1, 'HOLDOUT_ZIP = "RL_Cashew_holdout (2).zip"',
        'HOLDOUT_ZIP = "RL_Cashew_holdout (3).zip"')
replace(1, 'HOLDOUT_SHA256 = "47a5722660e2c8dc7cc04c9ff720e1f889c2481ec99112b41bc0606958a7fdd7"',
        'HOLDOUT_SHA256 = "134072def34a6d766068b7c2de68085e3a6876d8544450d51b9428d0dfa6f6c9"')
first = body(1)
start = first.index('EXPECTED_TEST_COUNTS = {')
end = first.index('EXPECTED_HOLDOUT_COUNTS = {', start)
put(1, first[:start] + first[end:])
replace(1, '"healthy": 20,', '"healthy": 19,')

replace(2, 'dataset_cache/exp019', 'dataset_cache/exp020')
replace(2, '"dataset_cache" / "exp019"', '"dataset_cache" / "exp020"')
replace(2, 'EXP-019 supports Colab or Windows', 'EXP-020 supports Colab or Windows')
replace(2, 'print("EXP-019:", out_dir)', 'print("EXP-020:", out_dir)')

put(3, """## Verify the EXP-018 lock and revised holdout archive

Any archive, manifest or checkpoint mismatch stops before holdout inference.
""")
replace(4, 'Split 9 or Holdout 2 archive SHA-256 mismatch',
        'Split 9 or Holdout 3 archive SHA-256 mismatch')

cell5 = body(5)
cell5 = cell5.replace('cache_root / "holdout02"', 'cache_root / "holdout03"')
start = cell5.index('test_manifest = source_manifest[')
end = cell5.index('val_manifest = source_manifest[', start)
cell5 = cell5[:start] + cell5[end:]
cell5 = cell5.replace('if not test_manifest["image_path"].map(lambda x: Path(x).is_file()).all():\n'
                      '    raise FileNotFoundError("Some Test images are missing")\n', '')
cell5 = cell5.replace('if len(holdout_manifest) != 127', 'if len(holdout_manifest) != 126')
cell5 = cell5.replace('Holdout 2 contains missing or duplicate images',
                      'Holdout 3 contains missing or duplicate images')
cell5 = cell5.replace('test_manifest.to_csv(out_dir / "test_dataset_manifest.csv", index=False)\n', '')
cell5 = cell5.replace('    "test_counts": EXPECTED_TEST_COUNTS, "holdout_zip_sha256": HOLDOUT_SHA256,',
                      '    "holdout_zip_sha256": HOLDOUT_SHA256,')
cell5 = cell5.replace('print("Dataset audit PASS: Test", len(test_manifest), "Holdout", len(holdout_manifest))',
                      'print("Dataset audit PASS: Holdout 3", len(holdout_manifest))')
put(5, cell5)

replace(8, '# 06. Freeze evaluation inputs before any Test or holdout inference',
        '# 06. Freeze the holdout evaluation inputs before inference')
replace(8, '"test_images": 686, "holdout_images": 127,',
        '"holdout_images": 126, "internal_test": "NOT_RUN",')
replace(8, 'EXP-019 evaluation lock changed', 'EXP-020 evaluation lock changed')
put(9, """## External real holdout (3)

The five EXP-018 checkpoints stay fixed. All five class probabilities are
exported per image; macro scores use the four true classes in this holdout.
""")

replace(10, 'supported = [CLASSES.index(x) for x in (CLASSES if split_name == "test" else HOLDOUT_CLASSES)]',
        'supported = [CLASSES.index(x) for x in HOLDOUT_CLASSES]')
replace(10, '"dataset_version": ("Cashew_dataV09_more_real" if split_name == "test" else\n'
            '                            "RL_Cashew_holdout_02"),\n'
            '        "dataset_sha256": SPLIT_SHA256 if split_name == "test" else HOLDOUT_SHA256,',
        '"dataset_version": "RL_Cashew_holdout_03", "dataset_sha256": HOLDOUT_SHA256,')
replace(10, 'for split_name, frame in (("test", test_manifest), ("real_holdout", holdout_manifest)):',
        'for split_name, frame in (("real_holdout", holdout_manifest),):')
replace(10, 'print("Final Test and holdout inference complete")',
        'print("Holdout 3 inference complete")')
replace(11, 'for split_name in ("test", "real_holdout"):',
        'for split_name in ("real_holdout",):')
aggregate = body(11)
start = aggregate.index('gap = pd.DataFrame(')
put(11, aggregate[:start] + 'print("Holdout 3 aggregate tables and figures complete")\n')

archive = body(12)
archive = re.sub(r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]+"$',
                 'NOTEBOOK_SOURCE_ZLIB_B64 = ""', archive, count=1, flags=re.M)
if 'NOTEBOOK_SOURCE_ZLIB_B64 = ""' not in archive:
    raise RuntimeError("Could not remove EXP-019 embedded source")
archive = archive.replace('Evaluation-only continuation of {SOURCE_ID}. No training or selection on Test/holdout.',
                          'Holdout-only continuation of {SOURCE_ID}. No training, Final Test or holdout-based selection.')
archive = archive.replace('Test: 686 images, five true classes. Holdout: 127 images, four true classes.',
                          'Holdout 3: 126 images, four true classes. Internal Final Test is NOT_RUN here.')
archive = archive.replace('holdout 2 SHA-256', 'holdout 3 SHA-256')
archive = archive.replace('F1 is undefined because it has zero true support.',
                          'F1 is undefined because it has zero true support.\n'
                          'Relative to Holdout (2), one healthy image was removed and eight\n'
                          'healthy images changed bytes; compare versions with that provenance.')
archive = archive.replace('Per-seed outputs: `seed_*/test|real_holdout/` contains metrics, standard',
                          'Per-seed outputs: `seed_*/real_holdout/` contains metrics, standard')
archive = archive.replace('    "training": "REUSED_EXP018", "final_test": "COMPLETE", "real_holdout": "COMPLETE",',
                          '    "training": "REUSED_EXP018", "final_test": "NOT_RUN", "real_holdout": "COMPLETE",')
archive = archive.replace('    "test_dataset_manifest.csv", "holdout_dataset_manifest.csv",\n'
                          '            "aggregate/test_results_5seeds.csv", "aggregate/test_mean_std.csv",\n'
                          '            "aggregate/real_holdout_results_5seeds.csv", "aggregate/real_holdout_mean_std.csv"]',
                          '    "holdout_dataset_manifest.csv",\n'
                          '            "aggregate/real_holdout_results_5seeds.csv", "aggregate/real_holdout_mean_std.csv"]')
archive = archive.replace('for split_name in ("test", "real_holdout"):',
                          'for split_name in ("real_holdout",):')
archive = archive.replace('densenet121_transfer_5seed_019.ipynb',
                          'densenet121_transfer_5seed_020.ipynb')
put(12, archive)
put(13, """## Done

EXP-020 reads the EXP-018 Validation lock and five checkpoints. It writes
only Holdout (3) predictions, four-class summaries, plots, manifests and
FULL/ANALYSIS ZIPs. No training or Final Test is run. Holdout (2) and (3)
are different datasets, so their results should keep separate provenance.
""")

for cell in book["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

embedded = base64.b64encode(zlib.compress(json.dumps(
    book, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
archive = body(12).replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                           f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"', 1)
put(12, archive)
TARGET.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(book['cells'])} cells, {TARGET.stat().st_size} bytes")
