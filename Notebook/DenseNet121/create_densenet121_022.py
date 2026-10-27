"""Build a new five-seed DenseNet121 training run on Split (10) and Holdout (4)."""

import base64
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "densenet121_transfer_5seed_018.ipynb"
TARGET = HERE / "densenet121_transfer_5seed_022.ipynb"
book = json.loads(SOURCE.read_text(encoding="utf-8"))


def body(index):
    return "".join(book["cells"][index]["source"])


def put(index, value):
    book["cells"][index]["source"] = value.splitlines(keepends=True)


def change(index, old, new, expected=1):
    current = body(index)
    if current.count(old) != expected:
        raise RuntimeError(f"Cell {index}: expected {expected}, found {current.count(old)}: {old[:80]}")
    put(index, current.replace(old, new))


for index, cell in enumerate(book["cells"]):
    content = "".join(cell["source"])
    if index == 30:
        content, count = re.subn(r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]+"$',
                                 'NOTEBOOK_SOURCE_ZLIB_B64 = ""', content,
                                 count=1, flags=re.M)
        if count != 1:
            raise RuntimeError("EXP-018 embedded notebook source was not found")
    for old, new in [
        ("EXP-DENSENET121-TRANSFER-5SEEDS-018", "EXP-DENSENET121-TRANSFER-5SEEDS-022"),
        ("EXP-018", "EXP-022"),
        ("CashewData_Split (9).zip", "CashewData_Split (10).zip"),
        ("Split (9)", "Split (10)"),
        ("Split 9", "Split 10"),
        ("split_09", "split_10"),
        ("Cashew_dataV09_more_real", "Cashew_dataV10_more_real"),
        ("RL_Cashew_holdout (2).zip", "RL_Cashew_holdout (4).zip"),
        ("holdout02", "holdout04"),
        ("Holdout02", "Holdout04"),
        ("densenet121_transfer_5seed_018.ipynb", "densenet121_transfer_5seed_022.ipynb"),
        ("real_holdout_018", "real_holdout_022"),
        ("1eb8c46921de19ab53ef8d50e05d8785f94dcb45a5511dcf32e2e7e01516c6f4",
         "b24f0faa5e918fb43d7d16500c6b8234df1cd771cc4a27eb2354051bce84edac"),
        ("47A5722660E2C8DC7CC04C9FF720E1F889C2481EC99112B41BC0606958A7FDD7",
         "3B340EA830418638FA610FB2176DCC171739CC7FE0BD8E8430C8ED2A614717CC"),
    ]:
        content = content.replace(old, new)
    cell["source"] = content.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

put(0, """# EXP-DENSENET121-TRANSFER-5SEEDS-022 — Split (10)

New five-seed DenseNet121 ImageNet transfer training run, using the EXP-018
architecture and two-stage recipe but **new weights and a new result namespace**.
The frozen inputs are `CashewData_Split (10).zip` and
`RL_Cashew_holdout (4).zip`.

Run all cells once, in order. The notebook audits both ZIPs, runs pipeline
smoke checks, trains all five seeds, writes a Validation-only checkpoint and
deployment selection lock, then runs Final Test and the external holdout.
Test and holdout predictions never select a seed, stage, threshold or
hyperparameter. Existing completed evaluation outputs are read back on rerun.

Split (10): Train 4,920, Val 1,445, Test 686. Holdout (4): 120 images in four
true classes; `not_cashew_leaf` remains a possible prediction but has no true
support. These two evaluation sets have different class support and are
reported separately.

Colab locates the project shortcut in the active account's My Drive. Set
`FORCE_REMOUNT_DRIVE=True` if the runtime still mounts the previous account.
Use the Drive path overrides if the shortcut has a custom location or the
shared project is read-only.

**Status:** code prepared; model training and numerical results pending.
""")

change(4, '"healthy": {"train": 828, "val": 216, "test": 118, "total": 1162}',
       '"healthy": {"train": 834, "val": 216, "test": 118, "total": 1168}')
change(4, 'NOTEBOOK_VERSION = "018"', 'NOTEBOOK_VERSION = "022"')
change(4, 'Optional path to this exact 018 .ipynb on Drive.',
       'Optional path to this exact 022 .ipynb on Drive.')
change(4, 'EXPECTED_TOTALS = {"train": 4914, "val": 1445, "test": 686, "total": 7045}',
       'EXPECTED_TOTALS = {"train": 4920, "val": 1445, "test": 686, "total": 7051}')
change(4, 'EXPECTED_REAL_FILENAME_COUNTS = {"train": 236, "val": 50, "test": 28}',
       'EXPECTED_REAL_FILENAME_COUNTS = {"train": 242, "val": 50, "test": 28}')
change(4, 'RUN_FULL_TRAINING = False', 'RUN_FULL_TRAINING = True')
change(4, 'RUN_REAL_HOLDOUT = True  # New 127-image archive; evaluate after Validation lock.',
       'RUN_REAL_HOLDOUT = True  # Holdout (4), run after the five-seed Validation lock.')
change(4, 'REAL_HOLDOUT_EXPECTED_COUNTS = {"anthracnose": 46, "healthy": 20, "leaf_miner": 36, "red_rust": 25}',
       'REAL_HOLDOUT_EXPECTED_COUNTS = {"anthracnose": 46, "healthy": 13, "leaf_miner": 36, "red_rust": 25}')
change(4, 'if RUN_FULL_TRAINING and (RUN_FINAL_TEST or RUN_REAL_HOLDOUT):\n'
       '    raise RuntimeError("Finish and lock all five Validation runs before enabling Test or holdout.")\n\n',
       '# Cell order enforces training -> Validation lock -> Test -> holdout.\n\n')
change(4, 'REQUIRED_ABSENT_IMAGES = []  # V08-specific removals are not asserted for V09.',
       'REQUIRED_ABSENT_IMAGES = []  # The Split (10) ZIP hash freezes exact membership.')

# The generic replacement above already changed the Split archive checksum.
change(10, 'len(holdout_preflight_df) != 127', 'len(holdout_preflight_df) != 120')
change(16, '"comparison_scope": "new V09 dataset; compare only with runs sharing the exact manifest",',
       '"comparison_scope": "new V10 dataset; compare only with runs sharing the exact manifest",')
change(16, '"baseline_source": "EXP-014 protocol; no checkpoints or prior metrics reused",',
       '"baseline_source": "EXP-018 protocol; no checkpoints or prior metrics reused",')
change(16, '"test_protocol": "disabled by default; evaluate only after Validation lock",',
       '"test_protocol": "run once after five-seed Validation selection lock in the same execution",')
change(16, '"holdout_protocol": "new 127-image four-class archive; Validation lock precedes evaluation",',
       '"holdout_protocol": "120-image four-class Holdout (4); Validation lock precedes evaluation",')
put(19, """## Final Test after the Validation lock

All five seeds and the Validation selection lock are completed first. This
cell then scores the 686-image internal Test once with the frozen checkpoints.
A completed result is read back on rerun. Test never changes the recipe.
""")
put(21, """## External real holdout (4)

The 120-image archive has 46 anthracnose, 13 healthy, 36 leaf miner and
25 red rust images. It is audited for exact byte overlap with every Train,
Validation and Test image in Split (10). All five locked models are scored.
Four-class macro metrics and support are reported separately from Test.
""")
change(22, 'if len(holdout_manifest) != 127:', 'if len(holdout_manifest) != 120:')
change(22, 'Holdout04 must have 127 scored images after exact-overlap audit',
       'Holdout04 must have 120 scored images after exact-overlap audit')
change(22, '"status": "EVALUATED_SEPARATELY", "evaluated_images": 127,',
       '"status": "EVALUATED_SEPARATELY", "evaluated_images": 120,')
change(22, 'V09/holdout04', 'V10/holdout04')
change(29, '"real_holdout": 127}[split]', '"real_holdout": 120}[split]')
change(30, 'Holdout has 127 images and four true classes.',
       'Holdout has 120 images and four true classes.')
change(30, 'two-stage protocol from EXP-014.', 'two-stage protocol from EXP-018.')
put(31, """## Completion check

- Split (10) and Holdout (4) archive SHA-256 values, classes and counts match.
- The dataset audit checks corrupt images, exact cross-split overlap and
  pHash candidates. The holdout preflight checks exact overlap before training.
- All five seeds finish both stages and Validation before selection is locked.
- Final Test and holdout run after the lock and write separate reports.
- EXP-022 archives contain the notebook, metrics, predictions, figures and
  checkpoints. No EXP-018 checkpoints are reused.
""")

embedded = base64.b64encode(zlib.compress(json.dumps(
    book, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 9)).decode("ascii")
archive = body(30)
if not re.search(r'^NOTEBOOK_SOURCE_ZLIB_B64 = ""$', archive, flags=re.M):
    raise RuntimeError("Empty embedded-source placeholder missing")
archive = archive.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                          f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"', 1)
put(30, archive)
TARGET.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(f"Wrote {TARGET.name}: {len(book['cells'])} cells, {TARGET.stat().st_size} bytes")
