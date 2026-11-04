"""Repair EXP-029 notebook-source export without changing its trained recipe."""

import ast
import base64
import copy
import json
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTED = HERE / "densenet121_transfer_photometric_029.ipynb"
CLEAN = HERE / "densenet121_transfer_photometric_029_fixed.ipynb"
REVISION = "9c23ac0551785a34e06a1de362382bd57af785093ce4ddc4664d0ae83bae1086"


def source(cell):
    return "".join(cell["source"])


def put(cell, value):
    cell["source"] = value.splitlines(keepends=True)


book = json.loads(EXECUTED.read_text(encoding="utf-8"))
if len(book["cells"]) != 36:
    raise ValueError("Unexpected EXP-029 cell count")
config = source(book["cells"][4])
if (f'CODE_REVISION_SHA256 = "{REVISION}"' not in config or
        'NOTEBOOK_VERSION = "029"' not in config):
    raise ValueError("This is not the completed EXP-029 recipe")
original_export = source(book["cells"][32])
old_header = ("NOTEBOOK_SOURCE_ZLIB_B64 = None  # EXP-029: set "
              "NOTEBOOK_SOURCE_PATH instead of embedding the source.\n")
if not original_export.startswith(old_header):
    raise ValueError("EXP-029 export cell has changed; inspect before applying repair")
old_copy = '''    notebook_copy = notebook_dir / "densenet121_transfer_split12_holdout_new_027.ipynb"
    if NOTEBOOK_SOURCE_PATH is not None:
        shutil.copyfile(NOTEBOOK_SOURCE_PATH, notebook_copy)
    elif NOTEBOOK_SOURCE_ZLIB_B64:
        source_book = json.loads(zlib.decompress(base64.b64decode(NOTEBOOK_SOURCE_ZLIB_B64)))
        notebook_copy.write_text(json.dumps(source_book, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        raise RuntimeError("Set NOTEBOOK_SOURCE_PATH to this notebook when rerunning an archived copy")'''
new_copy = '''    notebook_copy = notebook_dir / "densenet121_transfer_photometric_029.ipynb"
    if NOTEBOOK_SOURCE_PATH is not None:
        source_book = json.loads(Path(NOTEBOOK_SOURCE_PATH).read_text(encoding="utf-8"))
    elif NOTEBOOK_SOURCE_ZLIB_B64:
        source_book = json.loads(zlib.decompress(base64.b64decode(NOTEBOOK_SOURCE_ZLIB_B64)))
        # The compressed snapshot uses an empty placeholder to avoid self-reference.
        # Restore its payload so archived copies can run this export cell again.
        source_book["cells"][32]["source"][0] = (
            f'NOTEBOOK_SOURCE_ZLIB_B64 = "{NOTEBOOK_SOURCE_ZLIB_B64}"\\n')
    else:
        raise RuntimeError("Notebook source missing; use the corrected EXP-029 notebook")
    source_config = "".join(source_book["cells"][4]["source"])
    if f'CODE_REVISION_SHA256 = "{CODE_REVISION_SHA256}"' not in source_config:
        raise RuntimeError("Notebook source revision differs from this trained run")
    notebook_copy.write_text(json.dumps(source_book, ensure_ascii=False, indent=1),
                             encoding="utf-8")'''
if original_export.count(old_copy) != 1:
    raise ValueError("EXP-029 source-copy block is not the expected one")

clean = copy.deepcopy(book)
for cell in clean["cells"]:
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
clean_export = original_export.replace(old_copy, new_copy, 1).replace(
    old_header, 'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n', 1)
put(clean["cells"][32], clean_export)
snapshot = copy.deepcopy(clean)
encoded = base64.b64encode(
    zlib.compress(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"), 9)
).decode("ascii")
clean_export = clean_export.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""\n',
                                    f'NOTEBOOK_SOURCE_ZLIB_B64 = "{encoded}"\n', 1)
put(clean["cells"][32], clean_export)
put(book["cells"][32], clean_export)
book["cells"][32]["execution_count"] = None
book["cells"][32]["outputs"] = []  # Clear the resolved traceback only.

for value in (book, clean):
    for cell in value["cells"]:
        if cell["cell_type"] == "code":
            code = "\n".join(
                "pass" if line.lstrip().startswith("!") else line
                for line in source(cell).splitlines()
            )
            ast.parse(code)
    payload = source(value["cells"][32]).split('"', 2)[1]
    restored = json.loads(zlib.decompress(base64.b64decode(payload)))
    if len(restored["cells"]) != 36:
        raise ValueError("Embedded EXP-029 notebook is incomplete")

for path, value in ((EXECUTED, book), (CLEAN, clean)):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    print(path, path.stat().st_size, "bytes")
