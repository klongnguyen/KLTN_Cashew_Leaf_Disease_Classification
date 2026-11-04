"""Keep the completed EXP-029 run and fix its Drive export target."""

import ast
import base64
import copy
import json
import re
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXECUTED = HERE / "densenet121_transfer_photometric_029.ipynb"
CLEAN = HERE / "densenet121_transfer_photometric_029_fixed.ipynb"
FOLDER_ID = "1-Mrqs8Y9dwLexSKIVKBvyYvvQ8X8fXti"
REVISION = "9c23ac0551785a34e06a1de362382bd57af785093ce4ddc4664d0ae83bae1086"


def source(cell):
    return "".join(cell["source"])


def put(cell, value):
    cell["source"] = value.splitlines(keepends=True)


executed = json.loads(EXECUTED.read_text(encoding="utf-8"))
clean = json.loads(CLEAN.read_text(encoding="utf-8"))
if len(executed["cells"]) != len(clean["cells"]) or len(clean["cells"]) != 36:
    raise ValueError("Unexpected EXP-029 notebook shape")
if any(source(a) != source(b) for a, b in zip(executed["cells"], clean["cells"])):
    raise ValueError("Executed and clean EXP-029 sources differ")

config = source(clean["cells"][4])
if f'CODE_REVISION_SHA256 = "{REVISION}"' not in config:
    raise ValueError("EXP-029 recipe revision changed")
anchor = 'NOTEBOOK_SOURCE_PATH = None  # Optional path to this exact 029 .ipynb on Drive.\n'
if config.count(anchor) != 1:
    raise ValueError("Cannot locate EXP-029 Drive configuration anchor")
config = config.replace(anchor, anchor +
                        f'DRIVE_TARGET_FOLDER_ID = "{FOLDER_ID}"\n', 1)
put(clean["cells"][4], config)
put(executed["cells"][4], config)

export = source(clean["cells"][32])
export, count = re.subn(r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]+"\n',
                        'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n', export, count=1)
if count != 1:
    raise ValueError("EXP-029 embedded source is missing")
old_query = "'{DRIVE_TARGET_FOLDER_ID}' in parents"
old_parent = '"parents": [DRIVE_TARGET_FOLDER_ID]'
if export.count(old_query) != 1 or export.count(old_parent) != 1:
    raise ValueError("EXP-029 Drive upload target changed")
export = export.replace(old_query, "'{drive_target_folder_id}' in parents", 1)
export = export.replace(old_parent, '"parents": [drive_target_folder_id]', 1)
anchor = '        folder_name = EXPERIMENT_ID\n'
if export.count(anchor) != 1:
    raise ValueError("Cannot locate EXP-029 Drive upload block")
export = export.replace(
    anchor, anchor +
    f'        drive_target_folder_id = globals().get("DRIVE_TARGET_FOLDER_ID") or "{FOLDER_ID}"\n',
    1,
)
put(clean["cells"][32], export)
snapshot = copy.deepcopy(clean)
payload = base64.b64encode(
    zlib.compress(json.dumps(snapshot, ensure_ascii=False).encode("utf-8"), 9)
).decode("ascii")
export = export.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""\n',
                        f'NOTEBOOK_SOURCE_ZLIB_B64 = "{payload}"\n', 1)
put(clean["cells"][32], export)
put(executed["cells"][32], export)
executed["cells"][32]["execution_count"] = None
executed["cells"][32]["outputs"] = []

for book in (executed, clean):
    for cell in book["cells"]:
        if cell["cell_type"] == "code":
            code = "\n".join(
                "pass" if line.lstrip().startswith("!") else line
                for line in source(cell).splitlines()
            )
            ast.parse(code)
    if f'CODE_REVISION_SHA256 = "{REVISION}"' not in source(book["cells"][4]):
        raise ValueError("EXP-029 revision changed")
    if 'drive_target_folder_id = globals().get' not in source(book["cells"][32]):
        raise ValueError("Drive fallback missing")

for path, book in ((EXECUTED, executed), (CLEAN, clean)):
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(book, ensure_ascii=False, indent=1) + "\n",
                         encoding="utf-8")
    temporary.replace(path)
    print(path, path.stat().st_size, "bytes")
