"""Send both EXP-029 result ZIPs directly to the confirmed Drive folder."""

import ast
import base64
import copy
import json
import re
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent
BOOKS = [HERE / 'densenet121_transfer_photometric_029_fixed.ipynb']
FOLDER_ID = '1DGKTnzsvVc60RIEn5lQgflR3s-loM24G'
REVISION = '9c23ac0551785a34e06a1de362382bd57af785093ce4ddc4664d0ae83bae1086'


def source(cell):
    return ''.join(cell['source'])


def put(cell, value):
    cell['source'] = value.splitlines(keepends=True)


books = [json.loads(path.read_text(encoding='utf-8')) for path in BOOKS]
if any(len(book['cells']) != 36 for book in books):
    raise ValueError('Unexpected EXP-029 notebook shape')
book = copy.deepcopy(books[0])

config = source(book['cells'][4])
if f'CODE_REVISION_SHA256 = "{REVISION}"' not in config:
    raise ValueError('EXP-029 recipe revision changed')
config = re.sub(r'^DRIVE_TARGET_FOLDER_ID = "[^"]+"$',
                f'DRIVE_TARGET_FOLDER_ID = "{FOLDER_ID}"', config, count=1, flags=re.M)
if f'DRIVE_TARGET_FOLDER_ID = "{FOLDER_ID}"' not in config:
    raise AssertionError('Drive target missing')
if 'UPLOAD_RESULT_ZIPS_TO_DRIVE = True' not in config:
    config = config.replace(f'DRIVE_TARGET_FOLDER_ID = "{FOLDER_ID}"',
                            f'DRIVE_TARGET_FOLDER_ID = "{FOLDER_ID}"\n'
                            'UPLOAD_RESULT_ZIPS_TO_DRIVE = True', 1)
put(book['cells'][4], config)

export = source(book['cells'][32])
export, n = re.subn(r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]+"\n',
                    'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n', export, count=1, flags=re.M)
if n != 1:
    raise AssertionError('Embedded notebook source not found')
export = re.sub(r'drive_target_folder_id = globals\(\)\.get\("DRIVE_TARGET_FOLDER_ID"\) or "[^"]+"',
                f'drive_target_folder_id = globals().get("DRIVE_TARGET_FOLDER_ID") or "{FOLDER_ID}"',
                export, count=1)
old_folder_block = '''        folder_name = EXPERIMENT_ID
        drive_target_folder_id = globals().get("DRIVE_TARGET_FOLDER_ID") or "1DGKTnzsvVc60RIEn5lQgflR3s-loM24G"
        query = f"name = '{folder_name}' and '{drive_target_folder_id}' in parents and trashed = false"
        found = service.files().list(q=query, fields="nextPageToken,files(id,name,mimeType)",
                                     pageSize=100, supportsAllDrives=True,
                                     includeItemsFromAllDrives=True).execute()
        folders = [x for x in found.get("files", []) if x["mimeType"] == "application/vnd.google-apps.folder"]
        if found.get("nextPageToken") or len(folders) > 1:
            raise RuntimeError("Ambiguous Drive experiment folder")
        folder_id = (folders[0]["id"] if folders else service.files().create(
            body={"name": folder_name, "mimeType": "application/vnd.google-apps.folder",
                  "parents": [drive_target_folder_id]}, fields="id", supportsAllDrives=True).execute()["id"])
'''
new_folder_block = '''        drive_target_folder_id = DRIVE_TARGET_FOLDER_ID
        destination = service.files().get(
            fileId=drive_target_folder_id, fields="id,name,mimeType,trashed",
            supportsAllDrives=True).execute()
        if (destination.get("mimeType") != "application/vnd.google-apps.folder" or
                destination.get("trashed")):
            raise RuntimeError(f"Invalid Drive ZIP destination: {drive_target_folder_id}")
        folder_id = drive_target_folder_id
        print("Uploading both EXP-029 ZIPs directly to Drive folder:",
              destination["name"], folder_id)
'''
if export.count(old_folder_block) == 1:
    export = export.replace(old_folder_block, new_folder_block, 1)
elif new_folder_block not in export:
    raise AssertionError('Unexpected Drive destination block')
export = re.sub(r'^    if RUNTIME == "Colab":$',
                '    if UPLOAD_RESULT_ZIPS_TO_DRIVE:', export, count=1, flags=re.M)
if '    if UPLOAD_RESULT_ZIPS_TO_DRIVE:\n' not in export:
    raise AssertionError('ZIP upload gate missing')
export = export.replace('''        from google.colab import auth
        auth.authenticate_user()
''', '''        if RUNTIME == "Colab":
            from google.colab import auth
            auth.authenticate_user()
''', 1)
put(book['cells'][32], export)

# Archive copies must retain the revised destination and a runnable export cell.
snapshot = copy.deepcopy(book)
payload = base64.b64encode(zlib.compress(
    json.dumps(snapshot, ensure_ascii=False).encode('utf-8'), 9)).decode('ascii')
put(book['cells'][32], export.replace('NOTEBOOK_SOURCE_ZLIB_B64 = ""\n',
                                     f'NOTEBOOK_SOURCE_ZLIB_B64 = "{payload}"\n', 1))

for cell in book['cells']:
    if cell['cell_type'] == 'code':
        body = '\n'.join('pass' if line.lstrip().startswith('!') else line
                         for line in source(cell).splitlines())
        ast.parse(body)
embedded = json.loads(zlib.decompress(base64.b64decode(payload)))
assert f'DRIVE_TARGET_FOLDER_ID = "{FOLDER_ID}"' in source(embedded['cells'][4])
assert 'folder_id = drive_target_folder_id' in source(embedded['cells'][32])

for path, original in zip(BOOKS, books):
    revised = copy.deepcopy(original)
    put(revised['cells'][4], config)
    put(revised['cells'][32], source(book['cells'][32]))
    revised['cells'][32]['execution_count'] = None
    revised['cells'][32]['outputs'] = []
    path.write_text(json.dumps(revised, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    print(path.name, path.stat().st_size, 'bytes')
