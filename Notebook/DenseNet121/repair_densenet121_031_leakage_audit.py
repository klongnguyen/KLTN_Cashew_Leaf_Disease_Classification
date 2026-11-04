"""Fail clearly when EXP-031 leakage-audit image inputs are absent or empty."""

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parent / 'densenet121_yolo_evidence_gate_031.ipynb'
book = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
cell = next(c for c in book['cells'] if c['cell_type'] == 'code' and
            '# 03. LEAKAGE AUDIT' in ''.join(c['source']))
body = ''.join(cell['source'])

old = '''    def fingerprint_folder(root, label):
        rows = []
        for path in sorted(Path(root).rglob("*")):
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                rows.append({"source": label, "path": str(path), "basename": path.name,
                             "pixel_sha256": pixel_sha256(path), "phash64": phash64(path)})
        frame = pd.DataFrame(rows)
        print(f"  fingerprinted {label:<24} {len(frame):5d} images")
        return frame
'''
new = '''    IMAGE_INPUT_HINTS = {
        "yolo_train": "YOLO_TRAIN_IMAGE_DIR; extract NEG V03 dataset ZIP (Drive ID 1cRQ-kR1lsEvyqvs0Qn8rhUS4mvdi4nD2) and select its images/train folder",
        "classifier_holdout09": "HOLDOUT_IMAGE_DIR; extract RL_Cashew_holdout (9).zip (Drive ID 1QW2JYx8K6zkTs9y34f39OTBlxUOjWosZ)",
        "classifier_validation": "SPLIT13_VAL_DIR; extract CashewData_Split (13).zip (Drive ID 1HUE5ObO1vW0mP2QjgsdObMuW4RVPzNk7) and select val",
        "classifier_test": "SPLIT13_TEST_DIR; select test from the same Split 13 archive",
    }

    def image_paths(root, label):
        root = Path(root)
        hint = IMAGE_INPUT_HINTS[label]
        if not root.is_dir():
            raise FileNotFoundError(f"{label}: image directory does not exist: {root}\\nSet {hint}")
        paths = sorted(path for path in root.rglob("*")
                       if path.is_file() and path.suffix.lower() in
                       {".jpg", ".jpeg", ".png", ".bmp", ".webp"})
        if not paths:
            raise FileNotFoundError(f"{label}: no images found under {root}\\nSet {hint}")
        return paths

    def fingerprint_folder(root, label):
        rows = []
        for path in image_paths(root, label):
            rows.append({"source": label, "path": str(path), "basename": path.name,
                         "pixel_sha256": pixel_sha256(path), "phash64": phash64(path)})
        frame = pd.DataFrame(rows, columns=["source", "path", "basename", "pixel_sha256", "phash64"])
        print(f"  fingerprinted {label:<24} {len(frame):5d} images")
        return frame
'''
if old in body:
    body = body.replace(old, new, 1)
elif new not in body:
    raise AssertionError('Unexpected EXP-031 fingerprint_folder implementation')
cell['source'] = body.splitlines(keepends=True)

for i, entry in enumerate(book['cells']):
    if entry['cell_type'] == 'code':
        entry['execution_count'] = None
        entry['outputs'] = []
        text = ''.join(entry['source']).replace('    !pip install -q ultralytics\n', '    pass\n')
        ast.parse(text, filename=f'cell-{i}')
NOTEBOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding='utf-8')
print(NOTEBOOK.name, 'bytes', NOTEBOOK.stat().st_size)
