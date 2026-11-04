"""Check that EXP-031 can stage and load the real EXP-029 probabilities."""

import ast
import csv
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


here = Path(__file__).resolve().parent
book = json.loads((here / 'densenet121_yolo_evidence_gate_031.ipynb').read_text(encoding='utf-8'))
cells = [''.join(cell['source']) for cell in book['cells']]
assert len(cells) == 19
for i, cell in enumerate(book['cells']):
    if cell['cell_type'] == 'code':
        text = cells[i].replace('    !pip install -q ultralytics\n', '    pass\n')
        ast.parse(text, filename=f'cell-{i}')
        assert not cell['outputs']

namespace = {'Path': Path, 'zipfile': zipfile, 'hashlib': hashlib,
             'json': json}
namespace['file_sha256'] = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
old_cwd = Path.cwd()
with tempfile.TemporaryDirectory() as temporary:
    try:
        os.chdir(temporary)
        for i in (2, 3, 4):
            if i == 4:
                weights = Path(temporary) / 'detector' / 'best.pt'
                weights.parent.mkdir(parents=True, exist_ok=True)
                weights.write_bytes(b'local-synthetic-weights')
                namespace['DETECTOR_WEIGHTS_SHA256'] = hashlib.sha256(
                    weights.read_bytes()).hexdigest()
            exec(compile(cells[i], f'cell-{i}', 'exec'), namespace)
        root = namespace['ANALYSIS_ROOT']
        assert sum(p.name == 'predictions.csv' for p in root.rglob('predictions.csv')) == 15
        classes = namespace['CLASS_NAMES']
        accuracies = {}
        for split in ('validation', 'test', 'holdout'):
            values = []
            for seed in namespace['SEEDS']:
                path = (root / 'real_holdout' / f'seed_{seed}' / 'predictions.csv'
                        if split == 'holdout' else
                        root / f'seed_{seed}' / split / 'predictions.csv')
                with path.open(encoding='utf-8', newline='') as stream:
                    rows = list(csv.DictReader(stream))
                correct = 0
                for row in rows:
                    truth = row.get('class') or row['true_label']
                    prediction = max(classes, key=lambda name: float(
                        row.get('probability_' + name, row.get('prob_' + name))))
                    correct += prediction == truth
                values.append(correct / len(rows))
            accuracies[split] = sum(values) / len(values)
        test = accuracies['test']
        holdout = accuracies['holdout']
        assert abs((test - holdout) * 100 - 13.54) < 0.2
        # Rerunning the resolver must not try to reuse a closed ZIP handle.
        exec(compile(cells[4], 'cell-4-rerun', 'exec'), namespace)
        # The new result ZIP cell must package completed output correctly.
        output = namespace['OUT_DIR']
        (output / 'exp031_summary.json').write_text('{"status":"test"}', encoding='utf-8')
        namespace['RUN_FINAL_EVALUATION'] = True
        namespace['UPLOAD_RESULT_ZIPS_TO_DRIVE'] = False  # local integrity test only
        namespace['save_json'] = lambda path, value: Path(path).write_text(
            json.dumps(value), encoding='utf-8')
        exec(compile(cells[-1], 'result-export', 'exec'), namespace)
        with zipfile.ZipFile(namespace['result_zip']) as archive:
            assert archive.testzip() is None
            assert any(name.endswith('/exp031_summary.json') for name in archive.namelist())
        audit = next(text for text in cells if '# 03. LEAKAGE AUDIT' in text)
        parsed = ast.parse(audit)
        conditional = next(node for node in parsed.body if isinstance(node, ast.If))
        image_paths_function = next(node for node in conditional.body
                                    if isinstance(node, ast.FunctionDef) and node.name == 'image_paths')
        isolated = ast.fix_missing_locations(ast.Module(body=[image_paths_function], type_ignores=[]))
        audit_namespace = {'Path': Path, 'IMAGE_INPUT_HINTS': {'yolo_train': 'YOLO_TRAIN_IMAGE_DIR'}}
        exec(compile(isolated, 'image-paths', 'exec'), audit_namespace)
        candidate = Path(temporary) / 'missing_yolo_images'
        for empty_directory in (False, True):
            if empty_directory:
                candidate.mkdir()
            try:
                audit_namespace['image_paths'](candidate, 'yolo_train')
            except FileNotFoundError as exc:
                assert 'YOLO_TRAIN_IMAGE_DIR' in str(exc)
            else:
                raise AssertionError('Missing/empty image folder must fail clearly')
        (candidate / 'sample.jpg').write_bytes(b'path-check-only')
        assert len(audit_namespace['image_paths'](candidate, 'yolo_train')) == 1
    finally:
        os.chdir(old_cwd)
print('EXP-031: 15 real EXP-029 tables loaded; baseline gap reproduced.')
