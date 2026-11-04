"""Add optional archive staging before the EXP-031 leakage audit."""

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / 'densenet121_yolo_evidence_gate_031.ipynb'
STAGING = (HERE / 'stage_exp031_images_cell.py').read_text(encoding='utf-8')
book = json.loads(NOTEBOOK.read_text(encoding='utf-8'))

def source(cell):
    return ''.join(cell['source'])

def put(cell, value):
    cell['source'] = value.splitlines(keepends=True)

config = next(c for c in book['cells'] if c['cell_type'] == 'code' and
              'EXPERIMENT_ID = "EXP-DENSENET121-YOLOGATE-031"' in source(c))
text = source(config)
text = text.replace('YOLO_DATASET_FILE_ID = None                 # Drive ID of the YOLO dataset ZIP',
                    'YOLO_DATASET_FILE_ID = "1cRQ-kR1lsEvyqvs0Qn8rhUS4mvdi4nD2"', 1)
if 'YOLO_DATASET_ARCHIVE_NAME = ' not in text:
    text = text.replace('YOLO_DATASET_OVERRIDE = None',
                        'YOLO_DATASET_OVERRIDE = None\n'
                        'YOLO_DATASET_ARCHIVE_NAME = "CASHEW_YOLO_MASTER_FINAL_NEG_V03 (2).zip"\n'
                        'YOLO_DATASET_EXPECTED_BYTES = 246419660', 1)
if 'AUTO_STAGE_MISSING_INPUTS = ' not in text:
    text = text.replace('RUN_LEAKAGE_AUDIT = True',
                        'AUTO_STAGE_MISSING_INPUTS = True  # download verified source ZIPs when image folders are absent\n'
                        'DATASET_ARCHIVE_OVERRIDE = None\n'
                        'REAL_HOLDOUT_ARCHIVE_OVERRIDE = None\n'
                        'RUN_LEAKAGE_AUDIT = True', 1)
put(config, text)

intro = book['cells'][0]
intro_text = source(intro)
old_intro = ('**Before running:** EXP-029 prediction tables are embedded and checksum-verified.\n'
             'Point\n'
             '`YOLO_TRAIN_IMAGE_DIR`, `SPLIT13_VAL_DIR`, `SPLIT13_TEST_DIR` and\n'
             '`HOLDOUT_IMAGE_DIR` at the extracted image folders.')
new_intro = ('**Before running:** EXP-029 prediction tables are embedded and checksum-verified.\n'
             'Missing Split 13, Holdout 09 and YOLO NEG V03 image folders are staged from Drive.\n'
             'Provide the verified YOLO26s `best.pt` through `DETECTOR_WEIGHTS_OVERRIDE`\n'
             'or place its FULL ZIP on the runtime before running the image-staging cell.')
if old_intro in intro_text:
    put(intro, intro_text.replace(old_intro, new_intro, 1))

book['cells'] = [c for c in book['cells'] if
                 not source(c).startswith('"""Stage the original EXP-029') and
                 not source(c).startswith('## Prepare image inputs for leakage audit')]
audit_index = next(i for i, c in enumerate(book['cells']) if
                   '# 03. LEAKAGE AUDIT' in source(c))
book['cells'][audit_index:audit_index] = [
    {'cell_type': 'markdown', 'metadata': {},
     'source': ['## Prepare image inputs for leakage audit\n', '\n',
                'If the original Split 13, Holdout 09 or YOLO NEG V03 image folders are missing, '
                'this cell downloads the exact source ZIPs from Drive, verifies archive size and '
                'checksum, extracts safely, and matches classifier filenames to the frozen EXP-029 predictions. '
                'Set the three `*_OVERRIDE` paths in CONFIG to use local ZIPs.\n']},
    {'cell_type': 'code', 'execution_count': None, 'metadata': {},
     'outputs': [], 'source': STAGING.splitlines(keepends=True)},
]
for i, cell in enumerate(book['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(source(cell).replace('    !pip install -q ultralytics\n', '    pass\n'),
                  filename=f'cell-{i}')
NOTEBOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding='utf-8')
print(NOTEBOOK.name, 'cells', len(book['cells']), 'bytes', NOTEBOOK.stat().st_size)
