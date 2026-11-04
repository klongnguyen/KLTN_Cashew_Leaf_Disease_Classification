"""Add verified Drive export for the completed EXP-031 result archive."""

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / 'densenet121_yolo_evidence_gate_031.ipynb'
EXPORT = (HERE / 'export_031_results_cell.py').read_text(encoding='utf-8')
book = json.loads(NOTEBOOK.read_text(encoding='utf-8'))

def source(cell):
    return ''.join(cell['source'])

def put(cell, value):
    cell['source'] = value.splitlines(keepends=True)

config = next(cell for cell in book['cells'] if
              cell['cell_type'] == 'code' and 'EXPERIMENT_ID = "EXP-DENSENET121-YOLOGATE-031"' in source(cell))
text = source(config)
block = ('DRIVE_TARGET_FOLDER_ID = "1DGKTnzsvVc60RIEn5lQgflR3s-loM24G"\n'
         'UPLOAD_RESULT_ZIPS_TO_DRIVE = True\n')
if block not in text:
    anchor = 'NOTEBOOK_VERSION = "031"\n'
    if text.count(anchor) != 1:
        raise AssertionError('EXP-031 config anchor missing')
    text = text.replace(anchor, anchor + block, 1)
put(config, text)

book['cells'] = [cell for cell in book['cells']
                 if not source(cell).startswith('## Archive and upload EXP-031 results') and
                 not source(cell).startswith('"""Archive EXP-031 results and upload')]
book['cells'].append({'cell_type': 'markdown', 'metadata': {},
    'source': ['## Archive and upload EXP-031 results\n', '\n',
               'After the final evaluation, this cell creates one `RESULTS.zip` and uploads it '
               'directly to the confirmed `current_dataset` Drive folder. '
               'The upload is checked by size, MD5 and parent folder.\n']})
book['cells'].append({'cell_type': 'code', 'execution_count': None,
                      'metadata': {}, 'outputs': [], 'source': EXPORT.splitlines(keepends=True)})
for i, cell in enumerate(book['cells']):
    if cell['cell_type'] == 'code':
        ast.parse(source(cell).replace('    !pip install -q ultralytics\n', '    pass\n'),
                  filename=f'cell-{i}')
NOTEBOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding='utf-8')
print(NOTEBOOK.name, 'cells', len(book['cells']), 'bytes', NOTEBOOK.stat().st_size)
