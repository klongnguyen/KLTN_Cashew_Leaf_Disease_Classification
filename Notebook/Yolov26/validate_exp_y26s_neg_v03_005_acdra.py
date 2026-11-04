"""Static and embedded-source checks for the EXP-005 Colab notebook."""

import ast
import base64
import json
import re
import zlib
from pathlib import Path

here = Path(__file__).resolve().parent
path = here / 'EXP-Y26S-NEG-V03-005_ACDRA.ipynb'
book = json.loads(path.read_text(encoding='utf-8'))
assert book['nbformat'] == 4
assert len(book['cells']) == 51
for i, cell in enumerate(book['cells']):
    if cell['cell_type'] == 'code':
        body = ''.join(cell['source'])
        if body.lstrip().startswith('!'):
            continue
        ast.parse(body, filename=f'cell-{i}')
        assert not cell['outputs']
        assert cell['execution_count'] is None

cells = [''.join(c['source']) for c in book['cells']]
config = next(c for c in cells if "EXPERIMENT_ID = f'EXP-" in c)
assert "MODEL_WEIGHTS = 'yolo26s.pt'" in config
assert "CLS_PW, MOSAIC, CLOSE_MOSAIC, MIXUP = 0.25, 0.4, 10, 0.0" in config
assert "'overlap_iou': 0.60" in config
assert "NMS_IOU = 0.70" in config
assert "FINAL_TEST_EVALUATION = False" in config
assert "DATASET_DRIVE_FILE_ID = '1cRQ-kR1lsEvyqvs0Qn8rhUS4mvdi4nD2'" in config
assert (here / 'adaptive_regions_v005.py').read_text(encoding='utf-8') in cells
assert (here / 'region_validation_v005.py').read_text(encoding='utf-8') in cells
export = next(c for c in cells if c.startswith('notebook_name = '))
encoded = re.search(r"^EMBEDDED_NOTEBOOK_B64 = '([A-Za-z0-9+/=]+)'$", export, re.M).group(1)
embedded = json.loads(zlib.decompress(base64.b64decode(encoded)))
assert len(embedded['cells']) == len(book['cells'])
embedded_export = next(''.join(c['source']) for c in embedded['cells']
                       if c['cell_type'] == 'code' and
                       ''.join(c['source']).startswith('notebook_name = '))
assert "EMBEDDED_NOTEBOOK_B64 = '__EMBEDDED_NOTEBOOK_B64__'" in embedded_export
assert "notebook_name = f'EXP-{MODEL_TAG}-NEG-V03-005_ACDRA.ipynb'" in export
print(f'Validated {len(book["cells"])} cells, source embedding, training and region config.')
