"""Use the pinned, hash-verified GitHub YOLO checkpoint in EXP-031."""

import ast
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / 'densenet121_yolo_evidence_gate_031.ipynb'
HELPER = (HERE / 'detector_weights_031.py').read_text(encoding='utf-8')
COMMIT = 'ac1efe16319052b6f08a3c41b4ec361bbbb19e2f'
URL = ('https://raw.githubusercontent.com/klongnguyen/'
       'KLTN_Cashew_Leaf_Disease_Classification/' + COMMIT +
       '/training_results/negative_evaluation/EXP-Y26S-NEG-V03-001_FULL.zip')


def source(cell):
    return ''.join(cell['source'])


def put(cell, value):
    cell['source'] = value.splitlines(keepends=True)


book = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
config = next(c for c in book['cells'] if c['cell_type'] == 'code' and
              'EXPERIMENT_ID = "EXP-DENSENET121-YOLOGATE-031"' in source(c))
config_text = source(config)
if 'DETECTOR_FULL_ZIP_URL = ' not in config_text:
    anchor = 'DETECTOR_FULL_ZIP = "EXP-Y26S-NEG-V03-001_FULL.zip"\n'
    if config_text.count(anchor) != 1:
        raise AssertionError('Checkpoint configuration anchor missing')
    config_text = config_text.replace(anchor,
        anchor + f'DETECTOR_FULL_ZIP_URL = "{URL}"\n'
        'DETECTOR_FULL_ZIP_EXPECTED_BYTES = 51515079\n', 1)
put(config, config_text)

resolver = next(c for c in book['cells'] if c['cell_type'] == 'code' and
                '# 01b. RESOLVE PATHS' in source(c))
resolver_text = source(resolver)
start = resolver_text.index('# best.pt, extracted from the FULL ZIP committed in the repo.')
end = resolver_text.index('for label, path in (', start)
replacement = (HELPER + '\n' +
    'DETECTOR_WEIGHTS_PATH = ensure_detector_weights(\n'
    '    PROJECT_ROOT, DETECTOR_WEIGHTS_OVERRIDE, DETECTOR_FULL_ZIP,\n'
    '    DETECTOR_FULL_ZIP_URL, DETECTOR_FULL_ZIP_EXPECTED_BYTES,\n'
    '    DETECTOR_FULL_ZIP_SHA256, DETECTOR_WEIGHTS_MEMBER, DETECTOR_WEIGHTS_SHA256)\n\n')
put(resolver, resolver_text[:start] + replacement + resolver_text[end:])

intro = book['cells'][0]
intro_text = source(intro)
intro_text = intro_text.replace(
    'Provide the verified YOLO26s `best.pt` through `DETECTOR_WEIGHTS_OVERRIDE`\n'
    'or place its FULL ZIP on the runtime before running the image-staging cell.',
    'The verified YOLO26s `best.pt` is loaded from an immutable GitHub commit\n'
    'when absent locally; `DETECTOR_WEIGHTS_OVERRIDE` remains available.')
put(intro, intro_text)

for i, cell in enumerate(book['cells']):
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
        ast.parse(source(cell).replace('    !pip install -q ultralytics\n', '    pass\n'),
                  filename=f'cell-{i}')
NOTEBOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding='utf-8')
print(NOTEBOOK.name, 'cells', len(book['cells']), 'bytes', NOTEBOOK.stat().st_size)
