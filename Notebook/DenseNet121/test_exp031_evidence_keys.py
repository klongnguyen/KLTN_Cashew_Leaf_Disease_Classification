"""Check split-qualified EXP-031 joins against the embedded EXP-029 predictions."""

import base64
import csv
import io
import json
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath


book = json.loads((Path(__file__).resolve().parent /
                   'densenet121_yolo_evidence_gate_031.ipynb').read_text(encoding='utf-8'))
cells = [''.join(cell['source']) for cell in book['cells']]
bundle = next(text for text in cells if 'EXP029_PREDICTION_BUNDLE_B64' in text)
encoded = re.search(r'^EXP029_PREDICTION_BUNDLE_B64 = [\'"]([A-Za-z0-9+/=]+)[\'"]$',
                    bundle, re.M).group(1)
archive = zipfile.ZipFile(io.BytesIO(base64.b64decode(encoded)))
baseline_id = 'EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05'
keys, basenames = [], []
for split in ('validation', 'test', 'holdout'):
    member = (f'{baseline_id}/real_holdout/seed_42/predictions.csv' if split == 'holdout'
              else f'{baseline_id}/seed_42/{split}/predictions.csv')
    rows = list(csv.DictReader(io.StringIO(archive.read(member).decode('utf-8'))))
    for row in rows:
        original = row.get('relative_path') or row['filename']
        relpath = '/'.join(PurePosixPath(original).parts[-2:])
        assert relpath.split('/')[0] == (row.get('class') or row['true_label'])
        keys.append(f'{split}/{relpath}')
        basenames.append(PurePosixPath(original).name)
assert len(keys) == 1452 + 685 + 126
assert len(set(keys)) == len(keys)
duplicates = {name for name, count in Counter(basenames).items() if count > 1}
assert duplicates == {'real_healthy_037.jpg'}
assert {key for key in keys if key.endswith('/real_healthy_037.jpg')} == {
    'validation/healthy/real_healthy_037.jpg',
    'holdout/healthy/real_healthy_037.jpg',
}
assert any('EVIDENCE_BY_KEY = evidence_df.set_index("evidence_key")' in text for text in cells)
assert all('EVIDENCE_BY_BASENAME' not in text for text in cells)
assert any('CALIBRATION_EXCLUDED = (set(val_exact["evidence_key"])' in text for text in cells)
assert any('predictions_df["evidence_key"].map(' in text for text in cells)
print('EXP-031: 2263 unique split/class/image keys; the cross-split basename collision is isolated.')
