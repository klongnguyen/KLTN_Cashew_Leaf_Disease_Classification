"""Make EXP-031 self-contained with the frozen EXP-029 prediction tables.

Usage: python repair_densenet121_031_predictions.py path/to/EXP-029_ANALYSIS.zip
"""

import argparse
import base64
import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
NOTEBOOK = HERE / 'densenet121_yolo_evidence_gate_031.ipynb'
BASELINE_ID = 'EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05'
SEEDS = (42, 123, 2026, 3407, 7777)
EXPECTED = [f'{BASELINE_ID}/seed_{seed}/{split}/predictions.csv'
            for split in ('validation', 'test') for seed in SEEDS]
EXPECTED += [f'{BASELINE_ID}/real_holdout/seed_{seed}/predictions.csv'
             for seed in SEEDS]
CELL_MARKER = '# 01a. FROZEN EXP-029 PREDICTION BUNDLE'


def cell_source(cell):
    return ''.join(cell['source'])


def set_source(cell, source):
    cell['source'] = source.splitlines(keepends=True)


def make_bundle(analysis_zip):
    source_hash = hashlib.sha256(analysis_zip.read_bytes()).hexdigest()
    payload = io.BytesIO()
    manifest = {'source_analysis_zip_sha256': source_hash, 'members': {}}
    with zipfile.ZipFile(analysis_zip) as archive, zipfile.ZipFile(
            payload, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        if archive.testzip() is not None:
            raise ValueError('EXP-029 ANALYSIS ZIP is corrupt')
        for name in EXPECTED:
            data = archive.read(name)
            if not data:
                raise ValueError(f'Empty EXP-029 predictions: {name}')
            manifest['members'][name] = hashlib.sha256(data).hexdigest()
            bundle.writestr(name, data)
        bundle.writestr('manifest.json', json.dumps(manifest, sort_keys=True))
    raw = payload.getvalue()
    return base64.b64encode(raw).decode('ascii'), hashlib.sha256(raw).hexdigest(), manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('analysis_zip', type=Path)
    args = parser.parse_args()
    if not args.analysis_zip.is_file():
        parser.error(f'not found: {args.analysis_zip}')
    encoded, bundle_hash, manifest = make_bundle(args.analysis_zip)
    book = json.loads(NOTEBOOK.read_text(encoding='utf-8'))
    book = copy.deepcopy(book)
    book['cells'] = [c for c in book['cells'] if CELL_MARKER not in cell_source(c)]
    for cell in book['cells']:
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

    config = next(c for c in book['cells'] if 'CLASSIFIER_ANALYSIS_ZIP = ' in cell_source(c))
    config_text = cell_source(config)
    config_text = config_text.replace(
        'CLASSIFIER_ANALYSIS_FILE_ID = None          # Drive ID of the EXP-029 ANALYSIS ZIP',
        'CLASSIFIER_ANALYSIS_FILE_ID = None          # unused when embedded tables are available')
    config_text = config_text.replace(
        'CLASSIFIER_ANALYSIS_OVERRIDE = None         # or an exact local path',
        'CLASSIFIER_ANALYSIS_OVERRIDE = None         # optional extracted folder or ANALYSIS ZIP path')
    set_source(config, config_text)

    bundle_cell = {
        'cell_type': 'code', 'execution_count': None,
        'metadata': {'jupyter': {'source_hidden': True}}, 'outputs': [],
        'source': [],
    }
    set_source(bundle_cell, (
        f'{CELL_MARKER}\n'
        '# The 15 real EXP-029 prediction CSVs, extracted from its ANALYSIS ZIP.\n'
        f'# Source ANALYSIS ZIP SHA256: {manifest["source_analysis_zip_sha256"]}\n'
        f'EXP029_PREDICTION_BUNDLE_SHA256 = {bundle_hash!r}\n'
        f'EXP029_PREDICTION_BUNDLE_B64 = {encoded!r}\n'))
    resolve_index = next(i for i, c in enumerate(book['cells'])
                         if '# 01b. RESOLVE PATHS' in cell_source(c))
    book['cells'].insert(resolve_index, bundle_cell)

    resolver = book['cells'][resolve_index + 1]
    old = '''# EXP-029 ANALYSIS ZIP, extracted:
ANALYSIS_ROOT = Path(CLASSIFIER_ANALYSIS_OVERRIDE) if CLASSIFIER_ANALYSIS_OVERRIDE else (
    PROJECT_ROOT / "exp029" / BASELINE_EXPERIMENT_ID)
'''
    new = '''# EXP-029: use an extracted folder or ZIP override, otherwise stage the
# verified 15-table prediction bundle included in this notebook.
import base64, io
_expected_029_members = [
    f"{BASELINE_EXPERIMENT_ID}/seed_{seed}/{split}/predictions.csv"
    for split in ("validation", "test") for seed in SEEDS
] + [f"{BASELINE_EXPERIMENT_ID}/real_holdout/seed_{seed}/predictions.csv"
     for seed in SEEDS]
_exp029_stage_root = PROJECT_ROOT / "exp029"
ANALYSIS_ROOT = (_exp029_stage_root / BASELINE_EXPERIMENT_ID)
_analysis_source = "already extracted"
_archive = None
if CLASSIFIER_ANALYSIS_OVERRIDE:
    _override = Path(CLASSIFIER_ANALYSIS_OVERRIDE).expanduser()
    if _override.is_dir():
        ANALYSIS_ROOT = _override
        _analysis_source = "configured extracted folder"
    elif _override.is_file() and _override.suffix.lower() == ".zip":
        _analysis_source = f"configured ZIP: {_override}"
        _archive = zipfile.ZipFile(_override)
    else:
        raise FileNotFoundError(f"CLASSIFIER_ANALYSIS_OVERRIDE is not a folder or ZIP: {_override}")
elif not all((ANALYSIS_ROOT / name.removeprefix(BASELINE_EXPERIMENT_ID + "/")).is_file()
             for name in _expected_029_members):
    _bundle_bytes = base64.b64decode(EXP029_PREDICTION_BUNDLE_B64, validate=True)
    if hashlib.sha256(_bundle_bytes).hexdigest() != EXP029_PREDICTION_BUNDLE_SHA256:
        raise ValueError("Embedded EXP-029 prediction bundle SHA256 mismatch")
    _archive = zipfile.ZipFile(io.BytesIO(_bundle_bytes))
    _analysis_source = "embedded real EXP-029 ANALYSIS predictions"
if _archive is not None:
    with _archive as _source:
        if _source.testzip() is not None:
            raise ValueError("EXP-029 prediction archive is corrupt")
        _manifest = (json.loads(_source.read("manifest.json"))
                     if "manifest.json" in _source.namelist() else None)
        for _member in _expected_029_members:
            _data = _source.read(_member)
            if _manifest and hashlib.sha256(_data).hexdigest() != _manifest["members"][_member]:
                raise ValueError(f"EXP-029 prediction SHA256 mismatch: {_member}")
            _target = _exp029_stage_root / _member
            if not _target.resolve().is_relative_to(_exp029_stage_root.resolve()):
                raise ValueError(f"Unsafe EXP-029 archive member: {_member}")
            _target.parent.mkdir(parents=True, exist_ok=True)
            _target.write_bytes(_data)
_missing_029 = [str(ANALYSIS_ROOT / name.removeprefix(BASELINE_EXPERIMENT_ID + "/"))
                for name in _expected_029_members
                if not (ANALYSIS_ROOT / name.removeprefix(BASELINE_EXPERIMENT_ID + "/")).is_file()]
if _missing_029:
    raise FileNotFoundError("EXP-029 needs 15 prediction tables; missing:\\n" +
                            "\\n".join(_missing_029[:5]) +
                            "\\nSet CLASSIFIER_ANALYSIS_OVERRIDE to its ANALYSIS ZIP or extracted folder.")
print("EXP-029 prediction source:", _analysis_source, "| files:", len(_expected_029_members))
'''
    resolver_text = cell_source(resolver)
    if old in resolver_text:
        set_source(resolver, resolver_text.replace(old, new, 1))
    elif '# EXP-029: use an extracted folder or ZIP override' in resolver_text:
        start = resolver_text.index('# EXP-029: use an extracted folder or ZIP override')
        end = resolver_text.index('# Classifier image folders', start)
        set_source(resolver, resolver_text[:start] + new + '\n' + resolver_text[end:])
    else:
        raise AssertionError('Expected resolver block not found')

    intro = cell_source(book['cells'][0])
    intro = intro.replace(
        '**Before running:** set `DETECTOR_WEIGHTS_FILE_ID` (or `_OVERRIDE`) to the\n'
        '`best.pt` of `EXP-Y26S-NEG-V03-001-CLASS-WEIGHTED`, and point',
        '**Before running:** EXP-029 prediction tables are embedded and checksum-verified.\n'
        'Point')
    set_source(book['cells'][0], intro)

    NOTEBOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding='utf-8')
    print(f'Updated {NOTEBOOK.name}: {len(book["cells"])} cells, '
          f'{len(EXPECTED)} predictions, bundle {len(encoded)} base64 chars.')


if __name__ == '__main__':
    main()
