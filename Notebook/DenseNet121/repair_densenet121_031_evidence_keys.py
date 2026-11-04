"""Replace ambiguous basename joins with split-qualified relative image keys."""

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).resolve().parent / 'densenet121_yolo_evidence_gate_031.ipynb'
book = json.loads(NOTEBOOK.read_text(encoding='utf-8'))


def source(cell):
    return ''.join(cell['source'])


def put(cell, text):
    cell['source'] = text.splitlines(keepends=True)


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise AssertionError(f'Expected one source anchor; found {text.count(old)}: {old[:80]}')
    return text.replace(old, new, 1)


prediction_cell = next(c for c in book['cells'] if c['cell_type'] == 'code' and
                       '# 02. LOAD FROZEN EXP-029 PREDICTIONS' in source(c))
text = source(prediction_cell)
if 'out["evidence_key"]' not in text:
    text = replace_once(text,
        '''    out["basename"] = out["image_key"].map(lambda p: Path(p).name)
    probabilities = out[PROB_COLUMNS].to_numpy()
''',
        '''    out["basename"] = out["image_key"].map(lambda p: Path(p).name)
    # EXP-029 paths end in class/image.jpg for all three splits. The absolute
    # Colab prefix in Val/Test is not stable across runtimes.
    out["image_relpath"] = out["image_key"].map(
        lambda p: "/".join(Path(p).parts[-2:]))
    if not (out["image_relpath"].str.split("/").str[0] == out["true_label"]).all():
        raise ValueError(f"{split}/{seed}: image path class differs from true_label")
    out["evidence_key"] = out["split"] + "/" + out["image_relpath"]
    if out["evidence_key"].duplicated().any():
        raise ValueError(f"{split}/{seed}: duplicate split/class/image prediction key")
    probabilities = out[PROB_COLUMNS].to_numpy()
''')
    text = replace_once(text,
        'per_seed = group.groupby("seed")["image_key"].apply(lambda s: tuple(sorted(s)))',
        'per_seed = group.groupby("seed")["evidence_key"].apply(lambda s: tuple(sorted(s)))')
put(prediction_cell, text)

audit_cell = next(c for c in book['cells'] if c['cell_type'] == 'code' and
                  '# 03. LEAKAGE AUDIT' in source(c))
text = source(audit_cell)
if '"evidence_key": f"{split}/{relpath}"' not in text:
    text = replace_once(text,
        '''        for path in image_paths(root, label):
            rows.append({"source": label, "path": str(path), "basename": path.name,
                         "pixel_sha256": pixel_sha256(path), "phash64": phash64(path)})
        frame = pd.DataFrame(rows, columns=["source", "path", "basename", "pixel_sha256", "phash64"])
''',
        '''        split = {"classifier_validation": "validation", "classifier_test": "test",
                 "classifier_holdout09": "holdout", "yolo_train": "yolo_train"}[label]
        for path in image_paths(root, label):
            relpath = path.relative_to(root).as_posix()
            rows.append({"source": label, "path": str(path), "basename": path.name,
                         "image_relpath": relpath, "evidence_key": f"{split}/{relpath}",
                         "pixel_sha256": pixel_sha256(path), "phash64": phash64(path)})
        frame = pd.DataFrame(rows, columns=["source", "path", "basename", "image_relpath",
                                           "evidence_key", "pixel_sha256", "phash64"])
''')
    text = replace_once(text,
        '''                near_rows.append({"target": row["path"], "target_basename": row["basename"],
                                  "yolo_train": yolo_train_fp.iloc[index]["path"],
''',
        '''                near_rows.append({"target": row["path"],
                                  "target_evidence_key": row["evidence_key"],
                                  "target_basename": row["basename"],
                                  "yolo_train": yolo_train_fp.iloc[index]["path"],
''')
    text = replace_once(text,
        '''    CALIBRATION_EXCLUDED = set(val_exact["basename"]) | set(val_near.get("target_basename", pd.Series(dtype=str)))
''',
        '''    CALIBRATION_EXCLUDED = (set(val_exact["evidence_key"]) |
                            set(val_near.get("target_evidence_key", pd.Series(dtype=str))))
''')
put(audit_cell, text)

inference_cell = next(c for c in book['cells'] if c['cell_type'] == 'code' and
                      '# 04. YOLO INFERENCE' in source(c))
text = source(inference_cell)
if 'EVIDENCE_BY_KEY = ' not in text:
    text = replace_once(text,
        '''                row = {"source": label, "image_key": str(path), "basename": path.name}
''',
        '''                split = {"classifier_validation": "validation", "classifier_test": "test",
                         "classifier_holdout09": "holdout", "yolo_validation": "yolo_validation"}[label]
                relpath = path.relative_to(image_dir).as_posix()
                row = {"source": label, "image_key": str(path), "basename": path.name,
                       "image_relpath": relpath, "evidence_key": f"{split}/{relpath}"}
''')
    text = replace_once(text,
        '''    evidence_yolo_val = pd.read_csv(OUT_DIR / "yolo_val_evidence_with_gt.csv")
    print("Loaded cached evidence for", len(evidence_df), "images")

EVIDENCE_BY_BASENAME = evidence_df.set_index("basename")
if EVIDENCE_BY_BASENAME.index.duplicated().any():
    raise ValueError("Image basenames are not unique across splits; key the join on full paths instead")
''',
        '''    yolo_val_cache = OUT_DIR / "yolo_val_evidence_with_gt.csv"
    evidence_yolo_val = pd.read_csv(yolo_val_cache) if yolo_val_cache.is_file() else None
    print("Loaded cached evidence for", len(evidence_df), "images")

# Migrate cached evidence written by an older EXP-031 cell in the same runtime.
# Cross-runtime cache paths are refused; the detector must score those images again.
if "evidence_key" not in evidence_df.columns:
    roots = {"classifier_validation": Path(SPLIT13_VAL_DIR),
             "classifier_test": Path(SPLIT13_TEST_DIR),
             "classifier_holdout09": Path(HOLDOUT_IMAGE_DIR)}
    names = {"classifier_validation": "validation", "classifier_test": "test",
             "classifier_holdout09": "holdout"}
    relpaths, keys = [], []
    for row in evidence_df.itertuples(index=False):
        root = roots[row.source]
        path = Path(row.image_key)
        if not path.is_relative_to(root):
            raise RuntimeError("Cached detector evidence uses different image paths. "
                               "Set RUN_YOLO_INFERENCE=True and rerun this cell.")
        relpath = path.relative_to(root).as_posix()
        relpaths.append(relpath)
        keys.append(f"{names[row.source]}/{relpath}")
    evidence_df["image_relpath"] = relpaths
    evidence_df["evidence_key"] = keys
    evidence_df.to_csv(OUT_DIR / "yolo_image_evidence.csv", index=False)
    print("Migrated cached detector evidence to split-qualified image keys")
if evidence_df["evidence_key"].duplicated().any():
    duplicated = evidence_df.loc[evidence_df["evidence_key"].duplicated(), "evidence_key"].tolist()
    raise ValueError(f"Duplicate detector evidence keys: {duplicated[:5]}")
expected_keys = set(predictions_df["evidence_key"])
actual_keys = set(evidence_df["evidence_key"])
if expected_keys != actual_keys:
    raise ValueError(f"Detector evidence image mismatch: "
                     f"missing={sorted(expected_keys - actual_keys)[:5]}, "
                     f"extra={sorted(actual_keys - expected_keys)[:5]}")
EVIDENCE_BY_KEY = evidence_df.set_index("evidence_key")
print("Verified unique detector evidence for", len(EVIDENCE_BY_KEY), "split-qualified images")
''')
if 'gt_rows.append({"basename": row["basename"],' in text:
    text = replace_once(text,
        '''            gt_rows.append({"basename": row["basename"],
''',
        '''            gt_rows.append({"evidence_key": row["evidence_key"],
''')
    text = replace_once(text,
        '''        evidence_yolo_val = evidence_yolo_val.merge(pd.DataFrame(gt_rows), on="basename",
''',
        '''        evidence_yolo_val = evidence_yolo_val.merge(pd.DataFrame(gt_rows), on="evidence_key",
''')
put(inference_cell, text)

calibration_cell = next(c for c in book['cells'] if c['cell_type'] == 'code' and
                        '# 05. THE GATE' in source(c))
text = source(calibration_cell)
if 'EVIDENCE_BY_BASENAME' in text:
    text = text.replace('predictions_df["basename"].map(\n        EVIDENCE_BY_BASENAME',
                        'predictions_df["evidence_key"].map(\n        EVIDENCE_BY_KEY')
    text = replace_once(text,
        '(~predictions_df["basename"].isin(CALIBRATION_EXCLUDED))',
        '(~predictions_df["evidence_key"].isin(CALIBRATION_EXCLUDED))')
    text = replace_once(text,
        '''    unique_images = calibration.drop_duplicates("basename")[
        ["basename", "true_label"] + [f"max_conf_{n}" for n in DETECTOR_CLASSES]]
''',
        '''    unique_images = calibration.drop_duplicates("evidence_key")[
        ["evidence_key", "true_label"] + [f"max_conf_{n}" for n in DETECTOR_CLASSES]]
''')
if 'EVIDENCE_BY_BASENAME' in text:
    raise AssertionError('Ambiguous basename evidence lookup remains')
put(calibration_cell, text)

for i, cell in enumerate(book['cells']):
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
        ast.parse(source(cell).replace('    !pip install -q ultralytics\n', '    pass\n'),
                  filename=f'cell-{i}')
NOTEBOOK.write_text(json.dumps(book, ensure_ascii=False, indent=1), encoding='utf-8')
print(NOTEBOOK.name, 'cells', len(book['cells']), 'bytes', NOTEBOOK.stat().st_size)
