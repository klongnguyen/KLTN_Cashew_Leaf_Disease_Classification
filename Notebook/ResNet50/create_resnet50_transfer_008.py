"""Create an independent frozen-backbone ResNet50 transfer experiment 008.

The 007 notebook and its results are never edited. 008 changes Stage 2 to a
short head-only refinement and keeps all evaluation and archive guards.
"""

import base64
import copy
import json
import re
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "resnet50_transfer_5seed_007.ipynb"
TARGET = HERE / "resnet50_transfer_5seed_008.ipynb"
if TARGET.exists():
    raise FileExistsError(f"Review existing 008 notebook before replacing: {TARGET}")

notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
cells = notebook["cells"]


def get(index):
    return "".join(cells[index]["source"])


def put(index, value):
    cells[index]["source"] = value.splitlines(keepends=True)


def swap(index, old, new, expected=1):
    source = get(index)
    if source.count(old) != expected:
        raise ValueError(f"cell {index}: expected {expected} occurrences of {old!r}, got {source.count(old)}")
    put(index, source.replace(old, new))


for index, cell in enumerate(cells):
    cell["id"] = f"resnet008_cell_{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

# Prevent the archived 007 notebook from becoming the 008 source payload.
source, count = re.subn(
    r'^NOTEBOOK_SOURCE_ZLIB_B64 = "[^"]*"$',
    'NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
    get(24), count=1, flags=re.MULTILINE)
if count != 1:
    raise ValueError("Expected exactly one embedded notebook assignment")
put(24, source)

for cell in cells:
    source = "".join(cell["source"])
    source = source.replace("EXP-RESNET50-TRANSFER-5SEEDS-007",
                            "EXP-RESNET50-TRANSFER-5SEEDS-008")
    source = source.replace("resnet50_transfer_5seed_007.ipynb",
                            "resnet50_transfer_5seed_008.ipynb")
    source = source.replace("real_holdout_007", "real_holdout_008")
    source = source.replace("EXP-007", "EXP-008")
    cell["source"] = source.splitlines(keepends=True)

put(0, """# EXP-RESNET50-TRANSFER-5SEEDS-008: frozen-backbone transfer ablation

**NOT_EXECUTED.** This is a lower-capacity ResNet50 ImageNet transfer run on
the same frozen V08 dataset. Both stages train only the classification head;
Stage 2 is a short, lower-learning-rate refinement. No backbone weights or
BatchNorm statistics are updated. The 006 and 007 experiments are preserved.

This tests whether ResNet50 fine-tuning caused the Train/Validation gap seen
in the partial 006 logs. High accuracy at the end of epoch 1 can still be
normal: it follows about 300 gradient steps on pretrained ImageNet features.
The pre-fit, first-batch, balanced Train probe, Validation and real holdout
diagnostics help distinguish learning speed from overfitting.

Five seeds, Validation-only selection, Final Test and the historical real
holdout remain enabled. Test and holdout run only after the selection lock.
Both result ZIPs include this exact 008 notebook. No 008 result is claimed.
""")

swap(4, 'NOTEBOOK_VERSION = "007"', 'NOTEBOOK_VERSION = "008"')
swap(4, 'STAGE2_TRAINABLE_PREFIX = "conv5_block3_"\n', '')
swap(4, 'STAGE2_LR, STAGE2_EPOCHS = 1e-5, 12',
        'STAGE2_LR, STAGE2_EPOCHS = 3e-5, 6')
swap(4, 'STAGE2_EARLY_PATIENCE, STAGE2_REDUCE_PATIENCE, STAGE2_MIN_LR = 4, 2, 1e-7',
        'STAGE2_EARLY_PATIENCE, STAGE2_REDUCE_PATIENCE, STAGE2_MIN_LR = 2, 1, 1e-7')

model = get(12)
start, end = model.index('def set_backbone_stage('), model.index('def compile_stage(')
model = model[:start] + '''def set_backbone_stage(backbone, stage):
    if stage not in ("stage1", "stage2"):
        raise ValueError(stage)
    # 008 is a frozen-feature transfer ablation in both stages.
    backbone.trainable = False
    if any(layer.trainable for layer in backbone.layers):
        raise RuntimeError("Backbone must remain frozen in both stages")
    return 0

''' + model[end:]
old = 'if not 0 < stage2_params < backbone.count_params():\n        raise AssertionError("Stage 2 parameter count is implausible")'
new = 'if stage2_params != 0 or not model.trainable_weights:\n        raise AssertionError("Stage 2 must train only the classification head")'
if model.count(old) != 1:
    raise ValueError("Stage 2 smoke assertion changed")
put(12, model.replace(old, new))

swap(14,
     '"comparison_scope": "reduced Stage 2 capacity versus ResNet50 Transfer-006; no claim until 007 is run",',
     '"comparison_scope": "frozen backbone versus ResNet50 Transfer-007 on the same V08; no claim until 008 is run",')
swap(14, '"trainable_backbone": "conv5_block3_* non-BatchNorm only"',
         '"trainable_backbone": "none; classification head only"')

swap(20, 'REAL_HOLDOUT_HELPER_VERSION = "007.1"',
         'REAL_HOLDOUT_HELPER_VERSION = "008.1"')

package = get(22)
old_builder = '''def set_stage(backbone, stage):
    backbone.trainable = stage == "stage2"
    if stage == "stage2":
        for layer in backbone.layers:
            layer.trainable = layer.name.startswith("conv5_block3_") and not isinstance(
                layer, layers.BatchNormalization)
    elif stage != "stage1":
        raise ValueError(stage)'''
new_builder = '''def set_stage(backbone, stage):
    if stage not in ("stage1", "stage2"):
        raise ValueError(stage)
    backbone.trainable = False'''
if package.count(old_builder) != 1:
    raise ValueError("Model package builder changed")
package = package.replace(old_builder, new_builder)
old_continue = '''backbone.trainable = True
for layer in backbone.layers:
    layer.trainable = layer.name.startswith("conv5_block3_") and not isinstance(layer, keras.layers.BatchNormalization)
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),'''
new_continue = '''backbone.trainable = False
model.compile(optimizer=keras.optimizers.AdamW(learning_rate=3e-5, weight_decay=1e-4),'''
if package.count(old_continue) != 1:
    raise ValueError("Continue-training example changed")
put(22, package.replace(old_continue, new_continue))

report = get(24)
report = report.replace(
    '- Stage 2: only conv5_block3_* non-BN + head trainable, fresh AdamW LR {STAGE2_LR}, up to {STAGE2_EPOCHS} epochs.',
    '- Stage 2: backbone frozen, head-only refinement, fresh AdamW LR {STAGE2_LR}, up to {STAGE2_EPOCHS} epochs.')
report = report.replace(
    '- EXP-006 is the same-V08 transfer reference. EXP-008 changes Stage 2 to the final ResNet block only and a shorter fixed schedule.',
    '- EXP-007 is the same-V08 transfer reference. EXP-008 freezes the backbone in both stages and shortens Stage 2.')
put(24, report)

put(25, """## Interpretation and completion

- This is a frozen-backbone capacity test. Stage 1 and Stage 2 update only
  the 10,245-parameter five-class dense head. Stage 2 uses a fresh optimizer
  and a short lower-rate schedule. The ImageNet feature extractor remains
  strong, so high first-epoch accuracy alone is not evidence of leakage.
- Compare the saved pre-fit score, first batch, online Train score and fixed
  Train inference probe. Use the Validation loss and real holdout diagnostics
  to judge whether weaker fine-tuning helps field-image performance.
- pHash pairs are screening candidates, not confirmed duplicates. Do not
  alter V08 or choose hyperparameters using Final Test or holdout.
- Report performance only after five seeds, the Validation lock, Final Test,
  real holdout and both verified ZIPs finish.
""")

notebook["metadata"].setdefault("colab", {})["name"] = TARGET.name
snapshot = copy.deepcopy(notebook)
snapshot_source = get(24).replace('NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
                                  'NOTEBOOK_SOURCE_ZLIB_B64 = ""')
snapshot["cells"][24]["source"] = snapshot_source.splitlines(keepends=True)
payload = base64.b64encode(zlib.compress(
    json.dumps(snapshot, ensure_ascii=False).encode("utf-8"))).decode("ascii")
put(24, get(24).replace('NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"',
                        f'NOTEBOOK_SOURCE_ZLIB_B64 = "{payload}"'))
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print("Created", TARGET, "with", len(cells), "output-free cells")
