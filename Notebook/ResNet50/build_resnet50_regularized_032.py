"""Build the locked five-seed ResNet50 Stage-2 regularization experiment."""
import ast
import base64
import copy
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = ROOT / "resnet50_transfer_5seed_031.ipynb"
TARGET = ROOT / "resnet50_transfer_regularized_5seed_032.ipynb"
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))


def get(index):
    return "".join(notebook["cells"][index]["source"])


def put(index, source):
    notebook["cells"][index]["source"] = source.splitlines(keepends=True)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["outputs"] = []


def replace_one(source, old, new):
    assert source.count(old) == 1, (old[:90], source.count(old))
    return source.replace(old, new)


put(0, """# EXP-RESNET50-REGULARIZED-5SEEDS-032 — giới hạn Stage 2

Notebook này huấn luyện **ResNet50 ImageNet Transfer với 5 seed**. Mục tiêu là giảm khả năng overfitting của Stage 2 bằng cách chỉ mở `conv5_block3` và hạ learning rate từ `1e-5` xuống **`5e-6`**. Cấu hình được khóa trước khi đánh giá Test/Holdout; không chọn cấu hình bằng kết quả của hai tập này.

- ResNet50 backbone: 23.587.712 tham số; `conv5_block3` không gồm BatchNorm: **4.459.520 tham số (18,9%)**. Bản 031 mở `conv5_block1–3`: 14.953.472 (63,4%).
- Giữ nguyên Split (13), Holdout (9), thứ tự 5 lớp, seeds `[42, 123, 2026, 3407, 7777]`, đầu vào 256×256, batch 16, augmentation, AdamW/weight decay 1e-4, dropout 0.40, Stage 1, số epoch tối đa, callbacks, BatchNorm đóng băng và luật chọn checkpoint chỉ bằng Validation.
- Hai khác biệt Stage 2 được ghi trong `experiment_config.json`. Vì vậy phép so sánh với DenseNet121 EXP-029 có cùng dữ liệu và các điều kiện chung, **không phải** so sánh hai kiến trúc với lịch Stage 2 hoàn toàn giống nhau.
- Mỗi seed có lịch sử hai giai đoạn, classification report và confusion matrix; báo cáo tổng hợp gồm biểu đồ 2×5. Test và Holdout chỉ chạy sau Validation lock. Cell cuối tạo, xác minh và upload `ANALYSIS.zip` cùng `FULL.zip`.

Chạy tuần tự trong Colab GPU. Đây là thí nghiệm mới; kết quả không ghi đè EXP-031. Holdout (9) là đánh giá mô tả đã được biết trước, không phải blind benchmark.
""")

s = get(4)
s = replace_one(s, "# 03. CONFIGURATION — EXP-RESNET50-TRANSFER-5SEEDS-031",
                "# 03. CONFIGURATION — EXP-RESNET50-REGULARIZED-5SEEDS-032")
s = replace_one(s, 'NOTEBOOK_VERSION = "031"', 'NOTEBOOK_VERSION = "032"')
s = replace_one(s, 'CODE_REVISION_SHA256 = "c954b7c4692e8bd608fa792f9d344a40fa8ac95ceb56ae18633972227f226025"',
                'CODE_REVISION_SHA256 = "__CODE_REVISION__"')
s = replace_one(s, 'Optional path to this exact 031 .ipynb on Drive.',
                'Optional path to this exact 032 .ipynb on Drive.')
s = replace_one(s, 'EXPERIMENT_ID = f"EXP-RESNET50-TRANSFER-5SEEDS-031-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"',
                'EXPERIMENT_ID = f"EXP-RESNET50-REGULARIZED-5SEEDS-032-B3-LR5e6-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"')
s = replace_one(s, 'STAGE2_LR, STAGE2_EPOCHS = 1e-5, 30',
                'STAGE2_LR, STAGE2_EPOCHS = 5e-6, 30')
s = replace_one(s, '# Stage 2 opens the last ResNet stage (conv5_block1..3) with the same literal\n# prefix rule DenseNet121 uses for its last block; BatchNorm stays frozen.\nSTAGE2_UNFREEZE_PREFIX = "conv5_"',
                '# Deliberate Stage 2 regularization: only the final residual block; BatchNorm stays frozen.\nSTAGE2_UNFREEZE_PREFIX = "conv5_block3"')
s = replace_one(s, 'EXPECTED_STAGE2_TRAINABLE_PARAMS = 14_953_472',
                'EXPECTED_STAGE2_TRAINABLE_PARAMS = 4_459_520')
s = s.replace('Only the backbone, its in-model preprocessing and the architecture-dependent\n# parameter count of the Stage 2 stage are allowed to differ.',
              'Only the backbone, in-model preprocessing, and the two declared Stage 2\n# changes (unfrozen block and learning rate) are allowed to differ.')
put(4, s)

s = get(5)
s = replace_one(s,
'''FAIRNESS_DRIFT = {key: {"reference": FAIRNESS_REFERENCE[key], "this_notebook": FAIRNESS_LIVE[key]}
                  for key in FAIRNESS_REFERENCE if FAIRNESS_REFERENCE[key] != FAIRNESS_LIVE[key]}
if FAIRNESS_DRIFT:
    raise ValueError("Shared settings drifted from the DenseNet121 EXP-029 reference recipe: "
                     + json.dumps(FAIRNESS_DRIFT, indent=2, default=str))''',
'''FAIRNESS_DRIFT = {key: {"reference": FAIRNESS_REFERENCE[key], "this_notebook": FAIRNESS_LIVE[key]}
                  for key in FAIRNESS_REFERENCE if key != "stage2" and
                  FAIRNESS_REFERENCE[key] != FAIRNESS_LIVE[key]}
if FAIRNESS_DRIFT:
    raise ValueError("Shared settings drifted from the DenseNet121 EXP-029 reference recipe: "
                     + json.dumps(FAIRNESS_DRIFT, indent=2, default=str))
EXPECTED_STAGE2_VARIANT = {**FAIRNESS_REFERENCE["stage2"],
                           "lr": 5e-6, "unfreeze_prefix": "conv5_block3"}
if FAIRNESS_LIVE["stage2"] != EXPECTED_STAGE2_VARIANT:
    raise ValueError("Stage 2 differs from the declared regularized variant: "
                     + json.dumps(FAIRNESS_LIVE["stage2"], indent=2, default=str))''')
s = replace_one(s,
    '"stage2_trainable_rule": "same prefix rule conv5_ / non-BatchNorm; the parameter count follows the architecture",',
    '"stage2_trainable_rule": "conv5_block3 only, non-BatchNorm: 4,459,520 backbone parameters",\n    "stage2_learning_rate": "5e-6 instead of the DenseNet121 reference 1e-5",')
s = replace_one(s,
    'print("Fairness contract PASS:", len(FAIRNESS_REFERENCE), "shared settings identical to the DenseNet121 EXP-029 reference.")',
    'print("Fairness contract PASS:", len(FAIRNESS_REFERENCE)-1, "shared groups match DenseNet121 EXP-029; Stage 2 block/LR are declared exceptions.")')
put(5, s)

s = get(16)
s = replace_one(s, 'if unfrozen_names != ["conv5_block1", "conv5_block2", "conv5_block3"]:',
                'if unfrozen_names != ["conv5_block3"]:')
put(16, s)

put(17, """## Khóa cấu hình rồi huấn luyện hai giai đoạn

Mỗi seed bắt đầu từ trọng số ImageNet mới. Stage 1 giữ nguyên bản 031. Stage 2 chỉ mở `conv5_block3` (BatchNorm vẫn đóng băng), dùng optimizer mới với LR `5e-6`. Checkpoint và seed triển khai được chọn chỉ từ Validation. Test/Holdout được đọc sau khi khóa lựa chọn.
""")

s = get(18)
s = replace_one(s,
    '"reference_recipe": f"{REFERENCE_EXPERIMENT_ID}: same dataset, augmentation, schedule, seeds and selection rule",',
    '"reference_recipe": f"{REFERENCE_EXPERIMENT_ID}: same dataset, augmentation, Stage 1, seeds and selection rule; declared Stage 2 block/LR changes",')
s = replace_one(s,
    'f"{STAGE2_UNFREEZE_PREFIX}* non-BatchNorm only (conv5_block1-3)"',
    'f"{STAGE2_UNFREEZE_PREFIX}* non-BatchNorm only (conv5_block3)"')
put(18, s)

s = get(24).replace('ResNet50 EXP-031', 'ResNet50 EXP-032').replace('real_holdout_031', 'real_holdout_032')
put(24, s)
s = get(26).replace('Do not overwrite EXP-031 checkpoints.', 'Do not overwrite completed experiment checkpoints.')
put(26, s)
s = get(38)
s = 'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n' + s.split('\n', 1)[1]
s = replace_one(s, 'resnet50_transfer_5seed_031.ipynb', 'resnet50_transfer_regularized_5seed_032.ipynb')
s = replace_one(s, 'use the EXP-031 ResNet50 notebook', 'use the EXP-032 ResNet50 notebook')
s = replace_one(s,
    'f"DenseNet121 recipe {REFERENCE_EXPERIMENT_ID} (same dataset, augmentation, schedule and selection rule).\\n\\n"',
    'f"DenseNet121 reference {REFERENCE_EXPERIMENT_ID}: same dataset, augmentation, Stage 1, seeds and Validation selection; Stage 2 block/LR are declared changes.\\n\\n"')
s = replace_one(s, 'f"Stage 2 unfreezes conv5_block1-3 non-BatchNorm layers ({EXPECTED_STAGE2_TRAINABLE_PARAMS:,} of "',
                'f"Stage 2 unfreezes conv5_block3 non-BatchNorm layers at LR {STAGE2_LR:g} ({EXPECTED_STAGE2_TRAINABLE_PARAMS:,} of "')
put(38, s)

for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"resnet032-cell-{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        if index != 1:
            ast.parse(get(index))

# Hash the source with placeholders, so the experiment ID changes with this recipe.
canonical = json.dumps(notebook, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
put(4, replace_one(get(4), "__CODE_REVISION__", revision))

# The export cell carries a self-contained notebook snapshot without recursion.
snapshot = copy.deepcopy(notebook)
snapshot["cells"][38]["source"][0] = 'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n'
blob = base64.b64encode(zlib.compress(
    json.dumps(snapshot, ensure_ascii=False, indent=1).encode("utf-8"), 9)).decode("ascii")
put(38, replace_one(get(38), 'NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                    'NOTEBOOK_SOURCE_ZLIB_B64 = ' + json.dumps(blob)))
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code" and index != 1:
        ast.parse(get(index))
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET, "revision", revision[:8], "cells", len(notebook["cells"]))
