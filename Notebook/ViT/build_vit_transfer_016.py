"""Build the DeiT-Small Transfer notebook for EXP-016 (layer-wise LR decay and unfreeze depth).

Plan: KE_HOACH_CAI_TIEN_VIT_SAU_EXP-015.md, section "EXP-016".

The notebook is derived from the EXP-015 notebook. Every shared cell (dataset audit, leakage
audit, augmentation, Stage 1, callbacks, Validation lock, Final Test, Holdout, exports) is kept;
only the cells that carry the Stage 2 recipe are replaced. The single factor under test is the
Stage 2 learning-rate scheme and unfreeze depth.

Usage:  python build_vit_transfer_016.py [SOURCE_015_NOTEBOOK] [TARGET_NOTEBOOK]
"""
import ast
import base64
import copy
import hashlib
import json
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "vit_deit_small_transfer_5seeds_015_256px.ipynb"
TARGET = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "vit_deit_small_transfer_llrd_016_256px.ipynb"
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
assert len(notebook["cells"]) == 40, len(notebook["cells"])


def get(index):
    return "".join(notebook["cells"][index]["source"])


def put(index, source):
    notebook["cells"][index]["source"] = source.splitlines(keepends=True)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["outputs"] = []


def find(marker, kind=None, head=None):
    """Index of the single cell containing marker (optionally only within its first `head` characters)."""
    hits = [i for i, c in enumerate(notebook["cells"])
            if marker in (get(i)[:head] if head else get(i)) and (kind is None or c["cell_type"] == kind)]
    assert len(hits) == 1, (marker, hits)
    return hits[0]


def replace_one(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new)


def new_cell(kind, source):
    cell = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        cell.update(execution_count=None, outputs=[])
    return cell


# ----------------------------------------------------------------------------------------
# Cell 0: introduction
# ----------------------------------------------------------------------------------------
put(0, """# EXP-VIT-DEITSMALL-TRANSFER-LLRD-016 — DeiT-Small Transfer: layer-wise LR decay và độ sâu mở

**Mục đích:** thực hiện EXP-016 của `KE_HOACH_CAI_TIEN_VIT_SAU_EXP-015.md`: thay LR phẳng 1e-5 của Stage 2 (EXP-015) bằng **LR giảm dần theo khối** (layer-wise LR decay, LLRD) và khảo sát **số khối encoder được mở**. Đây là thay đổi ưu tiên cao nhất vì EXP-015 overfit mạnh ở Stage 2 (7,1 M tham số mở trên 4.972 ảnh) và độ lệch trên ảnh thật (Holdout) nằm ở đây.

- **Một yếu tố duy nhất thay đổi so với EXP-015:** công thức Stage 2 (LR theo khối + số khối mở). Dataset, augmentation, Stage 1, scheduler, early stopping, batch 16, seed, luật chọn checkpoint đều giữ nguyên (đối chiếu bằng *fairness contract* ở cell 03a).
- **Cơ sở / đối chứng:** `EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015-256px-d4379a48` (baseline); `EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05` (công thức Transfer chuẩn).
- Dataset: `Cashew_dataV13` / `CashewData_Split (13).zip`, Train 4.972 / Val 1.452 / Test 685. Holdout 09 (126 ảnh, 4 lớp) chỉ dùng mô tả, **không bao giờ dùng để chọn cấu hình**.

## Công thức LR theo khối

`lr(khối i) = base_lr × 0.75^(11 − i)`, với i = 0…11 (khối 12 nhận `base_lr`, khối 1 nhận 0,042·`base_lr`). LayerNorm cuối và hai đầu phân loại dùng `base_lr`. Embedding (patch + position) **luôn đóng băng** ở mọi cấu hình. Adam chuẩn hóa gradient nên không thể nhân gradient; notebook dùng lớp con `LayerwiseLRAdamW` của `AdamW` ghi đè `update_step` để nhân `learning_rate` theo từng biến. Weight decay (1e-4) vẫn dùng `base_lr` chưa nhân hệ số: độ co mỗi bước chỉ khoảng 5e-9, không đáng kể. Smoke check đo trực tiếp tỉ lệ cập nhật từng khối so với AdamW thường và dừng nếu lệch quá 2%.

## Hai pha (đổi biến `PHASE` ở cell 03)

| Pha | Việc làm | Dùng dữ liệu |
|---|---|---|
| `SCREEN` (mặc định) | Lưới số khối mở N ∈ {4, 8, 12} × `base_lr` ∈ {3e-5, 5e-5} = 6 cấu hình, **cộng 1 cấu hình tái lập baseline (N=4, LR phẳng 1e-5)**, mỗi cấu hình 3 seed (42, 123, 2026). Stage 1 huấn luyện **một lần mỗi seed** rồi mọi cấu hình bắt đầu Stage 2 từ cùng checkpoint Stage 1 (so sánh theo cặp). | Train + Validation. **Không đọc Test và Holdout.** |
| `CONFIRM` | Chạy đủ 5 seed với cấu hình thắng, khóa Validation, rồi mới mở Test và Holdout một lần (đúng quy trình EXP-015). | Train, Validation, sau đó Test và Holdout. |

**Quy tắc chấp nhận (chỉ Validation, kế hoạch mục 3):** một cấu hình thắng baseline nếu val_loss trung bình giảm ≥ 0,005, val macro-F1 không giảm quá 0,3 điểm và `false_healthy_rate` không tăng. Nếu nhiều cấu hình qua cổng và val_loss cách nhau ≤ 0,005 thì chọn cấu hình đơn giản hơn (ít khối mở hơn, rồi LR thấp hơn). Nếu không cấu hình nào qua cổng, kết luận là **giữ nguyên baseline** và pha `CONFIRM` từ chối chạy (trừ khi đặt `CONFIRM_CONFIG_OVERRIDE` có chủ đích). Số cấu hình đã thử được ghi trong khóa sàng lọc để báo cáo.

Thứ tự chạy: (1) `PHASE = "SCREEN"`, chạy hết notebook; (2) đọc `screening_decision.json`; (3) đổi `PHASE = "CONFIRM"`, chạy lại từ đầu (cùng runtime để khóa sàng lọc còn trên đĩa, hoặc đặt `CONFIRM_CONFIG_OVERRIDE`).

- Seeds xác nhận: `[42, 123, 2026, 3407, 7777]`. Input 256×256, batch 16, Dropout 0,40, Stage 1 LR 3e-4 tối đa 12 epoch, Stage 2 tối đa 30 epoch, EarlyStopping patience 8, ReduceLROnPlateau patience 4 (hệ số 0,5, min 1e-7).
- Checkpoint selection (chỉ Validation): val_loss thấp nhất, tie-break macro-F1 cao hơn.
- Output schema version 2 (STANDARD_TRAINING_NOTEBOOK_FORMAT V2).

**Trạng thái:** notebook chưa được huấn luyện; mọi chỉ số chỉ có sau khi chạy trên Colab GPU. Nhiều cấu hình được thử trên cùng Validation, nên con số Validation của cấu hình thắng có thể hơi lạc quan.
""")

# ----------------------------------------------------------------------------------------
# Configuration cell
# ----------------------------------------------------------------------------------------
ci = find("# 03. CONFIGURATION", head=200)
s = get(ci)
s = replace_one(s, "# 03. CONFIGURATION — EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015",
                "# 03. CONFIGURATION — EXP-VIT-DEITSMALL-TRANSFER-LLRD-016")
s = replace_one(s, """# DeiT-Small Transfer trained under the DenseNet121 EXP-029 recipe (the same recipe
# ResNet50 EXP-031 follows). Every shared setting is checked against that recipe by
# the fairness-contract cell below. Only the backbone, its classification head, and
# the architecture-dependent set of blocks opened in Stage 2 are allowed to differ.""",
"""# EXP-016: DeiT-Small Transfer, EXP-015 recipe with ONE changed factor, the Stage 2
# learning-rate scheme (layer-wise LR decay) and the number of unfrozen encoder blocks.
# Everything else is checked against the DenseNet121 EXP-029 / EXP-015 recipe by the
# fairness-contract cell below. See KE_HOACH_CAI_TIEN_VIT_SAU_EXP-015.md.
#
# PHASE = "SCREEN"  : 7 Stage 2 configurations x 3 seeds, Validation only (no Test, no Holdout).
# PHASE = "CONFIRM" : five seeds with the screening winner, then Validation lock, Test, Holdout.""")
s = replace_one(s, 'NOTEBOOK_VERSION = "015"', 'NOTEBOOK_VERSION = "016"\nPHASE = "SCREEN"  # "SCREEN" -> "CONFIRM" after reading screening_decision.json')
s = replace_one(s, 'CODE_REVISION_SHA256 = "d4379a482305619cb253546516874091c3219c78c07aeb35e09434c142e49e19"',
                'CODE_REVISION_SHA256 = "__CODE_REVISION__"')
s = replace_one(s, "Optional path to this exact 015 .ipynb on Drive.", "Optional path to this exact 016 .ipynb on Drive.")
s = replace_one(s, 'EXPERIMENT_ID = f"EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"',
'''if PHASE not in ("SCREEN", "CONFIRM"):
    raise ValueError('PHASE must be "SCREEN" or "CONFIRM"')
SCREEN_EXPERIMENT_ID = f"EXP-VIT-DEITSMALL-TRANSFER-LLRD-SCREEN-016-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"
CONFIRM_EXPERIMENT_ID = f"EXP-VIT-DEITSMALL-TRANSFER-LLRD-5SEEDS-016-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"
EXPERIMENT_ID = SCREEN_EXPERIMENT_ID if PHASE == "SCREEN" else CONFIRM_EXPERIMENT_ID
BASELINE_EXPERIMENT_ID = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015-256px-d4379a48"  # incumbent (flat LR 1e-5, 4 blocks)
PLAN_DOCUMENT = "KE_HOACH_CAI_TIEN_VIT_SAU_EXP-015.md (EXP-016)"''')
s = replace_one(s, 'MODEL_NAME = "DeiT-Small (distilled)"', 'MODEL_NAME = "DeiT-Small (distilled)"')  # sanity: anchor exists

s = replace_one(s, "STAGE2_LR, STAGE2_EPOCHS = 1e-5, 30\n",
                "STAGE2_EPOCHS = 30\nBASELINE_STAGE2_LR = 1e-5  # EXP-015 / DenseNet121 EXP-029 flat Stage 2 LR; the base_lr of EXP-016 is the variable\n")
s = replace_one(s, """# Stage 2 opens the LAST part of the network, the ViT analogue of conv5_: the last N of the
# 12 encoder blocks plus the final LayerNorm. Embeddings and earlier blocks stay frozen.
STAGE2_UNFREEZE_LAST_BLOCKS = 4
STAGE2_UNFREEZE_FINAL_LAYERNORM = True
STAGE2_RULE_TEXT = (f"last {STAGE2_UNFREEZE_LAST_BLOCKS} of {DEIT_SMALL_BACKBONE_CONFIG['num_layers']} "
                    "encoder blocks + final LayerNorm (embeddings and earlier blocks frozen)")
EXPECTED_BACKBONE_PARAMS = 21_689_472          # at 256 px; the 224 px preset has 21,666,432
EXPECTED_PRETRAINED_224_PARAMS = 21_666_432
EXPECTED_STAGE2_TRAINABLE_PARAMS = 7_098_624   # 4 x 1,774,464 + 768 (final LayerNorm)
""", '''# Stage 2 opens the LAST N of the 12 encoder blocks plus the final LayerNorm; embeddings
# (patch + position) and the earlier blocks stay frozen at every N. N and the learning
# rates are the factors of EXP-016 and are resolved in the "EXP-016 Stage 2" block below.
STAGE2_UNFREEZE_FINAL_LAYERNORM = True
PARAMS_PER_ENCODER_BLOCK = 1_774_464
FINAL_LAYERNORM_PARAMS = 768
EXPECTED_BACKBONE_PARAMS = 21_689_472          # at 256 px; the 224 px preset has 21,666,432
EXPECTED_PRETRAINED_224_PARAMS = 21_666_432


def stage2_trainable_params(last_blocks):
    """Backbone parameters opened in Stage 2: N encoder blocks + final LayerNorm (4 -> 7,098,624)."""
    return int(last_blocks) * PARAMS_PER_ENCODER_BLOCK + (FINAL_LAYERNORM_PARAMS if STAGE2_UNFREEZE_FINAL_LAYERNORM else 0)


# ------------------------------------------------------------
# EXP-016 factors: layer-wise LR decay and unfreeze depth (KE_HOACH ... EXP-016)
# ------------------------------------------------------------
LLRD_DECAY = 0.75  # lr(block i) = base_lr * LLRD_DECAY ** (11 - i); block 12, final LayerNorm and heads use base_lr
SCREEN_SEEDS = [42, 123, 2026]
SCREEN_UNFREEZE_LAST_BLOCKS = (4, 8, 12)
SCREEN_BASE_LRS = (3e-5, 5e-5)
INCLUDE_BASELINE_REPRODUCTION = True   # re-run the EXP-015 recipe (4 blocks, flat 1e-5) inside this notebook
BASELINE_015_VALIDATION_CSV = None     # optional: EXP-015 aggregate/validation_results_5seeds.csv; if set, B0 is not re-run
VAL_LOSS_MIN_GAIN = 0.005              # acceptance rule (plan section 3): val_loss mean must drop by at least this
VAL_F1_MAX_DROP = 0.003                # ... val macro-F1 may drop by at most 0.3 points
CONFIRM_CONFIG_OVERRIDE = None         # e.g. {"unfreeze_last_blocks": 8, "base_lr": 5e-5}; None = read the screening lock
SCREENING_LOCK_PATH_OVERRIDE = None    # path to screening_selection_lock.json from a previous runtime
STAGE2_LR_RULE_TEXT = (f"lr(encoder block i) = base_lr * {LLRD_DECAY} ** (11 - i) for the unfrozen blocks; "
                       "final LayerNorm and both heads use base_lr; embeddings frozen; "
                       "weight decay applies the unscaled base_lr (about 5e-9 relative shrink per step)")


def make_stage2_config(last_blocks, base_lr, decay=None, config_id=None):
    decay = LLRD_DECAY if decay is None else float(decay)
    last_blocks = int(last_blocks)
    if not 1 <= last_blocks <= DEIT_SMALL_BACKBONE_CONFIG["num_layers"]:
        raise ValueError(f"unfreeze_last_blocks must be 1..{DEIT_SMALL_BACKBONE_CONFIG['num_layers']}")
    flat = decay == 1.0
    return {
        "id": config_id or f"n{last_blocks:02d}_lr{float(base_lr):.0e}",
        "unfreeze_last_blocks": last_blocks, "base_lr": float(base_lr), "decay": decay,
        "trainable_params": stage2_trainable_params(last_blocks),
        "rule_text": (f"last {last_blocks} of {DEIT_SMALL_BACKBONE_CONFIG['num_layers']} encoder blocks + final LayerNorm "
                      f"(embeddings and earlier blocks frozen); " +
                      (f"flat LR {float(base_lr):g}" if flat else
                       f"LR = {float(base_lr):g} x {decay} ** (11 - block index)")),
    }


BASELINE_CONFIG = make_stage2_config(4, BASELINE_STAGE2_LR, decay=1.0, config_id="B0_flat_n04_lr1e-05")
SCREEN_GRID = [make_stage2_config(n, lr) for n in SCREEN_UNFREEZE_LAST_BLOCKS for lr in SCREEN_BASE_LRS]
SCREEN_CONFIGS = ([BASELINE_CONFIG] if (INCLUDE_BASELINE_REPRODUCTION and BASELINE_015_VALIDATION_CSV is None) else []) + SCREEN_GRID
if len({c["id"] for c in SCREEN_CONFIGS}) != len(SCREEN_CONFIGS):
    raise ValueError("Duplicate screening configuration ids")
''')
s = replace_one(s, "DENSENET121_REFERENCE_STAGE2_PARAMS = 2_129_920\nRESNET50_PEER_STAGE2_PARAMS = 14_953_472\n",
                "DENSENET121_REFERENCE_STAGE2_PARAMS = 2_129_920\nRESNET50_PEER_STAGE2_PARAMS = 14_953_472\nEXPECTED_STAGE2_TRAINABLE_PARAMS_EXP015 = 7_098_624  # EXP-015: 4 blocks + LayerNorm\n")
s = replace_one(s, """RUN_FULL_TRAINING = True
RUN_FINAL_TEST = True
RUN_REAL_HOLDOUT = True  # Runs after the five-seed Validation lock.""",
"""# The screening phase reads Train and Validation only. Test and Holdout are opened exclusively
# in the CONFIRM phase, after the five-seed Validation lock.
RUN_SCREENING = PHASE == "SCREEN" and not VIEW_ONLY and not RUN_SMOKE_ONLY
RUN_FULL_TRAINING = PHASE == "CONFIRM" and not VIEW_ONLY and not RUN_SMOKE_ONLY
RUN_FINAL_TEST = PHASE == "CONFIRM" and not VIEW_ONLY and not RUN_SMOKE_ONLY
RUN_REAL_HOLDOUT = PHASE == "CONFIRM" and not VIEW_ONLY and not RUN_SMOKE_ONLY  # Runs after the five-seed Validation lock.""")
s = replace_one(s, "# Separate result namespace; EXP-013 is never modified.",
                "# Separate result namespace; EXP-015 is never modified.")

# Resolve the active Stage 2 configuration once RESULTS_ROOT exists.
s = replace_one(s, "OUT_DIR = RESULTS_ROOT / EXPERIMENT_ID\n", '''# ------------------------------------------------------------
# EXP-016 Stage 2: resolve the configuration this run uses
# ------------------------------------------------------------
SCREENING_LOCK_PATH = (Path(SCREENING_LOCK_PATH_OVERRIDE).expanduser() if SCREENING_LOCK_PATH_OVERRIDE
                       else RESULTS_ROOT / SCREEN_EXPERIMENT_ID / "screening_selection_lock.json")
if PHASE == "SCREEN":
    # The screening cell runs every configuration; the smoke check uses the deepest one (N=12, 5e-5)
    # because it exercises the learning-rate multiplier of all twelve blocks.
    ACTIVE_STAGE2 = make_stage2_config(12, 5e-5)
    STAGE2_CONFIG_SOURCE = "smoke representative; the screening cell runs every configuration of SCREEN_CONFIGS"
    STAGE2_RECORD = {"phase": "screening", "configs": SCREEN_CONFIGS, "seeds": SCREEN_SEEDS,
                     "shared_stage1_per_seed": True,
                     "baseline_source": ("csv: " + str(BASELINE_015_VALIDATION_CSV)) if BASELINE_015_VALIDATION_CSV
                                        else "B0 re-run in this notebook" if INCLUDE_BASELINE_REPRODUCTION else "none",
                     "acceptance_rule": {"val_loss_min_gain": VAL_LOSS_MIN_GAIN, "val_macro_f1_max_drop": VAL_F1_MAX_DROP,
                                         "false_healthy_rate": "must not increase",
                                         "tie": "within val_loss_min_gain of the best passing config -> fewer blocks, then lower base_lr"}}
else:
    _saved_config = RESULTS_ROOT / CONFIRM_EXPERIMENT_ID / "experiment_config.json"
    if CONFIRM_CONFIG_OVERRIDE is not None:
        _chosen, STAGE2_CONFIG_SOURCE = dict(CONFIRM_CONFIG_OVERRIDE), "manual CONFIRM_CONFIG_OVERRIDE"
    elif _saved_config.is_file():
        _chosen = json.loads(_saved_config.read_text(encoding="utf-8"))["stage2"]["variable_under_test"]["config"]
        STAGE2_CONFIG_SOURCE = "saved experiment_config.json of this experiment"
    elif SCREENING_LOCK_PATH.is_file():
        _lock = json.loads(SCREENING_LOCK_PATH.read_text(encoding="utf-8"))
        if _lock.get("decision") != "ADOPT_CANDIDATE":
            raise RuntimeError(
                "The screening found no Stage 2 configuration that passes the acceptance rule against the baseline "
                f"({SCREENING_LOCK_PATH}). The result is: keep the EXP-015 recipe. Set CONFIRM_CONFIG_OVERRIDE only if you "
                "deliberately want to confirm a configuration that did not pass, and say so in the report.")
        _chosen, STAGE2_CONFIG_SOURCE = _lock["winner"], f"screening lock {SCREENING_LOCK_PATH}"
    else:
        raise FileNotFoundError(
            f"No screening lock at {SCREENING_LOCK_PATH}. Run PHASE = 'SCREEN' first in this runtime, or set "
            "SCREENING_LOCK_PATH_OVERRIDE / CONFIRM_CONFIG_OVERRIDE.")
    ACTIVE_STAGE2 = make_stage2_config(_chosen["unfreeze_last_blocks"], _chosen["base_lr"], _chosen.get("decay"))
    STAGE2_RECORD = {"phase": "confirmation", "config": ACTIVE_STAGE2, "config_source": STAGE2_CONFIG_SOURCE,
                     "screening_experiment": SCREEN_EXPERIMENT_ID}
STAGE2_UNFREEZE_LAST_BLOCKS = ACTIVE_STAGE2["unfreeze_last_blocks"]
STAGE2_LR = ACTIVE_STAGE2["base_lr"]
STAGE2_RULE_TEXT = ACTIVE_STAGE2["rule_text"]
EXPECTED_STAGE2_TRAINABLE_PARAMS = ACTIVE_STAGE2["trainable_params"]

OUT_DIR = RESULTS_ROOT / EXPERIMENT_ID
''')
s = replace_one(s, 'print("Dataset:", DATASET_ARCHIVE_NAME, "| seeds:", SEEDS)',
                'print("Phase:", PHASE, "| Stage 2 config:", ACTIVE_STAGE2["id"] if PHASE == "CONFIRM" else [c["id"] for c in SCREEN_CONFIGS])\n'
                'print("Dataset:", DATASET_ARCHIVE_NAME, "| seeds:", SEEDS if PHASE == "CONFIRM" else SCREEN_SEEDS)')
s = replace_one(s, 'print("Flags:", RUN_FULL_TRAINING, RUN_FINAL_TEST, RUN_REAL_HOLDOUT)',
                'print("Flags (screening, full training, final test, holdout):", RUN_SCREENING, RUN_FULL_TRAINING, RUN_FINAL_TEST, RUN_REAL_HOLDOUT)')
put(ci, s)

# ----------------------------------------------------------------------------------------
# Fairness contract: Stage 2 LR is the declared deviation of EXP-016
# ----------------------------------------------------------------------------------------
fi = find("# 03a. FAIRNESS CONTRACT")
s = get(fi)
s = replace_one(s, """FAIRNESS_DRIFT = {key: {"reference": FAIRNESS_REFERENCE[key], "this_notebook": FAIRNESS_LIVE[key]}
                  for key in FAIRNESS_REFERENCE if FAIRNESS_REFERENCE[key] != FAIRNESS_LIVE[key]}""",
"""# EXP-016 varies exactly one thing on purpose: the Stage 2 learning-rate scheme (and, with it,
# how many blocks are unfrozen). The Stage 2 LR is therefore excluded from the drift check and
# recorded as a declared deviation below. Every other shared setting must still match.
DECLARED_DEVIATION_KEYS = {"stage2": ("lr",)}


def _without_declared(key, value):
    if key in DECLARED_DEVIATION_KEYS and isinstance(value, dict):
        return {k: v for k, v in value.items() if k not in DECLARED_DEVIATION_KEYS[key]}
    return value


FAIRNESS_DRIFT = {key: {"reference": FAIRNESS_REFERENCE[key], "this_notebook": FAIRNESS_LIVE[key]}
                  for key in FAIRNESS_REFERENCE
                  if _without_declared(key, FAIRNESS_REFERENCE[key]) != _without_declared(key, FAIRNESS_LIVE[key])}""")
s = replace_one(s, '''if not (RUN_FINAL_TEST and RUN_REAL_HOLDOUT) and not (VIEW_ONLY or RUN_SMOKE_ONLY):
    raise ValueError("Project policy: Final Test and Real Holdout must always be ON "
                     "(they still run only after the Validation lock).")''',
'''if not (RUN_FINAL_TEST and RUN_REAL_HOLDOUT) and not (VIEW_ONLY or RUN_SMOKE_ONLY or PHASE == "SCREEN"):
    raise ValueError("Project policy: Final Test and Real Holdout must always be ON in the CONFIRM phase "
                     "(they still run only after the Validation lock). The SCREEN phase never reads them.")''')
s = replace_one(s, '    "stage2_trainable_rule": STAGE2_RULE_TEXT + f":',
                '    "stage2_trainable_rule": ("[screening phase: smoke-check representative only; the screening cell runs every configuration] "\n'
                '                              if PHASE == "SCREEN" else "") + STAGE2_RULE_TEXT + f":')
s = replace_one(s, '''    "normalization": "LayerNorm has no running statistics, so no BatchNorm-style freeze applies (the CNNs keep BatchNorm frozen)",
}''', '''    "normalization": "LayerNorm has no running statistics, so no BatchNorm-style freeze applies (the CNNs keep BatchNorm frozen)",
    "exp016_declared_deviation": "Stage 2 learning rate is NOT the flat 1e-5 of the references: " + STAGE2_LR_RULE_TEXT +
                                 f". Baseline for comparison: {BASELINE_EXPERIMENT_ID}.",
}''')
s = replace_one(s, 'print("Fairness contract PASS:", len(FAIRNESS_REFERENCE), "shared settings identical to the DenseNet121 EXP-029 reference.")',
                'print("Fairness contract PASS:", len(FAIRNESS_REFERENCE), "shared settings checked; identical to the DenseNet121 EXP-029 reference except the declared Stage 2 learning rate.")')
put(fi, s)

# ----------------------------------------------------------------------------------------
# Model cell: variable depth, LayerwiseLRAdamW, smoke gates
# ----------------------------------------------------------------------------------------
mi = find("# 08. DeiT-Small backbone (256 px)")
s = get(mi)
s = replace_one(s, """    def set_backbone_stage(backbone, stage):
        if stage == "stage1":""", """    def set_backbone_stage(backbone, stage, last_blocks=None):
        if stage == "stage1":""")
s = replace_one(s, """            embeddings.trainable = False
            first_open_block = DEIT_SMALL_BACKBONE_CONFIG["num_layers"] - STAGE2_UNFREEZE_LAST_BLOCKS""",
"""            embeddings.trainable = False
            last_blocks = STAGE2_UNFREEZE_LAST_BLOCKS if last_blocks is None else int(last_blocks)
            if not 1 <= last_blocks <= DEIT_SMALL_BACKBONE_CONFIG["num_layers"]:
                raise ValueError(f"last_blocks out of range: {last_blocks}")
            first_open_block = DEIT_SMALL_BACKBONE_CONFIG["num_layers"] - last_blocks""")
s = replace_one(s, """    def compile_stage(model, lr):
        model.compile(optimizer=keras.optimizers.AdamW(learning_rate=lr,
                                                      weight_decay=WEIGHT_DECAY),
                      loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"], steps_per_execution=1, jit_compile=False)
""", '''    class LayerwiseLRAdamW(keras.optimizers.AdamW):
        """AdamW whose step for each variable uses ``learning_rate * scale[variable]``.

        Adam normalises the gradient, so a per-layer learning rate cannot be obtained by scaling
        gradients; the multiplier has to act on the learning rate inside ``update_step``. Variables
        that are not in ``lr_scale_by_variable`` (the two heads) use multiplier 1.0, i.e. base_lr.
        The schedule/ReduceLROnPlateau still drives ``learning_rate`` (the base value).
        """

        def __init__(self, lr_scale_by_variable=None, **kwargs):
            super().__init__(**kwargs)
            self._scale_source = list(lr_scale_by_variable or [])
            self._scale_by_key = {}

        def build(self, var_list):
            super().build(var_list)
            by_id = {id(variable): float(scale) for variable, scale in self._scale_source}
            self._scale_by_key = {self._var_key(variable): by_id.get(id(variable), 1.0) for variable in var_list}

        def update_step(self, gradient, variable, learning_rate):
            scale = self._scale_by_key.get(self._var_key(variable), 1.0)
            super().update_step(gradient, variable, learning_rate * scale)

    def stage2_lr_scales(backbone, last_blocks, decay):
        """(variable, multiplier) for the unfrozen backbone variables; block i gets decay ** (11 - i)."""
        encoder = backbone.get_layer("deit_encoder")
        num_blocks = DEIT_SMALL_BACKBONE_CONFIG["num_layers"]
        pairs = []
        for block_index, block in enumerate(encoder.encoder_layers):
            if block_index >= num_blocks - last_blocks:
                scale = float(decay) ** (num_blocks - 1 - block_index)
                pairs += [(variable, scale) for variable in block.trainable_weights]
        pairs += [(variable, 1.0) for variable in encoder.layer_norm.trainable_weights]
        return pairs

    def compile_stage(model, lr, lr_scales=None):
        if lr_scales is None:
            optimizer = keras.optimizers.AdamW(learning_rate=lr, weight_decay=WEIGHT_DECAY)
        else:
            optimizer = LayerwiseLRAdamW(lr_scale_by_variable=lr_scales, learning_rate=lr, weight_decay=WEIGHT_DECAY)
        model.compile(optimizer=optimizer, loss=keras.losses.SparseCategoricalCrossentropy(),
                      metrics=["accuracy"], steps_per_execution=1, jit_compile=False)

    def compile_stage2(model, backbone, config):
        """Fresh optimizer for Stage 2; call after set_backbone_stage(backbone, 'stage2', N)."""
        compile_stage(model, config["base_lr"],
                      stage2_lr_scales(backbone, config["unfreeze_last_blocks"], config["decay"]))
''')

s = replace_one(s, """        for key in ("embedding", "first_block"):
            if not np.array_equal(stage2_probe_before[key], frozen_probe[key].numpy()):
                raise AssertionError(f"Frozen DeiT part changed in Stage 2: {key}")
""", """        all_blocks_open = ACTIVE_STAGE2["unfreeze_last_blocks"] >= DEIT_SMALL_BACKBONE_CONFIG["num_layers"]
        for key in ("embedding",) + (() if all_blocks_open else ("first_block",)):
            if not np.array_equal(stage2_probe_before[key], frozen_probe[key].numpy()):
                raise AssertionError(f"Frozen DeiT part changed in Stage 2: {key}")
        if all_blocks_open and np.array_equal(stage2_probe_before["first_block"], frozen_probe["first_block"].numpy()):
            raise AssertionError("Block 1 must train when all 12 blocks are open")
""")
# smoke: Stage 2 uses the layer-wise optimizer and the active depth
s = replace_one(s, """        stage2_params = set_backbone_stage(backbone, "stage2")
        if stage2_params != EXPECTED_STAGE2_TRAINABLE_PARAMS:""",
"""        stage2_params = set_backbone_stage(backbone, "stage2", ACTIVE_STAGE2["unfreeze_last_blocks"])
        if stage2_params != EXPECTED_STAGE2_TRAINABLE_PARAMS:""")
s = replace_one(s, """        compile_stage(model, STAGE2_LR)
        if model.optimizer is old_optimizer:
            raise AssertionError("Stage 2 reused the Stage 1 optimizer")
        model.train_on_batch(*smoke_train)""",
"""        compile_stage2(model, backbone, ACTIVE_STAGE2)
        if model.optimizer is old_optimizer:
            raise AssertionError("Stage 2 reused the Stage 1 optimizer")
        if not isinstance(model.optimizer, LayerwiseLRAdamW):
            raise AssertionError("Stage 2 must use LayerwiseLRAdamW")
        model.train_on_batch(*smoke_train)""")
s = replace_one(s, """        with tempfile.TemporaryDirectory() as temporary:
            weights_path = Path(temporary) / "smoke.weights.h5\"""", '''        # ---- EXP-016 gate 1: the unfreeze depth rule gives the planned parameter counts ----
        depth_gate = {}
        for depth in sorted({4, 8, 12, *SCREEN_UNFREEZE_LAST_BLOCKS, STAGE2_UNFREEZE_LAST_BLOCKS}):
            opened = set_backbone_stage(backbone, "stage2", depth)
            if opened != stage2_trainable_params(depth):
                raise AssertionError(f"Stage 2 with {depth} blocks opened {opened}, expected {stage2_trainable_params(depth)}")
            if embedding_layer.trainable_weights:
                raise AssertionError("Embeddings must stay frozen at every depth")
            depth_gate[str(depth)] = opened
        if depth_gate.get("4") != EXPECTED_STAGE2_TRAINABLE_PARAMS_EXP015:
            raise AssertionError("4-block depth no longer matches EXP-015")

        # ---- EXP-016 gate 2: LayerwiseLRAdamW scales each block's step by decay ** (11 - i) ----
        # Same gradients, same starting weights, plain AdamW versus LayerwiseLRAdamW. The Adam step is
        # linear in the learning rate, so ||step_llrd|| / ||step_plain|| must equal the multiplier.
        set_backbone_stage(backbone, "stage2", DEIT_SMALL_BACKBONE_CONFIG["num_layers"])
        probe_scales = stage2_lr_scales(backbone, DEIT_SMALL_BACKBONE_CONFIG["num_layers"], LLRD_DECAY)
        probe_variables = list(model.trainable_variables)
        probe_x, probe_y = smoke_train
        with tf.GradientTape() as tape:
            probe_loss = keras.losses.SparseCategoricalCrossentropy()(probe_y, model(probe_x, training=False))
        probe_grads = tape.gradient(probe_loss, probe_variables)
        if any(g is None for g in probe_grads):
            raise AssertionError("A trainable variable received no gradient")
        probe_start = [variable.numpy().copy() for variable in probe_variables]

        def one_step(optimizer):
            for variable, start in zip(probe_variables, probe_start):
                variable.assign(start)
            optimizer.apply_gradients(zip(probe_grads, probe_variables))
            return [variable.numpy() - start for variable, start in zip(probe_variables, probe_start)]

        probe_lr = 1e-3
        step_plain = one_step(keras.optimizers.AdamW(learning_rate=probe_lr, weight_decay=WEIGHT_DECAY))
        step_llrd = one_step(LayerwiseLRAdamW(lr_scale_by_variable=probe_scales, learning_rate=probe_lr,
                                              weight_decay=WEIGHT_DECAY))
        for variable, start in zip(probe_variables, probe_start):
            variable.assign(start)
        expected_scale = {id(variable): scale for variable, scale in probe_scales}
        block_of = {}
        for block_index, block in enumerate(backbone.get_layer("deit_encoder").encoder_layers):
            for variable in block.weights:
                block_of[id(variable)] = block_index + 1
        llrd_rows = []
        for variable, plain, llrd in zip(probe_variables, step_plain, step_llrd):
            ratio = float(np.linalg.norm(llrd) / max(np.linalg.norm(plain), 1e-30))
            target = expected_scale.get(id(variable), 1.0)
            if abs(ratio / target - 1.0) > 0.02:
                raise AssertionError(f"Layer-wise LR multiplier wrong for {variable.path}: {ratio:.4f} vs {target:.4f}")
            llrd_rows.append({"block": block_of.get(id(variable), 0), "ratio": ratio, "expected": target})
        by_block = {}
        for row in llrd_rows:
            by_block.setdefault(row["block"], []).append(row["ratio"])
        llrd_gate = {"decay": LLRD_DECAY, "variables_checked": len(llrd_rows), "tolerance": 0.02,
                     "measured_multiplier_by_block": {str(b): float(np.mean(v)) for b, v in sorted(by_block.items())},
                     "expected_multiplier_by_block": {str(b): LLRD_DECAY ** (12 - b) for b in range(1, 13)},
                     "head_and_final_layernorm_multiplier": 1.0, "status": "PASS"}
        if abs(llrd_gate["measured_multiplier_by_block"]["1"] - LLRD_DECAY ** 11) > 0.02 * LLRD_DECAY ** 11:
            raise AssertionError("Block 1 multiplier is not decay ** 11")
        print("Layer-wise LR gate PASS:", {b: round(v, 4) for b, v in llrd_gate["measured_multiplier_by_block"].items()})
        set_backbone_stage(backbone, "stage2", ACTIVE_STAGE2["unfreeze_last_blocks"])
        compile_stage2(model, backbone, ACTIVE_STAGE2)  # back to the active optimizer for the save/reload test

        with tempfile.TemporaryDirectory() as temporary:
            weights_path = Path(temporary) / "smoke.weights.h5"''')
s = replace_one(s, """            "stage2_unfrozen_blocks": unfrozen_names,""",
                """            "stage2_unfrozen_blocks": unfrozen_names,
            "exp016_stage2_depth_gate": depth_gate,
            "exp016_layerwise_lr_gate": llrd_gate,
            "exp016_active_config_for_smoke": ACTIVE_STAGE2,""")
put(mi, s)

# ----------------------------------------------------------------------------------------
# Markdown above the recipe / training cells
# ----------------------------------------------------------------------------------------
put(find("## Freeze the recipe, then train both stages"), """## Freeze the recipe, then train

The configuration and dataset fingerprint are written before fitting. In the `SCREEN` phase the
recipe lists the whole grid; in the `CONFIRM` phase it freezes the single winning Stage 2
configuration and where it came from. Stage 1 is identical to EXP-015. Stage 2 starts from the best Stage 1
weights, uses a fresh `LayerwiseLRAdamW`, and opens the last N encoder blocks plus the final
LayerNorm with learning rates `base_lr × 0.75^(11 − i)`. Selection uses Validation only.
""")

# ----------------------------------------------------------------------------------------
# Recipe JSON
# ----------------------------------------------------------------------------------------
ri = find("# 09. Immutable experiment recipe")
s = get(ri)
s = replace_one(s, """        "peer_experiment": PEER_EXPERIMENT_ID,
        "comparison_scope": "cross-architecture comparison (DeiT-Small versus DenseNet121 EXP-029 and ResNet50 EXP-031) under one frozen Transfer recipe",""",
"""        "peer_experiment": PEER_EXPERIMENT_ID,
        "baseline_experiment": BASELINE_EXPERIMENT_ID, "plan_document": PLAN_DOCUMENT, "phase": PHASE,
        "comparison_scope": "single-factor change versus the DeiT-Small EXP-015 baseline: Stage 2 learning-rate scheme (layer-wise decay) and unfreeze depth; all other settings identical to the DenseNet121 EXP-029 / EXP-015 recipe",""")
s = replace_one(s, """        "seeds": SEEDS, "input": f"raw RGB""", """        "seeds": SEEDS, "screening_seeds": SCREEN_SEEDS,
        "input": f"raw RGB""")
s = replace_one(s, """        "stage2": {"lr": STAGE2_LR, "max_epochs": STAGE2_EPOCHS,
                   "early_patience": STAGE2_EARLY_PATIENCE,
                   "reduce_patience": STAGE2_REDUCE_PATIENCE, "min_lr": STAGE2_MIN_LR,
                   "trainable_backbone": STAGE2_RULE_TEXT,
                   "unfreeze_last_blocks": STAGE2_UNFREEZE_LAST_BLOCKS,
                   "unfreeze_final_layernorm": STAGE2_UNFREEZE_FINAL_LAYERNORM,
                   "trainable_backbone_params": EXPECTED_STAGE2_TRAINABLE_PARAMS,
                   "backbone_params": EXPECTED_BACKBONE_PARAMS,""",
"""        "stage2": {"max_epochs": STAGE2_EPOCHS,
                   "early_patience": STAGE2_EARLY_PATIENCE,
                   "reduce_patience": STAGE2_REDUCE_PATIENCE, "min_lr": STAGE2_MIN_LR,
                   "optimizer": "LayerwiseLRAdamW (AdamW with a per-variable learning-rate multiplier)",
                   "learning_rate_rule": STAGE2_LR_RULE_TEXT,
                   "llrd_decay": LLRD_DECAY, "baseline_flat_lr": BASELINE_STAGE2_LR,
                   "variable_under_test": STAGE2_RECORD,
                   "unfreeze_final_layernorm": STAGE2_UNFREEZE_FINAL_LAYERNORM,
                   "embeddings": "always frozen",
                   "exp015_trainable_backbone_params": EXPECTED_STAGE2_TRAINABLE_PARAMS_EXP015,
                   "backbone_params": EXPECTED_BACKBONE_PARAMS,""")
s = replace_one(s, '''        "test_protocol": "enabled Final Test after five-seed Validation lock; no model selection",
        "holdout_protocol": "holdout leakage audited before training; predictions only after Validation lock; descriptive audit, not a blind benchmark",''',
'''        "test_protocol": ("enabled Final Test after five-seed Validation lock; no model selection" if PHASE == "CONFIRM"
                          else "not run in the screening phase; Test stays unread"),
        "holdout_protocol": ("holdout leakage audited before training; predictions only after Validation lock; descriptive audit, not a blind benchmark"
                             if PHASE == "CONFIRM" else "not run in the screening phase; Holdout predictions stay unread and never select a configuration"),''')
s = replace_one(s, """    if RUN_FULL_TRAINING:
        smoke_path = OUT_DIR / "pipeline_smoke.json\"""", """    if RUN_FULL_TRAINING or RUN_SCREENING:
        smoke_path = OUT_DIR / "pipeline_smoke.json\"""")
s = replace_one(s, 'raise RuntimeError("Pipeline smoke must PASS before five-seed training")',
                'raise RuntimeError("Pipeline smoke must PASS before training")')
put(ri, s)

# ----------------------------------------------------------------------------------------
# Training cell: per-seed unit, shared by the confirmation loop and the screening cell
# ----------------------------------------------------------------------------------------
ti = find("Two-stage five-seed training and Validation-only selection")
s = get(ti)
start = s.index("training_rows, validation_rows = [], []\n")
end = s.index("if RUN_FULL_TRAINING:\n    training_summary_df = pd.DataFrame(training_rows)")
UNIT_CODE = '''def fit_stage1(model, backbone, train_ds, val_ds, stage1_dir):
    """Stage 1 (frozen backbone, new heads) exactly as in EXP-015."""
    params = set_backbone_stage(backbone, "stage1")
    compile_stage(model, STAGE1_LR)
    audit1, callbacks1, _ = stage_callbacks(stage1_dir, STAGE1_EARLY_PATIENCE,
                                            STAGE1_REDUCE_PATIENCE, STAGE1_MIN_LR)
    started = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=STAGE1_EPOCHS,
              callbacks=callbacks1, verbose=1)
    seconds = time.perf_counter() - started
    candidate = evaluate_candidate(model, val_ds, stage1_dir, "stage1", audit1, seconds, params)
    return candidate, seconds, params

def train_stage1_shared(seed, stage1_dir):
    """Screening only: train Stage 1 once per seed; every Stage 2 configuration starts from it."""
    stage1_dir.mkdir(parents=True)
    keras.backend.clear_session(); gc.collect(); set_global_seed(seed)
    train_ds = make_dataset(train_paths, train_labels, seed, training=True)
    val_ds = make_dataset(val_paths, val_labels, seed, training=False)
    model, backbone = build_transfer_model(seed, weights="imagenet")
    candidate, seconds, _ = fit_stage1(model, backbone, train_ds, val_ds, stage1_dir)
    del model, backbone, train_ds, val_ds
    keras.backend.clear_session(); gc.collect()
    return {"dir": stage1_dir, "candidate": candidate, "seconds": seconds}

def train_unit(seed, seed_dir, config, shared_stage1=None):
    """Two-stage training for one seed and one Stage 2 configuration.

    Returns (training summary, Validation metrics). With shared_stage1 the Stage 1 result of
    train_stage1_shared is reused instead of training Stage 1 again.
    """
    stage2_dir = seed_dir / "stage2"
    selected_path = seed_dir / "best.weights.h5"
    summary_path = seed_dir / "training_summary.json"
    if seed_dir.exists() and any(seed_dir.rglob("*")):
        raise FileExistsError(f"Seed output already exists: {seed_dir}. Use a new experiment ID for a fresh run")
    stage2_dir.mkdir(parents=True)
    keras.backend.clear_session(); gc.collect(); set_global_seed(seed)
    train_ds = make_dataset(train_paths, train_labels, seed, training=True)
    val_ds = make_dataset(val_paths, val_labels, seed, training=False)
    model, backbone = build_transfer_model(seed, weights="imagenet")

    if shared_stage1 is None:
        stage1_dir = seed_dir / "stage1"
        stage1_dir.mkdir(parents=True)
        stage1, stage1_seconds, stage1_params = fit_stage1(model, backbone, train_ds, val_ds, stage1_dir)
    else:
        stage1_dir, stage1, stage1_seconds = (shared_stage1["dir"], dict(shared_stage1["candidate"]),
                                              shared_stage1["seconds"])
        stage1_params = set_backbone_stage(backbone, "stage1")
        compile_stage(model, STAGE1_LR)  # evaluate() needs a compiled model
        model.load_weights(stage1_dir / "best.weights.h5")
        reloaded_loss = float(model.evaluate(val_ds, verbose=0)[0])
        if not np.isclose(reloaded_loss, stage1["val_loss"], atol=1e-4):
            raise RuntimeError("Shared Stage 1 checkpoint failed reload parity")

    # Stage 1 best weights are in the model. Stage 2 starts from these exact weights.
    stage2_params = set_backbone_stage(backbone, "stage2", config["unfreeze_last_blocks"])
    if stage2_params != config["trainable_params"]:
        raise AssertionError(f"Stage 2 opened {stage2_params} parameters, expected {config['trainable_params']}")
    old_optimizer = model.optimizer
    compile_stage2(model, backbone, config)
    if model.optimizer is old_optimizer:
        raise AssertionError("Stage 2 must have a new optimizer")
    audit2, callbacks2, _ = stage_callbacks(stage2_dir, STAGE2_EARLY_PATIENCE,
                                            STAGE2_REDUCE_PATIENCE, STAGE2_MIN_LR)
    started = time.perf_counter()
    model.fit(train_ds, validation_data=val_ds, epochs=STAGE2_EPOCHS,
              callbacks=callbacks2, verbose=1)
    stage2_seconds = time.perf_counter() - started
    stage2 = evaluate_candidate(model, val_ds, stage2_dir, "stage2", audit2,
                                stage2_seconds, stage2_params)

    def stage_sort_key(row):
        tail = (row["val_loss"], -row["val_macro_f1"],
                0 if row["stage"] == "stage1" else 1)
        if SELECTION_PRIMARY_METRIC == "real_subgroup_accuracy":
            return (-row["real_subgroup_accuracy"],) + tail
        return tail

    selected = min((stage1, stage2), key=stage_sort_key)
    # Record what the val_loss rule would have picked.
    selected_by_val_loss = min((stage1, stage2),
                               key=lambda row: (row["val_loss"], -row["val_macro_f1"],
                                                0 if row["stage"] == "stage1" else 1))
    set_backbone_stage(backbone, selected["stage"], config["unfreeze_last_blocks"])
    source_checkpoint = stage1_dir if selected["stage"] == "stage1" else stage2_dir
    model.load_weights(source_checkpoint / "best.weights.h5")
    selected_reloaded_loss = float(model.evaluate(val_ds, verbose=0)[0])
    if not np.isclose(selected_reloaded_loss, selected["val_loss"], atol=1e-5):
        raise RuntimeError("Selected stage checkpoint failed reload parity")
    shutil.copyfile(source_checkpoint / "best.weights.h5", selected_path)
    stage_selection = {
        "seed": seed, "selected_stage": selected["stage"],
        "selected_epoch": selected["best_epoch"],
        "selection_metric": SELECTION_PRIMARY_METRIC,
        "val_loss": selected["val_loss"], "val_macro_f1": selected["val_macro_f1"],
        "real_subgroup_accuracy": selected["real_subgroup_accuracy"],
        "stage_by_val_loss_rule": selected_by_val_loss["stage"],
        "stage_rules_agree": selected["stage"] == selected_by_val_loss["stage"],
        "tie_break": "lower val_loss, higher val_macro_f1, then Stage 1",
        "checkpoint_sha256": file_sha256(selected_path),
        "model_params": int(model.count_params()),
        "stage2_config": config,
        "stage1_shared_across_configs": shared_stage1 is not None,
        "stage_candidates": {"stage1": stage1, "stage2": stage2},
    }
    save_json(seed_dir / "stage_selection.json", stage_selection)
    validation = save_selected_validation(seed, model, val_ds, seed_dir, selected)
    summary = {
        "seed": seed, "selected_stage": selected["stage"],
        "stage2_config_id": config["id"], "unfreeze_last_blocks": config["unfreeze_last_blocks"],
        "stage2_base_lr": config["base_lr"], "llrd_decay": config["decay"],
        "stage1_epochs": stage1["actual_epochs"], "stage2_epochs": stage2["actual_epochs"],
        "stage1_training_seconds": stage1_seconds, "stage2_training_seconds": stage2_seconds,
        "training_time_seconds": stage1_seconds + stage2_seconds,
        "stage1_backbone_trainable_params": stage1_params,
        "stage2_backbone_trainable_params": stage2_params,
        "total_parameters": int(model.count_params()),
        "selected_checkpoint_sha256": stage_selection["checkpoint_sha256"],
        "selected_checkpoint_size_mb": selected_path.stat().st_size / (1024 ** 2),
    }
    save_json(summary_path, summary)
    del model, backbone, train_ds, val_ds
    keras.backend.clear_session(); gc.collect()
    return summary, validation

# ---- CONFIRM phase: five seeds with the screening winner ----
training_rows, validation_rows = [], []
for run_index, seed in enumerate(SEEDS if RUN_FULL_TRAINING else [], start=1):
    print(f"SEED {run_index}/{len(SEEDS)}: {seed} | Stage 2 config {ACTIVE_STAGE2['id']}")
    unit_summary, unit_validation = train_unit(seed, OUT_DIR / f"seed_{seed}", ACTIVE_STAGE2)
    training_rows.append(unit_summary); validation_rows.append(unit_validation)

'''
s = s[:start] + UNIT_CODE + s[end:]
s = replace_one(s, '''        "validation_sha256": file_sha256(AGGREGATE_DIR / "validation_results_5seeds.csv"),''',
                '''        "validation_sha256": file_sha256(AGGREGATE_DIR / "validation_results_5seeds.csv"),
        "stage2_config": ACTIVE_STAGE2, "stage2_config_source": STAGE2_CONFIG_SOURCE,
        "screening_experiment": SCREEN_EXPERIMENT_ID, "baseline_experiment": BASELINE_EXPERIMENT_ID,''')
s = replace_one(s, "# 10–14. Two-stage five-seed training and Validation-only selection",
                "# 10–14. Two-stage training (screening units and five-seed confirmation) and Validation-only selection")
put(ti, s)

# ----------------------------------------------------------------------------------------
# Screening cell (new), inserted before the Validation figures cell
# ----------------------------------------------------------------------------------------
SCREEN_MD = """## Screening of Stage 2 configurations (`PHASE = "SCREEN"`, Validation only)

Seven configurations (the EXP-015 baseline re-run `B0` plus the six LLRD cells of the grid) × three seeds.
For each seed Stage 1 is trained once; every configuration then runs Stage 2 from that same Stage 1
checkpoint, so the comparison is paired. Nothing here reads Test or Holdout. The cell ends by writing
`screening_decision.json` and `screening_selection_lock.json`; the latter is what the `CONFIRM` phase reads.
"""

SCREEN_CODE = '''# 10b. EXP-016 screening: Stage 2 configurations x 3 seeds, Validation only
if RUN_SCREENING:
    SCREEN_DIR = OUT_DIR / "screening"
    SCREEN_DIR.mkdir(exist_ok=True)
    STAGE1_DIR_FOR = lambda seed: SCREEN_DIR / "_shared_stage1" / f"seed_{seed}"
    UNIT_DIR_FOR = lambda config, seed: SCREEN_DIR / config["id"] / f"seed_{seed}"
    METRIC_COLUMNS = ["val_loss", "val_accuracy", "val_macro_f1", "false_healthy_rate",
                      "real_subgroup_accuracy", "real_healthy_accuracy"]

    unit_rows = []
    for seed_number, seed in enumerate(SCREEN_SEEDS, start=1):
        # Stage 1: once per seed (resume if finished).
        stage1_dir = STAGE1_DIR_FOR(seed)
        marker = stage1_dir / "shared_stage1.json"
        if marker.is_file():
            record = json.loads(marker.read_text(encoding="utf-8"))
            shared = {"dir": stage1_dir, "candidate": record["candidate"], "seconds": record["seconds"]}
            print(f"[seed {seed}] reusing finished shared Stage 1")
        else:
            if stage1_dir.exists():
                shutil.rmtree(stage1_dir)  # partial output of an interrupted run
            print(f"[seed {seed}] ({seed_number}/{len(SCREEN_SEEDS)}) shared Stage 1")
            shared = train_stage1_shared(seed, stage1_dir)
            save_json(marker, {"candidate": shared["candidate"], "seconds": shared["seconds"]})
        # Stage 2: every configuration from the same Stage 1 checkpoint.
        for config_number, config in enumerate(SCREEN_CONFIGS, start=1):
            unit_dir = UNIT_DIR_FOR(config, seed)
            result_path = unit_dir / "unit_result.json"
            if result_path.is_file():
                unit_rows.append(json.loads(result_path.read_text(encoding="utf-8")))
                print(f"[seed {seed}] {config['id']}: reusing finished unit")
                continue
            if unit_dir.exists():
                shutil.rmtree(unit_dir)
            print(f"[seed {seed}] config {config_number}/{len(SCREEN_CONFIGS)}: {config['id']}  ({config['rule_text']})")
            unit_summary, unit_validation = train_unit(seed, unit_dir, config, shared_stage1=shared)
            row = {"config_id": config["id"], "seed": seed,
                   "unfreeze_last_blocks": config["unfreeze_last_blocks"], "base_lr": config["base_lr"],
                   "llrd_decay": config["decay"],
                   "selected_stage": unit_summary["selected_stage"], "selected_epoch": unit_validation["selected_epoch"],
                   "stage2_epochs": unit_summary["stage2_epochs"],
                   "stage2_training_seconds": unit_summary["stage2_training_seconds"]}
            row.update({column: float(unit_validation[column]) for column in METRIC_COLUMNS})
            save_json(result_path, row)
            unit_rows.append(row)
            print(f"    -> val_loss {row['val_loss']:.4f}  macro-F1 {row['val_macro_f1']:.4f}  "
                  f"false_healthy {row['false_healthy_rate']:.4f}  stage {row['selected_stage']} epoch {row['selected_epoch']}")

    units_df = pd.DataFrame(unit_rows).sort_values(["config_id", "seed"]).reset_index(drop=True)
    expected_units = len(SCREEN_CONFIGS) * len(SCREEN_SEEDS)
    if len(units_df) != expected_units:
        raise RuntimeError(f"Expected {expected_units} screening units, found {len(units_df)}")
    units_df.to_csv(SCREEN_DIR / "screening_unit_results.csv", index=False)

    # ---- baseline: B0 re-run, or EXP-015's saved Validation table ----
    if BASELINE_015_VALIDATION_CSV is not None:
        baseline_df = pd.read_csv(BASELINE_015_VALIDATION_CSV)
        baseline_df = baseline_df[baseline_df["seed"].isin(SCREEN_SEEDS)].sort_values("seed").reset_index(drop=True)
        baseline_label = f"EXP-015 saved Validation table ({BASELINE_015_VALIDATION_CSV})"
    else:
        baseline_df = units_df[units_df["config_id"] == BASELINE_CONFIG["id"]].sort_values("seed").reset_index(drop=True)
        baseline_label = f"{BASELINE_CONFIG['id']} re-run in this notebook"
    if baseline_df["seed"].tolist() != sorted(SCREEN_SEEDS):
        raise RuntimeError("The baseline must cover every screening seed")

    def mean_sd(frame, column):
        values = frame[column].astype(float)
        return float(values.mean()), float(values.std(ddof=1))

    baseline_mean = {column: mean_sd(baseline_df, column)[0] for column in METRIC_COLUMNS}
    summary_rows = []
    for config in SCREEN_GRID:
        frame = units_df[units_df["config_id"] == config["id"]].sort_values("seed").reset_index(drop=True)
        row = {"config_id": config["id"], "unfreeze_last_blocks": config["unfreeze_last_blocks"],
               "base_lr": config["base_lr"], "seeds": len(frame)}
        for column in METRIC_COLUMNS:
            row[f"{column}_mean"], row[f"{column}_sd"] = mean_sd(frame, column)
        delta = frame["val_loss"].to_numpy() - baseline_df["val_loss"].to_numpy()
        row["val_loss_delta_vs_baseline"] = float(delta.mean())
        row["seeds_better_than_baseline"] = int((delta < 0).sum())
        row["val_macro_f1_delta_vs_baseline"] = row["val_macro_f1_mean"] - baseline_mean["val_macro_f1"]
        row["false_healthy_rate_delta_vs_baseline"] = row["false_healthy_rate_mean"] - baseline_mean["false_healthy_rate"]
        row["passes_val_loss_gain"] = bool(row["val_loss_delta_vs_baseline"] <= -VAL_LOSS_MIN_GAIN)
        row["passes_f1_floor"] = bool(row["val_macro_f1_delta_vs_baseline"] >= -VAL_F1_MAX_DROP)
        row["passes_false_healthy"] = bool(row["false_healthy_rate_delta_vs_baseline"] <= 1e-12)
        row["accepted"] = row["passes_val_loss_gain"] and row["passes_f1_floor"] and row["passes_false_healthy"]
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    baseline_row = {"config_id": "BASELINE (" + baseline_label + ")", "unfreeze_last_blocks": 4, "base_lr": BASELINE_STAGE2_LR,
                    "seeds": len(baseline_df)}
    for column in METRIC_COLUMNS:
        baseline_row[f"{column}_mean"], baseline_row[f"{column}_sd"] = mean_sd(baseline_df, column)
    full_summary_df = pd.concat([pd.DataFrame([baseline_row]), summary_df], ignore_index=True)
    full_summary_df.to_csv(SCREEN_DIR / "screening_config_summary.csv", index=False)

    # ---- decision (Validation only; plan section 3) ----
    accepted = summary_df[summary_df["accepted"]]
    if accepted.empty:
        decision, winner = "KEEP_INCUMBENT", None
        reason = ("No configuration lowered the mean Validation loss by at least "
                  f"{VAL_LOSS_MIN_GAIN} without hurting macro-F1 (> {VAL_F1_MAX_DROP}) or false_healthy_rate. "
                  "Report: layer-wise LR decay and a different depth did not improve on the EXP-015 recipe.")
    else:
        best_loss = float(accepted["val_loss_mean"].min())
        near = accepted[accepted["val_loss_mean"] <= best_loss + VAL_LOSS_MIN_GAIN]
        chosen = near.sort_values(["unfreeze_last_blocks", "base_lr"]).iloc[0]
        winner = make_stage2_config(int(chosen["unfreeze_last_blocks"]), float(chosen["base_lr"]))
        decision = "ADOPT_CANDIDATE"
        reason = (f"{len(accepted)} configuration(s) passed; among those within {VAL_LOSS_MIN_GAIN} of the best "
                  f"Validation loss ({best_loss:.4f}) the simpler one was chosen: {winner['id']}.")
    decision_record = {
        "experiment_id": EXPERIMENT_ID, "decision": decision, "winner": winner, "reason": reason,
        "baseline": baseline_label, "baseline_mean": baseline_mean,
        "rule": {"val_loss_min_gain": VAL_LOSS_MIN_GAIN, "val_macro_f1_max_drop": VAL_F1_MAX_DROP,
                 "false_healthy_rate": "must not increase", "split": "validation only"},
        "configurations_tried": [c["id"] for c in SCREEN_GRID], "number_of_candidate_configurations": len(SCREEN_GRID),
        "baseline_reproduction_included": BASELINE_015_VALIDATION_CSV is None and INCLUDE_BASELINE_REPRODUCTION,
        "seeds": SCREEN_SEEDS,
        "caveat": "Several configurations were compared on one fixed Validation split with three seeds; the winner's Validation figures can be slightly optimistic, and the real_ subgroup has only 59 images (12 healthy).",
        "summary": summary_rows,
    }
    save_json(SCREEN_DIR / "screening_decision.json", decision_record)
    screening_lock = {
        "experiment_id": EXPERIMENT_ID, "config_sha256": file_sha256(config_path),
        "dataset_sha256": DATASET_ZIP_SHA256, "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "decision": decision, "winner": winner, "reason": reason, "selection_split": "validation",
        "seeds": SCREEN_SEEDS, "configurations_tried": [c["id"] for c in SCREEN_CONFIGS],
        "unit_results_sha256": file_sha256(SCREEN_DIR / "screening_unit_results.csv"),
        "config_summary_sha256": file_sha256(SCREEN_DIR / "screening_config_summary.csv"),
        "test_and_holdout_read": False,
    }
    lock_path = OUT_DIR / "screening_selection_lock.json"
    if lock_path.exists():
        previous = json.loads(lock_path.read_text(encoding="utf-8"))
        screening_lock["locked_at"] = previous.get("locked_at")
        if previous != screening_lock:
            raise ValueError("Screening lock changed; use a new experiment ID")
    else:
        screening_lock["locked_at"] = datetime.now().astimezone().isoformat()
        save_json(lock_path, screening_lock)
    save_json(OUT_DIR / "execution_status.json", {
        "experiment_id": EXPERIMENT_ID, "phase": PHASE, "screening": "COMPLETE", "decision": decision,
        "test": "NOT_RUN_IN_SCREENING", "real_holdout": "NOT_RUN_IN_SCREENING"})

    # ---- figure: Validation loss by configuration, per seed and mean ----
    fig, ax = plt.subplots(figsize=(11, 5), facecolor="white")
    labels = [BASELINE_CONFIG["id"] if BASELINE_015_VALIDATION_CSV is None else "EXP-015"] + [c["id"] for c in SCREEN_GRID]
    frames = [baseline_df] + [units_df[units_df["config_id"] == c["id"]].sort_values("seed") for c in SCREEN_GRID]
    for position, frame in enumerate(frames):
        ax.scatter([position] * len(frame), frame["val_loss"], color="#2563eb", alpha=0.7, zorder=3)
        ax.hlines(frame["val_loss"].mean(), position - 0.25, position + 0.25, color="#ea580c", linewidth=2.5, zorder=4)
    ax.axhline(baseline_mean["val_loss"], color="gray", linestyle="--", alpha=0.7, label="baseline mean")
    ax.axhline(baseline_mean["val_loss"] - VAL_LOSS_MIN_GAIN, color="#16a34a", linestyle=":", label="acceptance threshold")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right")
    ax.set_ylabel("Validation loss (selected checkpoint)")
    ax.set_title("EXP-016 screening: points = seeds, orange bar = mean")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout()
    fig.savefig(SCREEN_DIR / "screening_val_loss_by_config.png", dpi=170, bbox_inches="tight")
    display(fig); plt.close(fig)

    print("Screening summary (means over seeds; Validation only):")
    display(full_summary_df[["config_id", "val_loss_mean", "val_loss_sd", "val_macro_f1_mean", "false_healthy_rate_mean",
                             "real_subgroup_accuracy_mean"] +
                            [c for c in ("val_loss_delta_vs_baseline", "seeds_better_than_baseline", "accepted")
                             if c in full_summary_df.columns]])
    print("DECISION:", decision, "|", reason)
    if winner:
        print("Winner:", winner["id"], "->", winner["rule_text"])
        print('Next: set PHASE = "CONFIRM" in cell 03 and rerun from the top (same runtime).')

    # ---- archive the screening (no checkpoints) and upload it like the other result ZIPs ----
    screen_zip = RESULTS_ROOT / f"{EXPERIMENT_ID}_ANALYSIS.zip"
    temporary_zip = screen_zip.with_suffix(".zip.part")
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file() and path.suffix.lower() not in {".keras", ".h5"}:
                archive.write(path, arcname=f"{EXPERIMENT_ID}/{path.relative_to(OUT_DIR).as_posix()}")
    with zipfile.ZipFile(temporary_zip) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("Corrupt screening ZIP")
    temporary_zip.replace(screen_zip)
    print(screen_zip, screen_zip.stat().st_size, "SHA256", file_sha256(screen_zip))
    if RUNTIME == "Colab" and UPLOAD_RESULT_ZIPS_TO_DRIVE:
        import google.auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        from google.colab import auth
        auth.authenticate_user()
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        parent = service.files().get(fileId=DRIVE_TARGET_FOLDER_ID, fields="id,mimeType", supportsAllDrives=True).execute()
        if parent.get("mimeType") != "application/vnd.google-apps.folder":
            raise RuntimeError("Configured Drive upload target is not a folder")
        query = f"name = '{screen_zip.name}' and '{DRIVE_TARGET_FOLDER_ID}' in parents and trashed = false"
        existing = service.files().list(q=query, fields="files(id,name)", supportsAllDrives=True,
                                        includeItemsFromAllDrives=True).execute().get("files", [])
        if len(existing) > 1:
            raise RuntimeError(f"Ambiguous Drive archive: {screen_zip.name}")
        media = MediaFileUpload(str(screen_zip), mimetype="application/zip", resumable=True, chunksize=8 << 20)
        request = (service.files().update(fileId=existing[0]["id"], media_body=media, fields="id", supportsAllDrives=True)
                   if existing else
                   service.files().create(body={"name": screen_zip.name, "parents": [DRIVE_TARGET_FOLDER_ID]},
                                          media_body=media, fields="id", supportsAllDrives=True))
        uploaded = None
        while uploaded is None:
            progress, uploaded = request.next_chunk(num_retries=3)
            if progress:
                print(screen_zip.name, f"{progress.progress():.0%}")
        remote = service.files().get(fileId=uploaded["id"], fields="id,name,size,webViewLink", supportsAllDrives=True).execute()
        if remote["name"] != screen_zip.name or int(remote["size"]) != screen_zip.stat().st_size:
            raise RuntimeError("Drive upload verification failed for the screening ZIP")
        print("Verified Drive archive:", remote.get("webViewLink") or f"https://drive.google.com/file/d/{uploaded['id']}/view")
else:
    print("Screening skipped (PHASE != SCREEN)")
'''
figure_index = find("# 15–17. Validation tables")
notebook["cells"].insert(figure_index, new_cell("code", SCREEN_CODE))
notebook["cells"].insert(figure_index, new_cell("markdown", SCREEN_MD))

# ----------------------------------------------------------------------------------------
# Cells that only carry names or the Stage 2 recipe text
# ----------------------------------------------------------------------------------------
hi = find("External classification evaluation for DeiT-Small EXP-015")
s = get(hi)
s = replace_one(s, "External classification evaluation for DeiT-Small EXP-015.", "External classification evaluation for DeiT-Small EXP-016.")
assert s.count("real_holdout_015") == 3
s = s.replace("real_holdout_015", "real_holdout_016")
put(hi, s)

xi = find("continue_code")
s = get(xi)
s = replace_one(s, """    for block_index, block in enumerate(encoder.encoder_layers):
        block.trainable = block_index >= 12 - 4
    encoder.layer_norm.trainable = True
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    # Continue only in a NEW experiment ID. Do not overwrite EXP-015 checkpoints.
    \"\"\"""", """    for block_index, block in enumerate(encoder.encoder_layers):
        block.trainable = block_index >= 12 - __LAST_BLOCKS__
    encoder.layer_norm.trainable = True
    # EXP-016 trained Stage 2 with layer-wise LR: block i gets base_lr * 0.75 ** (11 - i). Reproducing it needs the
    # LayerwiseLRAdamW class from the training notebook (an AdamW whose update_step multiplies the learning rate per variable).
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=__BASE_LR__, weight_decay=1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    # Continue only in a NEW experiment ID. Do not overwrite EXP-016 checkpoints.
    \"\"\"
    continue_code = continue_code.replace("__LAST_BLOCKS__", str(STAGE2_UNFREEZE_LAST_BLOCKS)).replace("__BASE_LR__", repr(STAGE2_LR))""")
put(xi, s)

si = find("# 20c. STANDARD V2 ADD-ON")
s = get(si)
s = replace_one(s, '''f"Expected stage 2 trainable backbone parameters: {EXPECTED_STAGE2_TRAINABLE_PARAMS:,}"]''',
                '''f"Expected stage 2 trainable backbone parameters: {EXPECTED_STAGE2_TRAINABLE_PARAMS:,}",
                      f"Stage 2 learning rates (EXP-016): {STAGE2_LR_RULE_TEXT}", f"Stage 2 config: {ACTIVE_STAGE2['id']}"]''')
put(si, s)

fin = find("NOTEBOOK_SOURCE_ZLIB_B64 = ")
s = get(fin)
s = 'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n' + s.split('\n', 1)[1]
s = replace_one(s, 'vit_deit_small_transfer_5seeds_015_256px.ipynb', 'vit_deit_small_transfer_llrd_016_256px.ipynb')
s = replace_one(s, 'use the EXP-015 DeiT-Small notebook', 'use the EXP-016 DeiT-Small notebook')
s = replace_one(s, '''        f"DeiT-Small (distilled) ImageNet Transfer, two-stage, {IMG_SIZE}x{IMG_SIZE}, five seeds, trained under the "
        f"DenseNet121 recipe {REFERENCE_EXPERIMENT_ID} (same dataset, augmentation, schedule and selection rule; "
        f"ResNet50 peer: {PEER_EXPERIMENT_ID}).\\n\\n"''',
'''        f"DeiT-Small (distilled) ImageNet Transfer, two-stage, {IMG_SIZE}x{IMG_SIZE}, five seeds. EXP-016 changes ONE factor "
        f"versus the baseline {BASELINE_EXPERIMENT_ID}: the Stage 2 learning-rate scheme (layer-wise LR decay) and unfreeze depth. "
        f"Everything else follows the DenseNet121 recipe {REFERENCE_EXPERIMENT_ID} (dataset, augmentation, Stage 1, schedule and selection rule).\\n\\n"
        f"Stage 2 configuration: {ACTIVE_STAGE2['id']} ({ACTIVE_STAGE2['rule_text']}); source: {STAGE2_CONFIG_SOURCE}.\\n"
        f"Screening experiment: {SCREEN_EXPERIMENT_ID}. Several configurations were compared on Validation, so the Validation figures can be slightly optimistic.\\n"
        f"The archived notebook copy carries the default PHASE; this run used PHASE = {PHASE!r} (see experiment_config.json).\\n\\n"''')
put(fin, s)

# ----------------------------------------------------------------------------------------
# Finish: ids, syntax, revision hash, embedded snapshot
# ----------------------------------------------------------------------------------------
notebook["metadata"] = {
    "kernelspec": notebook["metadata"]["kernelspec"],
    "language_info": notebook["metadata"].get("language_info", {}),
    "exp016": {"derived_from": "vit_deit_small_transfer_5seeds_015_256px.ipynb",
               "plan": "KE_HOACH_CAI_TIEN_VIT_SAU_EXP-015.md, EXP-016",
               "single_factor": "Stage 2 learning-rate scheme (layer-wise LR decay) and unfreeze depth"},
}
for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"vit016-cell-{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        if "!nvidia-smi" not in get(index):
            ast.parse(get(index))

export_index = next(i for i, c in enumerate(notebook["cells"]) if get(i).startswith("NOTEBOOK_SOURCE_ZLIB_B64"))
canonical = json.dumps(notebook, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
config_index = find("# 03. CONFIGURATION", head=200)
put(config_index, replace_one(get(config_index), "__CODE_REVISION__", revision))

snapshot = copy.deepcopy(notebook)
snapshot["cells"][export_index]["source"][0] = 'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n'
blob = base64.b64encode(zlib.compress(
    json.dumps(snapshot, ensure_ascii=False, indent=1).encode("utf-8"), 9)).decode("ascii")
put(export_index, replace_one(get(export_index), 'NOTEBOOK_SOURCE_ZLIB_B64 = ""',
                              'NOTEBOOK_SOURCE_ZLIB_B64 = ' + json.dumps(blob)))
for index, cell in enumerate(notebook["cells"]):
    if cell["cell_type"] == "code" and "!nvidia-smi" not in get(index):
        ast.parse(get(index))
TARGET.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(TARGET, "revision", revision[:8], "cells", len(notebook["cells"]))
