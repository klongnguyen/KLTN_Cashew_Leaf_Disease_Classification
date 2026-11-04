"""Build the DeiT-Small ImageNet Transfer five-seed notebook (EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015).

The notebook is derived from resnet50_transfer_5seed_031.ipynb, which itself was built to
follow the DenseNet121 EXP-029 recipe. Every shared cell (dataset audit, leakage audit,
augmentation, schedule, callbacks, Validation lock, Final Test, Holdout, exports) is kept
byte-for-byte; only the cells that are architecture specific are replaced.

Usage:  python build_vit_transfer_015.py [SOURCE_031_NOTEBOOK] [TARGET_NOTEBOOK]
"""
import ast
import base64
import copy
import hashlib
import json
import sys
import textwrap
import zlib
from pathlib import Path

ROOT = Path(__file__).parent
SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "ResNet50" / "resnet50_transfer_5seed_031.ipynb"
TARGET = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "vit_deit_small_transfer_5seeds_015_256px.ipynb"
notebook = json.loads(SOURCE.read_text(encoding="utf-8"))
CELLS_031 = len(notebook["cells"])
assert CELLS_031 == 39, CELLS_031


def get(index):
    return "".join(notebook["cells"][index]["source"])


def put(index, source):
    notebook["cells"][index]["source"] = source.splitlines(keepends=True)
    if notebook["cells"][index]["cell_type"] == "code":
        notebook["cells"][index]["execution_count"] = None
        notebook["cells"][index]["outputs"] = []


def replace_one(source, old, new):
    assert source.count(old) == 1, (old[:100], source.count(old))
    return source.replace(old, new)


# ----------------------------------------------------------------------------------------
# Cell 0: introduction
# ----------------------------------------------------------------------------------------
put(0, """# EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015 — DeiT-Small ImageNet Transfer — 5 seeds

**Mục đích:** huấn luyện ViT (DeiT-Small distilled) ở chế độ Transfer trên đúng điều kiện của công thức DenseNet121 chuẩn **EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05** và của ResNet50 **EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4**, để ba kiến trúc được so sánh công bằng trên cùng dataset, cùng Holdout, cùng augmentation, lịch huấn luyện, seed và luật chọn checkpoint.

- Research question: DeiT-Small Transfer đạt chất lượng thế nào so với DenseNet121 và ResNet50 Transfer khi mọi điều kiện huấn luyện giống hệt nhau?
- Parent experiment / ablation factor: không có; đây là so sánh kiến trúc, **chỉ backbone, đầu phân loại và tập khối được mở khóa ở Stage 2 là khác** (bảng bên dưới).
- Dataset: `Cashew_dataV13` / `CashewData_Split (13).zip`, Train 4.972 / Val 1.452 / Test 685, ZIP SHA256 `e1647b72…4e04`, manifest SHA256 `2cee0fb4…07`.
- Real Holdout: `RL_Cashew_holdout (9).zip` (Holdout 09, 126 ảnh, 4 lớp có ground truth, **descriptive audit, không phải blind benchmark**).
- Class order từ runtime: `anthracnose, healthy, leaf_miner, red_rust, not_cashew_leaf`.
- Seeds: `[42, 123, 2026, 3407, 7777]`; mỗi seed khởi tạo lại từ cùng trọng số ImageNet.
- Input 256×256, batch 16, AdamW (weight decay 1e-4), Dropout 0.40; Stage 1 LR 3e-4 tối đa 12 epoch; Stage 2 LR 1e-5 tối đa 30 epoch; EarlyStopping/ReduceLROnPlateau theo đúng tham số của DenseNet121 (xem cell 03a).
- Augmentation (chỉ Train): HorizontalFlip, Rotation 0.08, Zoom 0.10, Contrast 0.25, Brightness 0.25, HSV jitter (saturation 0.6–1.4, hue 0.03), random grayscale 0.10. Không oversample.
- Checkpoint selection rule (chỉ Validation): Validation loss thấp nhất, tie-break Macro-F1 cao hơn.
- `RUN_FINAL_TEST = True`, `RUN_REAL_HOLDOUT = True` (luôn bật; chỉ chạy **sau** khi `validation_selection_lock.json` được ghi).
- Output schema version: 2 (STANDARD_TRAINING_NOTEBOOK_FORMAT V2).

**Khác biệt có chủ đích so với hai CNN (không phải biến thể tự do):**

| Hạng mục | DenseNet121 (EXP-029) | ResNet50 (EXP-031) | DeiT-Small (notebook này) |
|---|---|---|---|
| Backbone | `keras.applications` ImageNet | `keras.applications` ImageNet | KerasHub `deit_small_distilled_patch16_224_imagenet` (dự phòng: `hf://facebook/deit-small-distilled-patch16-224`) |
| Số tham số backbone | 7.037.504 | 23.587.712 | 21.689.472 (ở 256 px) |
| Độ phân giải | 256×256 | 256×256 | 256×256: preset 224 có lưới vị trí 14×14, được nội suy bicubic sang 16×16; hai hàng CLS và DIST giữ nguyên |
| Preprocessing trong model | `/255` rồi ImageNet mean/std | Caffe BGR | **`/255` rồi ImageNet mean/std (giống hệt DenseNet121)**, được kiểm tra với `keras.applications.densenet.preprocess_input` |
| Đầu phân loại | Dropout 0.40 + Dense 5 softmax | Dropout 0.40 + Dense 5 softmax | Hai đầu mới (token CLS và DIST), mỗi đầu Dropout 0.40 + Dense 5; **logit được lấy trung bình** rồi softmax (cách suy luận của DeiT distilled, như EXP-014) |
| Stage 2 mở khóa | tiền tố `conv5_`, không BatchNorm = 2.129.920 tham số (30,3%) | `conv5_block1–3` = 14.953.472 (63,4%) | **4 trong 12 khối encoder cuối + LayerNorm cuối = 7.098.624 tham số (32,7%)**; embedding và khối 1–8 đóng băng |
| Chuẩn hóa | BatchNorm đóng băng | BatchNorm đóng băng | LayerNorm không có thống kê chạy, nên không có phép đóng băng kiểu BatchNorm |

Quy tắc mở khóa là **một dòng cấu hình** (`STAGE2_UNFREEZE_LAST_BLOCKS = 4`): "phần cuối của mạng". ViT không có khái niệm stage như CNN, nên chọn 4/12 khối vì tỉ lệ tham số được mở (32,7%) gần nhất với DenseNet121 chuẩn (30,3%), không tinh chỉnh bằng Test/Holdout. Các thí nghiệm ViT cũ (EXP-011/014) dùng 2 khối cuối; đổi giá trị này sẽ phải dùng EXP ID mới.

**Trạng thái:** notebook chưa được huấn luyện; mọi chỉ số chỉ có sau khi chạy đủ trên Colab GPU. Cell smoke kiểm tra nội suy vị trí, tham số được mở khóa, nạp/lưu lại model và augmentation trước khi huấn luyện 5 seed.
""")

# ----------------------------------------------------------------------------------------
# Cell 4: configuration
# ----------------------------------------------------------------------------------------
s = get(4)
s = replace_one(s, "# 03. CONFIGURATION — EXP-RESNET50-TRANSFER-5SEEDS-031",
                "# 03. CONFIGURATION — EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015")
s = replace_one(s, """# ResNet50 Transfer trained under the DenseNet121 EXP-029 recipe. Every shared
# setting is checked against that recipe by the fairness-contract cell below.
# Only the backbone, its in-model preprocessing and the architecture-dependent
# parameter count of the Stage 2 stage are allowed to differ.""",
"""# DeiT-Small Transfer trained under the DenseNet121 EXP-029 recipe (the same recipe
# ResNet50 EXP-031 follows). Every shared setting is checked against that recipe by
# the fairness-contract cell below. Only the backbone, its classification head, and
# the architecture-dependent set of blocks opened in Stage 2 are allowed to differ.""")
s = replace_one(s, 'NOTEBOOK_VERSION = "031"', 'NOTEBOOK_VERSION = "015"')
s = replace_one(s, 'CODE_REVISION_SHA256 = "c954b7c4692e8bd608fa792f9d344a40fa8ac95ceb56ae18633972227f226025"',
                'CODE_REVISION_SHA256 = "__CODE_REVISION__"')
s = replace_one(s, "Optional path to this exact 031 .ipynb on Drive.", "Optional path to this exact 015 .ipynb on Drive.")
s = replace_one(s, 'EXPERIMENT_ID = f"EXP-RESNET50-TRANSFER-5SEEDS-031-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"',
                'EXPERIMENT_ID = f"EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-015-{IMG_SIZE}px-{CODE_REVISION_SHA256[:8]}"')
s = replace_one(s, 'REFERENCE_EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05"  # recipe reference',
                'REFERENCE_EXPERIMENT_ID = "EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05"  # recipe reference\n'
                'PEER_EXPERIMENT_ID = "EXP-RESNET50-TRANSFER-5SEEDS-031-256px-c954b7c4"  # second architecture under the same recipe')
s = replace_one(s, 'MODEL_NAME = "ResNet50"', 'MODEL_NAME = "DeiT-Small (distilled)"')
s = replace_one(s, '''# Stage 2 opens the last ResNet stage (conv5_block1..3) with the same literal
# prefix rule DenseNet121 uses for its last block; BatchNorm stays frozen.
STAGE2_UNFREEZE_PREFIX = "conv5_"
EXPECTED_BACKBONE_PARAMS = 23_587_712
EXPECTED_STAGE2_TRAINABLE_PARAMS = 14_953_472
EXPECTED_BACKBONE_BATCHNORM_LAYERS = 53
DENSENET121_REFERENCE_STAGE2_PARAMS = 2_129_920''',
'''# ------------------------------------------------------------
# DeiT-Small specifics (the only architecture-dependent block of the recipe)
# ------------------------------------------------------------
# The ImageNet checkpoint is trained at 224 px. KerasHub builds the backbone for one
# fixed image size, so the 256 px backbone is created empty and the 224 px weights are
# copied in; only the position embedding changes shape. Its CLS and DIST rows are kept
# and the 14x14 patch grid is resampled bicubically to 16x16 (DeiT interpolate_pos_encoding).
PRETRAINED_HANDLES = ("deit_small_distilled_patch16_224_imagenet",
                      "hf://facebook/deit-small-distilled-patch16-224")  # same checkpoint, tried second
PRETRAINED_IMG_SIZE = 224
PATCH_SIZE = 16
DEIT_BACKBONE_NAME = "deit_small_backbone"
DEIT_SMALL_BACKBONE_CONFIG = {  # asserted equal to the preset at load time
    "patch_size": (16, 16), "num_layers": 12, "num_heads": 6, "hidden_dim": 384,
    "intermediate_dim": 1536, "dropout_rate": 0.0, "attention_dropout": 0.0,
    "layer_norm_epsilon": 1e-12, "use_mha_bias": True,
}
# Stage 2 opens the LAST part of the network, the ViT analogue of conv5_: the last N of the
# 12 encoder blocks plus the final LayerNorm. Embeddings and earlier blocks stay frozen.
STAGE2_UNFREEZE_LAST_BLOCKS = 4
STAGE2_UNFREEZE_FINAL_LAYERNORM = True
STAGE2_RULE_TEXT = (f"last {STAGE2_UNFREEZE_LAST_BLOCKS} of {DEIT_SMALL_BACKBONE_CONFIG['num_layers']} "
                    "encoder blocks + final LayerNorm (embeddings and earlier blocks frozen)")
EXPECTED_BACKBONE_PARAMS = 21_689_472          # at 256 px; the 224 px preset has 21,666,432
EXPECTED_PRETRAINED_224_PARAMS = 21_666_432
EXPECTED_STAGE2_TRAINABLE_PARAMS = 7_098_624   # 4 x 1,774,464 + 768 (final LayerNorm)
DENSENET121_REFERENCE_STAGE2_PARAMS = 2_129_920
RESNET50_PEER_STAGE2_PARAMS = 14_953_472''')
s = replace_one(s, 'raise FileNotFoundError("A completed ResNet50 Validation lock is required")',
                'raise FileNotFoundError("A completed DeiT-Small Validation lock is required")')
put(4, s)

# ----------------------------------------------------------------------------------------
# Cell 5: fairness contract
# ----------------------------------------------------------------------------------------
s = get(5)
s = replace_one(s, """    "stage2": {"lr": 1e-5, "epochs": 30, "early_patience": 8, "reduce_patience": 4, "min_lr": 1e-7,
               "unfreeze_prefix": "conv5_", "batchnorm_trainable": False},""",
"""    "stage2": {"lr": 1e-5, "epochs": 30, "early_patience": 8, "reduce_patience": 4, "min_lr": 1e-7},""")
s = replace_one(s, """    "stage2": {"lr": STAGE2_LR, "epochs": STAGE2_EPOCHS, "early_patience": STAGE2_EARLY_PATIENCE,
               "reduce_patience": STAGE2_REDUCE_PATIENCE, "min_lr": STAGE2_MIN_LR,
               "unfreeze_prefix": STAGE2_UNFREEZE_PREFIX, "batchnorm_trainable": False},""",
"""    "stage2": {"lr": STAGE2_LR, "epochs": STAGE2_EPOCHS, "early_patience": STAGE2_EARLY_PATIENCE,
               "reduce_patience": STAGE2_REDUCE_PATIENCE, "min_lr": STAGE2_MIN_LR},""")
s = replace_one(s, """MODEL_SPECIFIC_DIFFERENCES = {
    "backbone": "ResNet50 (keras.applications, ImageNet) instead of DenseNet121",
    "preprocessing_inside_model": "Caffe BGR mean subtraction (matches the ImageNet weights)",
    "stage2_trainable_rule": "same prefix rule conv5_ / non-BatchNorm; the parameter count follows the architecture",
}""",
"""MODEL_SPECIFIC_DIFFERENCES = {
    "backbone": "DeiT-Small distilled (KerasHub, ImageNet-1k), 21,689,472 parameters at 256 px",
    "input_resolution": "256x256 like both references; the 224 px preset position grid (14x14) is resampled bicubically to 16x16, CLS and DIST rows kept",
    "preprocessing_inside_model": "x/255 then ImageNet mean/std, identical to the DenseNet121 reference",
    "head": "two new 5-class heads (CLS and DIST token), Dropout 0.40 each, logits averaged before softmax",
    "stage2_trainable_rule": STAGE2_RULE_TEXT + f": {EXPECTED_STAGE2_TRAINABLE_PARAMS:,} backbone parameters "
                             f"({EXPECTED_STAGE2_TRAINABLE_PARAMS / EXPECTED_BACKBONE_PARAMS:.1%}); DenseNet121 reference "
                             f"{DENSENET121_REFERENCE_STAGE2_PARAMS:,}, ResNet50 peer {RESNET50_PEER_STAGE2_PARAMS:,}",
    "normalization": "LayerNorm has no running statistics, so no BatchNorm-style freeze applies (the CNNs keep BatchNorm frozen)",
}""")
put(5, s)

# ----------------------------------------------------------------------------------------
# New cell A (after cell 1): install KerasHub before TensorFlow is imported
# ----------------------------------------------------------------------------------------
INSTALL_CELL = '''# ============================================================
# 01a. KERAS-HUB (DeiT backbone) — install before TensorFlow is imported
# ============================================================
# The project pins keras-hub 0.32.0 (the version used by the earlier DeiT experiments).
# If pip changes Keras after TensorFlow/Keras were already imported in this runtime,
# the runtime must be restarted once and the notebook rerun from the top.
import importlib.util
import subprocess
import sys
from importlib import metadata

KERAS_HUB_VERSION = "0.32.0"
try:
    installed_keras_hub = metadata.version("keras-hub")
except metadata.PackageNotFoundError:
    installed_keras_hub = None
already_imported = ("tensorflow" in sys.modules) or ("keras" in sys.modules)
if installed_keras_hub != KERAS_HUB_VERSION:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                           f"keras-hub=={KERAS_HUB_VERSION}"])
    if already_imported:
        raise RuntimeError("keras-hub was (re)installed after TensorFlow/Keras were imported. "
                           "Use Runtime > Restart session, then run the notebook from the top.")
import keras_hub
print("KerasHub:", keras_hub.__version__, "| DeiTBackbone available:",
      hasattr(keras_hub.models, "DeiTBackbone"))
'''
notebook["cells"].insert(2, {"cell_type": "code", "metadata": {}, "execution_count": None,
                             "outputs": [], "source": INSTALL_CELL.splitlines(keepends=True)})
OFFSET = 1  # every original index >= 2 is shifted by one from here on


def idx(original):
    return original + (OFFSET if original >= 2 else 0)


# imports cell: add keras_hub next to the tensorflow imports
s = get(idx(2))
s = replace_one(s, "from tensorflow.keras import layers\n", "from tensorflow.keras import layers\nimport keras_hub\n")
put(idx(2), s)
s = get(idx(3))
s = replace_one(s, 'print(f"Keras             : {keras.__version__ if hasattr(keras, \'__version__\') else \'tf.keras\'}")',
                'print(f"Keras             : {keras.__version__ if hasattr(keras, \'__version__\') else \'tf.keras\'}")\n'
                'print(f"KerasHub          : {keras_hub.__version__}")')
put(idx(3), s)

# ----------------------------------------------------------------------------------------
# Model markdown (original cell 15)
# ----------------------------------------------------------------------------------------
put(idx(15), """## Model and smoke checks

Raw RGB pixels (0–255) enter the model. Augmentation happens only during
training and is identical to the DenseNet121 and ResNet50 references. The in-model
preprocessing is `/255` followed by the ImageNet mean/std, exactly what the DenseNet121
reference uses; it is verified against `keras.applications.densenet.preprocess_input`
(and, when the preset converter can be loaded, against KerasHub's own DeiT converter) before training.

The DeiT-Small ImageNet weights are loaded once at 224 px and copied into a 256 px
backbone; only the position embedding is resampled (bicubic, CLS/DIST rows kept). The
smoke check verifies the copy tensor by tensor, the trainable parameter counts of both
stages, that the frozen part really stays frozen, and that the saved model reloads.
It uses one Train batch and one Validation batch; it never reads Test or holdout.
""")

# ----------------------------------------------------------------------------------------
# Cell 16: model, freezing rule, smoke test
# ----------------------------------------------------------------------------------------
MODEL_SEGMENT = textwrap.indent('''@keras.utils.register_keras_serializable(package="deit", name="DeiTImageNetPreprocess")
class DeiTImageNetPreprocess(layers.Layer):
    """Rescale RGB 0..255 to 0..1, then ImageNet mean/std (identical to the DenseNet121 reference)."""

    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)

    def call(self, inputs):
        x = inputs / 255.0
        return (x - tf.constant(self.MEAN, dtype=x.dtype)) / tf.constant(self.STD, dtype=x.dtype)

    def compute_output_shape(self, input_shape):
        return input_shape

def deit_preprocessing(inputs):
    return DeiTImageNetPreprocess(name="deit_imagenet_preprocess")(inputs)

@keras.utils.register_keras_serializable(package="deit", name="TokenSlice")
class TokenSlice(layers.Layer):
    """Select one token (0 = CLS, 1 = DIST) from a (batch, tokens, dim) sequence."""

    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = int(index)

    def call(self, inputs):
        return inputs[:, self.index, :]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[2])

    def get_config(self):
        config = super().get_config()
        config.update({"index": self.index})
        return config

# ---- Pretrained source: 224 px ImageNet DeiT-Small, loaded once per runtime ----
PRETRAINED_SOURCE = {}

def _same_backbone_value(key, actual, expected):
    if key in ("patch_size", "image_shape"):
        actual = tuple(actual) if isinstance(actual, (list, tuple)) else (actual, actual)
        return actual == tuple(expected)
    if isinstance(expected, float):
        return abs(float(actual) - expected) <= 1e-6 * max(1.0, abs(expected))
    return actual == expected

def load_pretrained_source():
    """Download the 224 px preset once; every seed copies exactly the same arrays."""
    if PRETRAINED_SOURCE:
        return PRETRAINED_SOURCE
    failures = []
    source = handle_used = None
    for handle in PRETRAINED_HANDLES:
        try:
            source = keras_hub.models.DeiTBackbone.from_preset(handle)
            handle_used = handle
            break
        except Exception as error:  # Kaggle/HF access can fail independently in Colab.
            failures.append(f"{handle}: {type(error).__name__}: {error}")
            print("Pretrained handle unavailable ->", failures[-1])
    if source is None:
        raise RuntimeError("Could not load the DeiT-Small ImageNet checkpoint:\\n" + "\\n".join(failures))
    config = source.get_config()
    if tuple(config["image_shape"]) != (PRETRAINED_IMG_SIZE, PRETRAINED_IMG_SIZE, 3):
        raise AssertionError(f"Unexpected preset resolution: {config['image_shape']}")
    wrong = {key: (config.get(key), expected) for key, expected in DEIT_SMALL_BACKBONE_CONFIG.items()
             if not _same_backbone_value(key, config.get(key), expected)}
    if wrong:
        raise AssertionError("Preset differs from DEIT_SMALL_BACKBONE_CONFIG (actual, expected): "
                             f"{wrong}. Update the constants only with a new experiment ID.")
    if int(source.count_params()) != EXPECTED_PRETRAINED_224_PARAMS:
        raise AssertionError(f"Unexpected preset size: {source.count_params()}")
    weights = [(variable.name, np.asarray(variable.numpy())) for variable in source.weights]
    digest = hashlib.sha256()
    for name, array in weights:
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(array).tobytes())
    PRETRAINED_SOURCE.update({"handle": handle_used, "weights": weights,
                              "sha256": digest.hexdigest(), "parameters": int(source.count_params())})
    del source
    print("Pretrained DeiT-Small:", handle_used, "| parameters:", PRETRAINED_SOURCE["parameters"],
          "| weights SHA256:", PRETRAINED_SOURCE["sha256"][:16])
    return PRETRAINED_SOURCE

def interpolate_position_embedding(position, source_size, target_size):
    """Keep CLS/DIST rows; resample the square patch grid bicubically (DeiT interpolate_pos_encoding)."""
    position = np.asarray(position, dtype=np.float32)
    special, patches = position[:, :2, :], position[:, 2:, :]
    source_grid, target_grid = source_size // PATCH_SIZE, target_size // PATCH_SIZE
    if patches.shape[1] != source_grid * source_grid:
        raise ValueError(f"Position embedding has {patches.shape[1]} patch rows, expected {source_grid ** 2}")
    if source_grid == target_grid:
        return position.copy()
    grid = patches.reshape(1, source_grid, source_grid, -1)
    resized = tf.image.resize(grid, (target_grid, target_grid), method="bicubic", antialias=False).numpy()
    return np.concatenate([special, resized.reshape(1, target_grid * target_grid, -1)], axis=1)

def build_deit_backbone(weights="imagenet"):
    """DeiT-Small at IMG_SIZE. weights=None only reconstructs the architecture before a checkpoint is restored."""
    backbone = keras_hub.models.DeiTBackbone(
        image_shape=(IMG_SIZE, IMG_SIZE, 3), name=DEIT_BACKBONE_NAME, **DEIT_SMALL_BACKBONE_CONFIG)
    if weights is None:
        return backbone
    if weights != "imagenet":
        raise ValueError(weights)
    pretrained = load_pretrained_source()
    if len(backbone.weights) != len(pretrained["weights"]):
        raise AssertionError("Backbone/preset weight lists differ in length")
    for variable, (name, array) in zip(backbone.weights, pretrained["weights"]):
        if variable.name != name:
            raise AssertionError(f"Weight order mismatch: {variable.name} vs {name}")
        if tuple(variable.shape) == array.shape:
            variable.assign(array)
        elif name == "position_embedding":
            variable.assign(interpolate_position_embedding(array, PRETRAINED_IMG_SIZE, IMG_SIZE))
        else:
            raise AssertionError(f"Unexpected shape change for {name}: {array.shape} -> {tuple(variable.shape)}")
    return backbone

def build_augmentation(seed):
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(ROTATION_FACTOR, seed=seed),
        layers.RandomZoom(ZOOM_FACTOR, seed=seed),
        RandomHSVJitter(saturation_range=SATURATION_RANGE, hue_delta=HUE_DELTA,
                        grayscale_probability=GRAYSCALE_PROBABILITY,
                        value_range=BRIGHTNESS_VALUE_RANGE, seed=seed + 2900),
        layers.RandomContrast(CONTRAST_FACTOR, seed=seed),
        layers.RandomBrightness(factor=BRIGHTNESS_FACTOR,
                                value_range=BRIGHTNESS_VALUE_RANGE, seed=seed + 1100),
    ], name=f"train_only_augmentation_seed_{seed}")

def build_transfer_model(seed, weights="imagenet"):
    set_global_seed(seed)
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="raw_rgb")
    augmented = build_augmentation(seed)(inputs)
    x = deit_preprocessing(augmented)
    backbone = build_deit_backbone(weights)
    backbone.trainable = False
    tokens = backbone(x)  # (batch, 2 + 256, 384); backbone dropout rates are 0, so no training flag is needed
    class_token = TokenSlice(0, name="class_token_output")(tokens)
    distillation_token = TokenSlice(1, name="distillation_token_output")(tokens)
    class_logits = layers.Dense(NUM_CLASSES, name="class_head")(
        layers.Dropout(HEAD_DROPOUT, seed=seed, name="class_head_dropout")(class_token))
    distillation_logits = layers.Dense(NUM_CLASSES, name="distillation_head")(
        layers.Dropout(HEAD_DROPOUT, seed=seed + 1, name="distillation_head_dropout")(distillation_token))
    logits = layers.Average(name="averaged_logits")([class_logits, distillation_logits])
    outputs = layers.Activation("softmax", dtype="float32", name="disease_output")(logits)
    return keras.Model(inputs, outputs, name=f"DeiTSmall_Transfer_seed_{seed}"), backbone

def set_backbone_stage(backbone, stage):
    if stage == "stage1":
        backbone.trainable = False
    elif stage == "stage2":
        backbone.trainable = True  # recursive; the parts that stay frozen are switched off below
        embeddings = backbone.get_layer("deit_patching_and_embedding")
        encoder = backbone.get_layer("deit_encoder")
        embeddings.trainable = False
        first_open_block = DEIT_SMALL_BACKBONE_CONFIG["num_layers"] - STAGE2_UNFREEZE_LAST_BLOCKS
        if len(encoder.encoder_layers) != DEIT_SMALL_BACKBONE_CONFIG["num_layers"]:
            raise RuntimeError("Unexpected number of encoder blocks")
        for block_index, block in enumerate(encoder.encoder_layers):
            block.trainable = block_index >= first_open_block
        encoder.layer_norm.trainable = STAGE2_UNFREEZE_FINAL_LAYERNORM
        if not backbone.trainable_weights:
            raise RuntimeError("No weighted Stage 2 layer was unfrozen")
    else:
        raise ValueError(stage)
    return int(sum(np.prod(v.shape) for v in backbone.trainable_weights))

''', "    ")

s = get(idx(16))
s = replace_one(s, "# 08. ImageNet backbone, frozen BN, photometric augmentation, and smoke test",
                "# 08. DeiT-Small backbone (256 px), photometric augmentation, and smoke test")
start = s.index('    @keras.utils.register_keras_serializable(package="resnet50", name="CaffeBGRPreprocess")')
end = s.index("    def compile_stage(model, lr):")
s = s[:start] + MODEL_SEGMENT + s[end:]

# --- smoke: preprocessing parity ---
s = replace_one(s, """        parity_model = keras.Model((p := keras.Input((IMG_SIZE, IMG_SIZE, 3))),
                                   resnet_preprocessing(p))
        parity_error = float(tf.reduce_max(tf.abs(
            parity_model(parity_input) - keras.applications.resnet50.preprocess_input(parity_input)
        )).numpy())
        if parity_error >= 1e-5:
            raise AssertionError(f"ResNet50 preprocessing parity failed: {parity_error}")""",
"""        parity_model = keras.Model((p := keras.Input((IMG_SIZE, IMG_SIZE, 3))),
                                   deit_preprocessing(p))
        parity_error = float(tf.reduce_max(tf.abs(
            parity_model(parity_input) - keras.applications.densenet.preprocess_input(parity_input)
        )).numpy())
        if parity_error >= 1e-5:
            raise AssertionError(f"Preprocessing differs from the DenseNet121 reference: {parity_error}")
        # Second, independent check against the converter shipped with the DeiT preset.
        try:
            converter = keras_hub.layers.DeiTImageConverter.from_preset(load_pretrained_source()["handle"])
            converter_probe = tf.random.uniform((2, PRETRAINED_IMG_SIZE, PRETRAINED_IMG_SIZE, 3), 0, 255)
            converter_error = float(tf.reduce_max(tf.abs(
                tf.cast(converter(converter_probe), tf.float32) -
                DeiTImageNetPreprocess()(converter_probe))).numpy())
            if converter_error > 1e-3:
                raise AssertionError(f"Preset image converter disagrees with the in-model preprocessing: {converter_error}")
            preset_converter_check = {"status": "PASS", "max_abs_error": converter_error}
        except AssertionError:
            raise
        except Exception as error:  # the converter API is not part of the training path
            preset_converter_check = {"status": "UNVERIFIED", "reason": f"{type(error).__name__}: {error}"}
        print("Preset converter check:", preset_converter_check)""")

# --- smoke: pretrained-weight transfer gate ---
s = replace_one(s, """        model, backbone = build_transfer_model(SEEDS[0], weights="imagenet")
""",
"""        model, backbone = build_transfer_model(SEEDS[0], weights="imagenet")
        pretrained = load_pretrained_source()
        if int(backbone.count_params()) != EXPECTED_BACKBONE_PARAMS:
            raise AssertionError(f"Unexpected 256 px DeiT-Small backbone size: {backbone.count_params()}")
        copied_tensors, resampled_tensors = 0, 0
        for variable, (name, array) in zip(backbone.weights, pretrained["weights"]):
            if tuple(variable.shape) == array.shape:
                if not np.array_equal(variable.numpy(), array):
                    raise AssertionError(f"Pretrained tensor was not copied exactly: {name}")
                copied_tensors += 1
            else:
                resampled_tensors += 1
                if name != "position_embedding":
                    raise AssertionError(f"Unexpected resampled tensor: {name}")
                source_position, new_position = array, variable.numpy()
                grid_tokens = (IMG_SIZE // PATCH_SIZE) ** 2
                if new_position.shape != (1, 2 + grid_tokens, array.shape[-1]):
                    raise AssertionError(f"Bad resampled position embedding shape: {new_position.shape}")
                if not np.array_equal(new_position[:, :2], source_position[:, :2]):
                    raise AssertionError("CLS/DIST position rows must not change")
                if not np.array_equal(interpolate_position_embedding(source_position, PRETRAINED_IMG_SIZE,
                                                                     PRETRAINED_IMG_SIZE), source_position):
                    raise AssertionError("Same-size position resampling must be the identity")
                grid_in = source_position[:, 2:, :]
                grid_out = new_position[:, 2:, :]
                # A bicubic resample of a smooth grid stays near the source range and keeps its mean.
                span = float(grid_in.max() - grid_in.min())
                if (grid_out.min() < grid_in.min() - 0.25 * span or grid_out.max() > grid_in.max() + 0.25 * span
                        or abs(float(grid_out.mean() - grid_in.mean())) > 0.05 * span + 1e-6):
                    raise AssertionError("Resampled position grid left the source value range")
        position_gate = {"copied_tensors": copied_tensors, "resampled_tensors": resampled_tensors,
                         "source_grid": PRETRAINED_IMG_SIZE // PATCH_SIZE, "target_grid": IMG_SIZE // PATCH_SIZE,
                         "method": "tf.image.resize bicubic, antialias=False; CLS/DIST rows unchanged",
                         "pretrained_handle": pretrained["handle"],
                         "pretrained_weights_sha256": pretrained["sha256"], "status": "PASS"}
        print("Pretrained weight transfer gate PASS:", position_gate)
""")

# --- smoke: normalization / sizes / unfrozen blocks ---
s = replace_one(s, """        bn_layers = [layer for layer in backbone.layers if isinstance(layer, layers.BatchNormalization)]
        if len(bn_layers) != EXPECTED_BACKBONE_BATCHNORM_LAYERS:
            raise AssertionError(f"Expected {EXPECTED_BACKBONE_BATCHNORM_LAYERS} BatchNorm layers")
        backbone_params = int(backbone.count_params())
        if backbone_params != EXPECTED_BACKBONE_PARAMS:
            raise AssertionError(f"Unexpected ResNet50 backbone size: {backbone_params}")
        bn_means = [layer.moving_mean.numpy().copy() for layer in bn_layers]
""",
"""        if any(isinstance(layer, layers.BatchNormalization) for layer in model._flatten_layers()):
            raise AssertionError("DeiT-Small must contain no BatchNorm layer")
        backbone_params = int(backbone.count_params())
        if backbone_params != EXPECTED_BACKBONE_PARAMS:
            raise AssertionError(f"Unexpected DeiT-Small backbone size: {backbone_params}")
        backbone_config = backbone.get_config()
        if backbone_config["dropout_rate"] != 0.0 or backbone_config["attention_dropout"] != 0.0:
            raise AssertionError("Backbone dropout must be 0 so a training flag cannot change it")
        embedding_layer = backbone.get_layer("deit_patching_and_embedding")
        encoder = backbone.get_layer("deit_encoder")
        frozen_probe = {"embedding": embedding_layer.patch_embedding.kernel,
                        "first_block": encoder.encoder_layers[0].weights[0],
                        "final_open_block": encoder.encoder_layers[-1].weights[0]}
""")
s = replace_one(s, """        unfrozen_names = sorted({layer.name.split("_")[0] + "_" + layer.name.split("_")[1]
                                 for layer in backbone.layers if layer.trainable and layer.weights})
        if unfrozen_names != ["conv5_block1", "conv5_block2", "conv5_block3"]:
            raise AssertionError(f"Unexpected unfrozen ResNet50 blocks: {unfrozen_names}")
""",
"""        num_blocks = DEIT_SMALL_BACKBONE_CONFIG["num_layers"]
        unfrozen_names = [f"transformer_block_{block_index + 1}"
                          for block_index, block in enumerate(encoder.encoder_layers)
                          if block.trainable_weights]
        expected_names = [f"transformer_block_{number}"
                          for number in range(num_blocks - STAGE2_UNFREEZE_LAST_BLOCKS + 1, num_blocks + 1)]
        if unfrozen_names != expected_names:
            raise AssertionError(f"Unexpected unfrozen DeiT blocks: {unfrozen_names}")
        if embedding_layer.trainable_weights:
            raise AssertionError("Embeddings must stay frozen in Stage 2")
        if bool(encoder.layer_norm.trainable_weights) != STAGE2_UNFREEZE_FINAL_LAYERNORM:
            raise AssertionError("Final LayerNorm trainability differs from the configuration")
        unfrozen_names = unfrozen_names + (["ln"] if STAGE2_UNFREEZE_FINAL_LAYERNORM else [])
        stage2_probe_before = {key: variable.numpy().copy() for key, variable in frozen_probe.items()}
""")
s = replace_one(s, """        if any(not np.array_equal(before, layer.moving_mean.numpy())
               for before, layer in zip(bn_means, bn_layers)):
            raise AssertionError("Backbone BatchNorm moving means changed")
""",
"""        for key in ("embedding", "first_block"):
            if not np.array_equal(stage2_probe_before[key], frozen_probe[key].numpy()):
                raise AssertionError(f"Frozen DeiT part changed in Stage 2: {key}")
        if np.array_equal(stage2_probe_before["final_open_block"], frozen_probe["final_open_block"].numpy()):
            raise AssertionError("Unfrozen DeiT block did not update in Stage 2")
""")

# --- smoke: save/reload mutation and report ---
s = replace_one(s, "            head_bias = model.layers[-1].bias\n",
                '            head_bias = model.get_layer("class_head").bias\n')
s = replace_one(s, '''            "backbone_batchnorm_layers": len(bn_layers),
            "weights_and_model_reload": "PASS", "imagenet_initialization": "PASS",''',
'''            "backbone_batchnorm_layers": 0,
            "backbone_normalization": "LayerNorm (no running statistics)",
            "backbone_parameters": backbone_params,
            "resnet50_peer_stage2_params": RESNET50_PEER_STAGE2_PARAMS,
            "pretrained_weight_transfer_gate": position_gate,
            "preset_converter_check": preset_converter_check,
            "weights_and_model_reload": "PASS", "imagenet_initialization": "PASS",''')
s = replace_one(s, 'print("Pipeline smoke PASS: ImageNet, preprocessing, trainability, BN, reload, photometric gate")',
                'print("Pipeline smoke PASS: ImageNet transfer, preprocessing, trainability, frozen parts, reload, photometric gate")')
put(idx(16), s)

# ----------------------------------------------------------------------------------------
# Cell 17 markdown and cell 18 (recipe JSON)
# ----------------------------------------------------------------------------------------
put(idx(17), """## Freeze the recipe, then train both stages

The configuration and dataset fingerprint are written before fitting. Each
seed has separate Stage 1 and Stage 2 histories and checkpoints. Stage 2
starts from the best Stage 1 weights, uses a fresh optimizer, and opens only the last
four encoder blocks and the final LayerNorm. The selected seed checkpoint comes from
Validation only.
""")

s = get(idx(18))
s = replace_one(s, 'raise ValueError("VIEW_ONLY notebook revision differs from saved ResNet50 recipe")',
                'raise ValueError("VIEW_ONLY notebook revision differs from saved DeiT-Small recipe")')
s = replace_one(s, """else:
    # 09. Immutable experiment recipe
    experiment_config = {""",
"""else:
    # 09. Immutable experiment recipe
    PRETRAINED_INFO = load_pretrained_source()  # same arrays for every seed; hash is frozen below
    experiment_config = {""")
s = replace_one(s, '''        "comparison_scope": "cross-architecture comparison (ResNet50 versus DenseNet121) under one frozen Transfer recipe",''',
                '''        "peer_experiment": PEER_EXPERIMENT_ID,
        "comparison_scope": "cross-architecture comparison (DeiT-Small versus DenseNet121 EXP-029 and ResNet50 EXP-031) under one frozen Transfer recipe",''')
s = replace_one(s, '''        "model": {"backbone": "ResNet50", "weights": "imagenet",
                  "include_top": False, "pooling": "avg", "head_dropout": HEAD_DROPOUT,
                  "preprocessing": "inside model: Keras ResNet50 caffe (RGB to BGR, ImageNet channel means, no /255)",
                  "augmentation": AUGMENTATION_CONFIG,
                  "backbone_call": "backbone(x, training=False)"},''',
'''        "model": {"backbone": "DeiT-Small distilled (KerasHub DeiTBackbone)", "weights": "imagenet",
                  "pretrained_handles": list(PRETRAINED_HANDLES),
                  "pretrained_weights_sha256": PRETRAINED_INFO["sha256"],
                  "pretrained_resolution": PRETRAINED_IMG_SIZE, "input_resolution": IMG_SIZE,
                  "position_embedding": "224 px 14x14 patch grid resampled bicubically to 16x16; CLS and DIST rows kept",
                  "backbone_config": {key: list(value) if isinstance(value, tuple) else value
                                      for key, value in DEIT_SMALL_BACKBONE_CONFIG.items()},
                  "backbone_parameters": EXPECTED_BACKBONE_PARAMS,
                  "head": "CLS and DIST tokens -> Dropout(0.40) -> Dense(5) each; logits averaged; softmax",
                  "head_dropout": HEAD_DROPOUT,
                  "preprocessing": "inside model: x/255, then ImageNet mean/std (identical to DenseNet121 reference)",
                  "augmentation": AUGMENTATION_CONFIG,
                  "backbone_call": "backbone(x); backbone dropout rates are 0"},''')
s = replace_one(s, '''                   "trainable_backbone": f"{STAGE2_UNFREEZE_PREFIX}* non-BatchNorm only (conv5_block1-3)",
                   "trainable_backbone_params": EXPECTED_STAGE2_TRAINABLE_PARAMS,
                   "backbone_params": EXPECTED_BACKBONE_PARAMS,
                   "densenet121_reference_trainable_params": DENSENET121_REFERENCE_STAGE2_PARAMS},''',
'''                   "trainable_backbone": STAGE2_RULE_TEXT,
                   "unfreeze_last_blocks": STAGE2_UNFREEZE_LAST_BLOCKS,
                   "unfreeze_final_layernorm": STAGE2_UNFREEZE_FINAL_LAYERNORM,
                   "trainable_backbone_params": EXPECTED_STAGE2_TRAINABLE_PARAMS,
                   "backbone_params": EXPECTED_BACKBONE_PARAMS,
                   "densenet121_reference_trainable_params": DENSENET121_REFERENCE_STAGE2_PARAMS,
                   "resnet50_peer_trainable_params": RESNET50_PEER_STAGE2_PARAMS},''')
s = replace_one(s, 'raise ValueError("Existing ResNet50 recipe differs. Use a new experiment ID for changes.")',
                'raise ValueError("Existing DeiT-Small recipe differs. Use a new experiment ID for changes.")')
s = replace_one(s, '''        "python": sys.version, "tensorflow": tf.__version__,
        "keras": getattr(keras, "__version__", "tf.keras"),
        "platform": platform.platform(), "gpu": [str(g) for g in gpus],''',
'''        "python": sys.version, "tensorflow": tf.__version__,
        "keras": getattr(keras, "__version__", "tf.keras"),
        "keras_hub": keras_hub.__version__,
        "pretrained_handle_used": PRETRAINED_SOURCE.get("handle"),
        "platform": platform.platform(), "gpu": [str(g) for g in gpus],''')
put(idx(18), s)

# ----------------------------------------------------------------------------------------
# Cells that only carry names: holdout (24), deployment (26), manual (31), statistics (36), export (38)
# ----------------------------------------------------------------------------------------
s = get(idx(24))
s = replace_one(s, "External classification evaluation for ResNet50 EXP-031.", "External classification evaluation for DeiT-Small EXP-015.")
assert s.count("real_holdout_031") == 3
s = s.replace("real_holdout_031", "real_holdout_015")
put(idx(24), s)

s = get(idx(26))
s = replace_one(s, '''    trained_backbone = deployment_model.get_layer("resnet50")
    feature_input = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    feature_x = resnet_preprocessing(feature_input)
    feature_backbone = keras.applications.ResNet50(include_top=False, weights=None,
                                                   input_shape=(IMG_SIZE, IMG_SIZE, 3))
    feature_backbone.set_weights(trained_backbone.get_weights())
    feature_map = feature_backbone(feature_x, training=False)
    feature_extractor = keras.Model(feature_input, feature_map, name="ResNet50_feature_extractor")
    embedding_model = keras.Model(feature_input, layers.GlobalAveragePooling2D()(feature_map),
                                  name="ResNet50_embedding_model")
    feature_extractor.save(MODEL_PACKAGE_DIR / "resnet50_feature_extractor.keras")
    embedding_model.save(MODEL_PACKAGE_DIR / "resnet50_embedding_model.keras")''',
'''    trained_backbone = deployment_model.get_layer(DEIT_BACKBONE_NAME)
    feature_input = keras.Input((IMG_SIZE, IMG_SIZE, 3), name="image")
    feature_x = deit_preprocessing(feature_input)
    token_sequence = trained_backbone(feature_x)  # (batch, 258, 384): CLS, DIST, 256 patch tokens
    feature_extractor = keras.Model(feature_input, token_sequence, name="DeiTSmall_token_extractor")
    embedding = layers.Concatenate(name="cls_dist_embedding")(
        [TokenSlice(0, name="embedding_cls")(token_sequence),
         TokenSlice(1, name="embedding_dist")(token_sequence)])
    embedding_model = keras.Model(feature_input, embedding, name="DeiTSmall_embedding_model")
    feature_extractor.save(MODEL_PACKAGE_DIR / "deit_small_token_extractor.keras")
    embedding_model.save(MODEL_PACKAGE_DIR / "deit_small_embedding_model.keras")''')
s = replace_one(s, '''    import tensorflow as tf
    from PIL import Image, ImageOps
    from tensorflow import keras

    MODEL_DIR''', '''    import tensorflow as tf
    from PIL import Image, ImageOps
    from tensorflow import keras
    import keras_hub  # registers DeiTBackbone; the custom layers (DeiTImageNetPreprocess, TokenSlice,
    # RandomHSVJitter) must be defined/imported from this notebook before load_model.

    MODEL_DIR''')
s = replace_one(s, '''    continue_code = """from tensorflow import keras
    model = keras.models.load_model("model_package/deployment_model.keras", compile=False)
    backbone = model.get_layer("resnet50")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith("conv5_") and not isinstance(layer, keras.layers.BatchNormalization)''',
'''    continue_code = """from tensorflow import keras
    import keras_hub  # registers DeiTBackbone
    model = keras.models.load_model("model_package/deployment_model.keras", compile=False)
    backbone = model.get_layer("deit_small_backbone")
    backbone.trainable = True
    backbone.get_layer("deit_patching_and_embedding").trainable = False
    encoder = backbone.get_layer("deit_encoder")
    for block_index, block in enumerate(encoder.encoder_layers):
        block.trainable = block_index >= 12 - 4
    encoder.layer_norm.trainable = True''')
s = replace_one(s, "Do not overwrite EXP-031 checkpoints.", "Do not overwrite EXP-015 checkpoints.")
s = replace_one(s, '''        f"tensorflow=={tf.__version__}\\nnumpy=={np.__version__}\\npandas=={pd.__version__}\\n"''',
                '''        f"tensorflow=={tf.__version__}\\nkeras-hub=={keras_hub.__version__}\\nnumpy=={np.__version__}\\npandas=={pd.__version__}\\n"''')
put(idx(26), s)

s = get(idx(31))
s = replace_one(s, "# The path goes through the same EXIF, RGB, bilinear-resize and in-model Caffe\n# preprocessing used for evaluation; no external /255 or mean subtraction.",
                "# The path goes through the same EXIF, RGB, bilinear-resize and in-model DeiT\n# preprocessing used for evaluation; no external /255 or mean subtraction.")
put(idx(31), s)

s = get(idx(36))
s = replace_one(s, '''f"Backbone: {MODEL_NAME} ({PRETRAINED_WEIGHTS}), stage 2 rule prefix '{STAGE2_UNFREEZE_PREFIX}' (non-BatchNorm)",''',
                '''f"Backbone: {MODEL_NAME} ({PRETRAINED_WEIGHTS}), stage 2 rule: {STAGE2_RULE_TEXT}",''')
put(idx(36), s)

s = get(idx(38))
s = 'NOTEBOOK_SOURCE_ZLIB_B64 = ""\n' + s.split('\n', 1)[1]
s = replace_one(s, 'resnet50_transfer_5seed_031.ipynb', 'vit_deit_small_transfer_5seeds_015_256px.ipynb')
s = replace_one(s, 'use the EXP-031 ResNet50 notebook', 'use the EXP-015 DeiT-Small notebook')
s = replace_one(s, '''        f"ResNet50 ImageNet Transfer, two-stage, {IMG_SIZE}x{IMG_SIZE}, five seeds, trained under the "
        f"DenseNet121 recipe {REFERENCE_EXPERIMENT_ID} (same dataset, augmentation, schedule and selection rule).\\n\\n"
        f"Dataset: {DATASET_ARCHIVE_NAME} (SHA256 {DATASET_ZIP_SHA256}, manifest {DATASET_MANIFEST_SHA256}).\\n"
        f"Stage 2 unfreezes conv5_block1-3 non-BatchNorm layers ({EXPECTED_STAGE2_TRAINABLE_PARAMS:,} of "
        f"{EXPECTED_BACKBONE_PARAMS:,} backbone parameters; DenseNet121 reference: {DENSENET121_REFERENCE_STAGE2_PARAMS:,}).\\n"''',
'''        f"DeiT-Small (distilled) ImageNet Transfer, two-stage, {IMG_SIZE}x{IMG_SIZE}, five seeds, trained under the "
        f"DenseNet121 recipe {REFERENCE_EXPERIMENT_ID} (same dataset, augmentation, schedule and selection rule; "
        f"ResNet50 peer: {PEER_EXPERIMENT_ID}).\\n\\n"
        f"Dataset: {DATASET_ARCHIVE_NAME} (SHA256 {DATASET_ZIP_SHA256}, manifest {DATASET_MANIFEST_SHA256}).\\n"
        f"Stage 2 rule: {STAGE2_RULE_TEXT} ({EXPECTED_STAGE2_TRAINABLE_PARAMS:,} of "
        f"{EXPECTED_BACKBONE_PARAMS:,} backbone parameters; DenseNet121 reference: {DENSENET121_REFERENCE_STAGE2_PARAMS:,}).\\n"
        f"Pretrained weights: {PRETRAINED_SOURCE.get('handle')} at 224 px, position embedding resampled to 256 px "
        f"(weights SHA256 {PRETRAINED_SOURCE.get('sha256')}).\\n"''')
put(idx(38), s)

# ----------------------------------------------------------------------------------------
# Finish: ids, syntax, revision hash, embedded snapshot
# ----------------------------------------------------------------------------------------
notebook["metadata"] = {
    "kernelspec": notebook["metadata"]["kernelspec"],
    "exp015": {"derived_from": "resnet50_transfer_5seed_031.ipynb (template) and "
                               "EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05 (recipe)",
               "single_factor": "architecture: backbone, head and Stage 2 block set"},
}
for index, cell in enumerate(notebook["cells"]):
    cell["id"] = f"vit015-cell-{index:02d}"
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []
        if "!nvidia-smi" not in get(index):
            ast.parse(get(index))

export_index = next(i for i, c in enumerate(notebook["cells"]) if get(i).startswith("NOTEBOOK_SOURCE_ZLIB_B64"))
canonical = json.dumps(notebook, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
revision = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
config_index = next(i for i, c in enumerate(notebook["cells"]) if "# 03. CONFIGURATION" in get(i))
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
