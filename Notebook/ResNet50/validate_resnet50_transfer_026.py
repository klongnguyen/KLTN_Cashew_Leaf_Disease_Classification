"""Static contract checks for the EXP-009-recipe Split-11 notebook."""

import ast
import base64
import hashlib
import json
import re
import zlib
from pathlib import Path


HERE = Path(__file__).resolve().parent


def load(name):
    return json.loads((HERE / name).read_text(encoding="utf-8"))


old = load("resnet50_transfer_5seed_009.ipynb")
new = load("resnet50_transfer_exp009_recipe_split11_026.ipynb")
split11 = load("resnet50_transfer_split11_025.ipynb")
dense = json.loads((HERE.parent / "DenseNet121" /
                    "densenet121_transfer_split11_resize_024.ipynb").read_text(encoding="utf-8"))


def source(book, index):
    value = book["cells"][index]["source"]
    return "".join(value) if isinstance(value, list) else value


def assignment(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    return ast.literal_eval(node.value)
                if isinstance(target, ast.Tuple):
                    names = [x.id for x in target.elts if isinstance(x, ast.Name)]
                    if name in names:
                        return ast.literal_eval(node.value)[names.index(name)]
    raise KeyError(name)


old_config = ast.parse(source(old, 4))
new_config = ast.parse(source(new, 4))
split_config = ast.parse(source(split11, 4))
dense_config = ast.parse(source(dense, 4))
recipe_keys = (
    "SEEDS", "IMG_SIZE", "BATCH_SIZE", "MODEL_NAME", "PRETRAINED_WEIGHTS",
    "HEAD_DROPOUT", "WEIGHT_DECAY", "STAGE1_LR", "STAGE1_EPOCHS",
    "STAGE1_EARLY_PATIENCE", "STAGE1_REDUCE_PATIENCE", "STAGE1_MIN_LR",
    "STAGE2_LR", "STAGE2_EPOCHS", "STAGE2_EARLY_PATIENCE",
    "STAGE2_REDUCE_PATIENCE", "STAGE2_MIN_LR", "LR_REDUCTION_FACTOR",
)
for key in recipe_keys:
    assert assignment(old_config, key) == assignment(new_config, key), key
for key in (
    "DATASET_VERSION", "DATASET_ARCHIVE_NAME", "EXPECTED_COUNTS",
    "EXPECTED_TOTALS", "EXPECTED_ZIP_SHA256", "REAL_HOLDOUT_ARCHIVE_NAME",
    "REAL_HOLDOUT_ARCHIVE_SHA256", "REAL_HOLDOUT_EXPECTED_COUNTS",
):
    assert assignment(new_config, key) == assignment(split_config, key), key
for key in ("DATASET_ARCHIVE_NAME", "EXPECTED_COUNTS", "EXPECTED_TOTALS",
            "EXPECTED_ZIP_SHA256", "REAL_HOLDOUT_ARCHIVE_SHA256"):
    assert assignment(new_config, key) == assignment(dense_config, key), key
assert assignment(new_config, "EXPERIMENT_ID") == "EXP-RESNET50-TRANSFER-5SEEDS-026"
assert assignment(new_config, "DATASET_ARCHIVE_FILE_ID") == "1rq2qBwdQbaL4MSUMSFfRp2QnIQxyI-xt"
assert assignment(new_config, "DRIVE_HOLDOUT_FILE_ID") == "1LCESEVpZvS7Yp0eWnLZKheeatI24KaZ5"
assert assignment(new_config, "DRIVE_DATASET_FOLDER_ID") == "1-Mrqs8Y9dwLexSKIVKBvyYvvQ8X8fXti"
assert assignment(new_config, "DRIVE_RESULTS_FOLDER_ID") == "1DGKTnzsvVc60RIEn5lQgflR3s-loM24G"
assert assignment(new_config, "DRIVE_RUN_FOLDER_ID") == "10kpdZ2cILx5u1mc5dpbgyClOkS4Rn14z"
all_sources = "\n".join(source(new, i) for i in range(len(new["cells"])))
assert "kaggle" not in all_sources.lower()
assert "drive.mount" not in all_sources
assert "auth.authenticate_user()" in source(new, 4)
assert "MediaIoBaseDownload" in source(new, 4)
assert "download_drive_archive(" in source(new, 6)
assert "download_drive_archive(" in source(new, 24)
for flag in ("RUN_FULL_TRAINING", "RUN_FINAL_TEST", "RUN_REAL_HOLDOUT"):
    assert assignment(new_config, flag) is True, flag
assert assignment(new_config, "VIEW_ONLY") is False

model_source = source(new, 14)
for term in (
    'layers.RandomFlip("horizontal", seed=seed)',
    'layers.RandomRotation(0.08, seed=seed)',
    'layers.RandomZoom(0.10, seed=seed)',
    'layers.RandomContrast(0.10, seed=seed)',
    'layer.name.startswith("conv5_block3_")',
    'backbone(x, training=False)',
):
    assert term in model_source, term
assert "layers.RandomBrightness" not in model_source
assert '"brightness": False' in model_source

train = source(new, 17)
assert "selected = choose_stage(stage1, stage2)" in train
assert "deployment_seed = choose_deployment_seed(validation_results_df)" in train
module = ast.parse(train)
namespace = {}
for node in module.body:
    if isinstance(node, ast.FunctionDef) and node.name in {
        "choose_stage", "choose_deployment_seed"
    }:
        exec(compile(ast.Module(body=[node], type_ignores=[]), "selection", "exec"), namespace)
s1 = {"stage": "stage1", "val_loss": 0.2000000, "val_macro_f1": 0.70}
s2 = {"stage": "stage2", "val_loss": 0.2000005, "val_macro_f1": 0.71}
assert namespace["choose_stage"](s1, s2)["stage"] == "stage2"
assert namespace["choose_stage"](s1, {**s2, "val_loss": 0.200002})["stage"] == "stage1"


class Rows:
    def to_dict(self, orient):
        assert orient == "records"
        return [
            {"seed": 42, "val_loss": 0.2, "val_macro_f1": 0.7},
            {"seed": 123, "val_loss": 0.2000005, "val_macro_f1": 0.71},
        ]


assert namespace["choose_deployment_seed"](Rows()) == 123
holdout = source(new, 24)
assert 'validation_results["val_loss"] <= lowest + 1e-6' in holdout
assert "real_holdout_025" not in holdout
assert "validation_selection_lock.json" in source(new, 22)
assert "assert_selection_lock()" in holdout

panels = source(new, 18) + source(new, 31)
for artifact in (
    "accuracy_loss_curves_5seeds.png",
    "validation_confusion_matrices_5seeds{suffix}.png",
    "predictions_standard.csv",
    "per_class_f1_",
    "calibration",
    "roc_pr_outputs",
    "domain_gap_descriptive.png",
):
    assert artifact in panels, artifact
export = source(new, 32)
for artifact in ("_FULL.zip", "_ANALYSIS.zip", "artifact_sha256.csv"):
    assert artifact in export, artifact
assert "MediaFileUpload" in export and "Drive upload verification failed" in export
assert all(not cell.get("outputs") and cell.get("execution_count") is None
           for cell in new["cells"] if cell["cell_type"] == "code")

match = re.match(r'NOTEBOOK_SOURCE_ZLIB_B64 = "([A-Za-z0-9+/=]+)"', export)
assert match
embedded = match.group(1)
snapshot = json.loads(zlib.decompress(base64.b64decode(embedded)))
marker = 'NOTEBOOK_SOURCE_ZLIB_B64 = "__EMBEDDED_SOURCE__"'
assert source(snapshot, 32).startswith(marker)
snapshot["cells"][32]["source"] = source(snapshot, 32).replace(
    marker, f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"', 1
)
assert snapshot == new, "Embedded source must reconstruct the exact clean notebook"
snapshot["cells"][32]["source"] = source(snapshot, 32).replace(
    f'NOTEBOOK_SOURCE_ZLIB_B64 = "{embedded}"', marker, 1
)
revision = hashlib.sha256("".join(source(snapshot, i) for i in
                           (14, 17, 20, 24, 31, 32, 35)).encode()).hexdigest()
assert assignment(new_config, "CODE_REVISION_SHA256") == revision

for index, cell in enumerate(new["cells"]):
    if cell["cell_type"] == "code":
        code = "\n".join(line for line in source(new, index).splitlines()
                         if not line.lstrip().startswith("!"))
        ast.parse(code)
print("EXP-026 contract PASS: recipe, data, selection, plots, ZIPs, source and syntax")
