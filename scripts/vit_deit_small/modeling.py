"""Frozen DeiT-Small architecture and trainable-layer contract for EXP-010."""

import keras
import keras_hub
import numpy as np

PRESET = "kaggle://keras/deit/keras/deit_small_distilled_patch16_224_imagenet/1"
CLASSES = ("anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf")
LAST_BLOCKS = ("transformer_block_11", "transformer_block_12")


def build_model(seed):
    keras.utils.set_random_seed(seed)
    classifier = keras_hub.models.DeiTImageClassifier.from_preset(
        PRESET, num_classes=len(CLASSES), activation=None, head_dtype="float32")
    if classifier.preprocessor is None:
        raise RuntimeError("DeiT-Small preset has no preprocessor")
    augmentation = keras.Sequential([
        keras.layers.RandomFlip("horizontal", seed=seed + 1),
        keras.layers.RandomRotation(0.04, seed=seed + 2),
        keras.layers.RandomZoom((-0.05, 0.05), (-0.05, 0.05), seed=seed + 3),
        keras.layers.RandomTranslation(0.03, 0.03, seed=seed + 4),
        keras.layers.RandomContrast(0.08, seed=seed + 5),
    ], name="augmentation")
    inputs = keras.Input(shape=(224, 224, 3), dtype="float32", name="image_0_255")
    # The nested KerasHub Task does not auto-run its preprocessor. Apply it once.
    outputs = classifier(classifier.preprocessor(augmentation(inputs)))
    model = keras.Model(inputs, outputs, name="deit_small_transfer")
    if model.output_shape != (None, 5) or classifier.output_dense.units != 5:
        raise RuntimeError(f"Wrong DeiT-Small head: {model.output_shape}")
    return model, classifier


def trainable_inventory(model):
    return [{"path": getattr(weight, "path", weight.name),
             "shape": tuple(int(x) for x in weight.shape),
             "parameters": int(np.prod(weight.shape))} for weight in model.trainable_weights]


def set_stage1(model, classifier):
    model.trainable = True
    classifier.trainable = True
    classifier.backbone.trainable = False
    classifier.output_dense.trainable = True
    inventory = trainable_inventory(model)
    if not inventory or any(classifier.output_dense.name not in item["path"] for item in inventory):
        raise RuntimeError(f"Stage 1 has non-head trainable weights: {inventory}")
    return inventory


def set_stage2(model, classifier):
    model.trainable = True
    classifier.trainable = True
    backbone = classifier.backbone
    backbone.trainable = True
    backbone.get_layer("deit_patching_and_embedding").trainable = False
    encoder = backbone.get_layer("deit_encoder")
    encoder.trainable = True
    blocks = list(encoder.encoder_layers)
    if len(blocks) != 12 or tuple(block.name for block in blocks[-2:]) != LAST_BLOCKS:
        raise RuntimeError("DeiT encoder block layout changed; stop before fine-tuning")
    for block in blocks:
        block.trainable = block.name in LAST_BLOCKS
    for layer in encoder._layers:
        if layer not in blocks:
            layer.trainable = False
    classifier.output_dense.trainable = True
    inventory = trainable_inventory(model)
    paths = [item["path"] for item in inventory]
    if not paths or not all(any(name in path for name in (*LAST_BLOCKS, classifier.output_dense.name)) for path in paths):
        raise RuntimeError(f"Stage 2 opened unexpected weights: {paths}")
    if not all(any(name in path for path in paths) for name in LAST_BLOCKS):
        raise RuntimeError(f"Stage 2 did not open both final transformer blocks: {paths}")
    return inventory


def preprocess_parity(model, classifier, images, rtol=1e-5, atol=1e-5):
    wrapped = np.asarray(model(images, training=False))
    direct = np.asarray(classifier(classifier.preprocessor(images), training=False))
    if wrapped.shape != (len(images), 5) or not np.isfinite(wrapped).all():
        raise RuntimeError("Invalid DeiT-Small logits")
    if not np.allclose(wrapped, direct, rtol=rtol, atol=atol):
        raise RuntimeError("Preset preprocessing parity failed; possible double preprocessing")
    return {"input": "RGB float32 0..255, bilinear 224x224", "preprocessor_passes": 1,
            "max_abs_logit_difference": float(np.max(np.abs(wrapped - direct)))}
