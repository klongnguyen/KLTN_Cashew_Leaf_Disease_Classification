
from tensorflow import keras
from tensorflow.keras import layers


def build_densenet121_model(seed, img_size=224, num_classes=5, dropout=0.30):
    keras.utils.set_random_seed(seed)
    inputs = keras.Input(shape=(img_size, img_size, 3), name="raw_rgb")
    x = layers.Rescaling(1 / 255.0, name="densenet_rescale")(inputs)
    x = layers.Normalization(mean=[0.485, 0.456, 0.406],
                             variance=[0.229**2, 0.224**2, 0.225**2],
                             axis=-1, name="densenet_preprocess")(x)
    x = keras.Sequential([
        layers.RandomFlip("horizontal", seed=seed),
        layers.RandomRotation(0.04, seed=seed),
        layers.RandomZoom(0.05, seed=seed),
    ], name=f"train_only_augmentation_seed_{seed}")(x)
    backbone = keras.applications.DenseNet121(
        include_top=False, weights=None, input_shape=(img_size, img_size, 3), pooling="avg")
    x = backbone(x)
    x = layers.Dropout(dropout, seed=seed)(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32",
                           name="disease_output")(x)
    return keras.Model(inputs, outputs, name=f"DenseNet121_Scratch_seed_{seed}")
