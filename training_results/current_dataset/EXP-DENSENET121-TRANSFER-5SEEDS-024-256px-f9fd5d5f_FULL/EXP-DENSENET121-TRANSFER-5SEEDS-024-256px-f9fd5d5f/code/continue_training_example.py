from tensorflow import keras
    model = keras.models.load_model("model_package/deployment_model.keras", compile=False)
    backbone = model.get_layer("densenet121")
    backbone.trainable = True
    for layer in backbone.layers:
        layer.trainable = layer.name.startswith("conv5_") and not isinstance(layer, keras.layers.BatchNormalization)
    model.compile(optimizer=keras.optimizers.AdamW(learning_rate=1e-5, weight_decay=1e-4),
                  loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    # Continue only in a NEW experiment ID. Do not overwrite EXP-024 checkpoints.
    