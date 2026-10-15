
from tensorflow import keras
model = keras.models.load_model("model_package/deployment_model.keras", compile=False)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
# model.fit(train_ds, validation_data=val_ds, ...)
