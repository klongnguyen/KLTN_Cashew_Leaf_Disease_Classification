import gc
import hashlib
import json
import random
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parents[2]
notebook = json.loads((ROOT / 'Notebook/ResNet50/resnet50_5seed_008.ipynb').read_text(encoding='utf-8'))
cell = lambda index: ''.join(notebook['cells'][index]['source'])
IMG_SIZE = 224
NUM_CLASSES = 5
HEAD_DROPOUT = 0.30
set_global_seed = lambda seed: (random.seed(seed), np.random.seed(seed), keras.utils.set_random_seed(seed))
exec(compile(cell(12), 'model_builder_cell', 'exec'))

inputs = keras.Input(shape=(1, 1, 3))
preprocess = keras.Model(inputs, resnet_preprocessing(inputs))
rgb = np.array([[[[10.0, 20.0, 30.0]]]], dtype=np.float32)
actual = preprocess(rgb, training=False).numpy()
expected = keras.applications.resnet50.preprocess_input(rgb.copy())
assert np.allclose(actual, expected, atol=1e-5), (actual, expected)
print('PASS: embedded ResNet50 preprocessing equals Keras preprocess_input')

model = build_resnet50_model(42)
assert model.output_shape == (None, 5)
assert model.count_params() == 23597969, model.count_params()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4),
              loss=keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
print('PASS: ResNet50 model builds and compiles with expected parameter count')
del model, preprocess
keras.backend.clear_session()
gc.collect()

EXPERIMENT_ID = 'EXP-RESNET50-SCRATCH-5SEEDS-008'
for use_file in (True, False):
    with tempfile.TemporaryDirectory(prefix='resnet50_008_archive_') as temporary:
        PROJECT_ROOT = ROOT if use_file else Path(temporary)
        NOTEBOOK_SOURCE_PATH = ROOT / 'Notebook/ResNet50/resnet50_5seed_008.ipynb' if use_file else None
        RESULTS_ROOT = Path(temporary) / 'results'
        OUT_DIR = RESULTS_ROOT / EXPERIMENT_ID
        OUT_DIR.mkdir(parents=True)
        (OUT_DIR / 'smoke.txt').write_text('archive smoke', encoding='utf-8')
        exec(compile(cell(35), 'archive_cell', 'exec'))
        for suffix in ('FULL', 'ANALYSIS'):
            archive = RESULTS_ROOT / f'{EXPERIMENT_ID}_{suffix}.zip'
            with zipfile.ZipFile(archive) as package:
                expected_name = f'{EXPERIMENT_ID}/notebook/resnet50_5seed_008.ipynb'
                assert expected_name in package.namelist()
                packaged = json.loads(package.read(expected_name))
                assert 'NOTEBOOK_VERSION = "008"' in ''.join(packaged['cells'][4]['source'])
        print('PASS: both ZIPs include notebook;', 'file source' if use_file else 'embedded fallback')
