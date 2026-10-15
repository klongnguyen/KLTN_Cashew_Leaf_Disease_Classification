# EXP-009 notebook verification

Notebook: `Notebook/DenseNet121/densenet121_5seed.ipynb`.

## Checks completed on CPU

- All 26 Python code cells (excluding the Colab shell command cell) compile.
- DenseNet121 has 7,042,629 parameters, matching the supplied one-seed notebook. Its BatchNorm momentum is the Keras default 0.99.
- Model preprocessing and `keras.applications.densenet.preprocess_input` differ by at most `7.15e-7` on a sampled RGB batch.
- One short fit (20 Train images, 10 Validation images, seed 42, 1 epoch) completed 2 optimizer updates, saved the minimum `val_loss` checkpoint, reloaded it, and reproduced the selected Validation loss (`1.6042039394`). These tiny-sample accuracy values do not measure dataset/model quality.
- A separate 9-image Test subset exercised prediction alignment, `real_` filename subgroup output, model packaging, and code export. The saved deployment, feature extractor, and embedding models reload. Spatial mean of the feature map and exported embedding differ by at most `2.98e-8`.
- The audit was executed on a 15-image synthetic split containing the **actual** known near-duplicate pair from the ZIP. It flagged the Train/Test pair with pHash distance 0 and grayscale 64×64 thumbnail MAD 0.0.
- The final class order was subsequently aligned with the supplied notebook (`anthracnose`, `healthy`, `leaf_miner`, `red_rust`, `not_cashew_leaf`); all code cells compiled again after this mapping change.

## Scope and limitations

The smoke tests used a handful of images and CPU. No full five-seed training has run. The Colab G4 runtime is not exposed to this Codex session (`apps=[]`, `browsers=[]`), so the provided Colab URL alone cannot execute the edited local notebook. The complete run should use `CashewData_Split (7).zip` with SHA-256 `6983b9ab8339489f18edb00f773b319ee6c62c2a3076c199b078ef52b464a3bf`.

The internally tested model/checkpoints were random-start diagnostics and were removed after verification. Historical one-seed Test results from the supplied notebook are 93.01% overall accuracy and 19/28 correct among `real_` filenames; they are not results of EXP-009.
