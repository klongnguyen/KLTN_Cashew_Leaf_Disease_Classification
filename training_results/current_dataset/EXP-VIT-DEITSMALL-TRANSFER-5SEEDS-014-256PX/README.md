# EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-014-256PX

**Status: NOT RUN / READY FOR GPU.** No Accuracy, F1, checkpoint, or Validation selection is claimed here.

Notebook: `Notebook/ViT/vit_deit_small_transfer_5seeds_014_256px.ipynb`. It runs on Google Colab GPU with the frozen V11 ZIP and writes measured results under this experiment ID on Drive. The local `experiment_config.json` describes the intended protocol; the notebook records exact runtime versions, source checkpoint revision/SHA256, Gate 0/1 evidence, five-seed outputs, selection, plots, and export only after execution.

The experiment uses the two-head Facebook DeiT-Small distilled ImageNet checkpoint with Hugging Face Transformers. This is a documented protocol revision from EXP-011's KerasHub CLS-head path. The 256×256 gate measures 256 patch tokens, CLS and DIST tokens, interpolation, gradients, and save/reload parity before training. Test is used only for ZIP integrity audit, and Holdout is never opened.

Open items: Colab Drive path, V11 ZIP/manifest hashes, pretrained artifact hash, GPU smoke, and full training must be checked in the actual runtime. pHash candidates require manual review and are not automatically labeled leakage.
