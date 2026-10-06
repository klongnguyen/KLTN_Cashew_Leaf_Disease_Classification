"""CPU-only tests: no TensorFlow, model training, or real Test prediction."""
import ast
import json
import io
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from real_holdout import evaluate_real_holdout, holdout_sha256, prepare_real_holdout
from real_holdout import save_real_holdout_seed

CLASSES = ["anthracnose", "healthy", "leaf_miner", "not_cashew_leaf", "red_rust"]
COUNTS = {"anthracnose": 1, "healthy": 1, "leaf_miner": 1, "red_rust": 1}
REPO_ROOT = Path(__file__).resolve().parents[2]


class HoldoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="test-cashew-holdout-006-")
        self.root = Path(self.temp.name)
        self.zip_path = self.root / "holdout.zip"
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            for index, name in enumerate(COUNTS):
                image_path = self.root / f"image_{index}.png"
                Image.new("RGB", (12, 12), (30 + 30 * index, 40, 50)).save(image_path)
                archive.write(image_path, f"RL_Cashew_holdout/{name}/image.png")
        self.references = []
        for index, split in enumerate(("train", "val", "test")):
            path = self.root / f"reference_{split}.png"
            Image.new("RGB", (12, 12), (200, index * 20, 200)).save(path)
            self.references.append({"split": split, "path": str(path)})
        self.expected_hash = holdout_sha256(self.zip_path)

    def tearDown(self):
        plt.close("all")
        self.temp.cleanup()

    def prepare(self, **kwargs):
        return prepare_real_holdout(self.zip_path, self.root / "extract", CLASSES,
            kwargs.get("expected_sha256", self.expected_hash), kwargs.get("expected_counts", COUNTS),
            kwargs.get("reference_rows", self.references), self.root / "results")

    def test_counts_mapping_and_no_holdout_training_overlap(self):
        manifest, audit = self.prepare()
        self.assertEqual(manifest["label"].tolist(), [0, 1, 2, 4])
        self.assertEqual(audit["unsupported_classes"], ["not_cashew_leaf"])
        self.assertEqual(audit["byte_overlap_count"], 0)
        self.assertEqual(audit["reference_images_checked"], 3)
        manifest_again, audit_again = self.prepare()
        self.assertEqual(audit, audit_again)
        pd.testing.assert_frame_equal(manifest, manifest_again)

    def test_archive_mismatch_and_wrong_counts_fail(self):
        with self.assertRaisesRegex(ValueError, "archive changed"):
            self.prepare(expected_sha256="0" * 64)
        with self.assertRaisesRegex(ValueError, "count"):
            self.prepare(expected_counts={**COUNTS, "healthy": 2})

    def test_byte_overlap_fails_before_evaluation(self):
        self.references[0]["path"] = str(self.root / "image_0.png")
        with self.assertRaisesRegex(ValueError, "overlaps V05"):
            self.prepare()

    def test_cached_image_changes_are_rejected(self):
        self.prepare()
        target = self.root / "extract/RL_Cashew_holdout/healthy/image.png"
        Image.new("RGB", (12, 12), (1, 2, 3)).save(target)
        with self.assertRaisesRegex(ValueError, "extraction changed"):
            self.prepare()

    def test_unsafe_zip_is_rejected(self):
        with zipfile.ZipFile(self.zip_path, "w") as archive:
            archive.writestr("../escape.txt", "not an instruction")
        with self.assertRaisesRegex(ValueError, "Unsafe ZIP"):
            self.prepare(expected_sha256=holdout_sha256(self.zip_path))
        self.assertFalse((self.root / "escape.txt").exists())

    def test_metrics_absent_class_and_readable_matrices(self):
        manifest, _ = self.prepare()
        probabilities = np.eye(5)[[0, 1, 3, 4]]  # leaf_miner wrongly predicted as absent not_cashew_leaf.
        captured = []
        original = plt.Figure.savefig

        def checked_save(figure, *args, **kwargs):
            axis = figure.axes[0]
            self.assertTrue(all(text.get_color() == "black" for text in axis.texts))
            self.assertTrue(np.allclose(axis.images[0].cmap(0.0)[:3], [1, 1, 1]))
            self.assertIn("N/A", [text.get_text() for text in axis.texts] if "normalized" in str(args[0]) else ["N/A"])
            captured.append(str(args[0]))
            return original(figure, *args, **kwargs)

        with patch.object(plt.Figure, "savefig", checked_save):
            metrics = save_real_holdout_seed(manifest, probabilities, CLASSES, self.root / "metrics")
        self.assertEqual(len(captured), 2)
        self.assertEqual(metrics["accuracy"], 0.75)
        self.assertEqual(metrics["macro_f1_present_classes"], 0.75)
        self.assertEqual(metrics["macro_class_count"], 4)
        per_class = pd.read_csv(self.root / "metrics/per_class_metrics.csv", index_col=0)
        self.assertTrue(np.isnan(per_class.loc["not_cashew_leaf", "f1-score"]))
        report = pd.read_csv(self.root / "metrics/classification_report.csv", index_col=0)
        self.assertEqual(report.loc["macro avg (present classes)", "f1-score"], 0.75)
        cm = pd.read_csv(self.root / "metrics/confusion_matrix.csv", index_col=0)
        self.assertEqual(cm.shape, (5, 5))
        self.assertEqual(cm.loc["leaf_miner", "not_cashew_leaf"], 1)
        with self.assertRaisesRegex(ValueError, "probabilities"):
            save_real_holdout_seed(manifest, np.ones((4, 5)), CLASSES, self.root / "invalid")

    def test_lock_validation_selection_logits_and_no_repeated_test(self):
        manifest, audit = self.prepare()
        seeds = [42, 123, 2026, 3407, 7777]
        validation = pd.DataFrame({"seed": seeds, "val_macro_f1": [.71334613461346137, .8, .6, .8, .5],
                                   "val_loss": [.3, .3, .3, .2, .3]})
        checkpoints = {}
        for seed in seeds:
            path = self.root / f"checkpoint_{seed}.txt"
            path.write_text(str(seed), encoding="utf-8")
            checkpoints[seed] = path
        calls = []

        class FakeModel:
            def predict(model, dataset, verbose=0):
                calls.append("predict")
                return np.eye(5)[manifest["label"].to_numpy()] * 10

        def run(frame=validation):
            return evaluate_real_holdout(manifest, audit, CLASSES, seeds, frame,
                checkpoints, lambda seed: FakeModel(), lambda rows: rows,
                lambda: None, self.root / "results", outputs_are_logits=True)

        results = run()
        self.assertEqual(len(calls), 5)
        self.assertTrue((results["accuracy"] == 1).all())
        lock = json.loads((self.root / "results/evaluation_lock.json").read_text())
        self.assertEqual(lock["deployment_seed"], 3407)  # chosen by Validation, not Holdout.
        self.assertEqual(len(lock["checkpoint_sha256"]), 5)
        run()
        self.assertEqual(len(calls), 5)
        run(pd.read_csv(io.StringIO(validation.to_csv(index=False))))
        self.assertEqual(len(calls), 5)
        with self.assertRaisesRegex(ValueError, "lock changed"):
            run(validation.assign(val_macro_f1=[1, .8, .6, .8, .5]))
        checkpoints[42].write_text("changed checkpoint", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "lock changed"):
            run()


class NotebookIntegrationTests(unittest.TestCase):
    def test_006_syntax_embedding_modes_and_locked_defaults(self):
        common_source = (REPO_ROOT / "Notebook/common/real_holdout.py").read_text(encoding="utf-8")
        notebook_paths = [
            "Notebook/DenseNet121/densenet121_5seed.ipynb",
            "Notebook/ResNet50/resnet50_5seed.ipynb",
            "Notebook/ViT/vit_scratch_5seed.ipynb",
            "Notebook/ViT/VIT_5seed.ipynb",
            "Notebook/ViT/deit_tiny_finetune_5seed.ipynb",
        ]
        deit_sources = []
        for relative in notebook_paths:
            notebook = json.loads((REPO_ROOT / relative).read_text(encoding="utf-8"))
            sources = []
            for index, cell in enumerate(notebook["cells"]):
                if cell["cell_type"] != "code":
                    continue
                source = "".join(cell["source"])
                stripped = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith(("!", "%")))
                ast.parse(stripped, filename=f"{relative}:cell{index}")
                sources.append(source)
            config = next(s for s in sources if 'NOTEBOOK_VERSION = "006"' in s)
            self.assertIn('RUN_FINAL_TEST = False', config)
            # Scratch 006 runs currently isolate training/validation; DeiT keeps
            # the optional holdout enabled behind the locked Final Test gate.
            expected_holdout = "True" if "FINETUNE" in config else "False"
            self.assertIn(f'RUN_REAL_HOLDOUT = {expected_holdout}', config)
            self.assertIn('-5SEEDS-006"', config)
            preflight = next(s for s in sources if "# --- Notebook-specific real holdout preflight" in s)
            self.assertTrue(preflight.startswith(common_source))
            first_fit_index = next(i for i, s in enumerate(sources) if "model.fit(" in s)
            self.assertLess(sources.index(preflight), first_fit_index)
            notebook_only = preflight.split("# --- Notebook-specific real holdout preflight", 1)[1]
            notebook_only = notebook_only.split("\n", 1)[1]
            # Locked preflight must not access the archive, references or model.
            exec(notebook_only, {"RUN_FINAL_TEST": False, "RUN_REAL_HOLDOUT": True})
            driver = next(s for s in sources if "def make_real_holdout_dataset" in s)
            exec(driver, {"RUN_FINAL_TEST": False, "RUN_REAL_HOLDOUT": True})
            self.assertNotIn(".shuffle(", driver)
            if "FINETUNE" in config:
                self.assertIn('deit_tiny_distilled_patch16_224_imagenet/1', config)
                self.assertIn("outputs_are_logits=True", driver)
                deit_sources.append(sources)
            else:
                self.assertIn('TRAINING_MODE = "Scratch"', config)
                self.assertIn('PRETRAINED_WEIGHTS = None', config)
                self.assertIn("outputs_are_logits=False", driver)
        self.assertEqual(deit_sources[0], deit_sources[1])
        python_source = (REPO_ROOT / "Notebook/ViT/deit_tiny_finetune_5seed.py").read_text(encoding="utf-8")
        ast.parse(python_source)
        self.assertIn(common_source, python_source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
