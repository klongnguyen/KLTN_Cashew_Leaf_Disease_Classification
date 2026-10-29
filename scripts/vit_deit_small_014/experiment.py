"""Colab implementation for EXP-014. No Test/Holdout inference paths exist here."""

import copy
import hashlib
import io
import json
import math
import os
import random
import shutil
import time
import zipfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode
from transformers import DeiTForImageClassificationWithTeacher
from huggingface_hub import snapshot_download, model_info
from safetensors.torch import save_file, load_file

EXPERIMENT_ID = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-014-256PX"
CLASSES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
SEEDS = [42, 123, 2026, 3407, 7777]
ZIP_SHA = "5d60295d26068f5d9f6427dc730a0f662c56c28ad9ee95f8b594faf546861f49"
MANIFEST_SHA = "91d8279239eaad2007b652f1859667359d0cf01f94af917dd27b0a934a7b7e75"
SOURCE_MODEL = "facebook/deit-small-distilled-patch16-224"
COUNTS = {"train": 4916, "val": 1447, "test": 685}
REAL_COUNTS = {"train": 238, "val": 54, "test": 28}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def sha(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def phash(raw):
    with Image.open(io.BytesIO(raw)) as image:
        pixels = np.asarray(ImageOps.exif_transpose(image).convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    values = dctn(pixels, type=2, norm="ortho")[:8, :8]
    bits = (values > np.median(values.ravel()[1:])).ravel()
    bits[0] = False
    return sum(int(bit) << i for i, bit in enumerate(bits))


def audit_archive(archive_path, out):
    """Gate 0. Test bytes are read for CRC/hash only and never extracted."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if sha(archive_path) != ZIP_SHA:
        raise ValueError("V11 archive SHA256 mismatch; stop before extraction")
    rows, hashes, members = [], [], defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        broken = archive.testzip()
        if broken:
            raise ValueError(f"ZIP CRC failed: {broken}")
        for item in archive.infolist():
            path = PurePosixPath(item.filename)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe ZIP path: {item.filename}")
            if item.is_dir():
                continue
            if len(path.parts) != 4 or path.parts[0] != "CashewData_Split":
                raise ValueError(f"Unexpected ZIP member: {item.filename}")
            _, split, klass, name = path.parts
            if split not in COUNTS or klass not in CLASSES or not name:
                raise ValueError(f"Unexpected split/class: {item.filename}")
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                raise ValueError(f"Unexpected image extension: {item.filename}")
            members[(split, klass)].append(item.filename)
        for split in COUNTS:
            for label, klass in enumerate(CLASSES):
                for name in sorted(members[(split, klass)]):
                    raw = archive.read(name)
                    if split != "test":
                        with Image.open(io.BytesIO(raw)) as image:
                            image.verify()
                        with Image.open(io.BytesIO(raw)) as image:
                            ImageOps.exif_transpose(image).convert("RGB").load()
                    relative = PurePosixPath(name).relative_to("CashewData_Split").as_posix()
                    rows.append(dict(split=split, **{"class": klass}, label=label,
                                     relative_path=relative,
                                     sha256=hashlib.sha256(raw).hexdigest(),
                                     is_real_filename=PurePosixPath(name).name.casefold().startswith("real_")))
                    hashes.append(phash(raw) if split != "test" else None)
    frame = pd.DataFrame(rows)
    if frame.groupby("split").size().to_dict() != COUNTS or len(frame) != 7048:
        raise ValueError("V11 split counts changed")
    real = frame.groupby("split").is_real_filename.sum().astype(int).to_dict()
    if real != REAL_COUNTS:
        raise ValueError(f"V11 real_ filename counts changed: {real}")
    if frame.relative_path.duplicated().any():
        raise ValueError("Duplicate relative paths")
    # Same record order and JSON serialization as the original V11 audit.
    records = frame[["relative_path", "label", "sha256"]].to_dict("records")
    manifest_hash = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest_hash != MANIFEST_SHA:
        raise ValueError(f"V11 manifest mismatch: {manifest_hash}")
    exact = frame[frame.sha256.duplicated(keep=False)].copy()
    cross = [g for _, g in exact.groupby("sha256") if g.split.nunique() > 1]
    if cross:
        raise ValueError(f"Exact cross-split duplicates: {len(cross)}")
    buckets = defaultdict(list)
    for i, value in enumerate(hashes):
        if value is None:
            continue
        for band in range(8):
            buckets[(band, (value >> (band * 8)) & 255)].append(i)
    pairs = set()
    splits = frame.split.to_numpy()
    for bucket in buckets.values():
        for n, i in enumerate(bucket):
            for j in bucket[n + 1:]:
                if splits[i] != splits[j] and (hashes[i] ^ hashes[j]).bit_count() <= 6:
                    pairs.add((min(i, j), max(i, j)))
    candidates = pd.DataFrame([dict(path_a=frame.relative_path.iloc[i], path_b=frame.relative_path.iloc[j],
                                    split_a=splits[i], split_b=splits[j],
                                    phash_distance=(hashes[i] ^ hashes[j]).bit_count(),
                                    status="MANUAL_REVIEW_REQUIRED") for i, j in sorted(pairs)],
                              columns=["path_a", "path_b", "split_a", "split_b", "phash_distance", "status"])
    frame.to_csv(out / "dataset_manifest.csv", index=False)
    exact.to_csv(out / "exact_duplicates.csv", index=False)
    candidates.to_csv(out / "phash_cross_split_candidates.csv", index=False)
    summary = dict(status="PASS_WITH_PHASH_CANDIDATES_PENDING_REVIEW", dataset_sha256=ZIP_SHA,
                   manifest_sha256=manifest_hash, counts=COUNTS, real_prefix_counts=real,
                   exact_cross_split_duplicates=0, phash_cross_split_candidates=len(candidates),
                   group_leakage_status="NOT_VERIFIED", test_use="ZIP integrity/hash only; no decode or pHash")
    write_json(out / "dataset_audit_summary.json", summary)
    return frame, summary


def extract_train_val(archive_path, frame, target):
    target = Path(target)
    with zipfile.ZipFile(archive_path) as archive:
        for relative in frame.loc[frame.split.isin(["train", "val"]), "relative_path"]:
            source = "CashewData_Split/" + relative
            dest = target / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.is_file() or sha(dest) != frame.loc[frame.relative_path.eq(relative), "sha256"].iloc[0]:
                with archive.open(source) as src, open(dest, "wb") as dst:
                    shutil.copyfileobj(src, dst)
    return target


def transform(train=False):
    sequence = [T.Resize((256, 256), interpolation=InterpolationMode.BILINEAR)]
    if train:
        sequence += [T.RandomHorizontalFlip(),
                     T.RandomRotation(28.8, interpolation=InterpolationMode.BILINEAR),
                     T.RandomAffine(0, scale=(0.9, 1.1), interpolation=InterpolationMode.BILINEAR),
                     T.ColorJitter(brightness=0.08, contrast=0.10)]
    sequence += [T.ToTensor(), T.Normalize(MEAN, STD)]
    return T.Compose(sequence)


class ImageRows(Dataset):
    def __init__(self, frame, root, train=False):
        self.rows = frame.reset_index(drop=True)
        self.root = Path(root)
        self.process = transform(train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows.iloc[index]
        with Image.open(self.root / row.relative_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.process(image)
        return tensor, int(row.label), index


def loaders(frame, root, seed):
    gen = torch.Generator().manual_seed(seed)
    def worker_seed(worker):
        value = torch.initial_seed() % 2**32
        random.seed(value)
        np.random.seed(value)
    train = ImageRows(frame[frame.split.eq("train")], root, train=True)
    val = ImageRows(frame[frame.split.eq("val")], root)
    return (DataLoader(train, batch_size=16, shuffle=True, generator=gen,
                       num_workers=2, worker_init_fn=worker_seed, pin_memory=True),
            DataLoader(val, batch_size=16, shuffle=False, num_workers=2, pin_memory=True), val.rows)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def source_snapshot(out):
    info = model_info(SOURCE_MODEL)
    snapshot = Path(snapshot_download(repo_id=SOURCE_MODEL, revision=info.sha,
                                      allow_patterns=["*.json", "*.safetensors", "*.bin"]))
    weights = [p for p in snapshot.iterdir() if p.name in ("model.safetensors", "pytorch_model.bin")]
    if not weights:
        raise FileNotFoundError("Pretrained weight artifact unavailable")
    files = {p.name: sha(p) for p in weights}
    processor_file = snapshot / "preprocessor_config.json"
    if not processor_file.is_file():
        raise FileNotFoundError("Pretrained preprocessor config unavailable")
    processor = json.loads(processor_file.read_text(encoding="utf-8"))
    if not np.allclose(processor.get("image_mean", []), MEAN) or not np.allclose(processor.get("image_std", []), STD):
        raise RuntimeError(f"Pretrained normalization differs from notebook: {processor}")
    evidence = dict(model=SOURCE_MODEL, revision=info.sha, artifact_sha256=files,
                    preprocessor_config_sha256=sha(processor_file),
                    source_protocol="Hugging Face distilled two-head checkpoint; differs from EXP-011 Keras preset")
    evidence_path = Path(out) / "pretrained_source.json"
    if evidence_path.is_file() and json.loads(evidence_path.read_text(encoding="utf-8")) != evidence:
        raise RuntimeError("Pretrained source/revision changed since this experiment began")
    write_json(evidence_path, evidence)
    return snapshot, evidence


def new_model(snapshot, seed):
    set_seed(seed)
    model, loading = DeiTForImageClassificationWithTeacher.from_pretrained(
        str(snapshot), num_labels=5, ignore_mismatched_sizes=True,
        output_loading_info=True, use_safetensors=(snapshot / "model.safetensors").is_file())
    if loading.get("missing_keys") or loading.get("unexpected_keys"):
        raise RuntimeError(f"Unexpected pretrained loading keys: {loading}")
    mismatch = [x[0] for x in loading.get("mismatched_keys", [])]
    expected = {"cls_classifier.weight", "cls_classifier.bias",
                "distillation_classifier.weight", "distillation_classifier.bias"}
    if set(mismatch) != expected:
        raise RuntimeError(f"Unexpected pretrained mismatch: {mismatch}")
    cfg = model.config
    if (cfg.image_size, cfg.patch_size, cfg.hidden_size, cfg.num_hidden_layers) != (224, 16, 384, 12):
        raise RuntimeError("Pretrained architecture changed")
    if tuple(model.deit.embeddings.position_embeddings.shape) != (1, 198, 384):
        raise RuntimeError("Expected CLS + DIST + 196 positions")
    source_weights = (load_file(snapshot / "model.safetensors") if (snapshot / "model.safetensors").is_file()
                      else torch.load(snapshot / "pytorch_model.bin", map_location="cpu", weights_only=True))
    current = model.state_dict()
    for name in ("deit.embeddings.patch_embeddings.projection.weight",
                 "deit.embeddings.position_embeddings",
                 "deit.encoder.layer.11.attention.attention.query.weight"):
        if name not in source_weights or not torch.equal(source_weights[name], current[name].cpu()):
            raise RuntimeError(f"Pretrained encoder tensor mismatch: {name}")
    del source_weights
    return model


def set_stage(model, stage):
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.cls_classifier.parameters():
        p.requires_grad_(True)
    for p in model.distillation_classifier.parameters():
        p.requires_grad_(True)
    if stage == "stage2":
        for block in model.deit.encoder.layer[-2:]:
            for p in block.parameters():
                p.requires_grad_(True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if stage == "stage1" and any("deit." in n for n in names):
        raise RuntimeError("Stage 1 backbone is trainable")
    if stage == "stage2" and (not any("layer.10." in n for n in names) or
                               not any("layer.11." in n for n in names) or
                               any("layer." in n and not ("layer.10." in n or "layer.11." in n) for n in names)):
        raise RuntimeError("Stage 2 block inventory invalid")
    return names


def logits(model, images):
    return model(pixel_values=images, interpolate_pos_encoding=True).logits


def smoke(snapshot, train_loader, out, device):
    """Gate 1 uses real Train images and never updates the model used for training."""
    model = new_model(snapshot, 42).to(device)
    images, labels, _ = next(iter(train_loader))
    images, labels = images[:2].to(device), labels[:2].to(device)
    evidence = dict(status="FAIL", backbone="DeiT-Small distilled",
                    pretrained_source=SOURCE_MODEL,
                    pretrained_artifact_sha256={p.name: sha(p) for p in snapshot.iterdir() if p.name in ("model.safetensors", "pytorch_model.bin")},
                    requested_image_shape=[256, 256, 3],
                    effective_model_image_shape=list(images.shape[-2:]) + [3], patch_size=16,
                    preprocessor_passes=1, position_method="Transformers interpolate_pos_encoding=True")
    try:
        assert images.shape == (2, 3, 256, 256)
        patch = model.deit.embeddings.patch_embeddings.projection(images)
        assert patch.shape[-2:] == (16, 16)
        assert model.deit.embeddings.position_embeddings.shape[1] == 198
        embedded = model.deit.embeddings(images, interpolate_pos_encoding=True)
        assert embedded.shape == (2, 258, 384)
        source_pos = model.deit.embeddings.position_embeddings.detach()
        new_pos = model.deit.embeddings.interpolate_pos_encoding(embedded, 256, 256)
        assert new_pos.shape == (1, 258, 384)
        assert torch.equal(new_pos[:, :2], source_pos[:, :2])
        evidence.update(patch_grid=[16, 16], num_patch_tokens=256,
                        sequence_tokens=258, special_tokens_checked=True,
                        position_embedding_interpolation_verified=True,
                        pretrained_position_shape=list(source_pos.shape), interpolated_position_shape=list(new_pos.shape))
        for stage in ("stage1", "stage2"):
            names = set_stage(model, stage)
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
            before = {n: p.detach().clone() for n, p in model.named_parameters()}
            optimizer.zero_grad()
            result = logits(model, images)
            loss = F.cross_entropy(result, labels)
            assert result.shape == (2, 5) and torch.isfinite(loss)
            loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for n, p in model.named_parameters() if n in names)
            optimizer.step()
            changed = {n for n, p in model.named_parameters() if not torch.equal(before[n], p.detach())}
            assert changed and changed.issubset(set(names))
            if stage == "stage2":
                assert any("layer.10." in n for n in changed) and any("layer.11." in n for n in changed)
            evidence[f"{stage}_gradient"] = "PASS"
            evidence[f"{stage}_changed_weights"] = len(changed)
            evidence["stage1_head_only_gradient" if stage == "stage1" else "stage2_last_two_blocks_gradient"] = "PASS"
        model.eval()
        with torch.no_grad():
            before_logits = logits(model, images).cpu()
        path = Path(out) / "gate1_smoke.safetensors"
        save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, path)
        restored = new_model(snapshot, 42)
        restored.load_state_dict(load_file(path))
        restored.to(device).eval()
        with torch.no_grad():
            after_logits = logits(restored, images).cpu()
        assert torch.allclose(before_logits, after_logits, rtol=1e-5, atol=1e-5)
        evidence.update(logit_shape=[2, 5], save_reload_logit_parity="PASS",
                        save_reload_max_abs_diff=float((before_logits - after_logits).abs().max()),
                        status="PASS", checkpoint_sha256=sha(path))
    finally:
        write_json(Path(out) / "gate1_smoke.json", evidence)
    return evidence


def score(y, probabilities):
    predicted = probabilities.argmax(1)
    cm = confusion_matrix(y, predicted, labels=range(5))
    precision, recall, f1, support = precision_recall_fscore_support(y, predicted, labels=range(5), zero_division=0)
    macro_p, macro_r, macro_f, _ = precision_recall_fscore_support(y, predicted, average="macro", zero_division=0)
    weighted_f = precision_recall_fscore_support(y, predicted, average="weighted", zero_division=0)[2]
    result = dict(accuracy=float(accuracy_score(y, predicted)), balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
                macro_precision=float(macro_p), macro_recall=float(macro_r), macro_f1=float(macro_f), weighted_f1=float(weighted_f),
                healthy_precision=float(precision[1]), healthy_recall=float(recall[1]), healthy_f1=float(f1[1]),
                disease_to_healthy_count=int(cm[[0, 2, 3], 1].sum()),
                disease_to_healthy_rate=float(cm[[0, 2, 3], 1].sum() / cm[[0, 2, 3], :].sum()),
                ood_to_healthy_count=int(cm[4, 1]),
                per_class=[dict(class_name=CLASSES[i], support=int(support[i]), tp=int(cm[i, i]),
                                fp=int(cm[:, i].sum() - cm[i, i]), fn=int(cm[i].sum() - cm[i, i]),
                                precision=float(precision[i]), recall=float(recall[i]), f1=float(f1[i])) for i in range(5)])
    result.update({f"healthy_to_{CLASSES[i]}": int(cm[1, i]) for i in (0, 2, 3, 4)})
    return result


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    all_logits, all_y = [], []
    for images, y, _ in loader:
        out = logits(model, images.to(device))
        all_logits.append(out.cpu())
        all_y.append(y)
    y = torch.cat(all_y)
    values = torch.cat(all_logits)
    loss = float(F.cross_entropy(values, y))
    probability = torch.softmax(values, dim=1).numpy()
    return y.numpy(), probability, loss


def checkpoint_save(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".part.safetensors")
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, temp)
    os.replace(temp, path)
    return sha(path)


def stage_train(model, train_loader, val_loader, val_frame, seed, stage, seed_dir, device, config_sha):
    names = set_stage(model, stage)
    lr = 1e-4 if stage == "stage1" else 1e-5
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    best_loss, stale, records = float("inf"), 0, []
    best_path = Path(seed_dir) / "checkpoints" / f"{stage}_best.safetensors"
    started = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(30):
        if stage == "stage2":
            rate = lr * (epoch + 1) / 3 if epoch < 3 else 1e-6 + (lr - 1e-6) * (1 + math.cos(math.pi * (epoch - 3) / 27)) / 2
            for group in optimizer.param_groups:
                group["lr"] = rate
        else:
            rate = lr
        model.train()
        losses, correct, count = [], 0, 0
        for images, labels, _ in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            result = logits(model, images)
            loss = F.cross_entropy(result, labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss")
            loss.backward()
            if stage == "stage2":
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            optimizer.step()
            losses.append(float(loss.detach()) * len(labels))
            correct += int((result.argmax(1) == labels).sum())
            count += len(labels)
        y, probs, val_loss = validate(model, val_loader, device)
        metrics = score(y, probs)
        row = dict(epoch=epoch + 1, train_loss=sum(losses) / count, train_accuracy=correct / count,
                   val_loss=val_loss, val_accuracy=metrics["accuracy"], val_macro_f1=metrics["macro_f1"],
                   val_disease_to_healthy_count=metrics["disease_to_healthy_count"], lr=rate)
        records.append(row)
        print(f"SEED {seed} — {stage.upper()} | epoch {epoch+1:02d}/30 | train_loss {row['train_loss']:.5f} | val_loss {val_loss:.5f} | val_macro_f1 {metrics['macro_f1']:.5f} | lr {rate:.2e}", flush=True)
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            checkpoint_save(model, best_path)
        else:
            stale += 1
        pd.DataFrame(records).to_csv(Path(seed_dir) / f"history_{stage}.csv", index=False)
        if stale >= 7:
            break
    model.load_state_dict(load_file(best_path))
    model.to(device)
    y, probs, val_loss = validate(model, val_loader, device)
    metrics = score(y, probs)
    metrics.update(seed=seed, stage=stage, val_loss=val_loss, best_epoch=int(pd.DataFrame(records).val_loss.idxmin() + 1),
                   epochs_ran=len(records), train_seconds=time.perf_counter() - started,
                   checkpoint_sha256=sha(best_path), checkpoint_bytes=best_path.stat().st_size,
                   total_parameters=sum(p.numel() for p in model.parameters()),
                   trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                   peak_gpu_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None)
    write_json(Path(seed_dir) / f"metrics_{stage}.json", metrics)
    save_validation(seed_dir, stage, seed, val_frame, y, probs, metrics)
    write_json(Path(seed_dir) / f"{stage}_complete.json", dict(seed=seed, stage=stage, config_sha256=config_sha,
               dataset_sha256=ZIP_SHA, checkpoint_sha256=metrics["checkpoint_sha256"],
               metrics_sha256=sha(Path(seed_dir) / f"metrics_{stage}.json")))
    return metrics


def save_validation(seed_dir, stage, seed, frame, y, probs, metrics):
    pred = probs.argmax(1)
    assert np.allclose(probs.sum(1), 1, atol=1e-5)
    rows = []
    for i, row in frame.reset_index(drop=True).iterrows():
        item = dict(experiment_id=EXPERIMENT_ID, architecture="DeiT-Small distilled two-head",
                    training_mode="ImageNet supervised transfer", dataset_version="Cashew_dataV11",
                    dataset_sha256=ZIP_SHA, split="val", seed=seed, stage=stage,
                    image_id=row.relative_path, relative_path=row.relative_path,
                    true_label=CLASSES[int(y[i])], predicted_label=CLASSES[int(pred[i])],
                    true_class_idx=int(y[i]), predicted_class_idx=int(pred[i]),
                    correct=bool(pred[i] == y[i]), confidence=float(probs[i, pred[i]]),
                    is_real_filename=bool(row.is_real_filename))
        item.update({f"prob_{name}": float(probs[i, j]) for j, name in enumerate(CLASSES)})
        rows.append(item)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(Path(seed_dir) / f"predictions_val_{stage}.csv", index=False)
    cm = confusion_matrix(y, pred, labels=range(5))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(Path(seed_dir) / f"confusion_matrix_val_{stage}.csv")
    normalized = np.divide(cm, cm.sum(axis=1, keepdims=True), out=np.zeros_like(cm, dtype=float), where=cm.sum(axis=1, keepdims=True) > 0)
    pd.DataFrame(normalized, index=CLASSES, columns=CLASSES).to_csv(Path(seed_dir) / f"confusion_matrix_normalized_val_{stage}.csv")
    pd.DataFrame(metrics["per_class"]).to_csv(Path(seed_dir) / f"classification_report_val_{stage}.csv", index=False)
    group = predictions.groupby(["is_real_filename", "true_label"]).agg(n=("correct", "size"), correct=("correct", "sum")).reset_index()
    group.to_csv(Path(seed_dir) / f"real_prefix_val_{stage}.csv", index=False)
    predictions.loc[~predictions.correct].sort_values("confidence", ascending=False).to_csv(
        Path(seed_dir) / f"errors_val_{stage}.csv", index=False)


def run_seed(seed, snapshot, frame, root, out, device, config_sha):
    seed_dir = Path(out) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, val_frame = loaders(frame, root, seed)
    marker1 = seed_dir / "stage1_complete.json"
    marker2 = seed_dir / "stage2_complete.json"
    for marker, stage in ((marker1, "stage1"), (marker2, "stage2")):
        if marker.is_file():
            data = json.loads(marker.read_text(encoding="utf-8"))
            checkpoint = seed_dir / "checkpoints" / f"{stage}_best.safetensors"
            if data["config_sha256"] != config_sha or data["dataset_sha256"] != ZIP_SHA or data["checkpoint_sha256"] != sha(checkpoint):
                raise RuntimeError(f"Resume marker mismatch: {marker}")
    if marker2.is_file() and not marker1.is_file():
        raise RuntimeError("Stage 2 exists without Stage 1")
    if marker1.is_file() and marker2.is_file():
        print(f"Seed {seed}: verified complete; skip")
        return
    if not marker1.is_file() and list(seed_dir.iterdir()):
        backup = Path(out) / "_incomplete_backups" / f"seed_{seed}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
        backup.parent.mkdir(exist_ok=True)
        shutil.move(str(seed_dir), str(backup))
        seed_dir.mkdir()
        print("Saved incomplete seed and restarted from pretrained:", backup)
    elif marker1.is_file() and not marker2.is_file():
        partial = [p for p in seed_dir.iterdir() if p.name.endswith("stage2.csv") or p.name.endswith("stage2.json")]
        partial += list((seed_dir / "checkpoints").glob("stage2*.safetensors"))
        if partial:
            backup = Path(out) / "_incomplete_backups" / f"seed_{seed}_stage2_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}"
            backup.mkdir(parents=True, exist_ok=False)
            for path in partial:
                shutil.move(str(path), str(backup / path.name))
            print("Saved incomplete Stage 2 and restarted from same-seed Stage 1 best:", backup)
    model = new_model(snapshot, seed).to(device)
    if not marker1.is_file():
        stage_train(model, train_loader, val_loader, val_frame, seed, "stage1", seed_dir, device, config_sha)
    else:
        model.load_state_dict(load_file(seed_dir / "checkpoints/stage1_best.safetensors"))
    if not marker2.is_file():
        stage_train(model, train_loader, val_loader, val_frame, seed, "stage2", seed_dir, device, config_sha)
    write_json(seed_dir / "seed_complete.json", dict(seed=seed, config_sha256=config_sha, dataset_sha256=ZIP_SHA,
               stage1_sha256=sha(seed_dir / "checkpoints/stage1_best.safetensors"),
               stage2_sha256=sha(seed_dir / "checkpoints/stage2_best.safetensors")))
    del model
    torch.cuda.empty_cache()


def aggregate(out):
    out = Path(out)
    rows = []
    for seed in SEEDS:
        directory = out / f"seed_{seed}"
        if not (directory / "seed_complete.json").is_file():
            raise RuntimeError(f"Five seed aggregation blocked: seed {seed} missing")
        for stage in ("stage1", "stage2"):
            row = json.loads((directory / f"metrics_{stage}.json").read_text(encoding="utf-8"))
            row.pop("per_class")
            rows.append(row)
    full = pd.DataFrame(rows)
    selected = full.sort_values(["seed", "val_loss", "macro_f1", "stage"], ascending=[True, True, False, True]).groupby("seed").head(1)
    # Explicit stage tie order: stage1 precedes stage2 alphabetically.
    selected = selected.sort_values("seed")
    aggregate_dir = out / "aggregate"
    aggregate_dir.mkdir(exist_ok=True)
    selected.to_csv(aggregate_dir / "validation_5seeds.csv", index=False)
    per_class, real_rows = [], []
    for _, row in selected.iterrows():
        seed, stage = int(row.seed), row.stage
        per_class.append(pd.read_csv(out / f"seed_{seed}" / f"classification_report_val_{stage}.csv").assign(seed=seed, stage=stage))
        real_rows.append(pd.read_csv(out / f"seed_{seed}" / f"real_prefix_val_{stage}.csv").assign(seed=seed, stage=stage))
    pd.concat(per_class, ignore_index=True).to_csv(aggregate_dir / "per_class_validation.csv", index=False)
    pd.concat(real_rows, ignore_index=True).to_csv(aggregate_dir / "real_prefix_validation.csv", index=False)
    selected[["seed", "stage", "disease_to_healthy_count", "disease_to_healthy_rate", "ood_to_healthy_count"]].to_csv(
        aggregate_dir / "disease_to_healthy_by_seed.csv", index=False)
    numeric = selected.select_dtypes(include="number").drop(columns=["seed"])
    summary = pd.DataFrame({"metric": numeric.columns, "mean": numeric.mean().to_numpy(), "sample_sd": numeric.std(ddof=1).to_numpy()})
    summary.to_csv(aggregate_dir / "validation_mean_std.csv", index=False)
    winner = selected.sort_values(["val_loss", "macro_f1", "seed"], ascending=[True, False, True]).iloc[0]
    lock = dict(experiment_id=EXPERIMENT_ID, selection_basis="Validation V11 only", seed=int(winner.seed),
                stage=winner.stage, val_loss=float(winner.val_loss), macro_f1=float(winner.macro_f1),
                checkpoint_sha256=winner.checkpoint_sha256, dataset_sha256=ZIP_SHA,
                created_at_utc=datetime.now(timezone.utc).isoformat(), test_status="NOT_RUN", holdout_status="NOT_RUN")
    write_json(out / "selection" / "validation_selection_lock.json", lock)
    return selected, summary, lock


def figures(out, selected):
    out = Path(out)
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(5, 2, figsize=(13, 20))
    for i, seed in enumerate(SEEDS):
        for stage in ("stage1", "stage2"):
            hist = pd.read_csv(out / f"seed_{seed}" / f"history_{stage}.csv")
            axes[i, 0].plot(hist.epoch, hist.val_loss, label=f"{stage} val")
            axes[i, 0].plot(hist.epoch, hist.train_loss, linestyle="--", label=f"{stage} train")
            axes[i, 1].plot(hist.epoch, hist.val_macro_f1, label=stage)
        axes[i, 0].set(ylabel=f"Seed {seed} · loss", xlabel="Epoch")
        axes[i, 1].set(ylabel="Validation Macro F1", xlabel="Epoch", ylim=(0, 1))
        axes[i, 0].legend(); axes[i, 1].legend()
    fig.tight_layout(); fig.savefig(figdir / "learning_curves_5seeds.png", dpi=160); plt.show(); plt.close(fig)
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for ax, (_, row) in zip(axes, selected.iterrows()):
        cm = pd.read_csv(out / f"seed_{int(row.seed)}" / f"confusion_matrix_val_{row.stage}.csv", index_col=0)
        ax.imshow(cm.to_numpy(), cmap="Blues")
        ax.set(title=f"Seed {int(row.seed)} · {row.stage}", xlabel="Predicted", ylabel="True")
        ax.set_xticks(range(5), CLASSES, rotation=90); ax.set_yticks(range(5), CLASSES)
        for r in range(5):
            for c in range(5):
                ax.text(c, r, int(cm.iloc[r, c]), ha="center", va="center", fontsize=8)
    fig.tight_layout(); fig.savefig(figdir / "confusion_matrices_5seeds.png", dpi=160); plt.show(); plt.close(fig)
    classes = pd.read_csv(out / "aggregate/per_class_validation.csv")
    means = classes.groupby("class_name")[["precision", "recall", "f1"]].mean().reindex(CLASSES)
    ax = means.plot.bar(figsize=(10, 5), ylim=(0, 1), title="Validation per-class metrics · selected stage per seed")
    ax.set(ylabel="Mean across five seeds", xlabel="Class")
    ax.figure.tight_layout(); ax.figure.savefig(figdir / "per_class_precision_recall_f1.png", dpi=160)
    plt.show(); plt.close(ax.figure)
    ax = selected.set_index("seed")[["healthy_precision", "healthy_recall"]].plot.bar(figsize=(8, 4), ylim=(0, 1), title="Healthy Validation metrics")
    ax.set(ylabel="Score", xlabel="Seed")
    ax.figure.tight_layout(); ax.figure.savefig(figdir / "healthy_recall_precision_by_seed.png", dpi=160)
    plt.show(); plt.close(ax.figure)
    ax = selected.set_index("seed")["disease_to_healthy_count"].plot.bar(figsize=(7, 4), title="Disease → Healthy errors on Validation")
    ax.set(ylabel="Images", xlabel="Seed")
    ax.figure.tight_layout(); ax.figure.savefig(figdir / "disease_to_healthy_by_seed.png", dpi=160)
    plt.show(); plt.close(ax.figure)


def export_analysis(out):
    out = Path(out)
    files = [p for p in out.rglob("*") if p.is_file() and "checkpoints" not in p.parts and
             p.suffix.lower() in (".csv", ".json", ".md", ".txt", ".png")]
    export = out / "export"
    export.mkdir(exist_ok=True)
    inventory = pd.DataFrame([{"path": p.relative_to(out).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size} for p in files])
    inventory.to_csv(export / "checksums_sha256.csv", index=False)
    target = export / f"{EXPERIMENT_ID}_ANALYSIS.zip"
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for p in files + [export / "checksums_sha256.csv"]:
            archive.write(p, p.relative_to(out).as_posix())
    with zipfile.ZipFile(target) as archive:
        assert archive.testzip() is None
    return target, sha(target)
