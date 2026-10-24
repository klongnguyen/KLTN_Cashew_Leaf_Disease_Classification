"""Generate the output-free EXP-VIT-TRANSFER-XGBOOST-008 notebook."""

import base64
import json
import zlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE / "vit_transfer_xgboost_008.ipynb"
if TARGET.exists():
    raise FileExistsError(TARGET)

cells = []


def md(number, title, body):
    cells.append({"cell_type": "markdown", "id": f"vit008_{number:02d}",
                  "metadata": {}, "source": f"## CELL {number:02d} — {title}\n\n{body}\n".splitlines(True)})


def code(number, title, source):
    cells.append({"cell_type": "code", "id": f"vit008_{number:02d}",
                  "metadata": {}, "execution_count": None, "outputs": [],
                  "source": (f"# CELL {number:02d} — {title}\n" + source.strip() + "\n").splitlines(True)})


md(0, "Experiment overview", """# EXP-VIT-TRANSFER-XGBOOST-008

**Source:** five immutable EXP-VIT-TRANSFER-5SEEDS-007 DeiT-tiny checkpoints, V08.
**Question:** do Train-only linear/XGBoost probes on true pre-head embeddings reduce disease→Healthy errors compared with the original softmax head? DeiT weights are never fitted here. Deployment seed stays **123**. The historical V08 Test and 48-image real holdout are previously exposed and disabled by default. Validation chooses a classifier under fixed safety guardrails; a new blind field set is required to confirm field performance.

Run cells in order. The final archives embed this exact output-free notebook. No 007 training curves or “full val for pre-fit” line are drawn: CELL 21 plots only Train-fold CV diagnostics. Results are **NOT_RUN** until execution completes.""")

code(1, "Dependencies and imports", r'''
# Colab: run once before the following imports, then restart only if pip asks.
import sys, subprocess
if "/content" in __import__("os").getcwd() or __import__("pathlib").Path("/content").exists():
    packages = ["keras-hub==0.32.0", "xgboost==3.1.1", "scikit-learn==1.7.2",
                "pandas==2.2.3", "matplotlib==3.10.0", "Pillow==11.3.0",
                "joblib==1.5.2", "google-api-python-client"]
    command = [sys.executable, "-m", "pip", "install", "--prefer-binary",
               "--only-binary=:all:", *packages]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode:
        print("pip install failed (last 12000 characters):\n" + (completed.stdout + "\n" + completed.stderr)[-12000:])
        raise RuntimeError(f"Dependency installation failed with exit code {completed.returncode}; see pip error above")
    print("Dependency installation complete for Python", sys.version.split()[0])
import base64, gc, hashlib, json, os, platform, random, shutil, time, zipfile, zlib
from collections import defaultdict
from pathlib import Path
import joblib, keras, keras_hub, matplotlib.pyplot as plt, numpy as np, pandas as pd, tensorflow as tf, xgboost
from PIL import Image
from matplotlib.colors import LinearSegmentedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, log_loss, precision_recall_fscore_support)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
CMAP = LinearSegmentedColormap.from_list("white_blue", ["#ffffff", "#dbeafe", "#60a5fa", "#1d4ed8"])
def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest().upper()
def write_json(p, obj):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))
EMBEDDED_NOTEBOOK_B64 = "__NOTEBOOK_PAYLOAD__"
def delivered_notebook_bytes():
    canonical = json.loads(zlib.decompress(base64.b64decode(EMBEDDED_NOTEBOOK_B64)))
    source = "".join(canonical["cells"][1]["source"])
    marker = 'EMBEDDED_NOTEBOOK_B64 = "__NOTEBOOK_PAYLOAD__"'
    if not source.startswith("# CELL 01") or marker not in source: raise AssertionError("Notebook payload marker changed")
    canonical["cells"][1]["source"] = source.replace(marker, 'EMBEDDED_NOTEBOOK_B64 = "' + EMBEDDED_NOTEBOOK_B64 + '"', 1).splitlines(True)
    return (json.dumps(canonical,ensure_ascii=False,indent=1)+"\n").encode("utf-8")
NOTEBOOK_BYTES = delivered_notebook_bytes()
''')

code(2, "Frozen configuration", r'''
EXPERIMENT_ID = "EXP-VIT-TRANSFER-XGBOOST-008"
SOURCE_ID = "EXP-VIT-TRANSFER-5SEEDS-007"
NOTEBOOK_FILENAME = "vit_transfer_xgboost_008.ipynb"
SEEDS = [42, 123, 2026, 3407, 7777]
DEPLOYMENT_SEED = 123
CLASSES = ["anthracnose", "healthy", "leaf_miner", "red_rust", "not_cashew_leaf"]
DISEASE_IDS, HEALTHY_ID, OTHER_ID = (0, 2, 3), 1, 4
EXPECTED_COUNTS = {"train": [939,803,875,1072,1101], "val": [308,216,264,328,314], "test": [122,118,132,158,156]}
DATASET_SHA256 = "75C5928B2F6432ACA901BC16831E91F9AA4BDACC1E2FF536753D71DA60C147E7"
SOURCE_MANIFEST_SHA256 = "C699A9ACA707DA25F143DB1FC3C9B5133EA2858BCDC468A158E5644E500EC715"
DATASET_DRIVE_ID = "1vaaOwAi72byaxFb7w_ixd_Y2HboKn2-G"
SOURCE_FULL_DRIVE_ID = "1WgYA9ZWzwNvGiOfBGV9jWSx4mdER4-px"
HISTORICAL_HOLDOUT_DRIVE_ID = "1PBry_y2bMsvd5_YdmEUy9erSFarS3O52"
HISTORICAL_HOLDOUT_SHA256 = "BDDCB61CA9AF6111E0B26DB7876D0A3309C587AEAFFAEF49B7A9D2D162309165"
DATASET_OVERRIDE = None                 # exact local/mounted CashewData_Split (8).zip
SOURCE_007_DIR_OVERRIDE = None          # extracted EXP-007 folder with seed_*/best_model.keras
SOURCE_007_FULL_ZIP_OVERRIDE = None      # exact EXP-007 FULL ZIP path
RESULT_ROOT_OVERRIDE = None
BATCH_SIZE = 16
CV_FOLDS = 3
LINEAR_C = [0.01, 0.1, 1.0, 10.0]
XGB_CANDIDATES = [
    (150,2,.03,.8,.8,1,1), (150,3,.05,1.,.8,3,3),
    (150,4,.08,.8,1.,5,10), (300,2,.05,1.,1.,1,3),
    (300,3,.03,.8,1.,3,10), (300,3,.05,.8,.8,1,3),
    (300,4,.08,1.,.8,5,1), (500,2,.03,1.,.8,3,1),
    (500,2,.05,.8,1.,5,10), (500,3,.05,1.,1.,3,3),
    (500,4,.03,.8,.8,1,10), (500,4,.08,1.,1.,5,3),
]
RUN_V08_TEST = False
RUN_HISTORICAL_HOLDOUT = False
RUN_FRESH_BLIND_TEST = False
HISTORICAL_HOLDOUT_OVERRIDE = None
FRESH_BLIND_MANIFEST = None
FRESH_BLIND_ROOT = None
ENABLE_HEALTHY_GATE = False  # separate future ablation; A/B/C here are raw classifiers
CONFIG = {k: globals()[k] for k in ["EXPERIMENT_ID","SOURCE_ID","SEEDS","DEPLOYMENT_SEED",
    "CLASSES","EXPECTED_COUNTS","DATASET_SHA256","SOURCE_MANIFEST_SHA256",
    "BATCH_SIZE","CV_FOLDS","LINEAR_C","XGB_CANDIDATES","RUN_V08_TEST",
    "RUN_HISTORICAL_HOLDOUT","RUN_FRESH_BLIND_TEST","ENABLE_HEALTHY_GATE"]}
# JSON stores XGB candidate tuples as lists. Normalize before comparing a resumed run.
CONFIG = json.loads(json.dumps(CONFIG))
''')

code(3, "Reproducibility and environment", r'''
random.seed(2026); np.random.seed(2026); tf.random.set_seed(2026)
if Path("/content").exists():
    from google.colab import drive
    drive.mount("/content/drive")
    project_root = Path("/content/drive/MyDrive/Cashew_Leaf_model_result")
    work_root = Path("/content/vit008_work")
else:
    project_root = Path.cwd()
    work_root = project_root / ".vit008_work"
results_root = Path(RESULT_ROOT_OVERRIDE) if RESULT_ROOT_OVERRIDE else project_root / "training_results/current_dataset"
OUT = results_root / EXPERIMENT_ID
if (OUT / "selection/validation_selection_lock.json").exists():
    raise FileExistsError("EXP-008 is already locked; do not rerun and overwrite it. Copy to a new experiment ID.")
for d in [OUT, work_root, OUT/"source", OUT/"dataset_audit", OUT/"figures", OUT/"selection", OUT/"model_package", OUT/"archives", OUT/"embeddings", OUT/"evaluation"]:
    d.mkdir(parents=True, exist_ok=True)
if (OUT/"experiment_config.json").is_file() and read_json(OUT/"experiment_config.json") != json.loads(json.dumps(CONFIG)):
    raise ValueError("Existing EXP-008 config differs. Use a new experiment ID.")
write_json(OUT/"experiment_config.json", CONFIG)
write_json(OUT/"environment.json", {"python":platform.python_version(),"tensorflow":tf.__version__,
    "keras":keras.__version__,"keras_hub":keras_hub.__version__,"sklearn":__import__("sklearn").__version__,
    "xgboost":xgboost.__version__,"joblib":joblib.__version__,"gpu":[str(x) for x in tf.config.list_physical_devices("GPU")]})
(OUT/"requirements.lock").write_text("\n".join([f"{k}=={v}" for k,v in {
    "tensorflow":tf.__version__,"keras":keras.__version__,"keras-hub":keras_hub.__version__,
    "scikit-learn":__import__("sklearn").__version__,"xgboost":xgboost.__version__,
    "joblib":joblib.__version__}.items()])+"\n", encoding="utf-8")
write_json(OUT/"execution_status.json", {"status":"RUNNING", "training_mode":"classifier_only", "test":"NOT_RUN", "holdout":"NOT_RUN"})
''')

code(4, "Resolve read-only EXP-007 source and V08 archive", r'''
def drive_download(file_id, target, expected_size=None):
    from google.colab import auth
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    auth.authenticate_user()
    credentials, _ = google.auth.default()
    service = build("drive","v3",credentials=credentials,cache_discovery=False)
    info = service.files().get(fileId=file_id,fields="id,name,size,capabilities(canDownload)",supportsAllDrives=True).execute()
    if info.get("capabilities",{}).get("canDownload") is False: raise PermissionError(info)
    if expected_size and int(info["size"]) != expected_size: raise ValueError(f"Drive size changed: {info}")
    target = Path(target); target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file() and target.stat().st_size == int(info["size"]): return target
    partial = target.with_name(target.name+".partial")
    with partial.open("wb") as f:
        request = service.files().get_media(fileId=file_id,supportsAllDrives=True)
        dl = MediaIoBaseDownload(f, request, chunksize=16*1024*1024)
        done = False
        while not done: _,done = dl.next_chunk(num_retries=3)
    if partial.stat().st_size != int(info["size"]): raise IOError("Incomplete Drive download")
    partial.replace(target)
    return target
def find_named(name, preferred):
    if preferred and Path(preferred).is_file(): return Path(preferred)
    candidates = [project_root/"Dataset"/name, project_root.parent/"Dataset"/name,
                  project_root/"training_results/current_dataset"/name,
                  Path("/content")/name, Path.cwd()/name]
    for p in candidates:
        if p.is_file(): return p
    if Path("/content/drive/MyDrive").exists():
        matches = list(Path("/content/drive/MyDrive").rglob(name))
        if matches: return matches[0]
    return None
dataset_zip = find_named("CashewData_Split (8).zip",DATASET_OVERRIDE)
if dataset_zip is None:
    if not Path("/content").exists(): raise FileNotFoundError("Set DATASET_OVERRIDE to the V08 ZIP")
    dataset_zip = drive_download(DATASET_DRIVE_ID, work_root/"CashewData_Split (8).zip",1058695554)
if sha256_file(dataset_zip) != DATASET_SHA256: raise ValueError("V08 dataset ZIP SHA-256 mismatch")
source_dir = Path(SOURCE_007_DIR_OVERRIDE) if SOURCE_007_DIR_OVERRIDE else results_root/SOURCE_ID
if not all((source_dir/f"seed_{s}"/"best_model.keras").is_file() for s in SEEDS):
    source_zip = find_named(f"{SOURCE_ID}_FULL.zip", SOURCE_007_FULL_ZIP_OVERRIDE)
    if source_zip is None:
        if not Path("/content").exists(): raise FileNotFoundError("Set SOURCE_007_FULL_ZIP_OVERRIDE or SOURCE_007_DIR_OVERRIDE")
        source_zip = drive_download(SOURCE_FULL_DRIVE_ID,work_root/f"{SOURCE_ID}_FULL.zip",782750036)
    source_dir = work_root/SOURCE_ID
    if not all((source_dir/f"seed_{s}"/"best_model.keras").is_file() for s in SEEDS):
        with zipfile.ZipFile(source_zip) as z:
            for item in z.infolist():
                rel = Path(item.filename)
                if rel.is_absolute() or ".." in rel.parts: raise ValueError(f"Unsafe source ZIP member: {item.filename}")
                if item.is_dir(): continue
                target = work_root/rel
                if work_root.resolve() not in target.resolve().parents: raise ValueError(item.filename)
                if (item.filename.endswith("best_model.keras") or
                    item.filename.endswith("validation_predictions.csv") or
                    item.filename.endswith("validation_selection_lock.json") or
                    item.filename.endswith("dataset_manifest.csv") or
                    item.filename.endswith("experiment_config.json") or
                    item.filename.endswith("real_holdout/dataset_manifest.csv")):
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with z.open(item) as src, target.open("wb") as dst: shutil.copyfileobj(src,dst)
for name in ["validation_selection_lock.json","dataset_manifest.csv","experiment_config.json"]:
    if not (source_dir/name).is_file(): raise FileNotFoundError(source_dir/name)
write_json(OUT/"source/provenance.json", {"source_experiment":SOURCE_ID,"source_dir":str(source_dir),
    "dataset_zip":str(dataset_zip),"dataset_sha256":sha256_file(dataset_zip),
    "source_lock_sha256":sha256_file(source_dir/"validation_selection_lock.json")})
''')

code(5, "Dataset manifest and exact-byte audit", r'''
extract_root = work_root/"dataset"
marker = extract_root/".sha256"
if not marker.exists():
    if extract_root.exists() and any(extract_root.iterdir()): raise RuntimeError("Unverified extracted dataset")
    extract_root.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(dataset_zip) as z:
        for item in z.infolist():
            rel = Path(item.filename)
            if rel.is_absolute() or ".." in rel.parts: raise ValueError(item.filename)
        z.extractall(extract_root)
    marker.write_text(DATASET_SHA256,encoding="utf-8")
elif marker.read_text().strip() != DATASET_SHA256: raise RuntimeError("Dataset extraction marker mismatch")
dataset_root = extract_root/"CashewData_Split"
rows = []
for split in ["train","val","test"]:
    if set(p.name for p in (dataset_root/split).iterdir() if p.is_dir()) != set(CLASSES): raise ValueError(f"Class directories changed: {split}")
    for label,klass in enumerate(CLASSES):
        files = sorted(p for p in (dataset_root/split/klass).rglob("*") if p.is_file())
        if len(files) != EXPECTED_COUNTS[split][label]: raise ValueError(f"Changed count: {split}/{klass}")
        for p in files:
            rows.append({"split":split,"class":klass,"label":label,
                         "relative_path":p.relative_to(dataset_root).as_posix(),"sha256":sha256_file(p)})
manifest = pd.DataFrame(rows)
if len(manifest) != 6906 or manifest.relative_path.duplicated().any(): raise ValueError("V08 manifest count/path mismatch")
manifest_sha = hashlib.sha256(manifest[["split","class","relative_path","sha256"]].to_csv(index=False,lineterminator="\n").encode()).hexdigest().upper()
if manifest_sha != SOURCE_MANIFEST_SHA256: raise ValueError(f"EXP-007 manifest mismatch: {manifest_sha}")
source_manifest = pd.read_csv(source_dir/"dataset_manifest.csv")
if not manifest.equals(source_manifest[manifest.columns]): raise ValueError("Manifest differs from EXP-007 rows/order")
duplicate = manifest[manifest.duplicated("sha256",keep=False)].copy()
duplicate.to_csv(OUT/"dataset_audit/exact_duplicates.csv",index=False)
if any(g.split.nunique()>1 for _,g in duplicate.groupby("sha256")): raise ValueError("Exact duplicate bytes across V08 splits")
manifest.to_csv(OUT/"dataset_manifest.csv",index=False)
# dHash is a visual shortlist only; it cannot prove group independence.
def dhash(path):
    with Image.open(path) as im:
        a=np.asarray(im.convert("L").resize((9,8),Image.Resampling.LANCZOS),dtype=np.int16)
    bits=(a[:,1:]>a[:,:-1]).reshape(-1)
    return sum(int(bit)<<i for i,bit in enumerate(bits))
hashes=[dhash(dataset_root/p) for p in manifest.relative_path]
bucket=defaultdict(list)
for i,h in enumerate(hashes):
    for part in range(4): bucket[(part,(h>>(part*16))&65535)].append(i)
candidate_pairs=set()
for members in bucket.values():
    for offset,i in enumerate(members):
        for j in members[offset+1:]:
            if manifest.split.iloc[i]!=manifest.split.iloc[j] and (hashes[i]^hashes[j]).bit_count()<=3:
                candidate_pairs.add((min(i,j),max(i,j)))
near_rows=[{"path_a":manifest.relative_path.iloc[i],"path_b":manifest.relative_path.iloc[j],
            "split_a":manifest.split.iloc[i],"split_b":manifest.split.iloc[j],
            "dhash_distance":(hashes[i]^hashes[j]).bit_count(),"status":"VISUAL_REVIEW_REQUIRED"}
           for i,j in sorted(candidate_pairs)]
pd.DataFrame(near_rows,columns=["path_a","path_b","split_a","split_b","dhash_distance","status"]).to_csv(
    OUT/"dataset_audit/near_duplicate_candidates.csv",index=False)
if near_rows:
    contact_dir=OUT/"dataset_audit/audit_contact_sheets"; contact_dir.mkdir(exist_ok=True)
    sample_pairs=near_rows[:12]; sheet=Image.new("RGB",(448,len(sample_pairs)*140),"white")
    for row_idx,pair in enumerate(sample_pairs):
        for col,key in enumerate(["path_a","path_b"]):
            with Image.open(dataset_root/pair[key]) as im:
                thumb=im.convert("RGB"); thumb.thumbnail((215,125))
                sheet.paste(thumb,(col*224,row_idx*140))
    sheet.save(contact_dir/"first_12_dhash_candidates.jpg",quality=90)
write_json(OUT/"dataset_audit/dataset_audit_summary.json", {"images":len(manifest),"manifest_sha256":manifest_sha,
    "cross_split_exact_duplicates":0,"dhash_candidate_pairs":len(candidate_pairs),
    "near_duplicate_status":"VISUAL_REVIEW_REQUIRED" if candidate_pairs else "NO_DHASH_LE_3_CANDIDATES",
    "group_leakage_status":"GROUP_LEAKAGE_NOT_VERIFIED"})
''')

code(6, "Class distribution and real_ subgroup", r'''
counts = manifest.groupby(["split","class"],sort=False).size().rename("count").reset_index()
counts.to_csv(OUT/"dataset_audit/class_distribution.csv",index=False)
pd.DataFrame([{"split":split,"real_prefix_count":int(manifest.loc[manifest.split==split,"relative_path"].map(lambda p:Path(p).name.startswith("real_")).sum())}
              for split in ["train","val","test"]]).to_csv(OUT/"dataset_audit/real_prefix_counts.csv",index=False)
display(counts)
''')

code(7, "EXP-007 input decoding, no augmentation", r'''
def decode_resize(path):
    raw = tf.io.read_file(path)
    image = tf.io.decode_image(raw,channels=3,expand_animations=False)
    image.set_shape((None,None,3))
    return tf.cast(tf.image.resize(image,(224,224),method="bilinear"),tf.float32)
def input_ds(frame,batch=BATCH_SIZE,root=None):
    root = dataset_root if root is None else Path(root)
    paths = [str(root/p) for p in frame.relative_path]
    return tf.data.Dataset.from_tensor_slices(paths).map(decode_resize,num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)
train_rows = manifest[manifest.split=="train"].reset_index(drop=True)
val_rows = manifest[manifest.split=="val"].reset_index(drop=True)
sample_image = decode_resize(str(dataset_root/train_rows.relative_path.iloc[0]))
if sample_image.shape != (224,224,3) or not np.isfinite(sample_image.numpy()).all(): raise ValueError("Input decode failed")
t0=time.perf_counter()
for relative in val_rows.relative_path.iloc[:64]: _=decode_resize(str(dataset_root/relative)).numpy()
write_json(OUT/"image_processing_time.json",{"sample_count":64,"decode_resize_ms_per_image":1000*(time.perf_counter()-t0)/64,
    "note":"RGB decode + bilinear resize on this runtime; not pure disk I/O or full inference"})
write_json(OUT/"source/preprocess_contract.json", {"image":"RGB","resize":"bilinear 224x224",
    "range":"float32 0..255","saved_model_preprocessor":"exactly once","augmentation":"disabled during extraction"})
''')

code(8, "Verify five immutable source checkpoints", r'''
source_lock = read_json(source_dir/"validation_selection_lock.json")
if source_lock["deployment_seed"] != 123 or source_lock["dataset_archive_sha256"].upper() != DATASET_SHA256: raise ValueError("Source lock mismatch")
if source_lock["dataset_manifest_sha256"].upper() != SOURCE_MANIFEST_SHA256: raise ValueError("Source manifest lock mismatch")
inventory = []
for seed in SEEDS:
    p = source_dir/f"seed_{seed}"/"best_model.keras"
    if not p.is_file(): raise FileNotFoundError(p)
    digest = sha256_file(p)
    expected = source_lock["checkpoint_sha256"].get(str(seed))
    if digest != str(expected).upper(): raise ValueError(f"Checkpoint SHA mismatch seed {seed}: {digest} != {expected}")
    inventory.append({"seed":seed,"path":str(p),"bytes":p.stat().st_size,"sha256":digest})
inventory_df = pd.DataFrame(inventory)
inventory_df.to_csv(OUT/"source/exp007_checkpoint_inventory.csv",index=False)
shutil.copy2(source_dir/"validation_selection_lock.json",OUT/"source/exp007_validation_selection_lock.json")
write_json(OUT/"source/exp007_config_sha256.json",{"sha256":sha256_file(source_dir/"experiment_config.json")})
''')

code(9, "Inspect saved DeiT and build true pre-head extractor", r'''
def load_extractor(seed):
    path = source_dir/f"seed_{seed}"/"best_model.keras"
    model = keras.models.load_model(path,compile=False)
    classifiers = [x for x in model.layers if isinstance(x,keras_hub.models.DeiTImageClassifier)]
    if len(classifiers)!=1: raise ValueError(f"Expected one DeiTImageClassifier, found {len(classifiers)}")
    classifier = classifiers[0]
    augmentations = [x for x in model.layers if x.name=="augmentation"]
    if len(augmentations)!=1 or not hasattr(classifier,"output_dense"): raise ValueError("EXP-007 model layout changed")
    prehead = keras.Model(classifier.input,classifier.output_dense.input,name="verified_prehead")
    x = augmentations[0](model.input,training=False)
    x = classifier.preprocessor(x)
    features = prehead(x,training=False)
    extractor = keras.Model(model.input,features,name="exp007_prehead_extractor")
    model.trainable = False; extractor.trainable = False
    dim = int(extractor.output_shape[-1])
    if dim <= len(CLASSES) or len(extractor.trainable_weights): raise ValueError("Invalid/fittable feature endpoint")
    descriptor={"seed":seed,"feature_dimension":dim,"pooling":str(classifier.pooling),
                "endpoint":"DeiTImageClassifier.output_dense.input","preprocessing":"augmentation(training=False) -> classifier.preprocessor once"}
    return model,classifier,extractor,descriptor
probe_model,probe_head,probe_extractor,probe_descriptor = load_extractor(123)
write_json(OUT/"source/feature_extractor_config.json",probe_descriptor)
del probe_model,probe_head,probe_extractor; gc.collect()
''')

code(10, "Pre-head/logit parity, determinism and batch smoke", r'''
def source_predictions(seed):
    p = source_dir/f"seed_{seed}"/"validation_predictions.csv"
    if not p.is_file(): raise FileNotFoundError(p)
    df = pd.read_csv(p)
    if df.relative_path.tolist()!=val_rows.relative_path.tolist() or df.true_label.astype(int).tolist()!=val_rows.label.astype(int).tolist():
        raise ValueError(f"EXP-007 prediction row/path/label order changed for seed {seed}")
    return df
def check_probe(model,head,extractor):
    x = tf.stack([decode_resize(str(dataset_root/p)) for p in val_rows.relative_path.iloc[:2]])
    f1 = extractor(x,training=False).numpy()
    f2 = extractor(x,training=False).numpy()
    f_single = np.concatenate([extractor(x[i:i+1],training=False).numpy() for i in range(2)])
    logits = model(x,training=False).numpy()
    reconstructed = head.output_dense(f1).numpy()
    repeated_logits = head.output_dense(f2).numpy()
    single_logits = head.output_dense(f_single).numpy()
    if not all(np.isfinite(a).all() for a in (f1,f2,f_single,logits,reconstructed)):
        raise AssertionError("Nonfinite features or logits")
    def relative_l2(a,b):
        return float(np.linalg.norm(a.astype("float64")-b.astype("float64")) /
                     max(float(np.linalg.norm(a.astype("float64"))),1.0))
    diagnostic = {"repeat_feature_max_abs":float(np.max(np.abs(f1-f2))),
                  "repeat_feature_relative_l2":relative_l2(f1,f2),
                  "single_vs_batch_feature_max_abs":float(np.max(np.abs(f1-f_single))),
                  "single_vs_batch_feature_relative_l2":relative_l2(f1,f_single),
                  "prehead_logit_max_abs":float(np.max(np.abs(logits-reconstructed))),
                  "prehead_logit_relative_l2":relative_l2(logits,reconstructed),
                  "single_vs_batch_logit_relative_l2":relative_l2(reconstructed,single_logits)}
    if diagnostic["repeat_feature_relative_l2"] > .005:
        raise AssertionError(f"Repeated same-batch features drift beyond 0.5%: {diagnostic}")
    if diagnostic["single_vs_batch_feature_relative_l2"] > .01:
        raise AssertionError(f"Single-image features drift beyond 1%: {diagnostic}")
    if diagnostic["prehead_logit_relative_l2"] > .001:
        raise AssertionError(f"Pre-head does not reconstruct original logits: {diagnostic}")
    if (not np.array_equal(logits.argmax(1),reconstructed.argmax(1)) or
        not np.array_equal(reconstructed.argmax(1),repeated_logits.argmax(1)) or
        not np.array_equal(reconstructed.argmax(1),single_logits.argmax(1))):
        raise AssertionError(f"Probe predicted labels differ by invocation/batch: {diagnostic}")
    print("Extractor probe:",diagnostic)
    return diagnostic
def predict_seed(seed, frame, root=None):
    model,head,extractor,descriptor = load_extractor(seed)
    smoke_error = check_probe(model,head,extractor)
    feats,probs = [],[]
    t0=time.perf_counter()
    for batch in input_ds(frame,root=root):
        f=extractor(batch,training=False).numpy()
        logits=head.output_dense(f).numpy()
        feats.append(f.astype("float32")); probs.append(tf.nn.softmax(logits,axis=1).numpy())
    elapsed=time.perf_counter()-t0
    features=np.concatenate(feats); probability=np.concatenate(probs)
    if features.shape!=(len(frame),descriptor["feature_dimension"]) or probability.shape!=(len(frame),5): raise ValueError("Batch output shape mismatch")
    if not np.isfinite(features).all() or not np.allclose(probability.sum(1),1,atol=1e-5): raise ValueError("Invalid output")
    del model,head,extractor; gc.collect(); keras.backend.clear_session()
    return features,probability,descriptor,smoke_error,1000*elapsed/len(frame)
''')

code(11, "Extract/cache Train and Val features, full source parity", r'''
feature_by_seed={}; baseline_by_seed={}; timing_rows=[]; parity_rows=[]
for seed in SEEDS:
    train_f,_,descriptor,error,train_ms=predict_seed(seed,train_rows)
    val_f,val_p,val_descriptor,val_error,val_ms=predict_seed(seed,val_rows)
    if descriptor["feature_dimension"]!=val_descriptor["feature_dimension"]: raise ValueError("Feature dimension changed")
    source_df=source_predictions(seed)
    mismatch=np.flatnonzero(np.argmax(val_p,axis=1)!=source_df.predicted_label.astype(int).to_numpy())
    if len(mismatch):
        pd.DataFrame({"relative_path":val_rows.relative_path.iloc[mismatch],"actual":np.argmax(val_p,axis=1)[mismatch],
                      "source":source_df.predicted_label.astype(int).iloc[mismatch]}).to_csv(OUT/f"source/seed_{seed}_parity_fail.csv",index=False)
        raise AssertionError(f"Full EXP-007 Validation prediction mismatch, seed {seed}: {len(mismatch)}")
    if seed==123:
        y=val_rows.label.to_numpy(); pred=np.argmax(val_p,axis=1)
        if int(np.sum(np.isin(y,DISEASE_IDS)&(pred==1)))!=18 or int(np.sum((y==1)&(pred==1)))!=210: raise AssertionError("Seed 123 baseline counts differ from EXP-007")
    seed_dir=OUT/f"seed_{seed}"; seed_dir.mkdir(exist_ok=True)
    fingerprint=hashlib.sha256(json.dumps({"manifest":SOURCE_MANIFEST_SHA256,"checkpoint":source_lock["checkpoint_sha256"][str(seed)],
        "endpoint":descriptor,"train_paths":train_rows.relative_path.tolist(),"val_paths":val_rows.relative_path.tolist()},sort_keys=True).encode()).hexdigest()
    fingerprint_path=OUT/f"embeddings/seed_{seed}_fingerprint.json"
    if fingerprint_path.is_file() and read_json(fingerprint_path)["sha256"]!=fingerprint:
        raise ValueError(f"Stale embedding cache fingerprint for seed {seed}")
    for split,frame,features in [("train",train_rows,train_f),("val",val_rows,val_f)]:
        np.save(OUT/f"embeddings/seed_{seed}_{split}.npy",features)
        frame.assign(row_id=np.arange(len(frame)),seed=seed,checkpoint_sha256=source_lock["checkpoint_sha256"][str(seed)]).to_csv(OUT/f"embeddings/seed_{seed}_{split}_metadata.csv",index=False)
    write_json(fingerprint_path,{"sha256":fingerprint,"shape_train":list(train_f.shape),"shape_val":list(val_f.shape)})
    feature_by_seed[seed]=(train_f,val_f); baseline_by_seed[seed]=val_p
    parity_rows.append({"seed":seed,"val_rows":len(val_rows),"prediction_mismatches":0,
        **{f"train_probe_{k}":v for k,v in error.items()},
        **{f"val_probe_{k}":v for k,v in val_error.items()}})
    timing_rows.append({"seed":seed,"train_extraction_ms_per_image":train_ms,"val_extraction_ms_per_image":val_ms})
pd.DataFrame(parity_rows).to_csv(OUT/"source/validation_parity_5seeds.csv",index=False)
write_json(OUT/"pipeline_smoke.json",{"status":"PASS","source_prediction_mismatches":0,"seeds":parity_rows})
''')

code(12, "Original softmax baseline A", r'''
def metric_record(y,p,seed,model_type):
    y=np.asarray(y,dtype=int); p=np.asarray(p,dtype=float)
    if p.shape!=(len(y),5) or not np.isfinite(p).all() or not np.allclose(p.sum(1),1,atol=1e-5): raise ValueError("Invalid class probabilities")
    pred=np.argmax(p,axis=1); cm=confusion_matrix(y,pred,labels=list(range(5)))
    if cm.sum()!=len(y): raise AssertionError("Confusion count mismatch")
    prec,rec,f1,support=precision_recall_fscore_support(y,pred,labels=list(range(5)),zero_division=0)
    disease=np.isin(y,DISEASE_IDS)
    result={"seed":seed,"classifier":model_type,"n":len(y),"accuracy":accuracy_score(y,pred),
        "balanced_accuracy":balanced_accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro"),
        "weighted_f1":f1_score(y,pred,average="weighted"),"macro_precision":float(np.mean(prec)),
        "macro_recall":float(np.mean(rec)),"log_loss":log_loss(y,p,labels=list(range(5))),
        "multiclass_brier":float(np.mean(np.sum((p-np.eye(5)[y])**2,axis=1))),
        "disease_to_healthy_count":int(np.sum(disease&(pred==HEALTHY_ID))),"disease_total":int(disease.sum()),
        "disease_to_healthy_rate":float(np.mean(pred[disease]==HEALTHY_ID)),
        "healthy_true_count":int(np.sum(y==1)),"healthy_correct":int(cm[1,1]),
        "healthy_predicted_count":int(np.sum(pred==1)),"healthy_recall":float(rec[1]),
        "healthy_precision":float(prec[1]),"not_cashew_leaf_recall":float(rec[4]),
        "not_cashew_leaf_precision":float(prec[4]),"disease_macro_recall":float(np.mean(rec[list(DISEASE_IDS)])),
        "confusion_matrix":cm.tolist(),"per_class":[{"class":CLASSES[i],"precision":float(prec[i]),
            "recall":float(rec[i]),"f1":float(f1[i]),"support":int(support[i]),
            "to_healthy":int(cm[i,HEALTHY_ID]) if i in DISEASE_IDS else None} for i in range(5)]}
    if len(y)==1430 and (result["disease_total"]!=900 or result["healthy_true_count"]!=216 or int(np.sum(y==4))!=314): raise ValueError("Validation class totals mismatch")
    return result
all_metrics=[]; all_probability={}; all_prediction={}
for seed in SEEDS:
    p=baseline_by_seed[seed]; result=metric_record(val_rows.label,p,seed,"A")
    all_metrics.append(result); all_probability[(seed,"A")]=p; all_prediction[(seed,"A")]=p.argmax(1)
    write_json(OUT/f"seed_{seed}/A_metrics.json",result)
''')

code(13, "Train-only CV folds", r'''
y_train=train_rows.label.to_numpy(dtype=int)
y_val=val_rows.label.to_numpy(dtype=int)
folds=list(StratifiedKFold(n_splits=CV_FOLDS,shuffle=True,random_state=2026).split(np.zeros(len(y_train)),y_train))
fold_rows=[]
for fold,(fit_idx,check_idx) in enumerate(folds):
    if set(fit_idx)&set(check_idx) or max(fit_idx.max(),check_idx.max())>=len(train_rows): raise AssertionError("CV leakage")
    fold_rows.extend({"fold":fold,"train_row_id":int(i),"relative_path":train_rows.relative_path.iloc[i],"role":"cv_holdout"} for i in check_idx)
pd.DataFrame(fold_rows).to_csv(OUT/"source/train_only_cv_folds.csv",index=False)
def d2h_count(y,p): return int(np.sum(np.isin(y,DISEASE_IDS)&(p==HEALTHY_ID)))
def choose_trial(rows,depth_key=None,trees_key=None):
    return sorted(rows,key=lambda r:(-round(r["cv_macro_f1_mean"],10),r["cv_d2h_total"],
         r.get(depth_key,0) if depth_key else 0,r.get(trees_key,0) if trees_key else 0,
         r.get("C",0)))[0]
''')

code(14, "Linear probe B: Train-fold scaler, CV and refit", r'''
cv_records=[]; classifier_models={}; classifier_prob={}; training_time=[]
for seed in SEEDS:
    xtr,xva=feature_by_seed[seed]; trials=[]; t0=time.perf_counter()
    for c in LINEAR_C:
        fold_f1=[]; fold_d2h=[]
        for fold,(fit_idx,check_idx) in enumerate(folds):
            pipe=make_pipeline(StandardScaler(),LogisticRegression(C=c,max_iter=3000,class_weight=None,random_state=seed))
            pipe.fit(xtr[fit_idx],y_train[fit_idx])
            pred=pipe.predict(xtr[check_idx]); score=f1_score(y_train[check_idx],pred,average="macro")
            fold_f1.append(score); fold_d2h.append(d2h_count(y_train[check_idx],pred))
            cv_records.append({"seed":seed,"classifier":"B","C":c,"fold":fold,"macro_f1":score,"d2h_count":fold_d2h[-1]})
        trials.append({"C":c,"cv_macro_f1_mean":float(np.mean(fold_f1)),"cv_d2h_total":sum(fold_d2h)})
    chosen=choose_trial(trials)
    model=make_pipeline(StandardScaler(),LogisticRegression(C=chosen["C"],max_iter=3000,class_weight=None,random_state=seed))
    model.fit(xtr,y_train)
    if not np.array_equal(model.classes_,np.arange(5)): raise ValueError("B class order mismatch")
    p=model.predict_proba(xva)
    joblib.dump(model,OUT/f"seed_{seed}/logistic_regression.joblib")
    classifier_models[(seed,"B")]=model; classifier_prob[(seed,"B")]=p
    write_json(OUT/f"seed_{seed}/B_cv_selection.json",{"trials":trials,"chosen":chosen})
    training_time.append({"seed":seed,"classifier":"B","cv_and_refit_seconds":time.perf_counter()-t0})
pd.DataFrame(cv_records).to_csv(OUT/"source/cv_folds_B.csv",index=False)
''')

code(15, "XGBoost C: fixed 12 candidates, Train-only CV", r'''
XGB_KEYS=["n_estimators","max_depth","learning_rate","subsample","colsample_bytree","min_child_weight","reg_lambda"]
def xgb_model(params,seed):
    return XGBClassifier(objective="multi:softprob",num_class=5,tree_method="hist",eval_metric="mlogloss",
        random_state=seed,n_jobs=4,**params)
for seed in SEEDS:
    xtr,xva=feature_by_seed[seed]; trials=[]; t0=time.perf_counter()
    for candidate_id,values in enumerate(XGB_CANDIDATES):
        params=dict(zip(XGB_KEYS,values)); fold_f1=[]; fold_d2h=[]
        for fold,(fit_idx,check_idx) in enumerate(folds):
            model=xgb_model(params,seed)
            model.fit(xtr[fit_idx],y_train[fit_idx])
            pred=model.predict(xtr[check_idx]); score=f1_score(y_train[check_idx],pred,average="macro")
            fold_f1.append(score); fold_d2h.append(d2h_count(y_train[check_idx],pred))
            cv_records.append({"seed":seed,"classifier":"C","candidate_id":candidate_id,"fold":fold,
                "macro_f1":score,"d2h_count":fold_d2h[-1],**params})
        trials.append({"candidate_id":candidate_id,**params,"cv_macro_f1_mean":float(np.mean(fold_f1)),"cv_d2h_total":sum(fold_d2h)})
    chosen=choose_trial(trials,"max_depth","n_estimators")
    params={k:chosen[k] for k in XGB_KEYS}; model=xgb_model(params,seed); model.fit(xtr,y_train)
    if not np.array_equal(model.classes_,np.arange(5)): raise ValueError("C class order mismatch")
    p=model.predict_proba(xva)
    model.save_model(OUT/f"seed_{seed}/xgboost.json")
    classifier_models[(seed,"C")]=model; classifier_prob[(seed,"C")]=p
    write_json(OUT/f"seed_{seed}/C_cv_selection.json",{"trials":trials,"chosen":chosen})
    training_time.append({"seed":seed,"classifier":"C","cv_and_refit_seconds":time.perf_counter()-t0})
pd.DataFrame(cv_records).to_csv(OUT/"source/cv_folds_all.csv",index=False)
''')

code(16, "Raw Validation A/B/C, same images/order", r'''
for seed in SEEDS:
    for typ in ["A","B","C"]:
        p=baseline_by_seed[seed] if typ=="A" else classifier_prob[(seed,typ)]
        rec=metric_record(y_val,p,seed,typ)
        if typ!="A": all_metrics.append(rec)
        all_probability[(seed,typ)]=p; all_prediction[(seed,typ)]=p.argmax(1)
        write_json(OUT/f"seed_{seed}/{typ}_metrics.json",rec)
        pd.DataFrame({"relative_path":val_rows.relative_path,"true_label":y_val,"predicted_label":p.argmax(1),
            **{f"p_{name}":p[:,i] for i,name in enumerate(CLASSES)}}).to_csv(OUT/f"seed_{seed}/{typ}_validation_predictions.csv",index=False)
        pd.DataFrame(rec["confusion_matrix"],index=CLASSES,columns=CLASSES).to_csv(OUT/f"seed_{seed}/{typ}_confusion_matrix.csv")
metrics_df=pd.DataFrame([{k:v for k,v in r.items() if k not in ["confusion_matrix","per_class"]} for r in all_metrics])
metrics_df.to_csv(OUT/"validation_metrics_5seeds_abc.csv",index=False)
''')

code(17, "Optional Healthy gate policy", r'''
if ENABLE_HEALTHY_GATE:
    raise NotImplementedError("Healthy threshold D is a separate registered ablation. EXP-008 A/B/C uses raw argmax.")
write_json(OUT/"selection/healthy_gate_status.json",{"enabled":False,"reason":"classifier-only A/B/C comparison"})
''')

code(18, "Aggregate and paired deltas", r'''
numeric_cols=[c for c in metrics_df.select_dtypes(include="number").columns if c!="seed"]
summary=metrics_df.groupby("classifier")[numeric_cols].agg(["mean","std"])
summary.to_csv(OUT/"validation_mean_std.csv")
delta_rows=[]
for seed in SEEDS:
    base=metrics_df[(metrics_df.seed==seed)&(metrics_df.classifier=="A")].iloc[0]
    for typ in ["B","C"]:
        cur=metrics_df[(metrics_df.seed==seed)&(metrics_df.classifier==typ)].iloc[0]
        bp=all_prediction[(seed,"A")]; cp=all_prediction[(seed,typ)]
        delta_rows.append({"seed":seed,"classifier":typ,"d2h_delta":int(cur.disease_to_healthy_count-base.disease_to_healthy_count),
            "macro_f1_delta":float(cur.macro_f1-base.macro_f1),"healthy_recall_delta":float(cur.healthy_recall-base.healthy_recall),
            "not_cashew_recall_delta":float(cur.not_cashew_leaf_recall-base.not_cashew_leaf_recall),
            "wrong_to_right":int(np.sum((bp!=y_val)&(cp==y_val))),"right_to_wrong":int(np.sum((bp==y_val)&(cp!=y_val)))})
delta_df=pd.DataFrame(delta_rows); delta_df.to_csv(OUT/"paired_deltas.csv",index=False)
''')

code(19, "D2H and real_ error audit", r'''
error_rows=[]
for seed in SEEDS:
    for typ in ["A","B","C"]:
        pred=all_prediction[(seed,typ)]
        for i in np.flatnonzero(pred!=y_val):
            error_rows.append({"seed":seed,"classifier":typ,"relative_path":val_rows.relative_path.iloc[i],
                "true_label":int(y_val[i]),"predicted_label":int(pred[i]),"disease_to_healthy":bool(y_val[i] in DISEASE_IDS and pred[i]==1),
                "real_prefix":Path(val_rows.relative_path.iloc[i]).name.startswith("real_")})
errors_df=pd.DataFrame(error_rows); errors_df.to_csv(OUT/"validation_errors.csv",index=False)
real_mask=val_rows.relative_path.map(lambda p:Path(p).name.startswith("real_")).to_numpy()
real_rows=[]
for seed in SEEDS:
    for typ in ["A","B","C"]:
        pred=all_prediction[(seed,typ)][real_mask]; y=y_val[real_mask]
        real_rows.append({"seed":seed,"classifier":typ,"n":len(y),"accuracy":float(np.mean(pred==y)),
            "d2h_count":d2h_count(y,pred),"disease_total":int(np.isin(y,DISEASE_IDS).sum())})
pd.DataFrame(real_rows).to_csv(OUT/"real_prefix_validation_subgroup.csv",index=False)
''')

code(20, "Validation selection and immutable lock", r'''
def passes_guardrail(cur,base):
    if cur["macro_f1"] < base["macro_f1"]-.01 or cur["healthy_recall"] < base["healthy_recall"]-.02 or cur["not_cashew_leaf_recall"] < base["not_cashew_leaf_recall"]-.02: return False
    for i in DISEASE_IDS:
        if cur["per_class"][i]["recall"] < base["per_class"][i]["recall"]-.02: return False
    return True
records={r["classifier"]:r for r in all_metrics if r["seed"]==DEPLOYMENT_SEED}
eligible=[r for r in records.values() if r["classifier"]=="A" or passes_guardrail(r,records["A"])]
rank={"A":0,"B":1,"C":2}
selected=min(eligible,key=lambda r:(r["disease_to_healthy_count"],-r["macro_f1"],r["log_loss"],rank[r["classifier"]]))
selected_type=selected["classifier"]
if selected_type!="A" and selected["disease_to_healthy_count"]>=records["A"]["disease_to_healthy_count"]:
    selected_type="A"; selected=records["A"]
lock_path=OUT/"selection/validation_selection_lock.json"
if lock_path.exists(): raise FileExistsError("Selection lock already exists")
lock={"experiment_id":EXPERIMENT_ID,"created_utc":pd.Timestamp.utcnow().isoformat(),
    "notebook_sha256":hashlib.sha256(NOTEBOOK_BYTES).hexdigest().upper(),"config_sha256":sha256_file(OUT/"experiment_config.json"),
    "dataset_archive_sha256":DATASET_SHA256,"manifest_sha256":SOURCE_MANIFEST_SHA256,
    "checkpoint_sha256":source_lock["checkpoint_sha256"],"deployment_seed":DEPLOYMENT_SEED,
    "selected_classifier":selected_type,"threshold":None,"selection_rule":"minimum D2H, guardrails, macro F1, log loss, simplicity; fallback A if no D2H improvement",
    "seed_123_validation_metrics":records,"all_seed_validation_metrics":all_metrics,
    "historical_holdout_not_for_selection":True,"v08_test_previously_exposed":True}
write_json(lock_path,lock)
write_json(OUT/"selection/selected_classifier_metrics.json",selected)
(OUT/"selection/selection_rationale.md").write_text(f"Selected **{selected_type}** for frozen seed 123. Raw Validation D2H {selected['disease_to_healthy_count']}/{selected['disease_total']}. This is selection-biased and needs fresh blind confirmation.\n",encoding="utf-8")
''')

code(21, "Train-only CV diagnostics, no pre-fit Val line", r'''
cv_df=pd.DataFrame(cv_records)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for typ,ax in [("B",axes[0]),("C",axes[1])]:
    sub=cv_df[cv_df.classifier==typ]
    axis="C" if typ=="B" else "candidate_id"
    agg=sub.groupby(axis).macro_f1.agg(["mean","std"])
    ax.errorbar(agg.index,agg["mean"],yerr=agg["std"],marker="o",color="#1d4ed8",capsize=3)
    ax.set(title=f"{typ}: Train-only 3-fold CV macro F1",xlabel=axis,ylabel="macro F1")
    ax.grid(alpha=.2)
    if typ=="B": ax.set_xscale("log")
fig.tight_layout(); fig.savefig(OUT/"figures/cv_model_selection.png",dpi=140); plt.close(fig)
''')

code(22, "White-blue confusion matrices", r'''
fig,axes=plt.subplots(5,3,figsize=(15,21),constrained_layout=True)
for row,seed in enumerate(SEEDS):
    for col,typ in enumerate(["A","B","C"]):
        rec=next(r for r in all_metrics if r["seed"]==seed and r["classifier"]==typ)
        cm=np.asarray(rec["confusion_matrix"]); ax=axes[row,col]; ax.imshow(cm,cmap=CMAP,vmin=0,vmax=max(1,cm.max()))
        for i in range(5):
            for j in range(5): ax.text(j,i,str(cm[i,j]),ha="center",va="center",color="white" if cm[i,j]>.68*cm.max() else "#0f172a",fontsize=8)
        ax.set(title=f"seed {seed} · {typ}",xticks=range(5),yticks=range(5),xticklabels=CLASSES,yticklabels=CLASSES)
        ax.tick_params(axis="x",labelrotation=50,labelsize=7); ax.tick_params(axis="y",labelsize=7)
fig.savefig(OUT/"figures/val_confusion_matrices_5seeds_abc.png",dpi=140); plt.close(fig)
''')

code(23, "Validation consistency and figures", r'''
for rec in all_metrics:
    cm=np.asarray(rec["confusion_matrix"])
    if cm.sum()!=1430 or not np.isclose(np.trace(cm)/1430,rec["accuracy"]): raise AssertionError("Validation metric inconsistency")
fig,ax=plt.subplots(figsize=(7,4))
for typ,group in metrics_df.groupby("classifier"):
    ax.plot(group.sort_values("seed").seed,group.sort_values("seed").disease_to_healthy_count,marker="o",label=typ)
ax.set(xlabel="DeiT seed",ylabel="Disease → Healthy count / 900",title="Validation D2H by classifier")
ax.legend(); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(OUT/"figures/d2h_counts_5seeds.png",dpi=140); plt.close(fig)
fig,ax=plt.subplots(figsize=(7,4))
for typ in ["A","B","C"]:
    p=all_probability[(123,typ)]; confidence=p.max(1); correct=(p.argmax(1)==y_val)
    xs=[]; ys=[]
    for lo,hi in zip(np.linspace(0,1,11)[:-1],np.linspace(0,1,11)[1:]):
        mask=(confidence>=lo)&(confidence<(hi if hi<1 else hi+1e-9))
        if mask.any(): xs.append(float(confidence[mask].mean())); ys.append(float(correct[mask].mean()))
    ax.plot(xs,ys,marker="o",label=typ)
ax.plot([0,1],[0,1],"--",color="#94a3b8",label="ideal")
ax.set(xlabel="Mean confidence",ylabel="Observed accuracy",title="Seed 123 Validation reliability (diagnostic)")
ax.grid(alpha=.2); ax.legend(); fig.tight_layout(); fig.savefig(OUT/"figures/calibration_curves.png",dpi=140); plt.close(fig)
''')

code(24, "Classifier and end-to-end timing", r'''
for seed in SEEDS:
    x=feature_by_seed[seed][1]
    for typ in ["B","C"]:
        model=classifier_models[(seed,typ)]
        t0=time.perf_counter(); _=model.predict_proba(x); ms=1000*(time.perf_counter()-t0)/len(x)
        timing_rows.append({"seed":seed,"classifier":typ,"classifier_predict_ms_per_image":ms,
            "feature_extraction_ms_per_image":next(r["val_extraction_ms_per_image"] for r in timing_rows if r["seed"]==seed and "val_extraction_ms_per_image" in r),
            "end_to_end_ms_per_image":ms+next(r["val_extraction_ms_per_image"] for r in timing_rows if r["seed"]==seed and "val_extraction_ms_per_image" in r)})
pd.DataFrame(timing_rows).to_csv(OUT/"timing.csv",index=False)
pd.DataFrame(training_time).to_csv(OUT/"classifier_cv_training_time.csv",index=False)
''')

code(25, "Traceable hard cases", r'''
selected_errors=errors_df[(errors_df.seed==123)&(errors_df.classifier==selected_type)]
selected_errors[selected_errors.disease_to_healthy].to_csv(OUT/"seed_123/recurrent_d2h.csv",index=False)
base=all_prediction[(123,"A")]; chosen=all_prediction[(123,selected_type)]
pd.DataFrame({"relative_path":val_rows.relative_path,"true_label":y_val,"A_pred":base,"selected_pred":chosen,
    "wrong_to_right":(base!=y_val)&(chosen==y_val),"right_to_wrong":(base==y_val)&(chosen!=y_val)}).to_csv(OUT/"seed_123/paired_error_changes.csv",index=False)
''')

code(26, "Locked retrospective gates", r'''
if not lock_path.is_file() or read_json(lock_path)["selected_classifier"]!=selected_type: raise RuntimeError("Selection lock invalid")
if RUN_FRESH_BLIND_TEST and FRESH_BLIND_MANIFEST is None: raise ValueError("Fresh blind manifest and provenance required")
def evaluate_locked(frame,root,tag):
    features,softmax,_,_,_=predict_seed(123,frame,root=root)
    probability=softmax if selected_type=="A" else classifier_models[(123,selected_type)].predict_proba(features)
    result=metric_record(frame.label,probability,123,selected_type)
    target=OUT/"evaluation"/tag; target.mkdir(parents=True,exist_ok=True)
    write_json(target/"selected_result.json",result)
    pd.DataFrame({"relative_path":frame.relative_path,"true_label":frame.label,"predicted_label":probability.argmax(1),
        **{f"p_{name}":probability[:,i] for i,name in enumerate(CLASSES)}}).to_csv(target/"predictions.csv",index=False)
    write_json(target/"evaluation_lock.json",{"selection_lock_sha256":sha256_file(lock_path),
        "status":"RETROSPECTIVE_PREVIOUSLY_EXPOSED" if tag!="fresh_blind_field_test" else "BLIND_PENDING_PROVENANCE_REVIEW"})
    return result
if RUN_V08_TEST:
    test_rows=manifest[manifest.split=="test"].reset_index(drop=True)
    evaluate_locked(test_rows,dataset_root,"retrospective_v08_test")
if RUN_HISTORICAL_HOLDOUT:
    holdout_zip=find_named("RL_Cashew_holdout.zip",HISTORICAL_HOLDOUT_OVERRIDE)
    if holdout_zip is None:
        if not Path("/content").exists(): raise FileNotFoundError("Set HISTORICAL_HOLDOUT_OVERRIDE")
        holdout_zip=drive_download(HISTORICAL_HOLDOUT_DRIVE_ID,work_root/"RL_Cashew_holdout.zip",150455582)
    if sha256_file(holdout_zip)!=HISTORICAL_HOLDOUT_SHA256: raise ValueError("Historical holdout ZIP SHA mismatch")
    holdout_extract=work_root/"historical_holdout"; holdout_root=holdout_extract/"RL_Cashew_holdout"
    if not holdout_root.exists():
        holdout_extract.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(holdout_zip) as z:
            for item in z.infolist():
                rel=Path(item.filename)
                if rel.is_absolute() or ".." in rel.parts: raise ValueError(item.filename)
            z.extractall(holdout_extract)
    expected_manifest=pd.read_csv(source_dir/"real_holdout/dataset_manifest.csv")
    expected_manifest=expected_manifest[["relative_path","class","label","sha256"]]
    if len(expected_manifest)!=49: raise ValueError("Historical holdout source manifest count changed")
    actual=[]
    for row in expected_manifest.itertuples(index=False):
        p=holdout_root/row.relative_path
        if not p.is_file(): raise FileNotFoundError(p)
        digest=sha256_file(p)
        if digest!=row.sha256: raise ValueError(f"Historical image changed: {row.relative_path}")
        actual.append(digest)
    duplicate_test=set(manifest.loc[manifest.split=="test","sha256"])&set(actual)
    if len(duplicate_test)!=1: raise ValueError(f"Expected exactly one previously known Test duplicate, found {len(duplicate_test)}")
    if set(actual)&set(manifest.loc[manifest.split.isin(["train","val"]),"sha256"]): raise ValueError("Historical holdout overlaps Train/Val bytes")
    holdout_rows=expected_manifest[~expected_manifest.sha256.isin(duplicate_test)].reset_index(drop=True)
    if len(holdout_rows)!=48: raise ValueError("Historical exclusion count mismatch")
    holdout_rows.to_csv(OUT/"evaluation/historical_holdout_48/verified_manifest.csv",index=False)
    evaluate_locked(holdout_rows,holdout_root,"historical_holdout_48")
if RUN_FRESH_BLIND_TEST:
    if FRESH_BLIND_ROOT is None: raise ValueError("FRESH_BLIND_ROOT is required")
    blind_root=Path(FRESH_BLIND_ROOT); blind=pd.read_csv(FRESH_BLIND_MANIFEST)
    required={"relative_path","class","label","sha256","leaf_group","source_id"}
    if not required.issubset(blind.columns) or blind.empty: raise ValueError("Fresh blind manifest lacks provenance columns")
    if set(blind["class"])!=set(CLASSES) or blind.relative_path.duplicated().any(): raise ValueError("Fresh blind classes/paths invalid")
    if blind[["leaf_group","source_id"]].isna().any().any(): raise ValueError("Fresh blind group/source provenance incomplete")
    if any(CLASSES[int(row.label)]!=row[1] for row in blind[["relative_path","class","label"]].itertuples(index=False)):
        raise ValueError("Fresh blind label map differs")
    for row in blind.itertuples(index=False):
        path=(blind_root/row.relative_path).resolve()
        if blind_root.resolve() not in path.parents or not path.is_file() or sha256_file(path)!=str(row.sha256).upper():
            raise ValueError(f"Fresh blind path/hash invalid: {row.relative_path}")
    if set(blind.sha256.str.upper())&set(manifest.sha256.str.upper()): raise ValueError("Fresh blind overlaps V08 bytes")
    blind_hashes=[dhash(blind_root/p) for p in blind.relative_path]
    if any((a^b).bit_count()<=3 for a in blind_hashes for b in hashes):
        raise ValueError("Fresh blind dHash shortlist overlaps V08; review source groups before calling blind")
    blind.to_csv(OUT/"evaluation/fresh_blind_field_test/verified_manifest.csv",index=False)
    evaluate_locked(blind,blind_root,"fresh_blind_field_test")
''')

code(27, "Model package and reload inference test", r'''
package=OUT/"model_package"
(package/"selected_deit_checkpoint.sha256").write_text(source_lock["checkpoint_sha256"]["123"]+"\n",encoding="utf-8")
write_json(package/"label_map.json",dict(enumerate(CLASSES)))
write_json(package/"preprocess_contract.json",read_json(OUT/"source/preprocess_contract.json"))
artifact=None
if selected_type=="B":
    artifact=package/"logistic_regression.joblib"; shutil.copy2(OUT/"seed_123/logistic_regression.joblib",artifact)
    reloaded=joblib.load(artifact)
elif selected_type=="C":
    artifact=package/"classifier_model.json"; shutil.copy2(OUT/"seed_123/xgboost.json",artifact)
    params=read_json(OUT/"seed_123/C_cv_selection.json")["chosen"]
    reloaded=xgb_model({k:params[k] for k in XGB_KEYS},123); reloaded.load_model(artifact)
if artifact:
    expected=all_probability[(123,selected_type)][:2]
    actual=reloaded.predict_proba(feature_by_seed[123][1][:2])
    if not np.allclose(actual,expected,atol=1e-5): raise AssertionError("Classifier reload parity failed")
write_json(package/"load_and_inference_test.json",{"status":"PASS","selected_classifier":selected_type,"n_probe":2})
write_json(package/"deployment_manifest.json",{"experiment_id":EXPERIMENT_ID,"seed":123,"model_type":selected_type,
    "source_checkpoint_sha256":source_lock["checkpoint_sha256"]["123"],"source_checkpoint_path":str(source_dir/"seed_123/best_model.keras"),
    "classifier_sha256":sha256_file(artifact) if artifact else None,"classes":CLASSES,"threshold":None,
    "source_checkpoint_in_full_zip":False,"full_zip_self_contained":False})
demo_model,demo_head,demo_extractor,_=load_extractor(123)
demo_input=decode_resize(str(dataset_root/val_rows.relative_path.iloc[0]))[None,...]
demo_features=demo_extractor(demo_input,training=False).numpy()
demo_p=(tf.nn.softmax(demo_head.output_dense(demo_features),axis=1).numpy() if selected_type=="A"
        else reloaded.predict_proba(demo_features))
demo_reference=all_probability[(123,selected_type)][:1]
demo_probability_error=float(np.max(np.abs(demo_p-demo_reference)))
if int(demo_p.argmax(1)[0])!=int(demo_reference.argmax(1)[0]):
    raise AssertionError(f"Manual RGB image predicted label differs from Validation; max probability drift {demo_probability_error:.6g}")
write_json(package/"representative_inference.json",{"relative_path":val_rows.relative_path.iloc[0],
    "predicted_label":CLASSES[int(demo_p.argmax(1)[0])],"probabilities":demo_p[0].tolist(),
    "single_vs_val_batch_probability_max_abs_error":demo_probability_error})
del demo_model,demo_head,demo_extractor; gc.collect(); keras.backend.clear_session()
''')

code(28, "README and model report", r'''
selected_metric=next(r for r in all_metrics if r["seed"]==123 and r["classifier"]==selected_type)
status={"status":"COMPLETED_VALIDATION","training_mode":"classifier_only","selected_classifier":selected_type,
    "deployment_seed":123,"v08_test":"RETROSPECTIVE_COMPLETED" if RUN_V08_TEST else "NOT_RUN",
    "historical_holdout":"RETROSPECTIVE_COMPLETED" if RUN_HISTORICAL_HOLDOUT else "NOT_RUN",
    "fresh_blind_field_test":"COMPLETED" if RUN_FRESH_BLIND_TEST else "NOT_RUN"}
write_json(OUT/"execution_status.json",status)
report=f"""# {EXPERIMENT_ID} model report

Source: immutable EXP-007 five-seed DeiT-tiny checkpoints, V08 dataset SHA `{DATASET_SHA256}`.
No DeiT training in EXP-008. Input RGB, bilinear 224×224, float32 0–255; saved DeiT preprocessing once. Features are `DeiTImageClassifier.output_dense.input` with observed dimension {probe_descriptor['feature_dimension']}.

Classifiers: A original softmax, B per-fold StandardScaler + LogisticRegression, C XGBoost `multi:softprob` with fixed 12-candidate Train-only 3-fold CV. Deployment seed 123; selected **{selected_type}** by locked Validation guardrails. Seed 123 raw Validation D2H **{selected_metric['disease_to_healthy_count']}/900**, macro F1 **{selected_metric['macro_f1']:.4f}**, Healthy recall **{selected_metric['healthy_recall']:.4f}**, Not Cashew recall **{selected_metric['not_cashew_leaf_recall']:.4f}**. See `validation_metrics_5seeds_abc.csv` and `validation_mean_std.csv` for all five seeds and classifiers. CV/training and image/feature/classifier timing are recorded in CSV.

Dataset exact-byte cross-split audit passed. Same-leaf, tree and capture-session leakage remains **GROUP_LEAKAGE_NOT_VERIFIED**. Validation is used for classifier selection, so selected Validation results are selection-biased. Historical Test/holdout have been previously exposed and are not used to tune. Fresh independent field test: **{status['fresh_blind_field_test']}**. No field improvement claim without its independent review.

Full ZIP stores verified checkpoint references, not the five large EXP-007 checkpoint binaries; it is **not self-contained**. Supply those immutable files for inference. Model package is experimental and does not modify application deployment.
"""
(OUT/"MODEL_REPORT.md").write_text(report,encoding="utf-8")
(OUT/"README.md").write_text(f"# {EXPERIMENT_ID}\n\nStatus: COMPLETED_VALIDATION. Selected seed 123 classifier: {selected_type}. Run notebook `{NOTEBOOK_FILENAME}` on Colab with V08 and EXP-007 source checkpoints. See MODEL_REPORT.md, selection lock, per-seed metrics and plots. Full ZIP is not self-contained because EXP-007 checkpoints are referenced by verified SHA. Fresh blind test: {status['fresh_blind_field_test']}.\n",encoding="utf-8")
''')

code(29, "Export verified FULL and ANALYSIS ZIPs with notebook", r'''
notebook_copy=OUT/"notebook"/NOTEBOOK_FILENAME
notebook_copy.parent.mkdir(parents=True,exist_ok=True); notebook_copy.write_bytes(NOTEBOOK_BYTES)
manifest_rows=[]
for p in sorted(OUT.rglob("*")):
    if not p.is_file() or "archives" in p.relative_to(OUT).parts or p.name=="artifact_sha256_manifest.csv": continue
    manifest_rows.append({"relative_path":p.relative_to(OUT).as_posix(),"bytes":p.stat().st_size,"sha256":sha256_file(p)})
pd.DataFrame(manifest_rows).to_csv(OUT/"artifact_sha256_manifest.csv",index=False)
def make_archive(kind):
    name=f"{EXPERIMENT_ID}_{kind}.zip"; target=OUT/"archives"/name
    with zipfile.ZipFile(target,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for p in sorted(OUT.rglob("*")):
            if not p.is_file() or "archives" in p.relative_to(OUT).parts: continue
            rel=p.relative_to(OUT)
            if "audit_contact_sheets" in rel.parts: continue  # dataset image pixels stay outside ZIP
            if kind=="ANALYSIS" and ("embeddings" in rel.parts or p.suffix in [".joblib",".keras"]): continue
            z.write(p,arcname=f"{EXPERIMENT_ID}/{rel.as_posix()}")
    with zipfile.ZipFile(target) as z:
        bad=z.testzip()
        if bad: raise IOError(f"ZIP CRC failed: {bad}")
        member=f"{EXPERIMENT_ID}/notebook/{NOTEBOOK_FILENAME}"
        if z.read(member)!=NOTEBOOK_BYTES: raise AssertionError("Embedded notebook differs")
    return {"path":str(target),"bytes":target.stat().st_size,"sha256":sha256_file(target)}
archives=[make_archive("FULL"),make_archive("ANALYSIS")]
write_json(OUT/"archives/archive_registry.json",archives)
print(json.dumps(archives,indent=2))
''')

nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
snapshot = json.dumps(nb, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
payload = base64.b64encode(zlib.compress(snapshot, 9)).decode("ascii")
source = "".join(nb["cells"][1]["source"])
marker = 'EMBEDDED_NOTEBOOK_B64 = "__NOTEBOOK_PAYLOAD__"'
assert marker in source
nb["cells"][1]["source"] = source.replace(marker, 'EMBEDDED_NOTEBOOK_B64 = "' + payload + '"', 1).splitlines(True)
TARGET.write_bytes((json.dumps(nb, ensure_ascii=False, indent=1) + "\n").encode("utf-8"))
print(TARGET, len(nb["cells"]), "cells")
