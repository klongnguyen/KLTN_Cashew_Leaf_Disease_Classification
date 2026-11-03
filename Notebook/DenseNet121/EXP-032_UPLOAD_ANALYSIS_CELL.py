# EXP-032 — upload an ANALYSIS snapshot now; independent of the final export cell.
# Run after Part A/B tables exist, even if the pipeline consistency cell failed.
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

from google.colab import auth
import google.auth
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

TARGET_FOLDER_ID = "1DGKTnzsvVc60RIEn5lQgflR3s-loM24G"
if "OUT_DIR" in globals():
    source_root = Path(OUT_DIR)
else:
    candidates = [p for p in Path("/content/training_results/current_dataset").glob(
        "EXP-DENSENET121-YOLO-PIPELINE-032*") if p.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Set OUT_DIR to the intended EXP-032 result folder: {candidates}")
    source_root = candidates[0]
aggregate = source_root / "aggregate"
if not aggregate.is_dir():
    raise FileNotFoundError(f"No aggregate results in {source_root}")

def checksum(path, algorithm):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

files = sorted(p for p in source_root.rglob("*") if p.is_file() and (
    (p.is_relative_to(aggregate) and p.suffix.lower() not in {".joblib", ".keras", ".pt", ".h5", ".zip"})
    or (p.parent.name.startswith("seed_") and
        (p.name == "seed_summary.json" or p.name.endswith("_predictions.csv")))))
if not any(p.name.startswith("fusion_results_part") for p in files):
    raise RuntimeError("No fusion result table yet; run Part A or Part B first")
manifest = {
    "experiment": source_root.name,
    "kind": "analysis_snapshot",
    "status": "snapshot only; pipeline consistency is not certified by this export",
    "files": [{"path": p.relative_to(source_root).as_posix(),
               "bytes": p.stat().st_size, "sha256": checksum(p, "sha256")}
              for p in files],
}
manifest_bytes = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode("utf-8")
tag = hashlib.sha256(manifest_bytes).hexdigest()[:16]
archive_path = source_root.parent / f"{source_root.name}_ANALYSIS_SNAPSHOT_{tag}.zip"

with tempfile.NamedTemporaryFile(dir=source_root.parent, suffix=".zip.part", delete=False) as handle:
    temporary = Path(handle.name)
try:
    with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in files:
            name = f"{source_root.name}/{path.relative_to(source_root).as_posix()}"
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with path.open("rb") as src, archive.open(info, "w") as dst:
                shutil.copyfileobj(src, dst, length=8 << 20)
        info = zipfile.ZipInfo(f"{source_root.name}/analysis_export_manifest.json",
                               date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, manifest_bytes)
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ANALYSIS snapshot ZIP failed CRC check")
        for item in manifest["files"]:
            name = f"{source_root.name}/{item['path']}"
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for block in iter(lambda: stream.read(8 << 20), b""):
                    digest.update(block)
            if digest.hexdigest() != item["sha256"]:
                raise RuntimeError(f"Snapshot member SHA256 mismatch: {name}")
    if archive_path.exists():
        if checksum(archive_path, "sha256") != checksum(temporary, "sha256"):
            raise RuntimeError(f"Existing local snapshot differs: {archive_path}")
    else:
        temporary.replace(archive_path)
finally:
    temporary.unlink(missing_ok=True)

auth.authenticate_user()
credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
service = build("drive", "v3", credentials=credentials, cache_discovery=False)
folder = service.files().get(fileId=TARGET_FOLDER_ID,
    fields="id,mimeType,trashed", supportsAllDrives=True).execute()
if folder.get("mimeType") != "application/vnd.google-apps.folder" or folder.get("trashed"):
    raise RuntimeError("Invalid Drive target folder")
query = f"name = '{archive_path.name}' and '{TARGET_FOLDER_ID}' in parents and trashed = false"
listed = service.files().list(q=query, fields="nextPageToken,files(id,name,size,md5Checksum)",
    pageSize=100, supportsAllDrives=True, includeItemsFromAllDrives=True).execute()
matches = listed.get("files") or []
if listed.get("nextPageToken") or len(matches) > 1:
    raise RuntimeError(f"Ambiguous Drive archive: {archive_path.name}")
local_md5 = checksum(archive_path, "md5")
if matches:
    if (int(matches[0].get("size", -1)) != archive_path.stat().st_size or
            matches[0].get("md5Checksum") != local_md5):
        raise RuntimeError("A different Drive file already has this snapshot name; refusing overwrite")
    file_id = matches[0]["id"]
else:
    media = MediaFileUpload(str(archive_path), mimetype="application/zip",
                            resumable=True, chunksize=8 << 20)
    request = service.files().create(body={"name": archive_path.name,
        "parents": [TARGET_FOLDER_ID]}, media_body=media,
        fields="id", supportsAllDrives=True)
    uploaded = None
    while uploaded is None:
        progress, uploaded = request.next_chunk(num_retries=3)
        if progress:
            print(f"Upload {progress.progress():.0%}")
    file_id = uploaded["id"]
remote = service.files().get(fileId=file_id,
    fields="id,name,size,md5Checksum,parents,webViewLink",
    supportsAllDrives=True).execute()
if (remote["name"] != archive_path.name or
        int(remote["size"]) != archive_path.stat().st_size or
        remote.get("md5Checksum") != local_md5 or
        TARGET_FOLDER_ID not in remote.get("parents", [])):
    raise RuntimeError("Drive upload verification failed")
print("ANALYSIS snapshot uploaded:", remote.get("webViewLink") or file_id)
print("Drive file ID:", file_id, "bytes:", remote["size"],
      "SHA256:", checksum(archive_path, "sha256"))
print("This is an analysis snapshot; it does not certify the deployable pipeline.")
