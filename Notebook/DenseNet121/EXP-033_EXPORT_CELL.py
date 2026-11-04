# EXP-033 final audit, immutable FULL/ANALYSIS ZIPs, and verified Drive upload.
if not (AGGREGATE / "pipeline_consistency.json").is_file():
    raise RuntimeError("Pipeline consistency gate has not passed")
consistency = json.loads((AGGREGATE / "pipeline_consistency.json").read_text(encoding="utf-8"))
if consistency.get("status") != "PASS" or consistency.get("holdout_images", 0) < min(20, len(INVENTORY["holdout"])):
    raise RuntimeError("Pipeline consistency report is incomplete")

if USE_EXTRA_BLIND_SET:
    if not EXTRA_BLIND_DRIVE_FOLDER_ID.strip():
        raise ValueError("Set EXTRA_BLIND_DRIVE_FOLDER_ID for the optional blind set")
    from googleapiclient.http import MediaIoBaseDownload
    blind_service = drive_client()
    blind_root = STAGE_ROOT / "blind10"
    blind_root.mkdir(parents=True, exist_ok=True)
    def copy_folder(folder_id, target):
        page = None
        while True:
            response = blind_service.files().list(
                q=f"'{folder_id}' in parents and trashed = false",
                fields="nextPageToken,files(id,name,mimeType,size)", pageToken=page,
                pageSize=1000, supportsAllDrives=True,
                includeItemsFromAllDrives=True).execute()
            for item in response["files"]:
                dest = target / item["name"]
                assert dest.resolve().is_relative_to(blind_root.resolve())
                if item["mimeType"] == "application/vnd.google-apps.folder":
                    dest.mkdir(exist_ok=True)
                    copy_folder(item["id"], dest)
                else:
                    request = blind_service.files().get_media(fileId=item["id"], supportsAllDrives=True)
                    with dest.open("wb") as stream:
                        download = MediaIoBaseDownload(stream, request, chunksize=8 << 20)
                        done = False
                        while not done:
                            _, done = download.next_chunk(num_retries=3)
                    assert dest.stat().st_size == int(item["size"])
            page = response.get("nextPageToken")
            if not page:
                break
    copy_folder(EXTRA_BLIND_DRIVE_FOLDER_ID, blind_root)
    blind_rows = [{"key": p.relative_to(blind_root).as_posix(),
                   "true": p.parent.name, "pred": pipeline.predict(p)["label"]}
                  for p in image_files(blind_root)]
    pd.DataFrame(blind_rows).to_csv(AGGREGATE / "blind_predictions.csv", index=False)
    blind_acc = np.mean([r["true"] == r["pred"] for r in blind_rows]) if blind_rows else np.nan
    (AGGREGATE / "blind_report.md").write_text(
        f"# Frozen optional blind set\n\nN={len(blind_rows)}, accuracy={blind_acc:.4f}. "
        "No selection or threshold change.\n", encoding="utf-8")

required = ["final_test.csv", "real_holdout_n126.csv", "real_holdout_n124.csv",
            "validation_grouped_oof.csv", "paired_tests.csv", "traffic_light.csv",
            "localization_report.csv", "pipeline_consistency.json", "README.md",
            "frozen_config.json", "validation_selection_lock.json"]
for name in required:
    if not (AGGREGATE / name).is_file():
        raise FileNotFoundError(AGGREGATE / name)
assert len(list((AGGREGATE / "figures").glob("*.png"))) >= 16
assert (OUT_DIR / "cashew_pipeline.py").is_file()
assert all((OUT_DIR / "pipeline_bundle" / f"seed_{seed}" / "deployment_model.keras").is_file()
           for seed in SEEDS)

analysis_files = sorted([p for p in AGGREGATE.rglob("*") if p.is_file()] +
    [OUT_DIR / "cashew_pipeline.py", OUT_DIR / "pipeline_bundle" / "frozen_config.json",
     OUT_DIR / "pipeline_bundle" / "decision.json",
     OUT_DIR / "pipeline_bundle" / "artifact_sha256.csv"] +
    [p for seed in SEEDS for p in (OUT_DIR / f"seed_{seed}").glob("*_predictions.csv")] +
    [OUT_DIR / f"seed_{seed}" / "seed_summary.json" for seed in SEEDS])
analysis_manifest = [{"path": p.relative_to(OUT_DIR).as_posix(), "bytes": p.stat().st_size,
                      "sha256": file_sha256(p)} for p in analysis_files]
write_json(AGGREGATE / "analysis_manifest.json", analysis_manifest)
analysis_files = sorted(set(analysis_files + [AGGREGATE / "analysis_manifest.json"]))
full_files = sorted(p for p in OUT_DIR.rglob("*") if p.is_file())

def make_archive(files, suffix):
    import tempfile
    destination = OUT_DIR.parent / f"{OUT_DIR.name}_{suffix}.zip"
    with tempfile.NamedTemporaryFile(dir=OUT_DIR.parent, suffix=".zip.part", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in files:
                member = f"{OUT_DIR.name}/{path.relative_to(OUT_DIR).as_posix()}"
                info = zipfile.ZipInfo(member, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                with path.open("rb") as source, archive.open(info, "w") as target:
                    shutil.copyfileobj(source, target, length=8 << 20)
        with zipfile.ZipFile(temporary) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f"{suffix} ZIP CRC verification failed")
            names = set(archive.namelist())
            assert len(names) == len(files)
            assert all(f"{OUT_DIR.name}/{p.relative_to(OUT_DIR).as_posix()}" in names for p in files)
        if destination.exists():
            if file_sha256(destination) != file_sha256(temporary):
                raise RuntimeError(f"Existing {suffix} ZIP differs; do not overwrite: {destination}")
        else:
            temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print(suffix, "ZIP", destination, "bytes", destination.stat().st_size,
          "SHA256", file_sha256(destination))
    return destination

analysis_zip = make_archive(analysis_files, "ANALYSIS")
full_zip = make_archive(full_files, "FULL")
if UPLOAD_RESULT_ZIPS_TO_DRIVE:
    upload_service = drive_client()
    for path in (analysis_zip, full_zip):
        file_id = upload_immutable(upload_service, path)
        remote = upload_service.files().get(fileId=file_id,
            fields="id,name,size,md5Checksum,parents,webViewLink",
            supportsAllDrives=True).execute()
        if (remote["name"] != path.name or int(remote["size"]) != path.stat().st_size or
                remote.get("md5Checksum") != file_md5(path) or
                DRIVE_TARGET_FOLDER_ID not in remote.get("parents", [])):
            raise RuntimeError(f"Drive verification failed: {path.name}")
        print("Drive verified:", remote.get("webViewLink") or
              f"https://drive.google.com/file/d/{file_id}/view")
else:
    print("Drive upload disabled; ZIPs remain local only")
