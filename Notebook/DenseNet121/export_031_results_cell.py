"""Archive EXP-031 results and upload the ZIP to the confirmed Drive folder."""

if RUN_FINAL_EVALUATION:
    if not (OUT_DIR / "exp031_summary.json").is_file():
        raise RuntimeError("EXP-031 final evaluation is incomplete; refusing to export results")
    result_files = sorted(path for path in OUT_DIR.rglob("*") if path.is_file())
    if not result_files:
        raise RuntimeError("EXP-031 has no result files")
    manifest = [{"path": path.relative_to(OUT_DIR).as_posix(),
                 "bytes": path.stat().st_size, "sha256": file_sha256(path)}
                for path in result_files]
    save_json(OUT_DIR / "artifact_manifest.json", manifest)
    result_zip = OUT_DIR.parent / f"{EXPERIMENT_ID}_RESULTS.zip"
    with zipfile.ZipFile(result_zip, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6, allowZip64=True) as archive:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                archive.write(path, f"{EXPERIMENT_ID}/{path.relative_to(OUT_DIR).as_posix()}")
    with zipfile.ZipFile(result_zip) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("EXP-031 result ZIP failed integrity check")
    print("Created:", result_zip, "bytes:", result_zip.stat().st_size,
          "SHA256:", file_sha256(result_zip))

    if UPLOAD_RESULT_ZIPS_TO_DRIVE:
        import google.auth
        from googleapiclient.discovery import build
        from googleapiclient.http import MediaFileUpload
        if Path("/content").is_dir():
            from google.colab import auth
            auth.authenticate_user()
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/drive"])
        service = build("drive", "v3", credentials=credentials, cache_discovery=False)
        folder = service.files().get(fileId=DRIVE_TARGET_FOLDER_ID,
                                     fields="id,name,mimeType,trashed",
                                     supportsAllDrives=True).execute()
        if (folder.get("mimeType") != "application/vnd.google-apps.folder" or
                folder.get("trashed")):
            raise RuntimeError(f"Invalid Drive ZIP destination: {DRIVE_TARGET_FOLDER_ID}")
        query = (f"name = '{result_zip.name}' and "
                 f"'{DRIVE_TARGET_FOLDER_ID}' in parents and trashed = false")
        found = service.files().list(q=query, fields="nextPageToken,files(id,name,size,md5Checksum)",
                                     pageSize=100, supportsAllDrives=True,
                                     includeItemsFromAllDrives=True).execute()
        matches = found.get("files", [])
        if found.get("nextPageToken") or len(matches) > 1:
            raise RuntimeError(f"Ambiguous Drive archive: {result_zip.name}")
        digest = hashlib.md5()
        with result_zip.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b""):
                digest.update(chunk)
        local_md5 = digest.hexdigest()
        if matches and int(matches[0].get("size", -1)) == result_zip.stat().st_size and \
                matches[0].get("md5Checksum") == local_md5:
            file_id = matches[0]["id"]
        else:
            media = MediaFileUpload(str(result_zip), mimetype="application/zip",
                                    resumable=True, chunksize=8 << 20)
            request = (service.files().update(fileId=matches[0]["id"], media_body=media,
                                              fields="id", supportsAllDrives=True) if matches else
                       service.files().create(body={"name": result_zip.name,
                                                    "parents": [DRIVE_TARGET_FOLDER_ID]},
                                              media_body=media, fields="id",
                                              supportsAllDrives=True))
            uploaded = None
            while uploaded is None:
                progress, uploaded = request.next_chunk(num_retries=3)
                if progress:
                    print(result_zip.name, f"{progress.progress():.0%}")
            file_id = uploaded["id"]
        remote = service.files().get(fileId=file_id,
            fields="id,name,size,md5Checksum,parents,webViewLink",
            supportsAllDrives=True).execute()
        if (remote["name"] != result_zip.name or
                int(remote["size"]) != result_zip.stat().st_size or
                remote.get("md5Checksum") != local_md5 or
                DRIVE_TARGET_FOLDER_ID not in remote.get("parents", [])):
            raise RuntimeError(f"Drive upload verification failed: {result_zip.name}")
        print("Verified Drive archive:", remote.get("webViewLink") or
              f"https://drive.google.com/file/d/{file_id}/view")
else:
    print("EXP-031 result ZIP pending until final evaluation")
