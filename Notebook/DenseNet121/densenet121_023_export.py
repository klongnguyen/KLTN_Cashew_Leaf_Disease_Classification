# Create archives only after all five seeds and the Validation lock are complete.
if RUN_FULL_TRAINING or RUN_FINAL_TEST or RUN_REAL_HOLDOUT:
    assert_selection_lock()
    test_status = "EXPOSED_HISTORICAL_TEST" if (AGGREGATE_DIR / "test_results_5seeds.csv").is_file() else "PENDING"
    holdout_status = "EXPOSED_HOLDOUT_04" if (OUT_DIR / "real_holdout" / "evaluation_completed.json").is_file() else "PENDING"
    save_json(OUT_DIR / "execution_status.json", {
        "experiment_id": EXPERIMENT_ID, "training": "COMPLETE", "validation": "COMPLETE",
        "test": test_status, "real_holdout": holdout_status,
        "dataset_zip_sha256": DATASET_ZIP_SHA256,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "deployment_seed": deployment_seed, "selection_lock_sha256": file_sha256(lock_path)})
    (OUT_DIR / "README.md").write_text(
        f"# {EXPERIMENT_ID}\n\n"
        "DenseNet121 + RandomBrightness(0.08) versus historical EXP-022 on frozen V10.\n\n"
        f"Deployment seed selected by Validation loss: {deployment_seed}.\n"
        f"Test: {test_status}; Holdout (4): {holdout_status}. Historical evaluation cannot tune this recipe.\n"
        "See validation_decision_note.json. No automatic promotion or WebApp replacement.\n"
        "Validation has only 3 real_ Healthy images. Paired-seed CI is not a field-population CI.\n",
        encoding="utf-8")
    notebook_dir = OUT_DIR / "notebook"
    notebook_dir.mkdir(exist_ok=True)
    notebook_copy = notebook_dir / "densenet121_transfer_photometric_5seeds_023.ipynb"
    if NOTEBOOK_SOURCE_PATH is not None:
        shutil.copyfile(NOTEBOOK_SOURCE_PATH, notebook_copy)
    elif NOTEBOOK_SOURCE_ZLIB_B64:
        source_book = json.loads(zlib.decompress(base64.b64decode(NOTEBOOK_SOURCE_ZLIB_B64)))
        notebook_copy.write_text(json.dumps(source_book, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        raise RuntimeError("Set NOTEBOOK_SOURCE_PATH to this notebook when rerunning an archived copy")
    save_json(OUT_DIR / "code" / "source_notebook.json", {
        "path": str(notebook_copy), "sha256": file_sha256(notebook_copy)})
    pd.DataFrame([
        {"relative_path": path.relative_to(OUT_DIR).as_posix(),
         "sha256": file_sha256(path), "bytes": path.stat().st_size}
        for path in sorted(OUT_DIR.rglob("*"))
        if path.is_file() and path.name != "artifact_sha256.csv"
    ]).to_csv(OUT_DIR / "artifact_sha256.csv", index=False)
    full_zip = RESULTS_ROOT / f"{EXPERIMENT_ID}_FULL.zip"
    analysis_zip = RESULTS_ROOT / f"{EXPERIMENT_ID}_ANALYSIS.zip"
    for destination, analysis_only in ((full_zip, False), (analysis_zip, True)):
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for path in sorted(OUT_DIR.rglob("*")):
                if not path.is_file(): continue
                if analysis_only and path.suffix.lower() in {".keras", ".h5"}: continue
                archive.write(path, arcname=f"{EXPERIMENT_ID}/{path.relative_to(OUT_DIR).as_posix()}")
        print(destination, destination.stat().st_size, "SHA256", file_sha256(destination))
else:
    print("FULL/ANALYSIS ZIP PENDING until full training; no COMPLETE claim")
