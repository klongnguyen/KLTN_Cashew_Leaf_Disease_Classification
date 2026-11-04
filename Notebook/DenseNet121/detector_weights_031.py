"""Retrieve and verify the frozen YOLO26s NEG V03 001 checkpoint for EXP-031."""


def ensure_detector_weights(project_root, override, archive_name, archive_url,
                            archive_bytes, archive_sha256, weights_member,
                            weights_sha256):
    """Return a verified best.pt, reusing local files before downloading."""
    project_root = Path(project_root)
    weights = Path(override).expanduser() if override else project_root / 'detector' / 'best.pt'
    if weights.is_file():
        found = file_sha256(weights)
        if found != weights_sha256:
            raise ValueError(f'YOLO best.pt SHA256 mismatch: {found}')
        print('Verified YOLO best.pt:', weights, found[:16])
        return weights
    if override:
        raise FileNotFoundError(f'DETECTOR_WEIGHTS_OVERRIDE does not exist: {weights}')

    candidates = sorted(project_root.rglob(archive_name))
    if len(candidates) > 1:
        raise RuntimeError(f'Multiple YOLO FULL ZIPs found; select one explicitly: {candidates}')
    if candidates:
        archive_path = candidates[0]
        source = 'local archive'
    else:
        import urllib.request
        archive_path = project_root / 'detector' / archive_name
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        partial = archive_path.with_suffix('.zip.part')
        request = urllib.request.Request(archive_url,
                                         headers={'User-Agent': 'EXP-031-checkpoint-provenance'})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as target:
                while True:
                    block = response.read(8 << 20)
                    if not block:
                        break
                    target.write(block)
            if partial.stat().st_size != archive_bytes:
                raise ValueError(f'Incomplete YOLO FULL ZIP: {partial.stat().st_size} bytes')
            actual = file_sha256(partial)
            if actual != archive_sha256:
                raise ValueError(f'YOLO FULL ZIP SHA256 mismatch: {actual}')
            partial.replace(archive_path)
        except Exception as exc:
            partial.unlink(missing_ok=True)
            raise RuntimeError(
                'Could not fetch the pinned YOLO FULL ZIP from GitHub. '
                'Place the verified ZIP on the runtime or set DETECTOR_WEIGHTS_OVERRIDE.'
            ) from exc
        source = 'pinned GitHub commit'

    if archive_path.stat().st_size != archive_bytes:
        raise ValueError(f'YOLO FULL ZIP size mismatch: {archive_path}')
    found_zip = file_sha256(archive_path)
    if found_zip != archive_sha256:
        raise ValueError(f'YOLO FULL ZIP SHA256 mismatch: {found_zip}')
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError(f'Corrupt YOLO FULL ZIP: {archive_path}')
        checkpoint = archive.read(weights_member)
    found_weights = hashlib.sha256(checkpoint).hexdigest()
    if found_weights != weights_sha256:
        raise ValueError(f'YOLO best.pt SHA256 mismatch: {found_weights}')
    weights.parent.mkdir(parents=True, exist_ok=True)
    temp_weights = weights.with_suffix('.pt.part')
    temp_weights.write_bytes(checkpoint)
    temp_weights.replace(weights)
    print('Verified YOLO best.pt from', source, '|', archive_path,
          '| SHA256', found_weights[:16])
    return weights
