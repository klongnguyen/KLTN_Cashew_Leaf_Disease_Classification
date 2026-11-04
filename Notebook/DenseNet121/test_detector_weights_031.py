"""Network-free test of EXP-031's pinned checkpoint download and SHA checks."""

import hashlib
import tempfile
import zipfile
from pathlib import Path


source = (Path(__file__).resolve().parent / 'detector_weights_031.py').read_text(encoding='utf-8')
namespace = {'Path': Path, 'hashlib': hashlib, 'zipfile': zipfile,
             'file_sha256': lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()}
exec(compile(source, 'detector-weights-helper', 'exec'), namespace)
ensure = namespace['ensure_detector_weights']

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    archive_path = root / 'source.zip'
    checkpoint = b'synthetic-checkpoint'
    with zipfile.ZipFile(archive_path, 'w') as archive:
        archive.writestr('train/weights/best.pt', checkpoint)
    expected_archive_hash = namespace['file_sha256'](archive_path)
    expected_weights_hash = hashlib.sha256(checkpoint).hexdigest()
    runtime = root / 'runtime'
    runtime.mkdir()
    args = (runtime, None, 'FULL.zip', archive_path.as_uri(), archive_path.stat().st_size,
            expected_archive_hash, 'train/weights/best.pt', expected_weights_hash)
    weights = ensure(*args)
    assert weights.read_bytes() == checkpoint
    assert ensure(*args) == weights  # reuse verified local checkpoint
    weights.write_bytes(b'tampered')
    try:
        ensure(*args)
    except ValueError as exc:
        assert 'SHA256 mismatch' in str(exc)
    else:
        raise AssertionError('Tampered checkpoint must fail')
print('EXP-031 checkpoint helper: pinned download, reuse and SHA checks passed.')
