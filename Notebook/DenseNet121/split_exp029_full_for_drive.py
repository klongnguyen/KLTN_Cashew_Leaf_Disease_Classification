"""Split the EXP-029 FULL ZIP into verified parts below the 100 MiB tool cap."""

import hashlib
import json
import zipfile
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = Path('D:/Study/Training result dowload/deep model/'
              'EXP-DENSENET121-TRANSFER-5SEEDS-029-256px-9c23ac05_FULL.zip')
OUTPUT = HERE.parent.parent / 'training_results' / 'current_dataset' / 'exp029_full_drive_parts'
RESTORE = (HERE / 'restore_exp029_full.py').read_bytes()
PART_COUNT = 8


def digest_stream(stream, limit=None):
    digest = hashlib.sha256()
    size = 0
    while limit is None or size < limit:
        block = stream.read(min(8 << 20, limit - size) if limit is not None else 8 << 20)
        if not block:
            break
        digest.update(block)
        size += len(block)
    return size, digest.hexdigest()


if not SOURCE.is_file():
    raise FileNotFoundError(SOURCE)
size = SOURCE.stat().st_size
part_sizes = [size // PART_COUNT + (i < size % PART_COUNT) for i in range(PART_COUNT)]
part_names = [f'{SOURCE.stem}_part{i:02d}.zip' for i in range(1, PART_COUNT + 1)]
member_names = [f'{SOURCE.name}.part{i:02d}' for i in range(1, PART_COUNT + 1)]
with SOURCE.open('rb') as stream:
    measured = [digest_stream(stream, chunk_size) for chunk_size in part_sizes]
if [n for n, _ in measured] != part_sizes:
    raise RuntimeError('Could not read all FULL ZIP bytes')
whole = hashlib.sha256()
with SOURCE.open('rb') as stream:
    for block in iter(lambda: stream.read(8 << 20), b''):
        whole.update(block)
manifest = {
    'original_name': SOURCE.name, 'original_bytes': size,
    'original_sha256': whole.hexdigest(),
    'parts': [{'zip_name': zip_name, 'member_name': member_name,
               'bytes': count, 'sha256': digest}
              for zip_name, member_name, (count, digest) in
              zip(part_names, member_names, measured)],
}
OUTPUT.mkdir(parents=True, exist_ok=True)
with SOURCE.open('rb') as stream:
    for part in manifest['parts']:
        destination = OUTPUT / part['zip_name']
        with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_STORED,
                             allowZip64=True) as archive:
            with archive.open(part['member_name'], 'w', force_zip64=True) as target:
                remaining = part['bytes']
                while remaining:
                    block = stream.read(min(8 << 20, remaining))
                    if not block:
                        raise RuntimeError('Unexpected EOF while splitting FULL ZIP')
                    target.write(block)
                    remaining -= len(block)
            archive.writestr('manifest.json', json.dumps(manifest, indent=2))
            archive.writestr('restore_exp029_full.py', RESTORE)
        if destination.stat().st_size >= 100 * 1024 * 1024:
            raise RuntimeError(f'Drive part is still over 100 MiB: {destination}')
        with zipfile.ZipFile(destination) as archive:
            if archive.testzip() is not None:
                raise RuntimeError(f'Corrupt Drive part: {destination}')
        print(destination, 'bytes', destination.stat().st_size)
print('Original SHA256:', manifest['original_sha256'])
