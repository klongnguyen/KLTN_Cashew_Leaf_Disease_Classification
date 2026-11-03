"""Reconstruct the byte-identical EXP-029 FULL ZIP from its Drive parts.

Download all *_FULL_part??.zip files into one folder, then run:
    python restore_exp029_full.py path/to/folder
"""

import hashlib
import json
import sys
import zipfile
from pathlib import Path


def restore(folder):
    folder = Path(folder)
    part_zips = sorted(folder.glob('*_FULL_part??.zip'))
    if not part_zips:
        raise FileNotFoundError('No *_FULL_part??.zip files found')
    with zipfile.ZipFile(part_zips[0]) as first:
        manifest = json.loads(first.read('manifest.json'))
    if len(part_zips) != len(manifest['parts']):
        raise FileNotFoundError(f'Expected {len(manifest["parts"])} FULL ZIP parts; found {len(part_zips)}')
    output = folder / manifest['original_name']
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    overall = hashlib.sha256()
    written = 0
    try:
        with output.open('wb') as target:
            for archive_path, part in zip(part_zips, manifest['parts']):
                if archive_path.name != part['zip_name']:
                    raise ValueError('Part ZIP order/name differs from manifest')
                digest = hashlib.sha256()
                with zipfile.ZipFile(archive_path) as archive:
                    if json.loads(archive.read('manifest.json')) != manifest:
                        raise ValueError(f'Manifest mismatch: {archive_path.name}')
                    with archive.open(part['member_name']) as stream:
                        for block in iter(lambda: stream.read(8 << 20), b''):
                            target.write(block)
                            overall.update(block)
                            digest.update(block)
                            written += len(block)
                if digest.hexdigest() != part['sha256']:
                    raise ValueError(f'Chunk SHA256 mismatch: {archive_path.name}')
        if written != manifest['original_bytes'] or overall.hexdigest() != manifest['original_sha256']:
            raise ValueError('Reconstructed FULL ZIP size/SHA256 mismatch')
        with zipfile.ZipFile(output) as archive:
            if archive.testzip() is not None:
                raise ValueError('Reconstructed FULL ZIP is corrupt')
    except Exception:
        output.unlink(missing_ok=True)
        raise
    print(output, 'bytes', written, 'SHA256', overall.hexdigest())
    return output


if __name__ == '__main__':
    if len(sys.argv) != 2:
        raise SystemExit('Usage: python restore_exp029_full.py FOLDER_WITH_BOTH_PART_ZIPS')
    restore(sys.argv[1])
