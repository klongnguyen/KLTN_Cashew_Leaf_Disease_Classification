"""Local, network-free checks of EXP-031 archive and path staging helpers."""

import ast
import hashlib
import stat
import tempfile
import zipfile
from pathlib import Path


source = (Path(__file__).resolve().parent / 'stage_exp031_images_cell.py').read_text(encoding='utf-8')
tree = ast.parse(source)
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
module = ast.fix_missing_locations(ast.Module(body=functions, type_ignores=[]))
namespace = {'Path': Path, 'hashlib': hashlib, 'stat': stat, 'zipfile': zipfile}
namespace['IMAGE_SUFFIXES'] = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
exec(compile(module, 'stage-helpers', 'exec'), namespace)

with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    namespace['STAGE_ROOT'] = root
    valid = root / 'split13.zip'
    with zipfile.ZipFile(valid, 'w') as archive:
        archive.writestr('CashewData_Split/train/a.jpg', b'a')
        archive.writestr('CashewData_Split/val/b.jpg', b'b')
        archive.writestr('CashewData_Split/test/c.jpg', b'c')
    chosen = namespace['source_archive'](None, valid.name, valid.stat().st_size, valid)
    assert chosen == valid
    extraction = namespace['extract_verified'](valid, root / 'extracted')
    assert namespace['extract_verified'](valid, extraction) == extraction
    split = namespace['unique_root'](extraction, 'split13')
    assert len(namespace['image_files'](split / 'val')) == 1
    assert len(namespace['image_files'](split / 'test')) == 1

    unsafe = root / 'unsafe.zip'
    with zipfile.ZipFile(unsafe, 'w') as archive:
        archive.writestr('../outside.txt', b'unsafe')
    try:
        namespace['extract_verified'](unsafe, root / 'unsafe_extract')
    except ValueError as exc:
        assert 'Unsafe archive member' in str(exc)
    else:
        raise AssertionError('Path-traversal ZIP must be rejected')
print('EXP-031 image staging helpers: archive size, extraction, root and traversal checks passed.')
