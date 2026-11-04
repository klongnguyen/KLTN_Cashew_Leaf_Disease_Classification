"""Stage the original EXP-029 and YOLO NEG V03 image archives for EXP-031."""

import stat
from collections import Counter

IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
STAGE_ROOT = PROJECT_ROOT / 'exp031_inputs'

def image_files(root):
    return sorted(path for path in Path(root).rglob('*')
                  if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

def archive_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def source_archive(file_id, name, expected_bytes, override):
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f'Archive override missing: {path}')
    else:
        path = STAGE_ROOT / 'archives' / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.is_file() or path.stat().st_size != expected_bytes or not zipfile.is_zipfile(path):
            if not file_id:
                raise FileNotFoundError(f'{name}: set a Drive file ID or archive override')
            import google.auth
            from googleapiclient.discovery import build
            from googleapiclient.http import MediaIoBaseDownload
            if Path('/content').is_dir():
                from google.colab import auth
                auth.authenticate_user()
            credentials, _ = google.auth.default(scopes=['https://www.googleapis.com/auth/drive.readonly'])
            service = build('drive', 'v3', credentials=credentials, cache_discovery=False)
            meta = service.files().get(fileId=file_id, fields='id,name,size,md5Checksum',
                                       supportsAllDrives=True).execute()
            if meta['name'] != name or int(meta['size']) != expected_bytes:
                raise ValueError(f'Drive source differs from expected {name}: {meta}')
            partial = path.with_suffix(path.suffix + '.part')
            request = service.files().get_media(fileId=file_id, supportsAllDrives=True)
            with partial.open('wb') as stream:
                downloader = MediaIoBaseDownload(stream, request, chunksize=8 << 20)
                done = False
                while not done:
                    progress, done = downloader.next_chunk(num_retries=3)
                    if progress:
                        print(name, f'{progress.progress():.0%}')
            if partial.stat().st_size != expected_bytes or not zipfile.is_zipfile(partial):
                raise RuntimeError(f'Incomplete archive download: {name}')
            if meta.get('md5Checksum'):
                digest = hashlib.md5()
                with partial.open('rb') as stream:
                    for block in iter(lambda: stream.read(8 << 20), b''):
                        digest.update(block)
                if digest.hexdigest() != meta['md5Checksum']:
                    raise RuntimeError(f'Drive MD5 mismatch: {name}')
            partial.replace(path)
    if path.stat().st_size != expected_bytes or not zipfile.is_zipfile(path):
        raise ValueError(f'Archive size/type mismatch: {path}')
    return path

def extract_verified(archive_path, destination):
    archive_path, destination = Path(archive_path), Path(destination)
    checksum = archive_sha256(archive_path)
    marker = destination / '.archive_sha256'
    if destination.is_dir():
        if not marker.is_file() or marker.read_text(encoding='utf-8').strip() != checksum:
            raise RuntimeError(f'Unverified/incomplete extraction: {destination}')
        return destination
    staging = destination.with_name(destination.name + '_staging')
    if staging.exists():
        raise RuntimeError(f'Incomplete staging directory: {staging}')
    staging.mkdir(parents=True)
    with zipfile.ZipFile(archive_path) as archive:
        resolved_root = staging.resolve()
        for member in archive.infolist():
            target = (staging / member.filename.replace('\\', '/')).resolve()
            if (stat.S_ISLNK(member.external_attr >> 16) or
                    not target.is_relative_to(resolved_root)):
                raise ValueError(f'Unsafe archive member: {member.filename}')
        archive.extractall(staging)
    (staging / '.archive_sha256').write_text(checksum + '\n', encoding='utf-8')
    staging.rename(destination)
    return destination

def unique_root(extracted, kind):
    candidates = [extracted, *[p for p in extracted.rglob('*') if p.is_dir()]]
    if kind == 'split13':
        matches = [p for p in candidates if (p / 'val').is_dir() and (p / 'test').is_dir()]
    elif kind == 'holdout09':
        holdout_classes = set(predictions_df.loc[predictions_df['split'] == 'holdout', 'true_label'])
        matches = [p for p in candidates if
                   {q.name for q in p.iterdir() if q.is_dir()} == holdout_classes]
    elif kind == 'yolo':
        matches = [p for p in candidates if (p / 'images' / 'train').is_dir() and
                   (p / 'labels' / 'train').is_dir()]
    else:
        raise ValueError(kind)
    if len(matches) != 1:
        raise RuntimeError(f'{kind}: expected one dataset root, found {matches}')
    return matches[0]

if not DETECTOR_WEIGHTS_PATH.is_file():
    raise FileNotFoundError(f'YOLO best.pt missing: {DETECTOR_WEIGHTS_PATH}; '
                            'set DETECTOR_WEIGHTS_OVERRIDE or place the verified FULL ZIP on this runtime')

if AUTO_STAGE_MISSING_INPUTS:
    if not Path(SPLIT13_VAL_DIR).is_dir() or not Path(SPLIT13_TEST_DIR).is_dir():
        archive = source_archive(DATASET_ARCHIVE_FILE_ID, DATASET_ARCHIVE_NAME,
                                 DATASET_ARCHIVE_EXPECTED_BYTES, DATASET_ARCHIVE_OVERRIDE)
        root = unique_root(extract_verified(archive, STAGE_ROOT / 'split13'), 'split13')
        SPLIT13_VAL_DIR, SPLIT13_TEST_DIR = root / 'val', root / 'test'
    if not Path(HOLDOUT_IMAGE_DIR).is_dir():
        archive = source_archive(REAL_HOLDOUT_ARCHIVE_FILE_ID, REAL_HOLDOUT_ARCHIVE_NAME,
                                 REAL_HOLDOUT_EXPECTED_BYTES, REAL_HOLDOUT_ARCHIVE_OVERRIDE)
        HOLDOUT_IMAGE_DIR = unique_root(
            extract_verified(archive, STAGE_ROOT / 'holdout09'), 'holdout09')
    if not Path(YOLO_TRAIN_IMAGE_DIR).is_dir():
        archive = source_archive(YOLO_DATASET_FILE_ID, YOLO_DATASET_ARCHIVE_NAME,
                                 YOLO_DATASET_EXPECTED_BYTES, YOLO_DATASET_OVERRIDE)
        yolo_root = unique_root(extract_verified(archive, STAGE_ROOT / 'yolo'), 'yolo')
        YOLO_TRAIN_IMAGE_DIR = yolo_root / 'images' / 'train'
        YOLO_VAL_IMAGE_DIR = yolo_root / 'images' / 'val'
        YOLO_VAL_LABEL_DIR = yolo_root / 'labels' / 'val'

for split, folder, expected_count in (
        ('validation', SPLIT13_VAL_DIR, 1452),
        ('test', SPLIT13_TEST_DIR, 685),
        ('holdout', HOLDOUT_IMAGE_DIR, 126)):
    files = image_files(folder) if Path(folder).is_dir() else []
    if len(files) != expected_count:
        raise FileNotFoundError(f'{split}: expected {expected_count} images under {folder}; found {len(files)}')
    expected_paths = Counter(predictions_df.loc[
        (predictions_df['split'] == split) & (predictions_df['seed'] == SEEDS[0]), 'image_relpath'])
    actual_paths = Counter(path.relative_to(folder).as_posix() for path in files)
    if actual_paths != expected_paths:
        raise ValueError(f'{split}: relative image paths do not match frozen EXP-029 predictions')
    print(f'  staged {split}: {len(files)} images at {folder}')
train_images = image_files(YOLO_TRAIN_IMAGE_DIR) if Path(YOLO_TRAIN_IMAGE_DIR).is_dir() else []
if not train_images:
    raise FileNotFoundError(f'YOLO training images missing: {YOLO_TRAIN_IMAGE_DIR}')
print(f'  staged YOLO train: {len(train_images)} images at {YOLO_TRAIN_IMAGE_DIR}')
