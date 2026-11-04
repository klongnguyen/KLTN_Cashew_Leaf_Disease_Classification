"""Notebook cell source: audit negative images through Ultralytics datasets and loaders."""

from ultralytics.cfg import IterableSimpleNamespace
from ultralytics.data import build_dataloader, build_yolo_dataset

PREFLIGHT_PATH = RUN_DIR / 'metadata' / 'negative_image_preflight.json'
PREFLIGHT_PATH.parent.mkdir(parents=True, exist_ok=True)
preflight = {
    'dataset': str(dataset_root),
    'data_yaml': str(DATA_YAML),
    'ultralytics_version': ultralytics.__version__,
    'class_names': CLASS_NAMES,
    'splits': {},
    'cross_split_exact_duplicates': [],
    'cross_split_same_names': [],
    'cross_split_perceptual_candidates': [],
    'errors': [],
    'warnings': [],
}

def negative_group(path):
    # Only the newly added *_neg_* images are negative by naming convention.
    # A legacy image such as healthy_1520.jpg may contain genuine disease GT.
    name = Path(path).stem.lower().replace('-', '_').replace(' ', '_')
    if name.startswith(('not_cashew_neg_', 'non_cashew_neg_')):
        return 'not_cashew_leaf'
    if name.startswith('healthy_neg_'):
        return 'healthy'
    return None

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

all_image_hashes = defaultdict(list)
all_image_names = defaultdict(list)
all_perceptual_hashes = defaultdict(list)
for split in ('train', 'val', 'test'):
    image_dir, label_dir = SPLITS[split]
    # Counter omits keys that never increment (for example Test can have 0
    # negatives). Keep all report fields present even when their value is 0.
    counts = Counter({'images': 0, 'boxes': 0, 'raw_box_lines': 0,
                      'duplicate_box_lines': 0, 'positive_images': 0,
                      'negative_images': 0, 'empty_label_files': 0,
                      'missing_label_files': 0})
    groups = Counter()
    negative_paths = set()
    duplicate_annotation_examples = []
    for image in SPLIT_IMAGES[split]:
        label = label_for(image, image_dir, label_dir)
        rows = read_labels(label)
        raw_box_lines = (sum(bool(line.strip()) for line in
                             label.read_text(encoding='utf-8-sig').splitlines())
                         if label.exists() else 0)
        group = negative_group(image.relative_to(image_dir))
        counts['images'] += 1
        counts['boxes'] += len(rows)
        counts['raw_box_lines'] += raw_box_lines
        counts['duplicate_box_lines'] += raw_box_lines - len(rows)
        if raw_box_lines > len(rows) and len(duplicate_annotation_examples) < 10:
            duplicate_annotation_examples.append(str(label))
        if rows:
            counts['positive_images'] += 1
            if group:
                preflight['errors'].append(f'{split}: {group} có box: {image}')
        else:
            counts['negative_images'] += 1
            negative_paths.add(str(image.resolve()))
            groups[group or 'unclassified'] += 1
            if label.exists():
                counts['empty_label_files'] += 1
                if label.stat().st_size:
                    preflight['warnings'].append(f'{split}: nhãn âm chỉ chứa whitespace: {label}')
            else:
                counts['missing_label_files'] += 1
                if group:
                    preflight['warnings'].append(f'{split}: {group} thiếu TXT rỗng (YOLO vẫn hỗ trợ): {label}')
            if any(token in image.stem.lower() for token in ('anthracnose', 'leaf_miner', 'red_rust')):
                preflight['errors'].append(f'{split}: tên ảnh bệnh nhưng không có box: {image}')
        digest = sha256_file(image)
        all_image_hashes[digest].append((split, str(image.relative_to(image_dir))))
        all_image_names[image.name.lower()].append((split, str(image.relative_to(image_dir))))
        with Image.open(image) as im:
            pixels = np.asarray(im.convert('L').resize((9, 8)))
        dhash = ''.join('1' if x else '0' for x in (pixels[:, 1:] > pixels[:, :-1]).flat)
        all_perceptual_hashes[dhash].append((split, str(image.relative_to(image_dir))))
    if split in ('train', 'val') and counts['positive_images'] == 0:
        preflight['errors'].append(f'{split}: không có nhãn dương hợp lệ')
    if split in ('train', 'val') and counts['negative_images'] == 0:
        preflight['errors'].append(f'{split}: không có ảnh âm để nạp vào dataloader')
    preflight['splits'][split] = {
        **dict(counts),
        'negative_groups': dict(groups),
        'negative_paths': negative_paths,
        'duplicate_annotation_examples': duplicate_annotation_examples,
        'orphan_empty_labels': {
            'count': len(ORPHAN_EMPTY_LABELS[split]),
            'examples': [str(p) for p in ORPHAN_EMPTY_LABELS[split][:10]],
        },
        'orphan_positive_labels': {
            'count': len(ORPHAN_POSITIVE_LABELS[split]),
            'examples': [str(p) for p in ORPHAN_POSITIVE_LABELS[split][:10]],
        },
    }
    if ORPHAN_EMPTY_LABELS[split]:
        preflight['warnings'].append(
            f'{split}: {len(ORPHAN_EMPTY_LABELS[split])} TXT rỗng không có ảnh; '
            'các file này không được dataloader dùng')
    if counts['duplicate_box_lines']:
        preflight['warnings'].append(
            f'{split}: {counts["duplicate_box_lines"]} dòng box trùng; '
            'Ultralytics tự khử trùng khi nạp, TXT nguồn không bị sửa')
    if ORPHAN_POSITIVE_LABELS[split]:
        preflight['errors'].append(
            f'{split}: {len(ORPHAN_POSITIVE_LABELS[split])} TXT có box không có ảnh tương ứng')
    if groups['unclassified']:
        preflight['warnings'].append(
            f'{split}: {groups["unclassified"]} ảnh 0 box không nhận ra nhóm từ tên; cần kiểm tra thủ công')

for digest, locations in all_image_hashes.items():
    if len({split for split, _ in locations}) > 1:
        preflight['cross_split_exact_duplicates'].append({'sha256': digest, 'locations': locations})
for name, locations in all_image_names.items():
    if len({split for split, _ in locations}) > 1:
        preflight['cross_split_same_names'].append({'name': name, 'locations': locations})
for dhash, locations in all_perceptual_hashes.items():
    if len({split for split, _ in locations}) > 1:
        preflight['cross_split_perceptual_candidates'].append(
            {'dhash': dhash, 'locations': locations})
if preflight['cross_split_exact_duplicates']:
    preflight['errors'].append('Có ảnh trùng byte giữa các split; xem cross_split_exact_duplicates')
if preflight['cross_split_same_names']:
    preflight['warnings'].append('Có tên file trùng giữa các split; xem cross_split_same_names')
if preflight['cross_split_perceptual_candidates']:
    preflight['warnings'].append('Có ảnh cùng perceptual hash giữa các split; kiểm tra nguồn thủ công')

train_cfg = {**DEFAULT_CFG_DICT, 'imgsz': IMGSZ, 'task': 'detect',
             'mosaic': MOSAIC, 'mixup': MIXUP, 'close_mosaic': CLOSE_MOSAIC,
             'cache': False, 'fraction': 1.0, 'classes': None,
             'single_cls': False, 'rect': False}
data_for_loader = yaml.safe_load(DATA_YAML.read_text(encoding='utf-8'))

def inspect_ultralytics_dataset(split, mode, config):
    image_dir, _ = SPLITS[split]
    dataset = build_yolo_dataset(IterableSimpleNamespace(**config),
                                 str(image_dir), BATCH, data_for_loader,
                                 mode=mode, rect=False, stride=32)
    expected = {str(p.resolve()) for p in SPLIT_IMAGES[split]}
    actual = {str(Path(p).resolve()) for p in dataset.im_files}
    label_lookup = {str(Path(row['im_file']).resolve()): len(row['cls'])
                    for row in dataset.labels}
    if actual != expected or set(label_lookup) != expected:
        preflight['errors'].append(f'{split}/{mode}: YOLODataset thiếu ảnh: '
                                   f'missing={len(expected-actual)}, extra={len(actual-expected)}')
    negative_paths = preflight['splits'][split]['negative_paths']
    loaded_boxes = sum(label_lookup.values())
    if loaded_boxes != preflight['splits'][split]['boxes']:
        preflight['errors'].append(
            f'{split}/{mode}: Ultralytics nạp {loaded_boxes} box, file nhãn có '
            f'{preflight["splits"][split]["boxes"]} box')
    for path in negative_paths & set(label_lookup):
        if label_lookup[path] != 0:
            preflight['errors'].append(f'{split}/{mode}: ảnh âm có GT box trong dataset: {path}')
    return dataset, {
        'images': len(dataset),
        'negative_images_accessible': len(negative_paths & actual),
        'positive_images_accessible': len(actual - negative_paths),
        'boxes_accessible': loaded_boxes,
    }

# Kiểm tra cấu hình train thật trên YOLODataset.labels. Mosaic có thể ghép thêm
# box vào tensor của một ảnh âm, vì vậy batch 0 GT được kiểm bằng train loader
# với augmentation trộn ảnh tắt riêng cho bước kiểm toán.
_, train_config_dataset = inspect_ultralytics_dataset('train', 'train', train_cfg)
preflight['splits']['train']['train_config_dataset'] = train_config_dataset

audit_cfg = {**train_cfg, 'mosaic': 0.0, 'mixup': 0.0,
             'cutmix': 0.0, 'copy_paste': 0.0}
for split, mode in (('train', 'train'), ('val', 'val')):
    dataset, dataset_summary = inspect_ultralytics_dataset(split, mode, audit_cfg)
    loader = build_dataloader(dataset, batch=BATCH, workers=0,
                              shuffle=False, rank=-1, device='cpu')
    seen = set()
    seen_negative = set()
    sample_batches = []
    for batch_number, batch in enumerate(loader):
        files = [str(Path(p).resolve()) for p in batch['im_file']]
        indices = batch['batch_idx'].reshape(-1).tolist()
        gt_per_image = Counter(int(i) for i in indices)
        batch_negatives = []
        for i, path in enumerate(files):
            if path in seen:
                preflight['errors'].append(f'{split}: dataloader lặp ảnh: {path}')
            seen.add(path)
            if path in preflight['splits'][split]['negative_paths']:
                seen_negative.add(path)
                batch_negatives.append(path)
                if gt_per_image[i] != 0:
                    preflight['errors'].append(f'{split}: batch ảnh âm có {gt_per_image[i]} GT box: {path}')
        if batch_negatives and len(sample_batches) < 5:
            sample_batches.append({'batch_index': batch_number,
                                   'images': len(files),
                                   'negative_images': [str(Path(p).name) for p in batch_negatives],
                                   'negative_gt_boxes': 0})
    expected = {str(p.resolve()) for p in SPLIT_IMAGES[split]}
    if seen != expected:
        preflight['errors'].append(f'{split}: dataloader thiếu {len(expected-seen)} ảnh, '
                                   f'dư {len(seen-expected)} ảnh')
    split_report = preflight['splits'][split]
    split_report['audit_dataset'] = dataset_summary
    split_report['dataloader'] = {
        'images_seen': len(seen),
        'negative_images_seen': len(seen_negative),
        'negative_images_expected': split_report['negative_images'],
        'sample_batches_with_negatives': sample_batches,
        'mix_augmentations_disabled_for_zero_gt_check': True,
    }
    if len(seen_negative) != split_report['negative_images']:
        preflight['errors'].append(f'{split}: loader không nạp đủ ảnh âm')

for split in preflight['splits']:
    preflight['splits'][split]['negative_paths'] = sorted(preflight['splits'][split]['negative_paths'])
preflight['status'] = 'pass' if not preflight['errors'] else 'fail'
save_json(PREFLIGHT_PATH, preflight)
print('Preflight:', preflight['status'], '| report:', PREFLIGHT_PATH)
for split, info in preflight['splits'].items():
    print(split, 'images:', info.get('images', 0), 'positive:', info.get('positive_images', 0),
          'negative:', info.get('negative_images', 0), 'groups:', info.get('negative_groups', {}))
    if 'dataloader' in info:
        print('  loader negatives:', info['dataloader']['negative_images_seen'])
if preflight['errors']:
    raise RuntimeError('Preflight phát hiện lỗi; xem metadata/negative_image_preflight.json: '
                       + '; '.join(preflight['errors'][:3]))
