"""Validation-only display-region analysis for the self-contained EXP-005 notebook.

This cell expects globals created by the preceding training and data cells.
"""

REGION_DIR = RUN_DIR / 'region_aggregation'
REGION_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR = REGION_DIR / 'visual_examples'
VIS_DIR.mkdir(exist_ok=True)

def gt_display_coverage(gt, candidates, threshold=0.5):
    """Count class-matched GT boxes with >= threshold GT area covered by one box.

    This is a display coverage diagnostic, not YOLO recall or mAP.
    """
    return sum(any(c['class_id'] == gt_class and
                   intersection_area(gt_box, c['bbox_xyxy']) / max(box_area(gt_box), 1e-9) >= threshold
                   for c in candidates) for gt_class, gt_box in gt)

def draw_region_comparison(image_path, gt, raw, regions, destination):
    with Image.open(image_path) as source:
        original = source.convert('RGB')
    left, right = original.copy(), original.copy()
    colors = ['#d77b27', '#d62d32', '#1f78b4']
    for canvas, boxes, is_region in ((left, raw, False), (right, regions, True)):
        draw = ImageDraw.Draw(canvas)
        for cls, box in gt:
            draw.rectangle(tuple(box), outline='#ffffff', width=3)
        for item in boxes:
            cls = item['class_id']
            box = item['bbox_xyxy']
            draw.rectangle(tuple(box), outline=colors[cls], width=3)
            label = (f"{CLASS_NAMES[cls]} n={item['n_detections']} | needs verification"
                     if is_region else f"{CLASS_NAMES[cls]} {item['confidence']:.2f}")
            draw.text((box[0], max(0, box[1] - 14)), label, fill=colors[cls])
    side_by_side = Image.new('RGB', (left.width * 2, left.height))
    side_by_side.paste(left, (0, 0))
    side_by_side.paste(right, (left.width, 0))
    side_by_side.save(destination, quality=90)

region_rows = []
region_examples = []
region_jsonl = REGION_DIR / 'validation_regions.jsonl'
raw_jsonl = REGION_DIR / 'validation_raw_detections.jsonl'
with region_jsonl.open('w', encoding='utf-8') as region_file, raw_jsonl.open('w', encoding='utf-8') as raw_file:
    for idx, im in enumerate(SPLIT_IMAGES['val']):
        image_id = str(im.relative_to(SPLITS['val'][0])).replace('\\', '/')
        with Image.open(im) as source:
            width, height = source.size
        gt_labels = read_labels(label_for(im, *SPLITS['val']))
        gt = [(int(row[0]), xywhn_to_xyxy(row, width, height).tolist()) for row in gt_labels]
        t0 = time.perf_counter()
        result = best_model.predict(source=str(im), imgsz=IMGSZ, conf=REGION_PROPOSAL_CONF,
                                    iou=NMS_IOU, agnostic_nms=AGNOSTIC_NMS,
                                    max_det=REGION_MAX_DET, device=DEVICE, verbose=False)[0]
        prediction_done = time.perf_counter()
        raw = [{'id': f'{idx:06d}_{j:04d}', 'image_id': image_id,
                'model_id': EXPERIMENT_ID, 'seed': SEED, 'class_id': int(cls),
                'confidence': float(conf), 'bbox_xyxy': [float(v) for v in box]}
               for j, (cls, conf, box) in enumerate(zip(
                   result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                   result.boxes.xyxy.cpu().numpy()))]
        regions = aggregate_detections(raw, (width, height), image_id, CLASS_NAMES, REGION_CONFIG)
        grouping_done = time.perf_counter()
        raw_file.write(json.dumps({'image_id': image_id, 'detections': raw}) + '\n')
        region_file.write(json.dumps({'image_id': image_id, 'regions': regions}) + '\n')
        raw_covered = gt_display_coverage(gt, raw)
        region_covered = gt_display_coverage(gt, regions)
        if region_covered < raw_covered:
            raise AssertionError(f'Gộp vùng làm mất GT coverage: {image_id}')
        unsupported_regions = sum(not any(
            region['class_id'] == gt_class and
            intersection_area(gt_box, region['bbox_xyxy']) / max(box_area(gt_box), 1e-9) >= 0.5
            for gt_class, gt_box in gt) for region in regions)
        gap_area = sum(box_area(r['bbox_xyxy']) * r['display_gap_area_ratio'] for r in regions)
        total_region_area = sum(box_area(r['bbox_xyxy']) for r in regions)
        region_rows.append({'image_id': image_id, 'gt_count': len(gt),
                            'is_negative': len(gt) == 0, 'raw_box_count': len(raw),
                            'display_region_count': len(regions),
                            'boxes_reduced': len(raw) - len(regions),
                            'raw_gt_covered': raw_covered, 'region_gt_covered': region_covered,
                            'regions_without_gt_support': unsupported_regions,
                            'display_gap_area_ratio': gap_area / total_region_area if total_region_area else 0.0,
                            'inference_ms': (prediction_done - t0) * 1000,
                            'aggregation_ms': (grouping_done - prediction_done) * 1000})
        region_examples.append((idx, im, gt, raw, regions))
        if (idx + 1) % 100 == 0:
            print('Region validation:', idx + 1, '/', len(SPLIT_IMAGES['val']))

region_frame = pd.DataFrame(region_rows)
region_frame.to_csv(REGION_DIR / 'validation_per_image.csv', index=False)
total_raw = int(region_frame['raw_box_count'].sum())
total_regions = int(region_frame['display_region_count'].sum())
total_gt = int(region_frame['gt_count'].sum())
checkpoint_digest = hashlib.sha256()
with BEST_PT.open('rb') as checkpoint_file:
    for block in iter(lambda: checkpoint_file.read(8 * 1024 * 1024), b''):
        checkpoint_digest.update(block)
archive_path = Path(DATASET_STATS['source_archive']) if DATASET_STATS['source_archive'] else None
archive_digest = None
if archive_path and archive_path.is_file():
    archive_hash = hashlib.sha256()
    with archive_path.open('rb') as archive_file:
        for block in iter(lambda: archive_file.read(8 * 1024 * 1024), b''):
            archive_hash.update(block)
    archive_digest = archive_hash.hexdigest()
REGION_SUMMARY = {
    'split': 'val', 'note': 'Display-region diagnostics; not detector mAP/recall.',
    'experiment_id': EXPERIMENT_ID, 'model_weights': MODEL_WEIGHTS,
    'checkpoint_sha256': checkpoint_digest.hexdigest(),
    'dataset_archive_sha256': archive_digest,
    'dataset_fingerprint': DATASET_STATS['dataset_fingerprint'],
    'seed': SEED, 'ultralytics_version': ultralytics.__version__,
    'config': {'proposal_conf': REGION_PROPOSAL_CONF, 'nms_iou': NMS_IOU,
               'agnostic_nms': AGNOSTIC_NMS, 'max_det': REGION_MAX_DET,
               'region_config': REGION_CONFIG, 'gt_coverage_ioa': 0.5},
    'images': len(region_frame), 'negative_images': int(region_frame['is_negative'].sum()),
    'negative_raw_boxes': int(region_frame.loc[region_frame['is_negative'], 'raw_box_count'].sum()),
    'negative_display_regions': int(region_frame.loc[region_frame['is_negative'], 'display_region_count'].sum()),
    'raw_boxes': total_raw, 'display_regions': total_regions,
    'box_reduction_ratio': (total_raw - total_regions) / total_raw if total_raw else 0.0,
    'gt_count': total_gt,
    'raw_gt_covered': int(region_frame['raw_gt_covered'].sum()),
    'region_gt_covered': int(region_frame['region_gt_covered'].sum()),
    'regions_without_gt_support': int(region_frame['regions_without_gt_support'].sum()),
    'mean_display_gap_area_ratio': float(region_frame['display_gap_area_ratio'].mean()),
    'mean_inference_ms': float(region_frame['inference_ms'].mean()),
    'mean_aggregation_ms': float(region_frame['aggregation_ms'].mean()),
}
save_json(REGION_DIR / 'validation_summary.json', REGION_SUMMARY)
with (RUN_DIR / 'README.md').open('a', encoding='utf-8') as readme:
    readme.write(f"\nACDRA Validation: raw boxes={total_raw}; display regions={total_regions}; "
                 f"GT covered={REGION_SUMMARY['raw_gt_covered']} raw / "
                 f"{REGION_SUMMARY['region_gt_covered']} regions. "
                 "Đây là metric hiển thị, không phải mAP/recall YOLO.\n")
print(json.dumps(REGION_SUMMARY, ensure_ascii=False, indent=2))

# Fixed, reproducible examples: highest box count, largest reduction, negative images.
chosen = []
for key in ('raw_box_count', 'boxes_reduced'):
    chosen.extend(region_frame.sort_values([key, 'image_id'], ascending=[False, True]).head(5).index.tolist())
chosen.extend(region_frame[region_frame['is_negative']].sort_values(
    ['raw_box_count', 'image_id'], ascending=[False, True]).head(5).index.tolist())
for i in dict.fromkeys(chosen):
    idx, im, gt, raw, regions = region_examples[i]
    draw_region_comparison(im, gt, raw, regions, VIS_DIR / f'{idx:04d}_{im.stem}.jpg')
