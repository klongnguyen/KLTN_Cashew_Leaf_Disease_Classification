"""Strict TP purity, exact same-class overlap, and crop-readiness diagnostics."""

from collections import defaultdict

from .geometry import area, intersection_area, iou
from .region_metrics import per_image_metrics


def overlap_area_at_least_two(boxes):
    """Area covered by at least two rectangles, counted once even for triples."""
    boxes = [box for box in boxes if area(box) > 0]
    if len(boxes) < 2:
        return 0.0
    xs = sorted({edge for box in boxes for edge in (box[0], box[2])})
    result = 0.0
    for left, right in zip(xs, xs[1:]):
        events = []
        for box in boxes:
            if box[0] < right and box[2] > left:
                events.extend(((box[1], 1), (box[3], -1)))
        if len(events) < 4:
            continue
        count = 0
        previous = None
        covered = 0.0
        for y, change in sorted(events):
            if previous is not None and count >= 2:
                covered += y - previous
            count += change
            previous = y
        result += (right - left) * covered
    return result


def same_class_overlap(items, iou_threshold=.1):
    """Count overlapping pairs and exact doubly covered area across classes."""
    by_class = defaultdict(list)
    for item in items:
        by_class[int(item["class_id"])].append(item["bbox_xyxy"])
    pairs = 0
    overlap_area = 0.0
    for boxes in by_class.values():
        if not any(intersection_area(a, b) > 0 for position, a in enumerate(boxes)
                   for b in boxes[position + 1:]):
            continue
        pairs += sum(iou(a, b) > iou_threshold for position, a in enumerate(boxes)
                     for b in boxes[position + 1:])
        overlap_area += overlap_area_at_least_two(boxes)
    display_area = sum(area(item["bbox_xyxy"]) for item in items)
    return pairs, overlap_area, overlap_area / display_area if display_area else 0.0


def per_image_metrics_v3(image_id, detections, regions, gt, image_size,
                         main_class=None):
    row, tp_ids = per_image_metrics(image_id, detections, regions, gt,
                                    image_size, main_class)
    strict = absorbed = mixed = 0
    for region in regions:
        member_ids = [str(member) for member in region["member_detection_ids"]]
        tp = sum(member in tp_ids for member in member_ids)
        strict += int(2 * tp >= len(member_ids))
        if tp:
            absorbed += len(member_ids) - tp
            mixed += int(tp < len(member_ids))
    raw_pairs, raw_overlap, raw_ratio = same_class_overlap(detections)
    region_pairs, region_overlap, region_ratio = same_class_overlap(regions)
    row.update({
        "strict_tp_regions": strict,
        "region_precision_strict": strict / len(regions) if regions else 0.0,
        "raw_fp_absorbed": absorbed,
        "raw_fp_absorbed_share": absorbed / row["raw_fp"] if row["raw_fp"] else 0.0,
        "regions_mixing_tp_fp": mixed,
        "raw_same_class_overlap_pairs": raw_pairs,
        "same_class_overlap_pairs": region_pairs,
        "raw_same_class_overlap_area": raw_overlap,
        "same_class_overlap_area": region_overlap,
        "raw_same_class_overlap_area_ratio": raw_ratio,
        "same_class_overlap_area_ratio": region_ratio,
        "has_same_class_overlap": bool(region_pairs),
        "sparse_changed": len(detections) <= 10 and len(regions) < len(detections),
        "sparse_boxes_merged": len(detections) - len(regions)
        if len(detections) <= 10 else 0,
    })
    return row, tp_ids


def summarize_v3(rows):
    """Pool counts and areas; avoid averaging per-image ratios."""
    rows = list(rows)
    regions = sum(row["region_count"] for row in rows)
    raw = sum(row["raw_box_count"] for row in rows)
    raw_fp = sum(row["raw_fp"] for row in rows)
    raw_area = sum(row["raw_display_area"] for row in rows)
    region_area = sum(row["region_display_area"] for row in rows)
    sparse = [row for row in rows if row["raw_box_count"] <= 10]
    return {
        "region_precision_strict": sum(row["strict_tp_regions"] for row in rows) /
            regions if regions else 0.0,
        "raw_precision_strict": sum(row["raw_tp"] for row in rows) / raw if raw else 0.0,
        "raw_fp_absorbed": sum(row["raw_fp_absorbed"] for row in rows),
        "raw_fp_absorbed_share": sum(row["raw_fp_absorbed"] for row in rows) /
            raw_fp if raw_fp else 0.0,
        "raw_same_class_overlap_pairs": sum(row["raw_same_class_overlap_pairs"] for row in rows),
        "same_class_overlap_pairs": sum(row["same_class_overlap_pairs"] for row in rows),
        "images_with_same_class_overlap": sum(row["has_same_class_overlap"] for row in rows),
        "images_with_same_class_overlap_share": sum(row["has_same_class_overlap"] for row in rows) /
            len(rows) if rows else 0.0,
        "raw_same_class_overlap_area_ratio": sum(row["raw_same_class_overlap_area"] for row in rows) /
            raw_area if raw_area else 0.0,
        "same_class_overlap_area_ratio": sum(row["same_class_overlap_area"] for row in rows) /
            region_area if region_area else 0.0,
        "sparse_images": len(sparse),
        "sparse_images_changed": sum(row["sparse_changed"] for row in sparse),
        "sparse_images_changed_share": sum(row["sparse_changed"] for row in sparse) /
            len(sparse) if sparse else 0.0,
        "sparse_boxes_merged": sum(row["sparse_boxes_merged"] for row in sparse),
        "sparse_review_flag": sum(row["sparse_changed"] for row in sparse) > .15 * len(sparse)
            if sparse else False,
    }
