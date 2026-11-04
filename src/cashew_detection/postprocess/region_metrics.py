"""GT-based metrics for raw detections and display regions."""

from collections import Counter

from .geometry import area, clipped_union_area, intersection_area, iou


def match_raw(detections, gt, iou_threshold=0.5):
    """Greedy confidence-descending, same-class one-to-one IoU matching."""
    matched_gt = set()
    tp_ids = set()
    for det in sorted(detections, key=lambda x: (-float(x["confidence"]), str(x["id"]))):
        choices = [(iou(det["bbox_xyxy"], target["bbox_xyxy"]), index)
                   for index, target in enumerate(gt)
                   if index not in matched_gt and int(target["class_id"]) == int(det["class_id"])]
        if not choices:
            continue
        best_iou, best_index = max(choices, key=lambda x: (x[0], -x[1]))
        if best_iou >= iou_threshold:
            matched_gt.add(best_index)
            tp_ids.add(str(det["id"]))
    return tp_ids, len(detections) - len(tp_ids), len(gt) - len(matched_gt)


def coverage(gt, candidates, center=False, ioa_threshold=0.5):
    total = 0
    for target in gt:
        box = target["bbox_xyxy"]
        if center:
            x, y = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            covered = any(int(item["class_id"]) == int(target["class_id"]) and
                          item["bbox_xyxy"][0] <= x <= item["bbox_xyxy"][2] and
                          item["bbox_xyxy"][1] <= y <= item["bbox_xyxy"][3]
                          for item in candidates)
        else:
            covered = any(int(item["class_id"]) == int(target["class_id"]) and
                          intersection_area(box, item["bbox_xyxy"]) / area(box) >= ioa_threshold
                          for item in candidates)
        total += int(covered)
    return total


def _gt_covered_area(candidates, gt, same_class):
    covered = 0.0
    for item in candidates:
        class_id = int(item["class_id"])
        related = [target["bbox_xyxy"] for target in gt
                   if (int(target["class_id"]) == class_id) == same_class and
                   intersection_area(target["bbox_xyxy"], item["bbox_xyxy"]) > 0]
        covered += clipped_union_area(related, item["bbox_xyxy"])
    return covered


def density_slice(n):
    if n <= 5:
        return "le5"
    if n <= 10:
        return "6_10"
    if n <= 20:
        return "11_20"
    if n <= 40:
        return "21_40"
    if n <= 80:
        return "41_80"
    if n <= 150:
        return "81_150"
    return "gt150"


def summarize_tp_composition(region_rows, raw_fp_count):
    """Describe TP purity and raw FP boxes hidden inside TP-containing regions.

    A region is TP-majority when at least half of its raw members matched a GT
    under the original one-to-one IoU rule. An absorbed FP belongs to a region
    containing at least one TP, so it is no longer shown as a separate box.
    """
    raw_fp_count = int(raw_fp_count)
    if raw_fp_count < 0:
        raise ValueError("raw_fp_count must be nonnegative")
    total = majority = aggregated = aggregated_majority = absorbed_fp = mixed_tp_fp = 0
    for row in region_rows:
        members = int(row["n_detections"])
        tp = int(row["tp_members"])
        if members < 1 or not 0 <= tp <= members:
            raise ValueError("Invalid TP composition in region")
        total += 1
        majority += int(2 * tp >= members)
        if members >= 2:
            aggregated += 1
            aggregated_majority += int(2 * tp >= members)
        if tp:
            absorbed_fp += members - tp
            mixed_tp_fp += int(tp < members)
    if absorbed_fp > raw_fp_count:
        raise ValueError("Absorbed FP count exceeds all raw FP boxes")
    return {
        "regions": total,
        "tp_majority_regions": majority,
        "tp_majority_region_share": majority / total if total else 0.0,
        "aggregated_regions": aggregated,
        "aggregated_tp_majority_regions": aggregated_majority,
        "aggregated_tp_majority_share": aggregated_majority / aggregated if aggregated else 0.0,
        "regions_mixing_tp_fp": mixed_tp_fp,
        "absorbed_raw_fp_boxes": absorbed_fp,
        "absorbed_raw_fp_share": absorbed_fp / raw_fp_count if raw_fp_count else 0.0,
        "raw_fp_boxes": raw_fp_count,
    }


def per_image_metrics(image_id, detections, regions, gt, image_size, main_class=None):
    """Use identical GT coverage, area and occupancy rules for raw and regions."""
    width, height = image_size
    image_area = width * height
    tp_ids, raw_fp, fn = match_raw(detections, gt)
    membership = {str(member): region for region in regions
                  for member in region["member_detection_ids"]}
    if (len(membership) != len(detections) or
            set(membership) != {str(item["id"]) for item in detections}):
        raise AssertionError("Each raw detection must belong to exactly one region")
    lost_tp = sum(int(membership[det_id]["class_id"] != int(next(
        item["class_id"] for item in detections if str(item["id"]) == det_id))) for det_id in tp_ids)
    if lost_tp:
        raise AssertionError("A TP changed class during region aggregation")
    region_tp = sum(any(str(det_id) in tp_ids for det_id in region["member_detection_ids"])
                    for region in regions)
    cross_pairs = sum(intersection_area(left["bbox_xyxy"], right["bbox_xyxy"]) > 0
                      for pos, left in enumerate(regions) for right in regions[pos + 1:]
                      if left["class_id"] != right["class_id"])
    gt_classes = {int(item["class_id"]) for item in gt}
    raw_area = sum(area(item["bbox_xyxy"]) for item in detections)
    region_area = sum(area(item["bbox_xyxy"]) for item in regions)
    raw_gt_area = _gt_covered_area(detections, gt, True)
    region_gt_area = _gt_covered_area(regions, gt, True)
    other_gt_area = _gt_covered_area(regions, gt, False)
    row = {
        "image_id": image_id, "main_class": main_class or "unknown",
        "mixed_disease": len(gt_classes) > 1, "is_negative": len(gt) == 0,
        "density_slice": density_slice(len(detections)), "raw_box_count": len(detections),
        "region_count": len(regions), "reduction": 1 - len(regions) / len(detections) if detections else 0.0,
        "gt_count": len(gt), "raw_tp": len(tp_ids), "raw_fp": raw_fp, "raw_fn": fn,
        "raw_precision": len(tp_ids) / len(detections) if detections else 0.0,
        "fp_regions": len(regions) - region_tp,
        "region_precision": region_tp / len(regions) if regions else 0.0,
        "lost_tp": lost_tp,
        "raw_gt_coverage": coverage(gt, detections),
        "region_gt_coverage": coverage(gt, regions),
        "raw_gt_center_coverage": coverage(gt, detections, center=True),
        "region_gt_center_coverage": coverage(gt, regions, center=True),
        "raw_display_area": raw_area, "region_display_area": region_area,
        "raw_gt_intersection_area": raw_gt_area,
        "region_gt_intersection_area": region_gt_area,
        "other_gt_intersection_area": other_gt_area,
        "raw_excess_area_ratio": 1 - raw_gt_area / raw_area if raw_area else 0.0,
        "region_excess_area_ratio": 1 - region_gt_area / region_area if region_area else 0.0,
        "other_class_contamination": other_gt_area / region_area if region_area else 0.0,
        "raw_display_occupancy": raw_area / image_area,
        "region_display_occupancy": region_area / image_area,
        "cross_class_overlap_pairs": cross_pairs,
        "has_cross_class_overlap": bool(cross_pairs),
        "regions_short_side_lt32": sum(min(item["bbox_xyxy"][2] - item["bbox_xyxy"][0],
                                         item["bbox_xyxy"][3] - item["bbox_xyxy"][1]) < 32
                                       for item in regions),
    }
    return row, tp_ids


def summarize(rows):
    rows = list(rows)
    raw = sum(row["raw_box_count"] for row in rows)
    regions = sum(row["region_count"] for row in rows)
    gt = sum(row["gt_count"] for row in rows)
    return {
        "images": len(rows), "raw_boxes": raw, "regions": regions,
        "reduction": 1 - regions / raw if raw else 0.0,
        "raw_per_image": raw / len(rows) if rows else 0.0,
        "regions_per_image": regions / len(rows) if rows else 0.0,
        "raw_fp_per_image": sum(row["raw_fp"] for row in rows) / len(rows) if rows else 0.0,
        "fp_regions_per_image": sum(row["fp_regions"] for row in rows) / len(rows) if rows else 0.0,
        "raw_precision": sum(row["raw_tp"] for row in rows) / raw if raw else 0.0,
        "region_precision": sum(row["region_count"] - row["fp_regions"] for row in rows) / regions if regions else 0.0,
        "raw_gt_coverage": sum(row["raw_gt_coverage"] for row in rows) / gt if gt else 0.0,
        "region_gt_coverage": sum(row["region_gt_coverage"] for row in rows) / gt if gt else 0.0,
        "raw_gt_center_coverage": sum(row["raw_gt_center_coverage"] for row in rows) / gt if gt else 0.0,
        "region_gt_center_coverage": sum(row["region_gt_center_coverage"] for row in rows) / gt if gt else 0.0,
        "raw_excess_area_ratio": 1 - sum(row["raw_gt_intersection_area"] for row in rows) /
            sum(row["raw_display_area"] for row in rows) if raw else 0.0,
        "region_excess_area_ratio": 1 - sum(row["region_gt_intersection_area"] for row in rows) /
            sum(row["region_display_area"] for row in rows) if regions else 0.0,
        "raw_display_occupancy": sum(row["raw_display_occupancy"] for row in rows) / len(rows) if rows else 0.0,
        "region_display_occupancy": sum(row["region_display_occupancy"] for row in rows) / len(rows) if rows else 0.0,
        "other_class_contamination": sum(row["other_gt_intersection_area"] for row in rows) /
            sum(row["region_display_area"] for row in rows) if regions else 0.0,
        "lost_tp": sum(row["lost_tp"] for row in rows),
        "cross_class_overlap_pairs": sum(row["cross_class_overlap_pairs"] for row in rows),
        "images_with_cross_class_overlap": sum(row["has_cross_class_overlap"] for row in rows),
        "region_short_side_lt32_share": sum(row["regions_short_side_lt32"] for row in rows) / regions if regions else 0.0,
    }
