"""Deterministic, class-aware display regions for EXP-Y26S-NEG-V03-005.

This is display post-processing. It never changes or discards raw detections.
Input boxes are xyxy pixels and image_size is (width, height).
"""

from hashlib import sha256
from math import hypot, isfinite


DEFAULT_CONFIG = {
    "overlap_iou": 0.60,
    "containment_ratio": 0.80,
    "cluster_gap_ratio": 0.02,
    "min_group_members": 5,
    "min_local_density": 40.0,
    "max_region_area_ratio": 0.25,
    "max_region_span_ratio": 0.55,
}


def box_area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_area(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1])
    )


def box_iou(a, b):
    intersection = intersection_area(a, b)
    union = box_area(a) + box_area(b) - intersection
    return intersection / union if union > 0 else 0.0


def containment_ratio(a, b):
    smaller = min(box_area(a), box_area(b))
    return intersection_area(a, b) / smaller if smaller > 0 else 0.0


def edge_gap(a, b):
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return hypot(dx, dy)


def bounding_box(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def rectangle_union_area(boxes):
    """Exact union area of axis-aligned rectangles, used as a display-gap proxy."""
    if not boxes:
        return 0.0
    x_edges = sorted({x for b in boxes for x in (b[0], b[2])})
    total = 0.0
    for x_left, x_right in zip(x_edges, x_edges[1:]):
        if x_right <= x_left:
            continue
        intervals = sorted((b[1], b[3]) for b in boxes
                           if b[0] < x_right and b[2] > x_left and b[3] > b[1])
        if not intervals:
            continue
        y0, y1 = intervals[0]
        height = 0.0
        for start, end in intervals[1:]:
            if start <= y1:
                y1 = max(y1, end)
            else:
                height += y1 - y0
                y0, y1 = start, end
        height += y1 - y0
        total += (x_right - x_left) * height
    return total


def aggregate_detections(detections, image_size, image_id, class_names,
                         config=None):
    """Return display regions while assigning each raw detection to one region.

    Required detection fields: id, class_id, confidence, bbox_xyxy.
    Different predicted classes are never compared or merged.
    """
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    width, height = map(int, image_size)
    class_names = (dict(enumerate(class_names)) if isinstance(class_names, (list, tuple))
                   else dict(class_names))
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    if not (0 < cfg["overlap_iou"] <= 1 and 0 < cfg["containment_ratio"] <= 1
            and 0 <= cfg["cluster_gap_ratio"] <= 1
            and cfg["min_group_members"] >= 2
            and cfg["min_local_density"] > 0
            and 0 < cfg["max_region_area_ratio"] <= 1
            and 0 < cfg["max_region_span_ratio"] <= 1):
        raise ValueError("Invalid aggregation configuration")
    data = []
    seen_ids = set()
    for item in detections:
        box = [float(v) for v in item["bbox_xyxy"]]
        class_id = int(item["class_id"])
        score = float(item["confidence"])
        det_id = str(item["id"])
        if (len(box) != 4 or not all(map(isfinite, box)) or
                not 0 <= box[0] < box[2] <= width + 1 or
                not 0 <= box[1] < box[3] <= height + 1 or
                class_id not in class_names or not 0 <= score <= 1 or
                det_id in seen_ids):
            raise ValueError(f"Invalid detection: {item}")
        seen_ids.add(det_id)
        data.append({"id": det_id, "class_id": class_id,
                     "confidence": score, "bbox_xyxy": box})
    if not data:
        return []

    image_area = width * height
    max_gap = cfg["cluster_gap_ratio"] * min(width, height)
    regions = []
    for class_id in sorted({item["class_id"] for item in data}):
        items = sorted((item for item in data if item["class_id"] == class_id),
                       key=lambda item: item["id"])
        n = len(items)
        parent = list(range(n))
        members = {i: [i] for i in range(n)}

        def root(index):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        def feasible(indices):
            box = bounding_box([items[i]["bbox_xyxy"] for i in indices])
            return (box_area(box) / image_area <= cfg["max_region_area_ratio"] and
                    (box[2] - box[0]) / width <= cfg["max_region_span_ratio"] and
                    (box[3] - box[1]) / height <= cfg["max_region_span_ratio"])

        edges = []
        strong_pairs = set()
        for i in range(n):
            for j in range(i + 1, n):
                a, b = items[i]["bbox_xyxy"], items[j]["bbox_xyxy"]
                overlap = box_iou(a, b)
                containment = containment_ratio(a, b)
                if overlap >= cfg["overlap_iou"] or containment >= cfg["containment_ratio"]:
                    edges.append((0, -max(overlap, containment), i, j))
                    strong_pairs.add((i, j))
                else:
                    gap = edge_gap(a, b)
                    if gap <= max_gap:
                        edges.append((1, gap, i, j))
        for _, _, i, j in sorted(edges):
            left, right = root(i), root(j)
            if left == right:
                continue
            candidate = members[left] + members[right]
            if not feasible(candidate):
                continue
            keep, drop = sorted((left, right))
            parent[drop] = keep
            members[keep] = sorted(candidate)
            del members[drop]

        class_groups = []
        for indices in members.values():
            boxes = [items[i]["bbox_xyxy"] for i in indices]
            bbox = bounding_box(boxes)
            density = len(indices) / (box_area(bbox) / image_area)
            strong_clique = all((min(i, j), max(i, j)) in strong_pairs
                                for position, i in enumerate(indices)
                                for j in indices[position + 1:])
            if len(indices) >= 2 and (strong_clique or
                    (len(indices) >= cfg["min_group_members"] and
                     density >= cfg["min_local_density"])):
                class_groups.append(indices)
            else:
                class_groups.extend([[i] for i in indices])
        class_groups.sort(key=lambda indices: min(items[i]["id"] for i in indices))
        for ordinal, indices in enumerate(class_groups, 1):
            boxes = [items[i]["bbox_xyxy"] for i in indices]
            bbox = bounding_box(boxes)
            scores = sorted((items[i]["confidence"] for i in indices), reverse=True)
            bbox_area = box_area(bbox)
            member_union = rectangle_union_area(boxes)
            regions.append({
                "region_id": f"{sha256(str(image_id).encode()).hexdigest()[:12]}_c{class_id}_r{ordinal:03d}",
                "image_id": str(image_id), "class_id": class_id,
                "class_name": class_names[class_id],
                "mode": "aggregated" if len(indices) > 1 else "individual",
                "bbox_xyxy": bbox,
                "member_detection_ids": [items[i]["id"] for i in indices],
                "n_detections": len(indices),
                "yolo_conf_max": scores[0],
                "yolo_conf_mean": sum(scores) / len(scores),
                "yolo_conf_topk_mean": sum(scores[:3]) / min(3, len(scores)),
                "local_density_per_image_area": len(indices) / (bbox_area / image_area),
                "display_gap_area_ratio": max(0.0, 1 - member_union / bbox_area),
                "aggregation_config": dict(cfg),
                "cnn_target_score": None,
                "verification_status": "not_evaluated",
            })
    assigned = [det_id for region in regions for det_id in region["member_detection_ids"]]
    if len(assigned) != len(data) or set(assigned) != seen_ids:
        raise AssertionError("Every raw detection must appear in exactly one region")
    return regions
