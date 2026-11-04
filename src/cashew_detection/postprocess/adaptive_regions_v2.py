"""Deterministic class-aware display regions from raw YOLO boxes."""

from collections import Counter
from hashlib import sha256
from heapq import heappop, heappush
from math import isfinite, sqrt
from pathlib import Path
import importlib.util

from .geometry import area, clipped_union_area, containment, edge_gap, envelope, iou, intersection_area, union_area


DEFAULT_CONFIG = {
    "variant": "B", "overlap_iou": 0.60, "containment_ratio": 0.80,
    "cluster_gap_ratio": 0.02, "k": 1.0, "tau_fill": 0.25,
    "min_fill_ratio": 0.15, "min_group_members": 5,
    "min_local_density": 40.0,
    "max_region_area_ratio": 0.25, "max_region_span_ratio": 0.55,
    "neighbor_gap_ratio": 0.12, "candidate_neighbors": 16,
}


def _v005():
    path = Path(__file__).resolve().parents[3] / "Notebook" / "Yolov26" / "adaptive_regions_v005.py"
    spec = importlib.util.spec_from_file_location("adaptive_regions_v005_frozen", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _valid(items, size, cfg):
    if len(items) <= 1:
        return True
    width, height = size
    box = envelope([item["bbox_xyxy"] for item in items])
    if (area(box) / (width * height) > cfg["max_region_area_ratio"] or
            (box[2] - box[0]) / width > cfg["max_region_span_ratio"] or
            (box[3] - box[1]) / height > cfg["max_region_span_ratio"]):
        return False
    fill = union_area([item["bbox_xyxy"] for item in items]) / area(box)
    return fill >= cfg["min_fill_ratio"]


def _partition(indices, items, size, cfg):
    """Recursively split oversized or sparse groups at the largest center gap."""
    group = [items[i] for i in indices]
    if len(indices) <= 1 or _valid(group, size, cfg):
        return [sorted(indices)]
    width, height = size
    scored = []
    for axis, scale in ((0, width), (1, height)):
        ordered = sorted(indices, key=lambda i: ((items[i]["bbox_xyxy"][axis] +
                                                  items[i]["bbox_xyxy"][axis + 2]) / 2, items[i]["id"]))
        centers = [(items[i]["bbox_xyxy"][axis] + items[i]["bbox_xyxy"][axis + 2]) / 2
                   for i in ordered]
        for position in range(1, len(ordered)):
            gap = (centers[position] - centers[position - 1]) / scale
            balance = min(position, len(ordered) - position) / len(ordered)
            scored.append((gap, balance, axis, position, ordered))
    _, _, _, position, ordered = max(scored, key=lambda x: (x[0], x[1], -x[2], -x[3]))
    return (_partition(ordered[:position], items, size, cfg) +
            _partition(ordered[position:], items, size, cfg))


def _strong_clique(indices, items, cfg):
    return all(iou(items[i]["bbox_xyxy"], items[j]["bbox_xyxy"]) >= cfg["overlap_iou"] or
               containment(items[i]["bbox_xyxy"], items[j]["bbox_xyxy"]) >= cfg["containment_ratio"]
               for pos, i in enumerate(indices) for j in indices[pos + 1:])


def _finish_groups(groups, items, cfg):
    finished = []
    for group in groups:
        if len(group) >= cfg["min_group_members"] or (len(group) > 1 and _strong_clique(group, items, cfg)):
            finished.append(group)
        else:
            finished.extend([[i] for i in group])
    return sorted(finished, key=lambda group: min(items[i]["id"] for i in group))


def _variant_b(items, size, cfg):
    width, height = size
    g_abs = cfg["cluster_gap_ratio"] * min(size)
    edges = []
    for i, left in enumerate(items):
        a = left["bbox_xyxy"]
        sa = sqrt(area(a))
        for j in range(i + 1, len(items)):
            b = items[j]["bbox_xyxy"]
            strong = iou(a, b) >= cfg["overlap_iou"] or containment(a, b) >= cfg["containment_ratio"]
            limit = max(g_abs, cfg["k"] * min(sa, sqrt(area(b))))
            gap = edge_gap(a, b)
            if strong or gap <= limit:
                edges.append((0 if strong else 1, gap / max(limit, 1e-9), i, j))
    parent = list(range(len(items)))
    groups = {i: [i] for i in range(len(items))}

    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for _, _, i, j in sorted(edges):
        a, b = root(i), root(j)
        if a == b:
            continue
        combined = sorted(groups[a] + groups[b])
        parts = _partition(combined, items, size, cfg)
        if len(parts) == 1:
            keep, drop = min(a, b), max(a, b)
            groups[keep] = combined
            del groups[drop]
            for index in combined:
                parent[index] = keep
        elif any(set(part) & set(groups[a]) and set(part) & set(groups[b]) for part in parts):
            # Rebuild the affected union-find components after a spatial split.
            del groups[a], groups[b]
            for part in parts:
                representative = min(part)
                groups[representative] = part
                for index in part:
                    parent[index] = representative
    return _finish_groups(groups.values(), items, cfg)


def _variant_c(items, size, cfg):
    groups = {i: [i] for i in range(len(items))}
    versions = Counter()
    heap = []
    attempted = set()
    next_id = len(items)
    neighbor_gap = max(cfg["cluster_gap_ratio"], cfg["neighbor_gap_ratio"]) * min(size)

    def bounds(group_id):
        return envelope([items[i]["bbox_xyxy"] for i in groups[group_id]])

    def candidates(group_id):
        box = bounds(group_id)
        nearby = sorted((edge_gap(box, bounds(other)), other)
                        for other in groups if other != group_id)
        return [other for gap, other in nearby[:cfg["candidate_neighbors"]]
                if gap <= neighbor_gap]

    def queue(a, b):
        if a == b or a not in groups or b not in groups:
            return
        a, b = sorted((a, b))
        combined = groups[a] + groups[b]
        box = envelope([items[i]["bbox_xyxy"] for i in combined])
        bbox_area = area(box)
        if (bbox_area / (size[0] * size[1]) > cfg["max_region_area_ratio"] or
                (box[2] - box[0]) / size[0] > cfg["max_region_span_ratio"] or
                (box[3] - box[1]) / size[1] > cfg["max_region_span_ratio"]):
            # Defer an oversized candidate to the recursive spatial splitter.
            # Process it after feasible high-fill merges to avoid long chains.
            if (len(combined) >= cfg["min_group_members"] and
                    edge_gap(bounds(a), bounds(b)) <= cfg["cluster_gap_ratio"] * min(size)):
                heappush(heap, (0.0, a, b, versions[a], versions[b]))
            return
        boxes = [items[i]["bbox_xyxy"] for i in combined]
        if sum(area(part) for part in boxes) / bbox_area < cfg["tau_fill"]:
            return
        fill = union_area(boxes) / bbox_area
        if fill >= cfg["tau_fill"]:
            heappush(heap, (-fill, a, b, versions[a], versions[b]))

    for a in list(groups):
        for b in candidates(a):
            queue(a, b)
    while heap:
        neg_fill, a, b, version_a, version_b = heappop(heap)
        if (a not in groups or b not in groups or
                versions[a] != version_a or versions[b] != version_b):
            continue
        signature = tuple(sorted((tuple(groups[a]), tuple(groups[b]))))
        if signature in attempted:
            continue
        attempted.add(signature)
        combined = sorted(groups[a] + groups[b])
        parts = _partition(combined, items, size, cfg)
        if len(parts) == 1:
            del groups[a], groups[b]
            group_id = next_id
            next_id += 1
            groups[group_id] = combined
            for other in candidates(group_id):
                queue(group_id, other)
        elif any(set(part) & set(groups[a]) and set(part) & set(groups[b]) for part in parts):
            del groups[a], groups[b]
            for part in parts:
                group_id = next_id
                next_id += 1
                groups[group_id] = part
                for other in candidates(group_id):
                    queue(group_id, other)
    return _finish_groups(groups.values(), items, cfg)


def aggregate_detections(detections, image_size, image_id, class_names, config=None):
    """Return regions; every detection is assigned exactly once to its predicted class."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg["variant"] not in {"A", "B", "C"}:
        raise ValueError("variant must be A, B, or C")
    width, height = map(int, image_size)
    if width <= 0 or height <= 0:
        raise ValueError("Invalid image size")
    class_names = dict(enumerate(class_names)) if isinstance(class_names, (list, tuple)) else dict(class_names)
    if cfg["variant"] == "A":
        frozen = _v005()
        old_cfg = {key: cfg[key] for key in frozen.DEFAULT_CONFIG}
        regions = frozen.aggregate_detections(detections, image_size, image_id, class_names, old_cfg)
    else:
        if not (0 < cfg["max_region_area_ratio"] <= 1 and 0 < cfg["max_region_span_ratio"] <= 1 and
                0 < cfg["min_fill_ratio"] <= 1 and 0 < cfg["tau_fill"] <= 1 and
                cfg["k"] >= 0 and cfg["min_group_members"] >= 2 and cfg["candidate_neighbors"] >= 1):
            raise ValueError("Invalid region configuration")
        seen = set()
        data = []
        for detection in detections:
            box = [float(value) for value in detection["bbox_xyxy"]]
            class_id = int(detection["class_id"])
            score = float(detection["confidence"])
            det_id = str(detection["id"])
            if (len(box) != 4 or not all(isfinite(v) for v in box) or
                    not 0 <= box[0] < box[2] <= width + 1 or
                    not 0 <= box[1] < box[3] <= height + 1 or
                    class_id not in class_names or not 0 <= score <= 1 or det_id in seen):
                raise ValueError(f"Invalid detection: {detection}")
            seen.add(det_id)
            data.append({"id": det_id, "class_id": class_id, "confidence": score,
                         "bbox_xyxy": box})
        regions = []
        for class_id in sorted({item["class_id"] for item in data}):
            items = sorted((item for item in data if item["class_id"] == class_id),
                           key=lambda item: item["id"])
            groups = (_variant_b(items, (width, height), cfg) if cfg["variant"] == "B"
                      else _variant_c(items, (width, height), cfg))
            for ordinal, group in enumerate(groups, 1):
                boxes = [items[i]["bbox_xyxy"] for i in group]
                bbox = envelope(boxes)
                scores = sorted((items[i]["confidence"] for i in group), reverse=True)
                union = union_area(boxes)
                bbox_area = area(bbox)
                regions.append({
                    "region_id": f"{sha256(str(image_id).encode()).hexdigest()[:12]}_c{class_id}_r{ordinal:03d}",
                    "image_id": str(image_id), "class_id": class_id,
                    "class_name": class_names[class_id],
                    "mode": "aggregated" if len(group) > 1 else "individual",
                    "bbox_xyxy": bbox,
                    "member_detection_ids": [items[i]["id"] for i in group],
                    "n_detections": len(group), "yolo_conf_max": scores[0],
                    "yolo_conf_mean": sum(scores) / len(scores),
                    "yolo_conf_topk_mean": sum(scores[:3]) / min(3, len(scores)),
                    "local_density_per_image_area": len(group) / (bbox_area / (width * height)),
                    "display_gap_area_ratio": 1 - union / bbox_area,
                    "aggregation_config": dict(cfg),
                    "cnn_target_score": None, "verification_status": "not_evaluated",
                })
    by_id = {str(item["id"]): item for item in detections}
    assigned = [det_id for region in regions for det_id in region["member_detection_ids"]]
    if len(assigned) != len(by_id) or set(assigned) != set(by_id):
        raise AssertionError("Every raw detection must belong to exactly one same-class region")
    for region in regions:
        members = [by_id[det_id] for det_id in region["member_detection_ids"]]
        if any(int(item["class_id"]) != region["class_id"] for item in members):
            raise AssertionError("Cross-class merge detected")
        bbox = region["bbox_xyxy"]
        member_boxes = [item["bbox_xyxy"] for item in members]
        union = union_area(member_boxes)
        bbox_area = area(bbox)
        region["member_union_area"] = union
        region["fill_ratio"] = union / bbox_area
        region["bbox_area_ratio_image"] = bbox_area / (width * height)
        other_boxes = [other["bbox_xyxy"] for other in regions
                       if other["class_id"] != region["class_id"] and
                       intersection_area(other["bbox_xyxy"], bbox) > 0]
        region["other_class_overlap_ratio"] = clipped_union_area(other_boxes, bbox) / bbox_area
    return regions
