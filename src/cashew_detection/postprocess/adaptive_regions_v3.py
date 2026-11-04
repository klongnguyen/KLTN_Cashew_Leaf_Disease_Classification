"""ACDRA v3: V006 A/B/C plus guarded same-class region post-merge D."""

from hashlib import sha256
from heapq import heappop, heappush

from .adaptive_regions_v2 import DEFAULT_CONFIG, _valid, aggregate_detections as aggregate_v2
from .geometry import (area, clipped_union_area, envelope, intersection_area,
                       union_area)


def _make_region(image_id, class_id, class_name, member_ids, detections,
                 image_size, config):
    members = [detections[member_id] for member_id in sorted(member_ids)]
    boxes = [member["bbox_xyxy"] for member in members]
    box = envelope(boxes)
    scores = sorted((float(member["confidence"]) for member in members), reverse=True)
    union = union_area(boxes)
    width, height = image_size
    bbox_area = area(box)
    signature = f"{image_id}|{class_id}|{'|'.join(sorted(member_ids))}"
    return {
        "region_id": f"v3_{sha256(signature.encode()).hexdigest()[:20]}",
        "image_id": str(image_id), "class_id": class_id,
        "class_name": class_name, "mode": "aggregated",
        "bbox_xyxy": box, "member_detection_ids": sorted(member_ids),
        "n_detections": len(members), "yolo_conf_max": scores[0],
        "yolo_conf_mean": sum(scores) / len(scores),
        "yolo_conf_topk_mean": sum(scores[:3]) / min(3, len(scores)),
        "member_union_area": union, "fill_ratio": union / bbox_area,
        "bbox_area_ratio_image": bbox_area / (width * height),
        "display_gap_area_ratio": 1 - union / bbox_area,
        "local_density_per_image_area": len(members) / (bbox_area / (width * height)),
        "aggregation_config": dict(config), "cnn_target_score": None,
        "verification_status": "not_evaluated",
    }


def aggregate_detections(detections, image_size, image_id, class_names, config=None):
    """Return deterministic class-aware regions; D merges only feasible overlaps."""
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    if cfg["variant"] in {"A", "B", "C"}:
        return aggregate_v2(detections, image_size, image_id, class_names, cfg)
    if cfg["variant"] != "D":
        raise ValueError("variant must be A, B, C, or D")
    if cfg.get("base_variant", "B") not in {"B", "C"}:
        raise ValueError("D base_variant must be B or C")
    theta = float(cfg.get("theta_ov", .25))
    tau = float(cfg["tau_fill"])
    if not (0 < theta <= 1 and 0 < tau <= 1):
        raise ValueError("Invalid D overlap or fill threshold")
    base = aggregate_v2(detections, image_size, image_id, class_names,
                        {**cfg, "variant": cfg["base_variant"]})
    if not base:
        return []
    lookup = {str(item["id"]): item for item in detections}
    active = {index: region for index, region in enumerate(base)}
    heap = []
    next_id = len(active)

    def queue(left_id, right_id):
        left, right = active[left_id], active[right_id]
        if left["class_id"] != right["class_id"]:
            return
        shared = intersection_area(left["bbox_xyxy"], right["bbox_xyxy"])
        smaller = min(area(left["bbox_xyxy"]), area(right["bbox_xyxy"]))
        ioa = shared / smaller if smaller else 0.0
        if ioa >= theta:
            heappush(heap, (-ioa, -shared, min(left_id, right_id),
                            max(left_id, right_id)))

    for left_id in active:
        for right_id in range(left_id + 1, len(base)):
            queue(left_id, right_id)
    while heap:
        _, _, left_id, right_id = heappop(heap)
        if left_id not in active or right_id not in active:
            continue
        left, right = active[left_id], active[right_id]
        member_ids = sorted(left["member_detection_ids"] +
                            right["member_detection_ids"])
        members = [lookup[member_id] for member_id in member_ids]
        if not _valid(members, image_size, cfg):
            continue
        box = envelope([item["bbox_xyxy"] for item in members])
        if union_area([item["bbox_xyxy"] for item in members]) / area(box) < tau:
            continue
        merged = _make_region(image_id, int(left["class_id"]), left["class_name"],
                              member_ids, lookup, image_size, cfg)
        del active[left_id], active[right_id]
        active[next_id] = merged
        for other_id in sorted(active):
            if other_id != next_id:
                queue(other_id, next_id)
        next_id += 1
    result = sorted(active.values(), key=lambda region:
                    (int(region["class_id"]), tuple(region["member_detection_ids"])))
    assigned = [str(member) for region in result
                for member in region["member_detection_ids"]]
    if (len(assigned) != len(lookup) or set(assigned) != set(lookup) or
            any(int(lookup[member]["class_id"]) != int(region["class_id"])
                for region in result for member in region["member_detection_ids"])):
        raise AssertionError("D changed or duplicated raw detection membership")
    for region in result:
        box = region["bbox_xyxy"]
        others = [other["bbox_xyxy"] for other in result
                  if other["class_id"] != region["class_id"] and
                  intersection_area(other["bbox_xyxy"], box) > 0]
        region["other_class_overlap_ratio"] = clipped_union_area(others, box) / area(box)
    return result
