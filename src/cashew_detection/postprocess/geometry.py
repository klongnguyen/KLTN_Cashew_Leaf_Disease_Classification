"""Exact axis-aligned rectangle geometry used by ACDRA v2."""

from math import hypot


def area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection(a, b):
    return [max(a[0], b[0]), max(a[1], b[1]),
            min(a[2], b[2]), min(a[3], b[3])]


def intersection_area(a, b):
    return area(intersection(a, b))


def iou(a, b):
    shared = intersection_area(a, b)
    union = area(a) + area(b) - shared
    return shared / union if union else 0.0


def containment(a, b):
    smaller = min(area(a), area(b))
    return intersection_area(a, b) / smaller if smaller else 0.0


def edge_gap(a, b):
    dx = max(0.0, a[0] - b[2], b[0] - a[2])
    dy = max(0.0, a[1] - b[3], b[1] - a[3])
    return hypot(dx, dy)


def envelope(boxes):
    return [min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes)]


def union_area(boxes):
    """Exact area of the union of axis-aligned rectangles."""
    boxes = [b for b in boxes if area(b) > 0]
    if not boxes:
        return 0.0
    xs = sorted({x for b in boxes for x in (b[0], b[2])})
    result = 0.0
    for left, right in zip(xs, xs[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes
                           if b[0] < right and b[2] > left)
        if not intervals:
            continue
        start, end = intervals[0]
        covered = 0.0
        for y0, y1 in intervals[1:]:
            if y0 <= end:
                end = max(end, y1)
            else:
                covered += end - start
                start, end = y0, y1
        covered += end - start
        result += (right - left) * covered
    return result


def clipped_union_area(boxes, clip):
    return union_area([intersection(box, clip) for box in boxes
                       if intersection_area(box, clip) > 0])
