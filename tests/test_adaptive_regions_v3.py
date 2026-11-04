"""ACDRA v3 guarded post-merge and strict evaluation edge cases."""

import pytest

from src.cashew_detection.postprocess import adaptive_regions_v3 as v3
from src.cashew_detection.postprocess.region_metrics_v3 import (
    overlap_area_at_least_two, per_image_metrics_v3, same_class_overlap)


def det(number, box, cls=0):
    return {"id": str(number), "bbox_xyxy": box, "class_id": cls,
            "confidence": .5}


def region(number, box, cls=0):
    return {"region_id": str(number), "bbox_xyxy": box, "class_id": cls,
            "class_name": {0: "a", 1: "b"}[cls],
            "member_detection_ids": [str(number)], "n_detections": 1}


def test_same_class_overlap_exact_area_and_pairs():
    items = [region(0, [0, 0, 10, 10]), region(1, [5, 0, 15, 10])]
    assert same_class_overlap(items) == (1, 50, .25)
    assert overlap_area_at_least_two([[0, 0, 10, 10], [0, 0, 10, 10],
                                      [0, 0, 10, 10]]) == 100
    assert same_class_overlap([items[0], region(1, [5, 0, 15, 10], 1)]) == (0, 0, 0)
    assert same_class_overlap([region(0, [0, 0, 10, 10]),
                               region(1, [2, 2, 4, 4])])[1] == 4


def test_strict_precision_one_vs_two_tp_of_three():
    raw = [det(i, [i * 30, 0, i * 30 + 10, 10]) for i in range(3)]
    merged = {**region(0, [0, 0, 70, 10]),
              "member_detection_ids": ["0", "1", "2"], "n_detections": 3}
    gt_one = [{"class_id": 0, "bbox_xyxy": raw[0]["bbox_xyxy"]}]
    gt_two = gt_one + [{"class_id": 0, "bbox_xyxy": raw[1]["bbox_xyxy"]}]
    assert per_image_metrics_v3("im", raw, [merged], gt_one, (100, 100))[0][
        "region_precision_strict"] == 0
    row, _ = per_image_metrics_v3("im", raw, [merged], gt_two, (100, 100))
    assert row["region_precision_strict"] == 1
    assert row["raw_fp_absorbed"] == 1
    assert row["lost_tp"] == 0


def test_d_merges_same_class_and_preserves_members(monkeypatch):
    raw = [det(0, [0, 0, 20, 20]), det(1, [10, 0, 30, 20])]
    monkeypatch.setattr(v3, "aggregate_v2", lambda *args: [
        region(0, raw[0]["bbox_xyxy"]), region(1, raw[1]["bbox_xyxy"])])
    config = {"variant": "D", "base_variant": "B", "theta_ov": .25,
              "tau_fill": .15, "max_region_area_ratio": .5,
              "max_region_span_ratio": .8, "min_fill_ratio": .15}
    first = v3.aggregate_detections(raw, (100, 100), "im", {0: "a"}, config)
    assert len(first) == 1
    assert first[0]["member_detection_ids"] == ["0", "1"]
    assert first == v3.aggregate_detections(list(reversed(raw)), (100, 100),
                                             "im", {0: "a"}, config)


def test_d_never_merges_other_class_or_chains_past_span(monkeypatch):
    raw = [det(0, [0, 0, 20, 20]), det(1, [10, 0, 30, 20], 1)]
    monkeypatch.setattr(v3, "aggregate_v2", lambda *args: [
        region(0, raw[0]["bbox_xyxy"]), region(1, raw[1]["bbox_xyxy"], 1)])
    config = {"variant": "D", "base_variant": "B", "theta_ov": .25,
              "tau_fill": .15, "max_region_area_ratio": .5,
              "max_region_span_ratio": .35, "min_fill_ratio": .15}
    assert len(v3.aggregate_detections(raw, (100, 100), "im", {0: "a", 1: "b"}, config)) == 2
    chain = [det(i, [i * 10, 0, i * 10 + 20, 20]) for i in range(5)]
    monkeypatch.setattr(v3, "aggregate_v2", lambda *args: [
        region(i, item["bbox_xyxy"]) for i, item in enumerate(chain)])
    result = v3.aggregate_detections(chain, (100, 100), "im", {0: "a"}, config)
    assert all((r["bbox_xyxy"][2] - r["bbox_xyxy"][0]) <= 35
               for r in result if r["n_detections"] > 1)
    assert sorted(x for r in result for x in r["member_detection_ids"]) == [str(i) for i in range(5)]


def test_tuning_rejects_test_split(monkeypatch):
    from scripts import run_acdra_v3
    monkeypatch.setattr("sys.argv", ["run_acdra_v3.py", "--split", "test",
                                 "--config-json", "unused.json"])
    with pytest.raises(SystemExit, match="Validation only"):
        run_acdra_v3.main()


def test_crop_window_flags_boundary_and_uses_minimum_side():
    from scripts.evaluate_crop_readiness import crop_window
    center, clipped = crop_window([45, 45, 55, 55], (100, 100), 64)
    assert center == [18, 18, 82, 82] and not clipped
    edge, clipped = crop_window([0, 0, 10, 10], (100, 100), 64)
    assert edge == [0, 0, 37, 37] and clipped
