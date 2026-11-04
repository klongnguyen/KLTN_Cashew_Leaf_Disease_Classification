"""Required ACDRA v2 edge cases and invariants."""

import io
import json
import time
import zipfile

import pytest

from src.cashew_detection.postprocess.adaptive_regions_v2 import (
    _partition, _valid, aggregate_detections)
from src.cashew_detection.postprocess.geometry import intersection_area
from src.cashew_detection.postprocess.region_metrics import per_image_metrics, summarize_tp_composition
from scripts.run_acdra_v2 import _main_class, load_class_names, load_split


CLASSES = {0: "anthracnose", 1: "leaf_miner", 2: "red_rust"}


def test_negative_image_filename_maps_to_report_slice():
    assert _main_class("not_cashew_neg_0101.jpg", CLASSES) == "not_cashew_leaf"


def det(i, box, cls=2, conf=.4):
    return {"id": str(i), "class_id": cls, "confidence": conf, "bbox_xyxy": box}


def aggregate(data, variant="B", size=(640, 640), **config):
    return aggregate_detections(data, size, "image.jpg", CLASSES,
                                {"variant": variant, **config})


@pytest.mark.parametrize("variant", ["B", "C"])
def test_single_lesion_keeps_box(variant):
    box = [10, 10, 20, 20]
    result = aggregate([det(1, box)], variant)
    assert result[0]["mode"] == "individual"
    assert result[0]["bbox_xyxy"] == box


@pytest.mark.parametrize("variant", ["B", "C"])
def test_sparse_three_remain_individual(variant):
    result = aggregate([det(i, [i * 140, 10, i * 140 + 12, 22]) for i in range(3)], variant)
    assert len(result) == 3 and all(x["mode"] == "individual" for x in result)


@pytest.mark.parametrize("variant", ["B", "C"])
def test_dense_twenty_merge_with_all_ids(variant):
    data = [det(i, [100 + i % 5 * 10, 100 + i // 5 * 10,
                    116 + i % 5 * 10, 116 + i // 5 * 10]) for i in range(20)]
    result = aggregate(data, variant, min_group_members=2)
    assert len(result) < 20
    assert {member for region in result for member in region["member_detection_ids"]} == {str(i) for i in range(20)}


@pytest.mark.parametrize("variant", ["B", "C"])
def test_distant_clusters_do_not_cross_merge(variant):
    data = [det(i, [50 + i * 8, 50, 68 + i * 8, 68]) for i in range(6)]
    data += [det(i + 6, [450 + i * 8, 450, 468 + i * 8, 468]) for i in range(6)]
    result = aggregate(data, variant, min_group_members=2)
    assert all(region["bbox_xyxy"][2] < 300 or region["bbox_xyxy"][0] > 300 for region in result)


@pytest.mark.parametrize("variant", ["B", "C"])
def test_cross_class_containment_keeps_both(variant):
    result = aggregate([det(1, [80, 80, 180, 180], 2),
                        det(2, [100, 100, 120, 120], 0)], variant)
    assert len(result) == 2 and {r["class_id"] for r in result} == {0, 2}
    assert all(r["mode"] == "individual" for r in result)
    assert all(r["other_class_overlap_ratio"] > 0 for r in result)


@pytest.mark.parametrize("variant", ["B", "C"])
def test_cross_class_overlap_is_reported(variant):
    raw = [det(1, [100, 100, 130, 130], 0), det(2, [110, 110, 140, 140], 2)]
    result = aggregate(raw, variant)
    row, _ = per_image_metrics("image.jpg", raw, result, [], (640, 640))
    assert row["cross_class_overlap_pairs"] == 1


@pytest.mark.parametrize("variant", ["B", "C"])
def test_long_chain_never_makes_giant_bbox(variant):
    raw = [det(i, [10 + i * 11, 100, 20 + i * 11, 110]) for i in range(50)]
    result = aggregate(raw, variant, min_group_members=2, k=4, tau_fill=.15)
    assert all((r["bbox_xyxy"][2] - r["bbox_xyxy"][0]) / 640 <= .55 or
               r["mode"] == "individual" for r in result)


@pytest.mark.parametrize("variant", ["B", "C"])
def test_low_confidence_100_boxes_not_discarded_or_inflated(variant):
    raw = [det(i, [20 + i % 10 * 12, 20 + i // 10 * 12,
                   30 + i % 10 * 12, 30 + i // 10 * 12], conf=.21) for i in range(100)]
    result = aggregate(raw, variant, min_group_members=2)
    assert sum(r["n_detections"] for r in result) == 100
    assert all(r["yolo_conf_max"] == pytest.approx(.21) for r in result)
    assert all("region_probability" not in r for r in result)


@pytest.mark.parametrize("variant", ["B", "C"])
def test_empty_and_relative_image_sizes(variant):
    assert aggregate([], variant) == []
    for size in ((320, 240), (1280, 720)):
        raw = [det(1, [10, 10, 20, 20]), det(2, [25, 10, 35, 20])]
        result = aggregate(raw, variant, size=size, min_group_members=2)
        assert all(0 <= v <= max(size) for r in result for v in r["bbox_xyxy"])


def test_class_mapping_from_yaml_order(tmp_path):
    archive_path = tmp_path / "mapping.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("data_fixed.yaml", "names:\n- red_rust\n- anthracnose\n- leaf_miner\n")
    with zipfile.ZipFile(archive_path) as archive:
        names = load_class_names(archive)
    result = aggregate_detections([det(1, [10, 10, 20, 20], cls=0)],
                                  (640, 640), "image.jpg", names, {"variant": "B"})
    assert result[0]["class_name"] == "red_rust"


@pytest.mark.parametrize("variant", ["B", "C"])
def test_membership_lost_tp_and_determinism(variant):
    raw = [det(i, [100 + i * 8, 100, 116 + i * 8, 116], cls=0, conf=.8) for i in range(8)]
    first = aggregate(raw, variant, min_group_members=2)
    assert first == aggregate(list(reversed(raw)), variant, min_group_members=2)
    assert sorted(x for r in first for x in r["member_detection_ids"]) == sorted(str(i) for i in range(8))
    gt = [{"class_id": 0, "bbox_xyxy": raw[i]["bbox_xyxy"]} for i in range(8)]
    row, _ = per_image_metrics("image.jpg", raw, first, gt, (640, 640))
    assert row["lost_tp"] == 0


def test_tuning_refuses_test_split():
    with pytest.raises(ValueError, match="Validation only"):
        load_split("test")


@pytest.mark.parametrize("variant", ["B", "C"])
def test_diagonal_sparse_fill_never_creates_empty_giant_region(variant):
    raw = [det(i, [20 + i * 23, 20 + i * 23, 30 + i * 23, 30 + i * 23]) for i in range(18)]
    result = aggregate(raw, variant, min_group_members=2, k=4, tau_fill=.35,
                       min_fill_ratio=.35)
    assert all(r["fill_ratio"] >= .35 for r in result if r["mode"] == "aggregated")


def test_recursive_spatial_partition_produces_feasible_pieces():
    items = [det(i, [20 + i * 25, 20 + i * 25,
                     30 + i * 25, 30 + i * 25]) for i in range(16)]
    config = {"max_region_area_ratio": .10, "max_region_span_ratio": .35,
              "min_fill_ratio": .35}
    parts = _partition(list(range(len(items))), items, (640, 640), config)
    assert len(parts) > 1
    assert sorted(i for part in parts for i in part) == list(range(len(items)))
    assert all(_valid([items[i] for i in part], (640, 640), config) for part in parts)


@pytest.mark.parametrize("variant", ["B", "C"])
def test_500_box_performance(variant):
    raw = [det(i, [10 + i % 25 * 20, 10 + i // 25 * 20,
                   25 + i % 25 * 20, 25 + i // 25 * 20]) for i in range(500)]
    start = time.perf_counter()
    result = aggregate(raw, variant, min_group_members=2)
    assert time.perf_counter() - start < 10.0
    assert sum(r["n_detections"] for r in result) == 500


def test_tp_composition_counts_majority_and_absorbed_fp():
    rows = [
        {"n_detections": 1, "tp_members": 1},
        {"n_detections": 4, "tp_members": 2},
        {"n_detections": 3, "tp_members": 1},
        {"n_detections": 2, "tp_members": 0},
    ]
    result = summarize_tp_composition(rows, raw_fp_count=6)
    assert result["tp_majority_regions"] == 2
    assert result["tp_majority_region_share"] == .5
    assert result["aggregated_tp_majority_regions"] == 1
    assert result["aggregated_regions"] == 3
    assert result["absorbed_raw_fp_boxes"] == 4
    assert result["absorbed_raw_fp_share"] == pytest.approx(4 / 6)
    assert result["regions_mixing_tp_fp"] == 2


def test_tp_composition_empty_and_invalid():
    assert summarize_tp_composition([], 0)["tp_majority_region_share"] == 0
    with pytest.raises(ValueError):
        summarize_tp_composition([{"n_detections": 2, "tp_members": 3}], 0)
