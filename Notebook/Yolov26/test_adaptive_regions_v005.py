"""Run with: python -m unittest Notebook/Yolov26/test_adaptive_regions_v005.py"""

import importlib.util
import unittest
from pathlib import Path


module_path = Path(__file__).with_name("adaptive_regions_v005.py")
spec = importlib.util.spec_from_file_location("adaptive_regions_v005", module_path)
regions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(regions)


CLASSES = {0: "anthracnose", 1: "leaf_miner", 2: "red_rust"}


def det(number, box, class_id=2, score=0.5):
    return {"id": str(number), "bbox_xyxy": box,
            "class_id": class_id, "confidence": score}


class AdaptiveRegionsTest(unittest.TestCase):
    def run_regions(self, detections, size=(640, 640)):
        return regions.aggregate_detections(detections, size, "image-1", CLASSES)

    def test_empty_and_single_preserve(self):
        self.assertEqual(self.run_regions([]), [])
        single = det(1, [10, 10, 20, 20])
        result = self.run_regions([single])
        self.assertEqual(result[0]["mode"], "individual")
        self.assertEqual(result[0]["bbox_xyxy"], single["bbox_xyxy"])
        self.assertEqual(result[0]["member_detection_ids"], ["1"])

    def test_sparse_three_stay_separate(self):
        data = [det(i, [i * 140, 10, i * 140 + 12, 22]) for i in range(3)]
        result = self.run_regions(data)
        self.assertEqual(len(result), 3)
        self.assertTrue(all(r["mode"] == "individual" for r in result))

    def test_strong_overlap_two_display_as_one(self):
        data = [det(1, [100, 100, 130, 130]), det(2, [102, 102, 132, 132])]
        result = self.run_regions(data)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["member_detection_ids"], ["1", "2"])

    def test_dense_twenty_merge_without_losing_members(self):
        data = [det(i, [100 + (i % 5) * 10, 100 + (i // 5) * 10,
                        116 + (i % 5) * 10, 116 + (i // 5) * 10])
                for i in range(20)]
        result = self.run_regions(data)
        self.assertLess(len(result), 20)
        self.assertEqual({str(i) for i in range(20)},
                         {m for r in result for m in r["member_detection_ids"]})
        self.assertTrue(any(r["mode"] == "aggregated" for r in result))

    def test_two_distant_clusters_do_not_cross_merge(self):
        data = [det(i, [100 + i * 8, 100, 118 + i * 8, 118]) for i in range(6)]
        data += [det(i + 6, [450 + i * 8, 450, 468 + i * 8, 468]) for i in range(6)]
        result = self.run_regions(data)
        self.assertGreaterEqual(len(result), 2)
        self.assertTrue(all(r["bbox_xyxy"][2] < 300 or
                            r["bbox_xyxy"][0] > 300 for r in result))

    def test_cross_class_overlap_keeps_both(self):
        data = [det(1, [100, 100, 130, 130], 0),
                det(2, [101, 101, 131, 131], 2)]
        result = self.run_regions(data)
        self.assertEqual(len(result), 2)
        self.assertEqual({r["class_id"] for r in result}, {0, 2})

    def test_chain_guard_and_determinism(self):
        data = [det(i, [10 + i * 11, 100, 20 + i * 11, 110]) for i in range(50)]
        first = self.run_regions(data)
        second = self.run_regions(list(reversed(data)))
        self.assertEqual(first, second)
        self.assertGreater(len(first), 1)
        self.assertTrue(all((r["bbox_xyxy"][2] - r["bbox_xyxy"][0]) / 640 <= .55
                            for r in first))

    def test_rejects_duplicate_ids(self):
        with self.assertRaises(ValueError):
            self.run_regions([det(1, [10, 10, 20, 20]), det(1, [30, 30, 40, 40])])


if __name__ == "__main__":
    unittest.main()
