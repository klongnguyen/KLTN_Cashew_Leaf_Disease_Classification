"""Recompute five V006 Validation references and stop on any metric mismatch."""

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_acdra_v2 import evaluate, load_split
from scripts.run_acdra_v3 import REPORT

REFERENCES = ("A_v005", "B_e6e7d65f", "B_08ff744d",
              "C2_6451497a", "C2_fc918698")
METRICS = ("dense_ge100.reduction", "region_excess_area_ratio",
           "sparse_le10.reduction")


def main():
    with (ROOT / "reports/EXP-006/validation_tuning.csv").open(
            newline="", encoding="utf-8") as stream:
        reference = {row["trial_id"]: row for row in csv.DictReader(stream)
                     if row["trial_id"] in REFERENCES}
    if set(reference) != set(REFERENCES):
        raise RuntimeError("A V006 reference trial is missing")
    records, names, _ = load_split("val")
    rows = []
    for trial_id in REFERENCES:
        row = reference[trial_id]
        config = json.loads(row["config_json"])
        started = time.perf_counter()
        summary, _, _ = evaluate(records, names, config,
                                 baseline_check=(trial_id == "A_v005"))
        actual = (summary["dense_ge100"]["reduction"],
                  summary["region_excess_area_ratio"],
                  summary["sparse_le10"]["reduction"])
        expected = tuple(float(row[key]) for key in METRICS)
        delta = max(abs(a - b) for a, b in zip(actual, expected))
        if delta > 1e-10 or summary["lost_tp"]:
            raise RuntimeError(f"V006 reference mismatch: {trial_id}: delta={delta}")
        rows.append({"trial_id": trial_id, "dense_reduction": actual[0],
                     "excess": actual[1], "sparse_reduction": actual[2],
                     "max_delta": delta, "elapsed_seconds": time.perf_counter() - started})
        print(trial_id, rows[-1], flush=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "baseline_reproduction.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
