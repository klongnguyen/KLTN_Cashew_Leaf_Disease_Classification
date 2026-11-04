"""Resumable Validation-only V007 sweep; never reads or optimizes on Test."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_acdra_v3 import EXP, REPORT, evaluate, load_validation
from scripts.tune_acdra_v2_on_val import flatten, write_csv

V006_TUNING = ROOT / "reports" / "EXP-006" / "validation_tuning.csv"


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def trial_configs():
    with V006_TUNING.open(newline="", encoding="utf-8") as stream:
        old = {row["trial_id"]: json.loads(row["config_json"])
               for row in csv.DictReader(stream)}
    for trial_id in ("A_v005", "B_e6e7d65f", "B_08ff744d",
                     "C2_6451497a", "C2_fc918698"):
        yield trial_id, old[trial_id], "V006 reference"
    base = {"variant": "B", "min_group_members": 2,
            "max_region_area_ratio": .10, "max_region_span_ratio": .35,
            "min_fill_ratio": .15}
    new = []
    for k in (.75, 1, 1.5, 2, 2.5, 3):
        new.append({**base, "k": k})
    new.extend((
        {**base, "k": 1.5, "max_region_area_ratio": .15,
         "max_region_span_ratio": .45, "min_fill_ratio": .20},
        {**base, "k": 2, "max_region_area_ratio": .20,
         "max_region_span_ratio": .45, "min_fill_ratio": .25},
        {"variant": "C", "tau_fill": .20, "min_fill_ratio": .20,
         "min_group_members": 3, "max_region_area_ratio": .25,
         "max_region_span_ratio": .45},
        {"variant": "C", "tau_fill": .25, "min_fill_ratio": .20,
         "min_group_members": 2, "max_region_area_ratio": .20,
         "max_region_span_ratio": .45},
    ))
    for config in new:
        yield "V7_" + config_hash(config)[:10], config, "V007 coarse/fine B-C"
    for theta in (.1, .25, .5):
        config = {**old["B_08ff744d"], "variant": "D", "base_variant": "B",
                  "theta_ov": theta, "tau_fill": .15}
        yield "V7_" + config_hash(config)[:10], config, "V007 D post-merge"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    if args.split != "val":
        raise SystemExit("V007 tuning refuses split=test")
    if (EXP / "config_locked_v007.yaml").exists():
        raise RuntimeError("V007 configuration is locked; tuning is forbidden")
    REPORT.mkdir(parents=True, exist_ok=True)
    result_dir = EXP / "results" / "trials"
    result_dir.mkdir(parents=True, exist_ok=True)
    csv_path = REPORT / "validation_tuning_v007.csv"
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8"))) if csv_path.exists() else []
    completed = {row["trial_id"] for row in rows}
    records, names, _ = load_validation()
    for trial_id, config, phase in trial_configs():
        if trial_id in completed:
            continue
        print("RUN", trial_id, config, flush=True)
        summary, _, _ = evaluate(records, names, config)
        if summary["lost_tp"] != 0:
            raise AssertionError(f"lost_tp != 0 in {trial_id}")
        row = flatten(summary, trial_id, config)
        row.update({"config_sha256": config_hash(config), "phase": phase,
                    "split": "val", "valid_for_selection": True})
        rows.append(row)
        write_csv(csv_path, rows)
        (result_dir / f"{trial_id}.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        completed.add(trial_id)
        print("DONE", trial_id, "dense", summary["dense_ge100"]["reduction"],
              "excess", summary["region_excess_area_ratio"],
              "strict", summary["region_precision_strict"],
              "overlap", summary["same_class_overlap_area_ratio"], flush=True)
    print("COMPLETE", len(rows), "trials", flush=True)


if __name__ == "__main__":
    main()
