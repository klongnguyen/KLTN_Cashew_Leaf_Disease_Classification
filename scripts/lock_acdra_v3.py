"""Lock one owner-approved V007 Validation candidate before any Test evaluation."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_acdra_v3 import EXP, REPORT


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-point", required=True,
                        choices=("conservative", "balanced", "aggressive"))
    args = parser.parse_args()
    lock_path = EXP / "config_locked_v007.yaml"
    if lock_path.exists():
        raise RuntimeError("V007 is already locked")
    if (REPORT / "test_evaluation_completed.json").exists():
        raise RuntimeError("V007 Test was already evaluated")
    points = yaml.safe_load((REPORT / "operating_points_v007.yaml").read_text(encoding="utf-8"))
    point = points[args.approved_point]
    required = ("validation_tuning_v007.csv", "crop_readiness.csv",
                "visual_gallery_manifest.csv", "EXP007_FINAL_REPORT.md")
    if any(not (REPORT / name).is_file() for name in required):
        raise RuntimeError("Validation review artifacts are incomplete")
    lock = {"experiment_id": EXP.name, "selection_source": "Validation only",
            "owner_approved_point": args.approved_point,
            "trial_id": point["trial_id"], "candidate_config": point["config"],
            "review_artifact_sha256": {name: sha256(REPORT / name) for name in required},
            "test_policy": "one internal evaluation; Test non-pristine since V006; no retuning"}
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    (EXP / "config_locked_v007.sha256").write_text(sha256(lock_path) + "\n",
                                                      encoding="utf-8")
    print(json.dumps({"locked_point": args.approved_point,
                      "trial_id": point["trial_id"],
                      "sha256": sha256(lock_path)}))


if __name__ == "__main__":
    main()
