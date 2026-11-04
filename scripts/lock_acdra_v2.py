"""Freeze one owner-approved Validation operating point before Test access."""

import argparse
import sys
from pathlib import Path

import yaml

from run_acdra_v2 import EXP, REPORT, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--approved-point", required=True,
                        choices=["conservative", "balanced", "aggressive"])
    args = parser.parse_args()
    lock_path = EXP / "config_locked.yaml"
    digest_path = EXP / "config_locked.sha256"
    if lock_path.exists() or digest_path.exists():
        raise RuntimeError("Locked config already exists; do not overwrite after Test exposure")
    gallery = REPORT / "visual_gallery_manifest.csv"
    if not gallery.is_file():
        raise RuntimeError("Owner must review the completed Validation gallery before lock")
    candidates = yaml.safe_load((EXP / "operating_points.yaml").read_text(encoding="utf-8"))
    chosen = candidates[args.approved_point]
    lock = {"experiment_id": "EXP-Y26S-NEG-V03-006-ACDRA-V2",
            "selection_source": "Validation only; owner approved after gallery review",
            "approved_point": args.approved_point,
            "trial_id": chosen["trial_id"],
            "candidate_config": chosen["config"],
            "validation_tuning_sha256": sha256(REPORT / "validation_tuning.csv"),
            "gallery_manifest_sha256": sha256(gallery),
            "test_policy": "one evaluation; no retuning after Test"}
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding="utf-8")
    digest_path.write_text(sha256(lock_path) + "\n", encoding="utf-8")
    print("Locked", args.approved_point, chosen["trial_id"],
          "config SHA256", sha256(lock_path))


if __name__ == "__main__":
    main()
