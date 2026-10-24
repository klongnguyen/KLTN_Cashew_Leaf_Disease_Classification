"""Run the frozen V08 ZIP audit and label its output for EXP-011."""

import argparse
import json
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "vit_deit_small"
sys.path.insert(0, str(BASE))
import dataset_audit

EXPERIMENT_ID = "EXP-VIT-DEITSMALL-TRANSFER-5SEEDS-011"


def run(archive, output):
    output = Path(output)
    report = dataset_audit.run(archive, output)
    if report["dataset_zip_sha256"] != dataset_audit.ZIP_SHA:
        raise ValueError("V08 fingerprint changed")
    report["experiment_id"] = EXPERIMENT_ID
    target = output / "dataset_audit.json"
    part = target.with_name(target.name + ".part")
    part.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                    encoding="utf-8")
    part.replace(target)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_zip")
    parser.add_argument("output")
    args = parser.parse_args()
    report = run(args.dataset_zip, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
