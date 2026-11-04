"""One-time internal V007 Test evaluation after explicit owner-selected lock."""

import hashlib
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_acdra_v2_locked import test_records
from scripts.run_acdra_v2 import EXP as V006_EXP, REPORT as V006_REPORT, sha256, verify_sources
from scripts.run_acdra_v3 import EXP, REPORT, evaluate


def main():
    lock_path = EXP / "config_locked_v007.yaml"
    lock_hash = EXP / "config_locked_v007.sha256"
    if not lock_path.is_file() or not lock_hash.is_file():
        raise RuntimeError("Owner-approved V007 configuration is not locked")
    if sha256(lock_path) != lock_hash.read_text(encoding="utf-8").strip():
        raise RuntimeError("V007 lock checksum mismatch")
    completion = REPORT / "test_evaluation_completed.json"
    if completion.exists():
        raise RuntimeError("V007 Test already evaluated once")
    verify_sources()
    v006_done = json.loads((V006_REPORT / "test_evaluation_completed.json").read_text(encoding="utf-8"))
    raw_path = V006_EXP / "results" / "test_raw_detections.jsonl"
    if sha256(raw_path) != v006_done["raw_jsonl_sha256"]:
        raise RuntimeError("Frozen V006 Test raw detections changed")
    locked = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if locked.get("selection_source") != "Validation only":
        raise RuntimeError("V007 lock selection provenance invalid")
    records, class_names = test_records(raw_path)
    summary, _, _ = evaluate(records, class_names, locked["candidate_config"],
                             EXP / "results" / "test_locked_internal",
                             save_regions=True, split_name="test")
    if summary["lost_tp"]:
        raise AssertionError("V007 Test lost TP")
    REPORT.mkdir(parents=True, exist_ok=True)
    result = {"status": "COMPLETE", "label": "đánh giá nội bộ, không còn là Test nguyên vẹn",
              "owner_approved_point": locked["owner_approved_point"],
              "trial_id": locked["trial_id"], "lock_sha256": sha256(lock_path),
              "raw_jsonl_sha256": sha256(raw_path),
              "summary": summary}
    completion.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print(result["label"], summary["images"], summary["regions"])


if __name__ == "__main__":
    main()
