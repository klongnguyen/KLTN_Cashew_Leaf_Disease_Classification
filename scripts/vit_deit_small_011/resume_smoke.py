"""Filesystem-only interruption/resume checks; never counts as a training seed."""

import json
import tempfile
from pathlib import Path


def run(_output):
    from runner import backup_stage2, marker_sha, validate_stage, write_json
    from dataset_audit import sha_file

    with tempfile.TemporaryDirectory(prefix="exp011_resume_smoke_") as temporary:
        root = Path(temporary)
        seed = root / "seed_42"
        checkpoint = seed / "checkpoints" / "stage1_best.keras"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"synthetic-checkpoint-for-resume-test")
        config_sha = "A" * 64
        metrics = seed / "metrics_stage1.json"
        write_json(metrics, {"checkpoint_sha256": sha_file(checkpoint)})
        write_json(seed / "stage1_complete.json", {
            "seed": 42, "stage": "stage1", "config_sha256": config_sha,
            "checkpoint_sha256": sha_file(checkpoint),
            "metrics_sha256": marker_sha(metrics)})
        assert validate_stage(seed, "stage1", config_sha)
        stage1_before = sha_file(checkpoint)
        partial = seed / "checkpoints" / "stage2_best.part.keras"
        partial.write_bytes(b"partial-stage2")
        backup_stage2(seed, root)
        assert not partial.exists()
        assert sha_file(checkpoint) == stage1_before
        assert validate_stage(seed, "stage1", config_sha)
        backups = list((root / "_incomplete_backups").glob("seed_42_stage2_*"))
        assert len(backups) == 1 and (backups[0] / partial.name).is_file()
        print(json.dumps({"status": "PASS", "stage1_reused_sha256": stage1_before,
                          "stage2_partial_preserved": True,
                          "no_training_invoked": True}, indent=2))


if __name__ == "__main__":
    run(None)
