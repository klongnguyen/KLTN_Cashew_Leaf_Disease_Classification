"""Upload the local classification result tree to a Google Drive folder.

The notebooks call this module through rclone because model checkpoints can be
large and rclone provides resumable, incremental transfers.  Authentication is
kept in rclone's own configuration instead of being embedded in a notebook.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _rclone_executable(executable: str | None = None) -> str:
    candidate = executable or os.environ.get("RCLONE_EXE") or shutil.which("rclone")
    if not candidate:
        raise FileNotFoundError(
            "rclone was not found. Install rclone, run 'rclone config', and "
            "create a Google Drive remote named 'gdrive' before training."
        )
    return str(candidate)


def _remote_name(remote: str) -> str:
    name = remote.strip().rstrip(":")
    if not name:
        raise ValueError("The rclone remote name cannot be empty.")
    return name


def verify_drive_access(
    target_folder_id: str,
    remote: str = "gdrive",
    executable: str | None = None,
) -> str:
    """Fail fast unless rclone can authenticate and read the target folder."""
    rclone = _rclone_executable(executable)
    remote = _remote_name(remote)

    configured = subprocess.run(
        [rclone, "listremotes"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if f"{remote}:" not in configured:
        raise RuntimeError(
            f"rclone remote '{remote}:' is not configured. Run 'rclone config' "
            "and create a Google Drive remote with that name."
        )

    subprocess.run(
        [
            rclone,
            "lsf",
            f"{remote}:",
            "--dirs-only",
            "--max-depth",
            "1",
            "--drive-root-folder-id",
            target_folder_id,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return rclone


def upload_current_dataset(
    local_results_root: str | Path,
    target_folder_id: str,
    remote: str = "gdrive",
    remote_folder: str = "current_dataset",
    executable: str | None = None,
) -> str:
    """Incrementally copy local results and verify every local file by size."""
    local_root = Path(local_results_root).expanduser().resolve(strict=True)
    if not local_root.is_dir():
        raise NotADirectoryError(local_root)

    remote = _remote_name(remote)
    rclone = verify_drive_access(target_folder_id, remote, executable)
    destination = f"{remote}:{remote_folder.strip('/')}"
    drive_scope = ["--drive-root-folder-id", target_folder_id]

    subprocess.run(
        [
            rclone,
            "copy",
            str(local_root),
            destination,
            *drive_scope,
            "--create-empty-src-dirs",
            "--progress",
            "--transfers",
            "4",
            "--checkers",
            "8",
            "--retries",
            "5",
            "--low-level-retries",
            "10",
        ],
        check=True,
    )

    subprocess.run(
        [
            rclone,
            "check",
            str(local_root),
            destination,
            *drive_scope,
            "--one-way",
            "--size-only",
        ],
        check=True,
    )
    return destination
