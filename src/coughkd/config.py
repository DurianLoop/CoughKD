"""Configuration and reproducibility helpers."""

from __future__ import annotations

import json
import os
import random
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunConfig:
    experiment_name: str = "smoke"
    seed: int = 7
    sample_rate: int = 16000
    min_duration_sec: float = 0.2
    max_clip_fraction: float = 0.05
    min_rms: float = 1e-4
    n_mels: int = 32
    frame_size: int = 400
    hop_size: int = 160


def set_seed(seed: int) -> None:
    """Seed standard-library randomness.

    NumPy/PyTorch are intentionally not imported here because the foundation
    smoke pipeline must run in minimal Python environments.
    """

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def git_hash() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def save_run_metadata(out_dir: Path, config: RunConfig, command: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "config": asdict(config),
        "command": command,
        "git_hash": git_hash(),
    }
    (out_dir / "config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
