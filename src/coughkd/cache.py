"""Config-hashed feature cache helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def stable_config_hash(config: Any) -> str:
    if is_dataclass(config):
        payload = asdict(config)
    elif isinstance(config, dict):
        payload = config
    else:
        payload = getattr(config, "__dict__", {"value": repr(config)})
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def feature_cache_path(out_dir: Path, recording_id: str, config: Any) -> Path:
    safe_id = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in recording_id)
    return out_dir / "feature_cache" / f"{safe_id}_{stable_config_hash(config)}.json"


def save_feature_cache(out_dir: Path, recording_id: str, config: Any, features: list[list[float]]) -> Path:
    path = feature_cache_path(out_dir, recording_id, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recording_id": recording_id,
        "config_hash": stable_config_hash(config),
        "shape": [len(features), len(features[0]) if features else 0],
        "features": features,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def prediction_cache_path(out_dir: Path, recording_id: str, model_id: str, config: Any) -> Path:
    safe_recording = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in recording_id)
    safe_model = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in model_id)
    return out_dir / "prediction_cache" / f"{safe_recording}_{safe_model}_{stable_config_hash(config)}.json"


def save_prediction_cache(
    out_dir: Path,
    recording_id: str,
    model_id: str,
    config: Any,
    logits: list[float],
    embedding: list[float],
) -> Path:
    path = prediction_cache_path(out_dir, recording_id, model_id, config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recording_id": recording_id,
        "model_id": model_id,
        "config_hash": stable_config_hash(config),
        "logits": logits,
        "embedding": embedding,
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
