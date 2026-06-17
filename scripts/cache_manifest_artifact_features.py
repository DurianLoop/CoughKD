from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.audio import audio_stats, read_wav_mono, resample_linear
from coughkd.config import RunConfig
from coughkd.manifest import read_manifest


FEATURE_NAMES = [
    "duration_sec",
    "log_rms",
    "peak",
    "clipping_fraction",
    "silence_ratio",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "active_ratio",
]


def _require_numpy() -> Any:
    import numpy as np

    return np


def _zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) <= 1:
        return 0.0
    crossings = 0
    prev = samples[0]
    for value in samples[1:]:
        if (prev < 0 <= value) or (prev >= 0 > value):
            crossings += 1
        prev = value
    return crossings / (len(samples) - 1)


def _spectral_features(samples: list[float], sample_rate: int) -> dict[str, float]:
    np = _require_numpy()
    if not samples:
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }
    x = np.asarray(samples, dtype=np.float32)
    if x.shape[0] > sample_rate:
        hop = max(1, x.shape[0] // sample_rate)
        x = x[::hop][:sample_rate]
    x = x - float(np.mean(x))
    if x.shape[0] < 256:
        x = np.pad(x, (0, 256 - x.shape[0]))
    window = np.hanning(x.shape[0]).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / sample_rate)
    total = float(spec.sum()) + 1e-12
    centroid = float((freqs * spec).sum() / total)
    bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spec).sum() / total))

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(spec[mask].sum() / total)

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "low_band_ratio": band(0.0, 1000.0),
        "mid_band_ratio": band(1000.0, 4000.0),
        "high_band_ratio": band(4000.0, sample_rate / 2.0 + 1.0),
    }


def _active_ratio(samples: list[float], frame: int = 400, hop: int = 160) -> float:
    if not samples:
        return 0.0
    energies: list[float] = []
    for start in range(0, max(1, len(samples) - frame + 1), hop):
        chunk = samples[start : start + frame]
        if not chunk:
            continue
        energies.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)))
    if not energies:
        return 0.0
    values = sorted(energies)
    lo = values[int(0.10 * (len(values) - 1))]
    hi = values[int(0.90 * (len(values) - 1))]
    return hi / max(1e-8, lo)


def _audio_features(row: dict[str, str], root: Path, config: RunConfig, max_duration_sec: float) -> dict[str, Any]:
    samples, sample_rate = read_wav_mono(root / row["path"])
    samples = resample_linear(samples, sample_rate, config.sample_rate)
    if max_duration_sec > 0:
        samples = samples[: int(max_duration_sec * config.sample_rate)]
    stats = audio_stats(samples, config.sample_rate, config)
    peak = max((abs(value) for value in samples), default=0.0)
    silence_ratio = sum(1 for value in samples if abs(value) < 0.005) / max(1, len(samples))
    return {
        "recording_id": row.get("recording_id", ""),
        "subject_id": row.get("subject_id", ""),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "label": row.get("label", ""),
        "path": row.get("path", ""),
        "duration_sec": stats.duration_sec,
        "log_rms": math.log10(max(stats.rms, 1e-8)),
        "peak": peak,
        "clipping_fraction": stats.clipping_fraction,
        "silence_ratio": silence_ratio,
        "zcr": _zero_crossing_rate(samples),
        "active_ratio": _active_ratio(samples),
        **_spectral_features(samples, config.sample_rate),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", action="append", default=[])
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    splits = set(args.split)
    rows = read_manifest(args.manifest)
    if splits:
        rows = [row for row in rows if row.get("split", "") in splits]
    if args.max_records:
        rows = rows[: args.max_records]
    config = RunConfig(experiment_name="artifact_feature_cache", seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["recording_id", "subject_id", "dataset", "split", "label", "path", *FEATURE_NAMES]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            try:
                writer.writerow(_audio_features(row, args.root, config, args.max_duration_sec))
            except Exception as exc:
                print(f"[warn] failed {row.get('recording_id', '')}: {exc}", flush=True)
            if idx == 1 or idx % 500 == 0:
                print(f"[features] {idx}/{len(rows)}", flush=True)
    print(str(args.out), flush=True)


if __name__ == "__main__":
    main()
