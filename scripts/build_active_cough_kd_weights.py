from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.audio import read_wav_mono, resample_linear
from coughkd.config import RunConfig
from coughkd.manifest import read_manifest


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    q = max(0.0, min(1.0, q))
    pos = q * (len(ordered) - 1)
    left = int(math.floor(pos))
    right = int(math.ceil(pos))
    if left == right:
        return ordered[left]
    frac = pos - left
    return ordered[left] * (1.0 - frac) + ordered[right] * frac


def _frame_rms(samples: list[float], sample_rate: int, frame_sec: float, hop_sec: float) -> list[float]:
    frame = max(1, int(round(frame_sec * sample_rate)))
    hop = max(1, int(round(hop_sec * sample_rate)))
    if not samples:
        return []
    if len(samples) < frame:
        padded = samples + [0.0] * (frame - len(samples))
        return [math.sqrt(sum(value * value for value in padded) / len(padded))]
    values: list[float] = []
    for start in range(0, len(samples) - frame + 1, hop):
        chunk = samples[start : start + frame]
        values.append(math.sqrt(sum(value * value for value in chunk) / frame))
    return values


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _active_stats(samples: list[float], sample_rate: int, args: argparse.Namespace) -> dict[str, float]:
    eps = 1e-9
    rms_values = _frame_rms(samples, sample_rate, args.frame_sec, args.hop_sec)
    if not rms_values:
        return {
            "kd_weight": args.weight_floor,
            "active_coverage": 0.0,
            "energy_concentration": 0.0,
            "contrast": 0.0,
            "full_rms": 0.0,
            "active_rms": 0.0,
            "noise_rms": 0.0,
            "threshold": 0.0,
            "n_frames": 0.0,
        }

    noise = _percentile(rms_values, args.noise_quantile)
    high = _percentile(rms_values, args.high_quantile)
    threshold = max(args.min_active_rms, noise + args.threshold_alpha * max(0.0, high - noise))
    active = [value for value in rms_values if value >= threshold]

    total_energy = sum(value * value for value in rms_values) + eps
    active_energy = sum(value * value for value in active)
    coverage = len(active) / len(rms_values)
    concentration = active_energy / total_energy
    full_rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, len(samples)))
    active_rms = math.sqrt(active_energy / max(1, len(active))) if active else 0.0
    contrast = active_rms / (noise + eps) if active else 0.0

    if coverage < args.min_active_coverage:
        coverage_score = coverage / max(eps, args.min_active_coverage)
    elif coverage > args.max_active_coverage:
        coverage_score = (1.0 - coverage) / max(eps, 1.0 - args.max_active_coverage)
    else:
        coverage_score = 1.0
    coverage_score = _clip01(coverage_score)

    contrast_score = _clip01((contrast - 1.0) / max(eps, args.contrast_ref - 1.0))
    concentration_score = math.sqrt(_clip01(concentration))
    quality = _clip01(
        args.coverage_weight * coverage_score
        + args.contrast_weight * contrast_score
        + args.concentration_weight * concentration_score
    )
    quality = quality ** args.quality_power
    kd_weight = args.weight_floor + (args.weight_ceil - args.weight_floor) * quality
    return {
        "kd_weight": _clip01(kd_weight),
        "active_coverage": coverage,
        "energy_concentration": concentration,
        "contrast": contrast,
        "full_rms": full_rms,
        "active_rms": active_rms,
        "noise_rms": noise,
        "threshold": threshold,
        "n_frames": float(len(rms_values)),
    }


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    weights = [float(row["kd_weight"]) for row in rows]
    coverages = [float(row["active_coverage"]) for row in rows]
    if not weights:
        return {"num_records": 0}
    return {
        "num_records": len(rows),
        "kd_weight": {
            "mean": sum(weights) / len(weights),
            "p10": _percentile(weights, 0.10),
            "p50": _percentile(weights, 0.50),
            "p90": _percentile(weights, 0.90),
            "min": min(weights),
            "max": max(weights),
        },
        "active_coverage": {
            "mean": sum(coverages) / len(coverages),
            "p10": _percentile(coverages, 0.10),
            "p50": _percentile(coverages, 0.50),
            "p90": _percentile(coverages, 0.90),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/active_cough_kd_weights/active_cough_kd_weights.csv")
    parser.add_argument("--splits", default="train")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--frame-sec", type=float, default=0.08)
    parser.add_argument("--hop-sec", type=float, default=0.04)
    parser.add_argument("--noise-quantile", type=float, default=0.40)
    parser.add_argument("--high-quantile", type=float, default=0.85)
    parser.add_argument("--threshold-alpha", type=float, default=0.45)
    parser.add_argument("--min-active-rms", type=float, default=1e-4)
    parser.add_argument("--min-active-coverage", type=float, default=0.05)
    parser.add_argument("--max-active-coverage", type=float, default=0.70)
    parser.add_argument("--contrast-ref", type=float, default=4.0)
    parser.add_argument("--weight-floor", type=float, default=0.25)
    parser.add_argument("--weight-ceil", type=float, default=1.0)
    parser.add_argument("--coverage-weight", type=float, default=0.40)
    parser.add_argument("--contrast-weight", type=float, default=0.35)
    parser.add_argument("--concentration-weight", type=float, default=0.25)
    parser.add_argument("--quality-power", type=float, default=1.0)
    args = parser.parse_args()

    config = RunConfig(sample_rate=args.sample_rate)
    selected_splits = {item.strip() for item in args.splits.split(",") if item.strip()}
    rows = []
    manifest_rows = [
        row for row in read_manifest(args.manifest) if not selected_splits or row.get("split", "") in selected_splits
    ]
    if args.max_records is not None:
        manifest_rows = manifest_rows[: args.max_records]

    max_samples = max(1, int(args.max_duration_sec * config.sample_rate)) if args.max_duration_sec else None
    for index, row in enumerate(manifest_rows, start=1):
        path = args.root / row["path"]
        samples, sample_rate = read_wav_mono(path)
        samples = resample_linear(samples, sample_rate, config.sample_rate)
        if max_samples is not None:
            samples = samples[:max_samples]
        stats = _active_stats(samples, config.sample_rate, args)
        out_row: dict[str, Any] = {
            "recording_id": row["recording_id"],
            "split": row.get("split", ""),
            "label": row.get("label", ""),
            "path": row.get("path", ""),
        }
        out_row.update({key: f"{value:.8f}" for key, value in stats.items()})
        rows.append(out_row)
        if index == 1 or index % 500 == 0:
            print(f"[weights] processed {index}/{len(manifest_rows)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_id",
        "split",
        "label",
        "path",
        "kd_weight",
        "active_coverage",
        "energy_concentration",
        "contrast",
        "full_rms",
        "active_rms",
        "noise_rms",
        "threshold",
        "n_frames",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    config_payload = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    summary = {"config": config_payload, "summary": _summarize(rows)}
    summary_path = args.out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"weights": str(args.out), "summary": str(summary_path), **summary["summary"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
