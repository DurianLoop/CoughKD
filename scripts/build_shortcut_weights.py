from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.audio import audio_stats, log_mel_like, read_wav_mono, resample_linear
from coughkd.config import RunConfig
from coughkd.manifest import read_manifest
from coughkd.torch_models import DepthwiseStudent, pad_collate


def _audio_features(row: dict[str, str], root: Path, config: RunConfig, max_duration_sec: float) -> dict[str, float]:
    samples, sample_rate = read_wav_mono(root / row["path"])
    samples = resample_linear(samples, sample_rate, config.sample_rate)
    if max_duration_sec > 0:
        samples = samples[: int(max_duration_sec * config.sample_rate)]
    stats = audio_stats(samples, config.sample_rate, config)
    peak = max((abs(value) for value in samples), default=0.0)
    silence_ratio = sum(1 for value in samples if abs(value) < 0.005) / max(1, len(samples))
    zcr = _zero_crossing_rate(samples)
    return {
        "duration_sec": stats.duration_sec,
        "rms": stats.rms,
        "log_rms": math.log10(max(stats.rms, 1e-8)),
        "clipping_fraction": stats.clipping_fraction,
        "silence_ratio": silence_ratio,
        "peak": peak,
        "zcr": zcr,
        "cough_detected": _safe_float(row.get("cough_detected", ""), 1.0),
    }


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


def _quality_weight(features: dict[str, float]) -> float:
    duration_w = _ramp(features["duration_sec"], lo=0.4, hi=2.0)
    rms_low = _ramp(features["rms"], lo=0.003, hi=0.025)
    rms_high = 1.0 - _ramp(features["rms"], lo=0.35, hi=0.70)
    clipping_w = 1.0 - _ramp(features["clipping_fraction"], lo=0.001, hi=0.05)
    silence_w = 1.0 - _ramp(features["silence_ratio"], lo=0.75, hi=0.98)
    cough_w = _ramp(features["cough_detected"], lo=0.60, hi=0.90)
    return _clip01(duration_w * rms_low * rms_high * clipping_w * silence_w * cough_w)


def _ramp(value: float, lo: float, hi: float) -> float:
    if value <= lo:
        return 0.0
    if value >= hi:
        return 1.0
    return (value - lo) / max(1e-12, hi - lo)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _safe_float(value: str, default: float) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def _domain_weights(
    source_records: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
    seed: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    if not target_records:
        return {}, {"enabled": False, "reason": "no target rows"}
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception as exc:
        return {}, {"enabled": False, "reason": f"sklearn unavailable: {exc}"}

    feature_names = ["duration_sec", "log_rms", "clipping_fraction", "silence_ratio", "peak", "zcr", "cough_detected"]
    rows = source_records + target_records
    x = np.array([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float32)
    y = np.array([0] * len(source_records) + [1] * len(target_records), dtype=np.int64)
    if len(set(y.tolist())) < 2:
        return {}, {"enabled": False, "reason": "single domain only"}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(x_train, y_train)
    auc = float(roc_auc_score(y_test, clf.predict_proba(x_test)[:, 1]))
    probs = clf.predict_proba(x[: len(source_records)])
    weights: dict[str, float] = {}
    for row, prob in zip(source_records, probs):
        source_confidence = float(prob[0])
        domain_typicality = abs(source_confidence - 0.5) * 2.0
        weights[str(row["recording_id"])] = 1.0 - _clip01(domain_typicality)
    return weights, {"enabled": True, "domain_probe_auc": auc, "feature_names": feature_names}


def _stability_weights(
    source_rows: list[dict[str, str]],
    root: Path,
    checkpoint: Path | None,
    seed: int,
    device_name: str,
    max_duration_sec: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    if checkpoint is None:
        return {}, {"enabled": False, "reason": "no stability checkpoint"}
    try:
        import torch
    except Exception as exc:
        return {}, {"enabled": False, "reason": f"torch unavailable: {exc}"}
    if not checkpoint.is_file():
        return {}, {"enabled": False, "reason": f"checkpoint not found: {checkpoint}"}

    config = RunConfig(experiment_name="shortcut_stability", seed=seed)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(k): int(v) for k, v in payload["label_to_index"].items()}
    model = DepthwiseStudent(num_classes=len(label_to_index))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(device)
    model.eval()

    rng = random.Random(seed)
    weights: dict[str, float] = {}
    divergences: list[float] = []
    with torch.no_grad():
        for idx, row in enumerate(source_rows, start=1):
            samples, sample_rate = read_wav_mono(root / row["path"])
            samples = resample_linear(samples, sample_rate, config.sample_rate)
            samples = samples[: int(max_duration_sec * config.sample_rate)]
            original = _probs_for_samples(model, samples, config, device)
            variants = [
                [value * 0.85 for value in samples],
                _time_shift(samples, int(0.05 * config.sample_rate)),
                _add_noise(samples, amplitude=0.003, rng=rng),
            ]
            max_js = max(_js_divergence(original, _probs_for_samples(model, variant, config, device)) for variant in variants)
            divergences.append(max_js)
            weights[row["recording_id"]] = 1.0 - _clip01(max_js / 0.20)
            if idx == 1 or idx % 200 == 0:
                print(f"[stability] {idx}/{len(source_rows)}", flush=True)
    summary = {
        "enabled": True,
        "checkpoint": str(checkpoint),
        "mean_js": sum(divergences) / max(1, len(divergences)),
        "max_js": max(divergences) if divergences else 0.0,
    }
    return weights, summary


def _probs_for_samples(model: Any, samples: list[float], config: RunConfig, device: Any) -> list[float]:
    import torch

    features = torch.tensor(log_mel_like(samples, config), dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    batch = pad_collate([{"features": features.squeeze(0), "waveform": torch.tensor(samples), "label": torch.tensor(0), "kd_weight": torch.tensor(1.0), "recording_id": "x"}])
    out = model(batch["features"].to(device))
    return [float(value) for value in torch.softmax(out["logits"], dim=1).squeeze(0).detach().cpu().tolist()]


def _time_shift(samples: list[float], shift: int) -> list[float]:
    if not samples:
        return samples
    shift = shift % len(samples)
    return samples[-shift:] + samples[:-shift] if shift else samples


def _add_noise(samples: list[float], amplitude: float, rng: random.Random) -> list[float]:
    return [max(-1.0, min(1.0, value + rng.uniform(-amplitude, amplitude))) for value in samples]


def _js_divergence(left: list[float], right: list[float]) -> float:
    mid = [(a + b) * 0.5 for a, b in zip(left, right)]
    return 0.5 * _kl(left, mid) + 0.5 * _kl(right, mid)


def _kl(left: list[float], right: list[float]) -> float:
    total = 0.0
    for a, b in zip(left, right):
        if a > 0:
            total += a * math.log(a / max(b, 1e-12))
    return total


def _combine_weights(
    rows: list[dict[str, Any]],
    domain_weights: dict[str, float],
    stability_weights: dict[str, float],
    quality_power: float,
    domain_power: float,
    stability_power: float,
    floor: float,
) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rid = str(row["recording_id"])
        quality = float(row["quality_weight"])
        domain = float(domain_weights.get(rid, 1.0))
        stability = float(stability_weights.get(rid, 1.0))
        kd_weight = (quality**quality_power) * (domain**domain_power) * (stability**stability_power)
        kd_weight = max(floor, min(1.0, kd_weight))
        out.append({**row, "domain_weight": domain, "stability_weight": stability, "kd_weight": kd_weight})
    return out


def _summarize(records: list[dict[str, Any]], field: str) -> dict[str, float]:
    values = sorted(float(row[field]) for row in records)
    if not values:
        return {"min": 0.0, "mean": 0.0, "max": 0.0}
    return {
        "min": values[0],
        "p10": values[int(0.10 * (len(values) - 1))],
        "mean": sum(values) / len(values),
        "p90": values[int(0.90 * (len(values) - 1))],
        "max": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--target-manifest", type=Path, default=ROOT / "manifests/coughvid_adapt_test.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/shortcut_weights/panns_depthwise_seed7")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--target-split", default="adapt")
    parser.add_argument("--max-source-records", type=int)
    parser.add_argument("--max-target-records", type=int, default=1500)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--stability-checkpoint", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--quality-power", type=float, default=1.0)
    parser.add_argument("--domain-power", type=float, default=1.0)
    parser.add_argument("--stability-power", type=float, default=1.0)
    parser.add_argument("--floor", type=float, default=0.05)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = RunConfig(experiment_name="shortcut_weights", seed=args.seed)
    source_rows = [row for row in read_manifest(args.source_manifest) if row.get("split", "") == args.source_split]
    target_rows = [row for row in read_manifest(args.target_manifest) if row.get("split", "") == args.target_split] if args.target_manifest.is_file() else []
    if args.max_source_records:
        source_rows = source_rows[: args.max_source_records]
    if args.max_target_records:
        target_rows = target_rows[: args.max_target_records]

    source_records: list[dict[str, Any]] = []
    target_records: list[dict[str, Any]] = []
    decode_errors: list[dict[str, str]] = []
    for name, rows, sink in (("source", source_rows, source_records), ("target", target_rows, target_records)):
        for idx, row in enumerate(rows, start=1):
            try:
                feats = _audio_features(row, args.root, config, args.max_duration_sec)
            except Exception as exc:
                decode_errors.append({"domain": name, "recording_id": row.get("recording_id", ""), "path": row.get("path", ""), "error": str(exc)})
                if len(decode_errors) <= 5:
                    print(f"[warn] failed to decode {name} {row.get('recording_id', '')}: {exc}", flush=True)
                continue
            sink.append({**row, **feats, "quality_weight": _quality_weight(feats)})
            if idx == 1 or idx % 500 == 0:
                print(f"[{name}] {idx}/{len(rows)}", flush=True)

    domain_weights, domain_summary = _domain_weights(source_records, target_records, args.seed)
    stability_weights, stability_summary = _stability_weights(
        source_rows,
        args.root,
        args.stability_checkpoint,
        args.seed,
        args.device,
        args.max_duration_sec,
    )
    weighted = _combine_weights(
        source_records,
        domain_weights,
        stability_weights,
        args.quality_power,
        args.domain_power,
        args.stability_power,
        args.floor,
    )

    fieldnames = [
        "recording_id",
        "subject_id",
        "dataset",
        "label",
        "split",
        "quality_weight",
        "domain_weight",
        "stability_weight",
        "kd_weight",
        "duration_sec",
        "rms",
        "clipping_fraction",
        "silence_ratio",
        "peak",
        "zcr",
        "cough_detected",
        "path",
    ]
    weights_path = args.out / "shortcut_weights.csv"
    with weights_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in weighted:
            writer.writerow({key: row.get(key, "") for key in fieldnames})

    summary = {
        "source_manifest": str(args.source_manifest),
        "target_manifest": str(args.target_manifest),
        "source_split": args.source_split,
        "target_split": args.target_split,
        "num_source": len(source_records),
        "num_target": len(target_records),
        "decode_errors": len(decode_errors),
        "decode_error_examples": decode_errors[:5],
        "quality": _summarize(weighted, "quality_weight"),
        "domain": _summarize(weighted, "domain_weight"),
        "stability": _summarize(weighted, "stability_weight"),
        "kd_weight": _summarize(weighted, "kd_weight"),
        "domain_summary": domain_summary,
        "stability_summary": stability_summary,
        "weights_path": str(weights_path),
    }
    (args.out / "shortcut_weight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Shortcut Weights",
        "",
        f"- Source: `{args.source_manifest}` split `{args.source_split}`",
        f"- Target: `{args.target_manifest}` split `{args.target_split}`",
        f"- Output: `{weights_path}`",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    (args.out / "SHORTCUT_WEIGHTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
