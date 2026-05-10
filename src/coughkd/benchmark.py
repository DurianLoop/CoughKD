"""Smoke export and efficiency benchmarking utilities."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

from .models import BaseAudioModel


def quantize_values(values: list[float], mode: str) -> list[float | int]:
    if mode == "fp32":
        return [float(value) for value in values]
    if mode == "fp16":
        return [round(float(value), 4) for value in values]
    if mode == "int8":
        return [max(-128, min(127, int(round(value * 127)))) for value in values]
    raise ValueError(f"unknown quantization mode: {mode}")


def export_smoke_model(model: BaseAudioModel, out_dir: Path, mode: str = "fp32") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": model.name,
        "parameter_count": model.parameter_count(),
        "mode": mode,
        "note": "Smoke export metadata, not a deployable binary model.",
    }
    path = out_dir / f"{model.name}_{mode}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def benchmark_latency(
    model: BaseAudioModel,
    features: list[list[float]],
    repeats: int = 20,
    warmup: int = 3,
) -> dict[str, float | int]:
    for _ in range(warmup):
        model.forward(features)
    times_ms: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        model.forward(features)
        end = time.perf_counter()
        times_ms.append((end - start) * 1000.0)
    sorted_times = sorted(times_ms)
    p95_idx = min(len(sorted_times) - 1, int(round(0.95 * (len(sorted_times) - 1))))
    return {
        "warmup": warmup,
        "repeats": repeats,
        "mean_ms": statistics.mean(times_ms),
        "median_ms": statistics.median(times_ms),
        "p95_ms": sorted_times[p95_idx],
    }


def benchmark_report(model: BaseAudioModel, features: list[list[float]], out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    exports = {}
    for mode in ["fp32", "fp16", "int8"]:
        export_path = export_smoke_model(model, out_dir / "exports", mode)
        exports[mode] = {
            "path": str(export_path),
            "size_bytes": export_path.stat().st_size,
        }
    output = model.forward(features)
    report = {
        "model": model.name,
        "parameter_count": model.parameter_count(),
        "latency": benchmark_latency(model, features),
        "exports": exports,
        "quantized_logits": {mode: quantize_values(output.logits, mode) for mode in ["fp32", "fp16", "int8"]},
    }
    (out_dir / "benchmark_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
