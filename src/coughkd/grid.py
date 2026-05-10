"""Experiment grid generation and smoke runner."""

from __future__ import annotations

import json
from pathlib import Path

from .metrics import binary_classification_report


def ablation_grid() -> list[dict[str, str]]:
    teachers = ["panns", "ast", "beats"]
    students = ["mobilenetv3", "efficientnet_b0", "bc_resnet", "ecapa_small"]
    kd_variants = ["ce_only", "response_kd", "feature_kd", "attention_kd", "relation_kd", "full_coughkd"]
    augmentations = ["none", "waveform", "specaugment", "combined"]
    aggregations = ["mean", "max", "topk", "quality_topk"]
    deployments = ["fp32", "fp16", "int8"]
    grid: list[dict[str, str]] = []
    for teacher in teachers:
        for student in students:
            for kd in kd_variants:
                for aug in augmentations:
                    for agg in aggregations:
                        for deploy in deployments:
                            grid.append(
                                {
                                    "teacher": teacher,
                                    "student": student,
                                    "kd": kd,
                                    "augmentation": aug,
                                    "aggregation": agg,
                                    "deployment": deploy,
                                }
                            )
    return grid


def run_smoke_grid(out_dir: Path, force: bool = False, limit: int = 2) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    failures = []
    for idx, config in enumerate(ablation_grid()[:limit]):
        run_id = f"smoke_grid_{idx:03d}"
        run_dir = out_dir / run_id
        metrics_path = run_dir / "metrics.json"
        if metrics_path.is_file() and not force:
            runs.append({"run_id": run_id, "status": "skipped", "config": config})
            continue
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            labels = [0, 0, 1, 1]
            offset = 0.01 * idx
            scores = [0.1 + offset, 0.2 + offset, 0.8 - offset, 0.9 - offset]
            metrics = binary_classification_report(labels, scores)
            payload = {"run_id": run_id, "config": config, "metrics": metrics}
            metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
            runs.append({"run_id": run_id, "status": "completed", "config": config})
        except Exception as exc:  # pragma: no cover - defensive failure log
            failure = {"run_id": run_id, "config": config, "error": str(exc)}
            failures.append(failure)
    failure_path = out_dir / "failures.json"
    failure_path.write_text(json.dumps(failures, indent=2), encoding="utf-8")
    summary = {"runs": runs, "failures": failures, "num_total_grid": len(ablation_grid())}
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def aggregate_results(runs_dir: Path, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        row = {
            "run_id": payload["run_id"],
            **payload["config"],
            **payload["metrics"],
        }
        rows.append(row)
    summary = {"num_runs": len(rows), "rows": rows}
    (out_dir / "aggregate_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = ["# Aggregate Results", ""]
    if rows:
        columns = list(rows[0].keys())
        lines.append("| " + " | ".join(columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    else:
        lines.append("No completed runs found.")
    (out_dir / "aggregate_results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
