"""Summarize KD-under-shift failure signals from completed CoughKD runs.

This script is intentionally analysis-only: it does not train another model.
It asks whether unlabeled-target confidence statistics and representation probes
track external AUROC deltas. The output is a first-pass evidence table for the
ICASSP-oriented question: when does KD help or fail for ultra-light cough models
under dataset shift?
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


METHOD_RUN_PREFIX = {
    "ce": {
        7: "external_coughvid_test_ce_baseline",
        11: "external_coughvid_test_ce_seed11",
        23: "external_coughvid_test_ce_seed23",
    },
    "kd": {
        7: "external_coughvid_test_kd_baseline",
        11: "external_coughvid_test_kd_seed11",
        23: "external_coughvid_test_kd_seed23",
    },
    "source_only": {
        7: "external_coughvid_test_stage3b_source_only_seed7",
        11: "external_coughvid_test_stage3c_source_only_seed11",
        23: "external_coughvid_test_stage3c_source_only_seed23",
    },
    "tcd_very_strong": {
        7: "external_coughvid_test_stage3b_tcd_very_strong_seed7",
        11: "external_coughvid_test_stage3c_tcd_very_strong_seed11",
        23: "external_coughvid_test_stage3c_tcd_very_strong_seed23",
    },
    "tcd_conf035": {
        7: "external_coughvid_test_stage3b_tcd_conf035_seed7",
        11: "external_coughvid_test_stage3c_tcd_conf035_seed11",
        23: "external_coughvid_test_stage3c_tcd_conf035_seed23",
    },
    "candidate_a": {
        7: "external_coughvid_test_candidate_a_seed7",
        11: "external_coughvid_test_candidate_a_seed11",
        23: "external_coughvid_test_candidate_a_seed23",
    },
    "candidate_b": {
        7: "external_coughvid_test_candidate_b_seed7",
        11: "external_coughvid_test_candidate_b_seed11",
        23: "external_coughvid_test_candidate_b_seed23",
    },
    "candidate_c": {
        7: "external_coughvid_test_candidate_c_seed7",
        11: "external_coughvid_test_candidate_c_seed11",
        23: "external_coughvid_test_candidate_c_seed23",
    },
}


def _load_summary(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _prediction_stats(path: Path) -> dict[str, float]:
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    if not rows:
        return {}
    prob_fields = [field for field in rows[0] if field.startswith("prob_")]
    class_count = max(1, len(prob_fields))
    max_probs: list[float] = []
    entropies: list[float] = []
    covid_probs: list[float] = []
    pred_counts: dict[str, int] = {}
    for row in rows:
        probs = [float(row[field]) for field in prob_fields]
        max_prob = max(probs) if probs else 0.0
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(class_count)
        max_probs.append(max_prob)
        entropies.append(entropy)
        for field, prob in zip(prob_fields, probs):
            if field == "prob_covid_positive":
                covid_probs.append(prob)
        pred = row.get("pred_label", "")
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
    total = len(rows)
    return {
        "target_confidence": mean(max_probs),
        "target_entropy": mean(entropies),
        "target_covid_prob": mean(covid_probs) if covid_probs else float("nan"),
        "pred_healthy_rate": pred_counts.get("healthy", 0) / total,
        "pred_covid_rate": pred_counts.get("covid_positive", 0) / total,
    }


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _fmt(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.6f}"


def main() -> None:
    summary = _load_summary(RUNS / "innovation_loop_summary.json")
    by_method = {item["method"]: item for item in summary["summaries"]}
    reference = by_method["source_only"]
    ref_macro = reference["external_macro_auroc_mean"]
    ref_covid = reference["external_covid_auroc_mean"]
    ref_domain = reference["probe_domain_auc_mean"]
    ref_task = reference["probe_task_auc_mean"]

    rows: list[dict[str, Any]] = []
    for method, run_map in METHOD_RUN_PREFIX.items():
        item = by_method.get(method)
        if item is None:
            continue
        seed_stats: list[dict[str, float]] = []
        for seed in [7, 11, 23]:
            run_name = run_map.get(seed)
            if not run_name:
                continue
            pred_path = RUNS / run_name / "predictions.csv"
            if not pred_path.is_file():
                continue
            seed_stats.append(_prediction_stats(pred_path))
        averaged: dict[str, float] = {}
        for key in ["target_confidence", "target_entropy", "target_covid_prob", "pred_healthy_rate", "pred_covid_rate"]:
            values = [stat[key] for stat in seed_stats if key in stat and not math.isnan(stat[key])]
            averaged[key] = mean(values) if values else float("nan")
            averaged[f"{key}_std"] = pstdev(values) if len(values) > 1 else 0.0
        rows.append(
            {
                "method": method,
                "kind": item["kind"],
                "macro_delta": item["external_macro_auroc_mean"] - ref_macro,
                "covid_delta": item["external_covid_auroc_mean"] - ref_covid,
                "domain_delta": item["probe_domain_auc_mean"] - ref_domain,
                "task_delta": item["probe_task_auc_mean"] - ref_task,
                **averaged,
            }
        )

    correlations: dict[str, dict[str, float | None]] = {}
    for signal in ["domain_delta", "task_delta", "target_confidence", "target_entropy", "target_covid_prob", "pred_healthy_rate", "pred_covid_rate"]:
        xs = [float(row[signal]) for row in rows if row["method"] != "source_only"]
        macro = [float(row["macro_delta"]) for row in rows if row["method"] != "source_only"]
        covid = [float(row["covid_delta"]) for row in rows if row["method"] != "source_only"]
        correlations[signal] = {
            "corr_with_macro_delta": _pearson(xs, macro),
            "corr_with_covid_delta": _pearson(xs, covid),
        }

    out_dir = RUNS / "kd_failure_signal_audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "signals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "correlations.json").write_text(json.dumps(correlations, indent=2), encoding="utf-8")

    lines = [
        "# KD Failure Signal Audit",
        "",
        "Reference method: `source_only`.",
        "",
        "| Method | Macro delta | COVID delta | Domain delta | Task delta | Target conf | Target entropy | Pred healthy | Pred COVID |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {method} | {macro_delta} | {covid_delta} | {domain_delta} | {task_delta} | {target_confidence} | {target_entropy} | {pred_healthy_rate} | {pred_covid_rate} |".format(
                **{key: (_fmt(value) if isinstance(value, float) else value) for key, value in row.items()}
            )
        )
    lines.extend(["", "## Signal Correlations", ""])
    lines.append("| Signal | Corr with macro delta | Corr with COVID delta |")
    lines.append("|---|---:|---:|")
    for signal, vals in correlations.items():
        lines.append(f"| {signal} | {_fmt(vals['corr_with_macro_delta'])} | {_fmt(vals['corr_with_covid_delta'])} |")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- This table is a diagnostic, not a success claim.",
            "- A useful ICASSP direction is viable only if unlabeled target signals can predict when KD will hurt or help across additional datasets/runs.",
            "- Current evidence should be used to design a bounded next experiment, not to add another open-ended KD loss.",
        ]
    )
    (out_dir / "KD_FAILURE_SIGNAL_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "KD_FAILURE_SIGNAL_AUDIT.md")


if __name__ == "__main__":
    main()
