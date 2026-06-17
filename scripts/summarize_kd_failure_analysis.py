"""Build an analysis-paper evidence report from completed CoughKD audits.

This report is intentionally not a new method search. It consolidates the
evidence collected by the bounded experiment loop into a paper-facing readout:
when vanilla and shortcut-aware KD help or fail for an ultra-light cough student
under Coswara -> COUGHVID dataset shift.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "kd_failure_analysis"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.6f}"
    return str(value)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def _method_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    source = next(item for item in summary["summaries"] if item["method"] == "source_only")
    source_macro = float(source["external_macro_auroc_mean"])
    source_covid = float(source["external_covid_auroc_mean"])
    for item in summary["summaries"]:
        macro = item.get("external_macro_auroc_mean")
        covid = item.get("external_covid_auroc_mean")
        rows.append(
            {
                "method": item["method"],
                "kind": item["kind"],
                "external_macro": macro,
                "external_macro_std": item.get("external_macro_auroc_std"),
                "external_macro_delta_vs_source": None if macro is None else float(macro) - source_macro,
                "external_covid": covid,
                "external_covid_delta_vs_source": None if covid is None else float(covid) - source_covid,
                "domain_probe": item.get("probe_domain_auc_mean"),
                "task_probe": item.get("probe_task_auc_mean"),
                "n": item.get("external_macro_auroc_n"),
            }
        )
    return rows


def _seed_delta_rows(summary: dict[str, Any]) -> list[dict[str, Any]]:
    by_method = {item["method"]: item for item in summary["summaries"]}
    source_by_seed = {
        int(seed_row["seed"]): seed_row
        for seed_row in by_method["source_only"]["seeds"]
        if seed_row.get("external_macro_auroc") is not None
    }
    rows = []
    for method, item in by_method.items():
        if method == "source_only":
            continue
        for seed_row in item.get("seeds", []):
            seed = int(seed_row["seed"])
            if seed not in source_by_seed or seed_row.get("external_macro_auroc") is None:
                continue
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "macro_delta_vs_source": float(seed_row["external_macro_auroc"]) - float(source_by_seed[seed]["external_macro_auroc"]),
                    "covid_delta_vs_source": float(seed_row["external_covid_auroc"]) - float(source_by_seed[seed]["external_covid_auroc"]),
                }
            )
    return rows


def _positive_seed_counts(seed_rows: list[dict[str, Any]]) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    totals: dict[str, int] = {}
    for row in seed_rows:
        method = str(row["method"])
        totals[method] = totals.get(method, 0) + 1
        counts.setdefault(method, [0])
        if float(row["macro_delta_vs_source"]) > 0:
            counts[method][0] += 1
    return {method: (counts.get(method, [0])[0], total) for method, total in totals.items()}


def main() -> None:
    loop_summary = _load_json(RUNS / "innovation_loop_summary.json")
    method_rows = _method_rows(loop_summary)
    seed_rows = _seed_delta_rows(loop_summary)
    positive_counts = _positive_seed_counts(seed_rows)
    slice_rows = _load_csv(RUNS / "coughvid_slice_guard" / "slice_guard_decisions.csv")
    correlations = _load_json(RUNS / "coughvid_slice_guard" / "slice_correlations.json")
    loso_summary = _load_json(RUNS / "coughvid_slice_guard_loso" / "summary.json")
    loso_rows = _load_csv(RUNS / "coughvid_slice_guard_loso" / "loso_guard_decisions.csv")

    best_method = max(
        [row for row in method_rows if row["method"] != "source_only" and row["external_macro"] is not None],
        key=lambda row: float(row["external_macro"]),
    )
    negative_slices = [row for row in slice_rows if str(row.get("selected_negative_transfer")).lower() == "true"]
    loso_negative = [row for row in loso_rows if str(row.get("selected_negative_transfer")).lower() == "true"]
    mean_best_posthoc = mean(float(row["best_macro_delta"]) for row in slice_rows if row.get("best_macro_delta") is not None)
    mean_selected = mean(float(row["selected_macro_delta"]) for row in slice_rows if row.get("selected_macro_delta") is not None)
    mean_kd = mean(float(row["always_kd_macro_delta"]) for row in slice_rows if row.get("always_kd_macro_delta") is not None)

    OUT.mkdir(parents=True, exist_ok=True)
    artifact = {
        "best_non_source_method": best_method,
        "positive_seed_counts": positive_counts,
        "slice_guard": {
            "slices": len(slice_rows),
            "negative_selected": len(negative_slices),
            "mean_selected_macro_delta": mean_selected,
            "mean_best_posthoc_macro_delta": mean_best_posthoc,
            "mean_always_kd_macro_delta": mean_kd,
        },
        "loso_guard": loso_summary,
        "correlations": correlations,
    }
    (OUT / "summary.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")

    lines = [
        "# KD Failure Analysis Evidence Pack",
        "",
        "This file consolidates the bounded experiment loop. It is meant to support an analysis-oriented paper direction, not to claim a successful new KD method.",
        "",
        "## Candidate Paper Claim",
        "",
        "A cautious, defensible claim is:",
        "",
        "> In ultra-light cough audio students trained on Coswara and evaluated on COUGHVID, common response KD and three shortcut-aware variants produce weak and unstable external gains. Probe-level shortcut reduction does not reliably translate into external generalization. Target-unlabeled prediction signals can expose some negative-transfer risk, but current label-free method selection is not stable enough to be used as a method claim.",
        "",
        "Working title:",
        "",
        "> When Does Knowledge Distillation Help or Fail for Ultra-Light Cough Audio Models under Dataset Shift?",
        "",
        "## External Method Summary",
        "",
        "| Method | Kind | External macro AUROC | Delta vs source-only | External COVID AUROC | COVID delta | Domain probe | Task probe | Positive seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in method_rows:
        pos, total = positive_counts.get(str(row["method"]), (0, 0))
        positive = "-" if total == 0 else f"{pos}/{total}"
        lines.append(
            "| {method} | {kind} | {macro} +/- {macro_std} | {macro_delta} | {covid} | {covid_delta} | {domain} | {task} | {positive} |".format(
                method=row["method"],
                kind=row["kind"],
                macro=_fmt(row["external_macro"]),
                macro_std=_fmt(row["external_macro_std"]),
                macro_delta=_fmt(row["external_macro_delta_vs_source"]),
                covid=_fmt(row["external_covid"]),
                covid_delta=_fmt(row["external_covid_delta_vs_source"]),
                domain=_fmt(row["domain_probe"]),
                task=_fmt(row["task_probe"]),
                positive=positive,
            )
        )

    lines.extend(
        [
            "",
            "## Slice Stress Test",
            "",
            f"- Slices: `{len(slice_rows)}`",
            f"- Fixed guard negative selected: `{len(negative_slices)}/{len(slice_rows)}`",
            f"- Fixed guard mean selected macro delta: `{_fmt(mean_selected)}`",
            f"- Best post-hoc mean macro delta: `{_fmt(mean_best_posthoc)}`",
            f"- Always vanilla KD mean macro delta: `{_fmt(mean_kd)}`",
            "",
            "| Slice | Selected | Selected delta | Best post-hoc | Always KD delta | Negative? |",
            "|---|---|---:|---|---:|---|",
        ]
    )
    for row in slice_rows:
        lines.append(
            f"| {row['slice']} | {row['selected']} | {_fmt(row['selected_macro_delta'])} | {row['best_macro']} ({_fmt(row['best_macro_delta'])}) | {_fmt(row['always_kd_macro_delta'])} | {row['selected_negative_transfer']} |"
        )

    lines.extend(
        [
            "",
            "## Target-Unlabeled Signal Readout",
            "",
            "| Signal | Pearson with macro delta | Spearman with macro delta |",
            "|---|---:|---:|",
        ]
    )
    for signal in [
        "guard_score",
        "target_covid_prob",
        "pred_healthy_rate",
        "target_confidence",
        "target_entropy",
        "agree_source_only",
        "l1_to_source_only",
        "confidence_gap_source_only",
        "entropy_gap_source_only",
    ]:
        item = correlations.get(signal, {})
        lines.append(f"| {signal} | {_fmt(item.get('pearson'))} | {_fmt(item.get('spearman'))} |")

    lines.extend(
        [
            "",
            "## Leave-One-Slice-Out Guard",
            "",
            f"- Negative selected: `{loso_summary['negative_selected']}/{loso_summary['slices']}`",
            f"- Mean selected macro delta: `{_fmt(loso_summary['mean_selected_macro_delta'])}`",
            f"- Mean best post-hoc macro delta: `{_fmt(loso_summary['mean_best_macro_delta'])}`",
            f"- Mean vanilla KD macro delta: `{_fmt(loso_summary['mean_kd_macro_delta'])}`",
            "",
            "## Decision",
            "",
            "The current evidence does not pass the method innovation gate. The most credible next paper route is an analysis/protocol contribution with additional independent external targets, calibration, and deployment metrics. Do not continue open-ended teacher x student x KD enumeration until that evidence gap is closed.",
        ]
    )
    (OUT / "KD_FAILURE_ANALYSIS_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "KD_FAILURE_ANALYSIS_REPORT.md")


if __name__ == "__main__":
    main()
