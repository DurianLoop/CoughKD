"""Multi-target CoughKD-Guard audit.

This script generalizes the single COUGHVID signal audit. It scans completed
external evaluation runs, aggregates target-unlabeled prediction signals, and
tests whether a simple guard score can avoid negative-transfer model choices.

Supported run name patterns:
- legacy COUGHVID runs from the current repository
- generic runs created by scripts/evaluate_external_model_set.py:
  runs/external_<target_tag>_<method>_seed<seed>
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
METHODS = ["source_only", "tcd_very_strong", "tcd_conf035", "candidate_a", "candidate_b", "candidate_c", "ce", "kd"]
SEEDS = [7, 11, 23]
REFERENCE = "source_only"


LEGACY_COUGHVID = {
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


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


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
        max_probs.append(max(probs) if probs else 0.0)
        entropies.append(-sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(class_count))
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


def _generic_runs(skip_legacy_coughvid: bool) -> dict[str, dict[str, dict[int, str]]]:
    found: dict[str, dict[str, dict[int, str]]] = {}
    method_pattern = "|".join(sorted((re.escape(item) for item in METHODS), key=len, reverse=True))
    pattern = re.compile(rf"^external_(?P<target>.+)_(?P<method>{method_pattern})_seed(?P<seed>7|11|23)$")
    for path in RUNS.iterdir():
        if not path.is_dir():
            continue
        match = pattern.match(path.name)
        if not match:
            continue
        target = match.group("target")
        if target.startswith("dryrun"):
            continue
        if skip_legacy_coughvid and target.startswith("coughvid"):
            continue
        method = match.group("method")
        seed = int(match.group("seed"))
        found.setdefault(target, {}).setdefault(method, {})[seed] = path.name
    return found


def _target_run_maps(include_legacy: bool) -> dict[str, dict[str, dict[int, str]]]:
    targets = _generic_runs(skip_legacy_coughvid=include_legacy)
    if include_legacy:
        targets["coughvid"] = LEGACY_COUGHVID
    return targets


def _aggregate_target(target: str, run_map: dict[str, dict[int, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        seed_runs = run_map.get(method, {})
        macro_values: list[float] = []
        covid_values: list[float] = []
        auprc_values: list[float] = []
        stat_buckets: dict[str, list[float]] = {
            "target_confidence": [],
            "target_entropy": [],
            "target_covid_prob": [],
            "pred_healthy_rate": [],
            "pred_covid_rate": [],
        }
        for seed in SEEDS:
            run_name = seed_runs.get(seed)
            if not run_name:
                continue
            run_dir = RUNS / run_name
            metrics = _load_json(run_dir / "metrics.json")
            if metrics is None:
                continue
            if "macro_ovr_auroc" in metrics:
                macro_values.append(float(metrics["macro_ovr_auroc"]))
            if "covid_positive" in metrics:
                covid_values.append(float(metrics["covid_positive"]))
            if "macro_ovr_auprc" in metrics:
                auprc_values.append(float(metrics["macro_ovr_auprc"]))
            pred_path = run_dir / "predictions.csv"
            if pred_path.is_file():
                stats = _prediction_stats(pred_path)
                for key in stat_buckets:
                    value = stats.get(key)
                    if value is not None and not math.isnan(value):
                        stat_buckets[key].append(float(value))
        if not macro_values:
            continue
        macro_mean, macro_std = _mean_std(macro_values)
        covid_mean, covid_std = _mean_std(covid_values)
        auprc_mean, auprc_std = _mean_std(auprc_values)
        row: dict[str, Any] = {
            "target": target,
            "method": method,
            "n": len(macro_values),
            "external_macro_auroc": macro_mean,
            "external_macro_auroc_std": macro_std,
            "external_covid_auroc": covid_mean,
            "external_covid_auroc_std": covid_std,
            "external_macro_auprc": auprc_mean,
            "external_macro_auprc_std": auprc_std,
        }
        for key, values in stat_buckets.items():
            row[key], row[f"{key}_std"] = _mean_std(values)
        rows.append(row)
    ref = next((row for row in rows if row["method"] == REFERENCE), None)
    if ref is None:
        return rows
    for row in rows:
        row["macro_delta"] = float(row["external_macro_auroc"]) - float(ref["external_macro_auroc"])
        row["covid_delta"] = (
            float(row["external_covid_auroc"]) - float(ref["external_covid_auroc"])
            if row["external_covid_auroc"] is not None and ref["external_covid_auroc"] is not None
            else None
        )
    return rows


def _zscore(rows: list[dict[str, Any]], key: str, value: float) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None]
    mu = mean(vals)
    sd = math.sqrt(max(mean([(item - mu) ** 2 for item in vals]), 1e-12))
    return (value - mu) / sd


def _guard_score(target_rows: list[dict[str, Any]], row: dict[str, Any]) -> float:
    return (
        0.40 * _zscore(target_rows, "target_covid_prob", float(row["target_covid_prob"]))
        + 0.25 * _zscore(target_rows, "pred_healthy_rate", float(row["pred_healthy_rate"]))
        + 0.20 * _zscore(target_rows, "target_confidence", float(row["target_confidence"]))
        - 0.15 * _zscore(target_rows, "target_entropy", float(row["target_entropy"]))
    )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for _, idx in ordered[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_rank(xs), _rank(ys))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=RUNS / "coughkd_guard_multitarget")
    parser.add_argument("--no-legacy-coughvid", action="store_true")
    args = parser.parse_args()

    targets = _target_run_maps(include_legacy=not args.no_legacy_coughvid)
    all_rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    for target, run_map in sorted(targets.items()):
        target_rows = _aggregate_target(target, run_map)
        if not target_rows:
            continue
        for row in target_rows:
            if row.get("target_covid_prob") is not None:
                row["guard_score"] = _guard_score(target_rows, row)
        candidates = [row for row in target_rows if row["method"] != REFERENCE and row.get("guard_score") is not None]
        if candidates:
            selected = max(candidates, key=lambda row: float(row["guard_score"]))
            best_macro = max(target_rows, key=lambda row: float(row["external_macro_auroc"]))
            always_kd = next((row for row in target_rows if row["method"] == "kd"), None)
            decisions.append(
                {
                    "target": target,
                    "selected": selected["method"],
                    "selected_macro_delta": selected.get("macro_delta"),
                    "selected_covid_delta": selected.get("covid_delta"),
                    "best_macro": best_macro["method"],
                    "best_macro_delta": best_macro.get("macro_delta"),
                    "always_kd_macro_delta": always_kd.get("macro_delta") if always_kd else None,
                    "selected_negative_transfer": bool(float(selected.get("macro_delta", 0.0)) < 0),
                }
            )
        all_rows.extend(target_rows)

    args.out.mkdir(parents=True, exist_ok=True)
    if all_rows:
        fieldnames = sorted({key for row in all_rows for key in row})
        with (args.out / "multitarget_signals.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
    if decisions:
        with (args.out / "guard_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
            writer.writeheader()
            writer.writerows(decisions)

    correlations: dict[str, dict[str, float | None]] = {}
    points = [row for row in all_rows if row["method"] != REFERENCE and row.get("macro_delta") is not None]
    for signal in ["guard_score", "target_covid_prob", "pred_healthy_rate", "target_confidence", "target_entropy", "pred_covid_rate"]:
        xs = [float(row[signal]) for row in points if row.get(signal) is not None]
        ys = [float(row["macro_delta"]) for row in points if row.get(signal) is not None]
        correlations[signal] = {"pearson_macro_delta": _pearson(xs, ys), "spearman_macro_delta": _spearman(xs, ys), "n": len(xs)}
    (args.out / "correlations.json").write_text(json.dumps(correlations, indent=2), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps({"decisions": decisions, "correlations": correlations}, indent=2), encoding="utf-8")

    lines = [
        "# CoughKD-Guard Multi-Target Audit",
        "",
        "This report uses target labels only for auditing the outcome, not for guard-score selection.",
        "",
        "## Target Decisions",
        "",
        "| Target | Selected | Selected macro delta | Best macro method | Always KD macro delta | Negative transfer? |",
        "|---|---|---:|---|---:|---|",
    ]
    for item in decisions:
        lines.append(
            f"| {item['target']} | {item['selected']} | {_fmt(item['selected_macro_delta'])} | {item['best_macro']} ({_fmt(item['best_macro_delta'])}) | {_fmt(item['always_kd_macro_delta'])} | {item['selected_negative_transfer']} |"
        )
    lines.extend(["", "## Correlations", "", "| Signal | Pearson with macro delta | Spearman with macro delta | n |", "|---|---:|---:|---:|"])
    for signal, item in correlations.items():
        lines.append(f"| {signal} | {_fmt(item['pearson_macro_delta'])} | {_fmt(item['spearman_macro_delta'])} | {item['n']} |")
    lines.extend(
        [
            "",
            "## Gate",
            "",
            "- Claimable only after at least two real external targets are present.",
            "- Current report is a readiness check if it contains only COUGHVID.",
        ]
    )
    (args.out / "COUGHKD_GUARD_MULTITARGET_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.out / "COUGHKD_GUARD_MULTITARGET_AUDIT.md")


if __name__ == "__main__":
    main()
