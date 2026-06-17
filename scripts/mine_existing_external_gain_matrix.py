from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
METRIC_KEY = "macro_ovr_auroc"
CANONICAL_METHODS = {
    "ce",
    "kd",
    "source_only",
    "tcd_conf035",
    "tcd_very_strong",
    "candidate_a",
    "candidate_b",
    "candidate_c",
    "candidate_d_active",
    "candidate_e_tga",
    "candidate_f_artifact_env_irm_ramp",
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_method(raw: str) -> str | None:
    raw = raw.removeprefix("stage3b_").removeprefix("stage3c_")
    raw = raw.removeprefix("stage3_")
    if raw in CANONICAL_METHODS:
        return raw
    return None


def _parse_run_dir(path: Path) -> tuple[str, str, int] | None:
    name = path.parent.name
    patterns = [
        (r"^external_coughvid_test_(?P<method>.+)_seed(?P<seed>\d+)$", "COUGHVID"),
        (r"^external_toscovid2021_test_(?P<method>.+)_seed(?P<seed>\d+)$", "Tos COVID-19"),
        (r"^external_virufy_(?P<method>.+)_seed(?P<seed>\d+)$", "Virufy"),
        (r"^external_virufyseg_(?P<method>.+)_seed(?P<seed>\d+)$", "Virufy segmented"),
    ]
    for pattern, target in patterns:
        match = re.match(pattern, name)
        if not match:
            continue
        method = _normalize_method(match.group("method"))
        if method is None:
            return None
        return target, method, int(match.group("seed"))
    return None


def _collect(metric_file_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((ROOT / "runs").glob(f"external_*/{metric_file_name}")):
        parsed = _parse_run_dir(path)
        if parsed is None:
            continue
        target, method, seed = parsed
        metrics = _load_json(path)
        if METRIC_KEY not in metrics:
            continue
        num_examples = int(metrics.get("num_examples", 0))
        if num_examples == 0 and metric_file_name == "subject_metrics.json":
            subject_predictions = path.parent / "subject_predictions.csv"
            if subject_predictions.is_file():
                with subject_predictions.open("r", encoding="utf-8", newline="") as handle:
                    num_examples = max(0, sum(1 for _ in csv.DictReader(handle)))
        rows.append(
            {
                "target": target,
                "unit": "subject" if metric_file_name == "subject_metrics.json" else "clip",
                "method": method,
                "seed": seed,
                "metric": float(metrics[METRIC_KEY]),
                "macro_ovr_auprc": float(metrics.get("macro_ovr_auprc", 0.0)),
                "accuracy": float(metrics.get("accuracy", 0.0)),
                "num_examples": num_examples,
                "path": str(path),
            }
        )
    return rows


def _summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["target"], row["unit"], row["method"])].append(row)
    baseline: dict[tuple[str, str], dict[str, Any]] = {}
    summaries: list[dict[str, Any]] = []
    for (target, unit, method), items in sorted(grouped.items()):
        by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for item in items:
            by_seed[int(item["seed"])].append(item)
        seed_items = []
        for seed, duplicates in sorted(by_seed.items()):
            if len(duplicates) == 1:
                seed_items.append(duplicates[0])
                continue
            seed_items.append(
                {
                    **duplicates[0],
                    "metric": mean(float(item["metric"]) for item in duplicates),
                    "macro_ovr_auprc": mean(float(item["macro_ovr_auprc"]) for item in duplicates),
                    "accuracy": mean(float(item["accuracy"]) for item in duplicates),
                    "num_examples": min(int(item["num_examples"]) for item in duplicates),
                    "path": ";".join(item["path"] for item in duplicates),
                }
            )
        values = [float(item["metric"]) for item in seed_items]
        seeds = sorted(int(item["seed"]) for item in seed_items)
        record = {
            "target": target,
            "unit": unit,
            "method": method,
            "seeds": ",".join(str(seed) for seed in seeds),
            "n_seeds": len(seeds),
            "mean_macro_ovr_auroc": mean(values),
            "std_macro_ovr_auroc": pstdev(values) if len(values) > 1 else 0.0,
            "min_macro_ovr_auroc": min(values),
            "max_macro_ovr_auroc": max(values),
            "num_examples_min": min(int(item["num_examples"]) for item in seed_items),
            "paths": ";".join(item["path"] for item in seed_items),
        }
        summaries.append(record)
        if method == "source_only":
            baseline[(target, unit)] = record
    for record in summaries:
        base = baseline.get((record["target"], record["unit"]))
        if base is None:
            record["delta_vs_source_only"] = ""
            record["clears_3pt_gate"] = False
            continue
        record["delta_vs_source_only"] = float(record["mean_macro_ovr_auroc"]) - float(base["mean_macro_ovr_auroc"])
        record["clears_3pt_gate"] = bool(float(record["delta_vs_source_only"]) >= 0.03)
    return summaries


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "target",
        "unit",
        "method",
        "seeds",
        "n_seeds",
        "mean_macro_ovr_auroc",
        "std_macro_ovr_auroc",
        "min_macro_ovr_auroc",
        "max_macro_ovr_auroc",
        "num_examples_min",
        "delta_vs_source_only",
        "clears_3pt_gate",
        "paths",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _fmt(value: Any) -> str:
    if value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_report(path: Path, summaries: list[dict[str, Any]]) -> None:
    eligible = [
        row
        for row in summaries
        if row["method"] != "source_only"
        and row.get("delta_vs_source_only") != ""
        and bool(row.get("clears_3pt_gate"))
    ]
    eligible = sorted(eligible, key=lambda row: float(row["delta_vs_source_only"]), reverse=True)
    lines = [
        "# Existing External Gain Matrix",
        "",
        "This mines already-computed external metrics. It does not train or select a new model.",
        "",
        "## 3-Point Positive Signals",
        "",
        "| Target | Unit | Method | Seeds | n | Mean AUROC | Delta vs source-only | Std | Min AUROC | Notes |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    if eligible:
        for row in eligible:
            notes = "tiny target; treat as auxiliary" if int(row["num_examples_min"]) < 100 else ""
            lines.append(
                f"| {row['target']} | {row['unit']} | {row['method']} | {row['seeds']} | {row['num_examples_min']} | "
                f"{_fmt(row['mean_macro_ovr_auroc'])} | {_fmt(row['delta_vs_source_only'])} | "
                f"{_fmt(row['std_macro_ovr_auroc'])} | {_fmt(row['min_macro_ovr_auroc'])} | {notes} |"
            )
    else:
        lines.append("| none |  |  |  |  |  |  |  |  |  |")
    lines.extend(
        [
            "",
            "## Full Summary",
            "",
            "| Target | Unit | Method | Seeds | Mean AUROC | Delta vs source-only | Std | n min |",
            "|---|---|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(summaries, key=lambda item: (item["target"], item["unit"], -float(item["mean_macro_ovr_auroc"]))):
        lines.append(
            f"| {row['target']} | {row['unit']} | {row['method']} | {row['seeds']} | "
            f"{_fmt(row['mean_macro_ovr_auroc'])} | {_fmt(row['delta_vs_source_only'])} | "
            f"{_fmt(row['std_macro_ovr_auroc'])} | {row['num_examples_min']} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/existing_external_gain_matrix")
    args = parser.parse_args()
    rows = _collect("metrics.json") + _collect("subject_metrics.json")
    summaries = _summaries(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "external_metric_rows.csv", rows)
    _write_csv(args.out / "external_gain_summary.csv", summaries)
    _write_report(args.out / "EXISTING_EXTERNAL_GAIN_MATRIX.md", summaries)
    print(str(args.out / "EXISTING_EXTERNAL_GAIN_MATRIX.md"), flush=True)


if __name__ == "__main__":
    main()
