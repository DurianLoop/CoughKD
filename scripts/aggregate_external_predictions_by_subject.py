"""Aggregate segmented external predictions to subject-level metrics.

This is mainly used for stress targets such as Virufy segmented cough clips,
where many short clips come from the same original recording. Clip-level
metrics can overstate evidence, so this script averages class probabilities
within each subject_id and recomputes the external metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.metrics import average_precision, multiclass_ovr_auroc


DEFAULT_METHODS = [
    "source_only",
    "ce",
    "kd",
    "tcd_very_strong",
    "tcd_conf035",
    "candidate_a",
    "candidate_b",
    "candidate_c",
]
DEFAULT_SEEDS = [7, 11, 23]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _metrics(labels: list[str], pred_labels: list[str], prob_rows: list[list[float]], classes: list[str]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "num_subjects": len(labels),
        "accuracy": sum(int(label == pred) for label, pred in zip(labels, pred_labels)) / max(1, len(labels)),
    }
    try:
        result.update(multiclass_ovr_auroc(labels, prob_rows, classes))
    except Exception as exc:
        result["macro_ovr_auroc_error"] = str(exc)
    auprc: dict[str, float] = {}
    for idx, class_name in enumerate(classes):
        binary = [1 if label == class_name else 0 for label in labels]
        if len(set(binary)) < 2:
            continue
        auprc[class_name] = average_precision(binary, [row[idx] for row in prob_rows])
    if auprc:
        result["macro_ovr_auprc"] = sum(auprc.values()) / len(auprc)
        for key, value in auprc.items():
            result[f"{key}_ovr_auprc"] = value
    return result


def _run_dir(target_tag: str, method: str, seed: int) -> Path:
    return ROOT / "runs" / f"external_{target_tag}_{method}_seed{seed}"


def _aggregate_run(manifest_rows: list[dict[str, str]], run_dir: Path) -> dict[str, Any] | None:
    predictions_path = run_dir / "predictions.csv"
    if not predictions_path.is_file():
        return None

    manifest_by_recording = {row["recording_id"]: row for row in manifest_rows}
    prediction_rows = _read_csv(predictions_path)
    if not prediction_rows:
        return None

    prob_cols = [key for key in prediction_rows[0] if key.startswith("prob_")]
    classes = [key.replace("prob_", "", 1) for key in prob_cols]
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    missing_manifest = 0
    for row in prediction_rows:
        manifest_row = manifest_by_recording.get(row["recording_id"])
        if manifest_row is None:
            missing_manifest += 1
            continue
        subject_id = manifest_row["subject_id"]
        merged = dict(row)
        merged["subject_id"] = subject_id
        grouped[subject_id].append(merged)

    subject_rows: list[dict[str, Any]] = []
    for subject_id, rows in sorted(grouped.items()):
        true_labels = {row["true_label"] for row in rows}
        if len(true_labels) != 1:
            raise ValueError(f"Subject {subject_id} has inconsistent labels: {sorted(true_labels)}")
        probs = []
        for col in prob_cols:
            probs.append(sum(float(row[col]) for row in rows) / len(rows))
        pred_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        out_row: dict[str, Any] = {
            "subject_id": subject_id,
            "num_clips": len(rows),
            "true_label": next(iter(true_labels)),
            "pred_label": classes[pred_idx],
        }
        out_row.update({f"prob_{label}": f"{probs[idx]:.8f}" for idx, label in enumerate(classes)})
        subject_rows.append(out_row)

    labels = [str(row["true_label"]) for row in subject_rows]
    pred_labels = [str(row["pred_label"]) for row in subject_rows]
    prob_rows = [[float(row[f"prob_{label}"]) for label in classes] for row in subject_rows]
    metrics = _metrics(labels, pred_labels, prob_rows, classes)
    metrics["num_clips"] = len(prediction_rows)
    metrics["num_matched_clips"] = sum(int(row["num_clips"]) for row in subject_rows)
    metrics["num_missing_manifest_clips"] = missing_manifest
    metrics["classes"] = classes

    _write_csv(
        run_dir / "subject_predictions.csv",
        subject_rows,
        ["subject_id", "num_clips", "true_label", "pred_label"] + [f"prob_{label}" for label in classes],
    )
    (run_dir / "subject_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target-tag", required=True)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    manifest_rows = _read_csv(args.manifest)
    out_dir = args.out or (ROOT / "runs" / f"subject_aggregate_{args.target_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for method in args.methods:
        for seed in args.seeds:
            run_dir = _run_dir(args.target_tag, method, seed)
            metrics = _aggregate_run(manifest_rows, run_dir)
            if metrics is None:
                print(f"[skip] missing predictions: {run_dir}")
                continue
            rows.append(
                {
                    "target": args.target_tag,
                    "method": method,
                    "seed": seed,
                    "num_subjects": metrics.get("num_subjects"),
                    "num_clips": metrics.get("num_clips"),
                    "accuracy": metrics.get("accuracy"),
                    "macro_ovr_auroc": metrics.get("macro_ovr_auroc"),
                    "covid_positive_ovr_auroc": metrics.get("covid_positive"),
                    "healthy_ovr_auroc": metrics.get("healthy"),
                    "macro_ovr_auprc": metrics.get("macro_ovr_auprc"),
                }
            )

    if not rows:
        raise SystemExit("No subject-level runs were aggregated.")

    fieldnames = list(rows[0])
    _write_csv(out_dir / "subject_level_runs.csv", rows, fieldnames)

    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[str(row["method"])].append(row)

    source_values = [
        float(row["macro_ovr_auroc"])
        for row in by_method.get("source_only", [])
        if row.get("macro_ovr_auroc") is not None
    ]
    source_mean = mean(source_values) if source_values else None

    summary_rows: list[dict[str, Any]] = []
    for method, method_rows in sorted(by_method.items()):
        macro_values = [
            float(row["macro_ovr_auroc"])
            for row in method_rows
            if row.get("macro_ovr_auroc") is not None
        ]
        covid_values = [
            float(row["covid_positive_ovr_auroc"])
            for row in method_rows
            if row.get("covid_positive_ovr_auroc") is not None
        ]
        auprc_values = [
            float(row["macro_ovr_auprc"])
            for row in method_rows
            if row.get("macro_ovr_auprc") is not None
        ]
        macro_mean = mean(macro_values) if macro_values else None
        summary_rows.append(
            {
                "target": args.target_tag,
                "method": method,
                "seeds": len(method_rows),
                "num_subjects": method_rows[0].get("num_subjects") if method_rows else None,
                "macro_ovr_auroc_mean": macro_mean,
                "macro_ovr_auroc_std": pstdev(macro_values) if len(macro_values) > 1 else 0.0,
                "macro_delta_vs_source": (macro_mean - source_mean) if macro_mean is not None and source_mean is not None else None,
                "covid_positive_ovr_auroc_mean": mean(covid_values) if covid_values else None,
                "macro_ovr_auprc_mean": mean(auprc_values) if auprc_values else None,
            }
        )
    summary_rows.sort(key=lambda row: (row["macro_ovr_auroc_mean"] is None, -(row["macro_ovr_auroc_mean"] or -999.0)))
    _write_csv(out_dir / "method_summary.csv", summary_rows, list(summary_rows[0]))

    lines = [
        f"# Subject-Level External Aggregation: {args.target_tag}",
        "",
        f"- Manifest: `{args.manifest}`",
        f"- Runs aggregated: {len(rows)}",
        f"- Subject count: {summary_rows[0]['num_subjects']}",
        "- Aggregation: mean class probability over clips belonging to the same `subject_id`, followed by subject-level AUROC/AUPRC.",
        "",
        "| Method | Seeds | Macro AUROC mean | Std | Delta vs source | COVID AUROC | Macro AUPRC |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {row['seeds']} | {_fmt(row['macro_ovr_auroc_mean'])} | {_fmt(row['macro_ovr_auroc_std'])} | {_fmt(row['macro_delta_vs_source'])} | {_fmt(row['covid_positive_ovr_auroc_mean'])} | {_fmt(row['macro_ovr_auprc_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- These metrics are intentionally stricter than clip-level Virufy segmented metrics because the effective independent sample size is the number of source subjects.",
            "- A large clip-level gain that disappears here should be treated as weak evidence, not as a deployable external-validation claim.",
        ]
    )
    (out_dir / "SUBJECT_AGGREGATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "SUBJECT_AGGREGATE_REPORT.md")


if __name__ == "__main__":
    main()
