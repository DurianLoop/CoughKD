"""Bootstrap target-subset stress test for CoughKD method stability.

This uses existing COUGHVID prediction CSV files. Labels are used only for the
audit metric after subsets are sampled; this is not a label-free selector.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DEFAULT_OUT = RUNS / "coughvid_bootstrap_stability"
REFERENCE = "source_only"
METHOD_RUNS = {
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


def _read_predictions(method: str, seed: int) -> pd.DataFrame:
    path = RUNS / METHOD_RUNS[method][seed] / "predictions.csv"
    return pd.read_csv(path).fillna("")


def _macro_auroc(df: pd.DataFrame) -> float:
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    classes = [col.removeprefix("prob_") for col in prob_cols]
    y_true = df["true_label"].astype(str).tolist()
    scores = df[prob_cols].astype(float).to_numpy()
    aucs: list[float] = []
    for idx, cls in enumerate(classes):
        binary = [1 if label == cls else 0 for label in y_true]
        if len(set(binary)) == 2:
            aucs.append(float(roc_auc_score(binary, scores[:, idx])))
    return mean(aucs) if aucs else float("nan")


def _mean_seed_metric(method_frames: dict[int, pd.DataFrame], ids: set[str]) -> float:
    values = []
    for frame in method_frames.values():
        sub = frame[frame["recording_id"].isin(ids)]
        if len(sub) >= 50:
            values.append(_macro_auroc(sub))
    values = [value for value in values if not math.isnan(value)]
    return mean(values) if values else float("nan")


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subsets", type=int, default=200)
    parser.add_argument("--subset-size", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out_dir = args.out if args.out.is_absolute() else ROOT / args.out

    rng = random.Random(args.seed)
    frames = {
        method: {seed: _read_predictions(method, seed) for seed in seed_runs}
        for method, seed_runs in METHOD_RUNS.items()
    }
    ref_ids = sorted(set(next(iter(frames[REFERENCE].values()))["recording_id"].astype(str)))
    subset_size = min(args.subset_size, len(ref_ids))
    rows: list[dict[str, Any]] = []
    for subset_idx in range(args.subsets):
        ids = set(rng.sample(ref_ids, subset_size))
        metrics = {method: _mean_seed_metric(method_frames, ids) for method, method_frames in frames.items()}
        ref = metrics[REFERENCE]
        deltas = {method: value - ref for method, value in metrics.items() if method != REFERENCE and not math.isnan(value)}
        best = max(deltas, key=deltas.get)
        for method, value in metrics.items():
            rows.append(
                {
                    "subset": subset_idx,
                    "method": method,
                    "macro_auroc": value,
                    "macro_delta": 0.0 if method == REFERENCE else value - ref,
                    "best_posthoc": method == best,
                }
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "bootstrap_method_rows.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for method in METHOD_RUNS:
        method_rows = [row for row in rows if row["method"] == method]
        deltas = [float(row["macro_delta"]) for row in method_rows]
        summary_rows.append(
            {
                "method": method,
                "mean_delta": mean(deltas),
                "std_delta": pstdev(deltas) if len(deltas) > 1 else 0.0,
                "positive_rate": sum(1 for value in deltas if value > 0) / len(deltas),
                "negative_rate": sum(1 for value in deltas if value < 0) / len(deltas),
                "best_posthoc_rate": sum(1 for row in method_rows if str(row["best_posthoc"]).lower() == "true") / len(method_rows),
            }
        )
    with (out_dir / "bootstrap_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)
    (out_dir / "summary.json").write_text(
        json.dumps({"subsets": args.subsets, "subset_size": subset_size, "methods": summary_rows}, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# COUGHVID Bootstrap Method Stability",
        "",
        "This audit samples target subsets from COUGHVID and evaluates existing predictions. Labels are used only after sampling for audit metrics.",
        "",
        f"- Subsets: `{args.subsets}`",
        f"- Subset size: `{subset_size}`",
        "",
        "| Method | Mean delta vs source-only | Std | Positive rate | Negative rate | Best post-hoc rate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['method']} | {_fmt(row['mean_delta'])} | {_fmt(row['std_delta'])} | {_fmt(row['positive_rate'])} | {_fmt(row['negative_rate'])} | {_fmt(row['best_posthoc_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- High best-posthoc dispersion means method ranking depends on target composition.",
            "- A method with high positive rate but tiny mean delta should not be claimed as a strong method without independent external validation.",
        ]
    )
    (out_dir / "BOOTSTRAP_STABILITY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out_dir / "BOOTSTRAP_STABILITY_REPORT.md")


if __name__ == "__main__":
    main()
