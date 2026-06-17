from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit_crossfit_metadata_slice_policy import _apply_policy, _fit_policy
from audit_metadata_slice_oracle import TARGETS, _method_predictions, _prepare_manifest
from audit_prediction_ensemble_frontier import _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _slice_macro_auc(
    labels: pd.Series,
    scores: np.ndarray,
    groups: pd.Series,
    idx: np.ndarray,
    min_eval_slice: int,
) -> tuple[float | None, list[dict[str, object]]]:
    rows = []
    values = []
    eval_groups = groups.iloc[idx].reset_index(drop=True)
    eval_labels = labels.iloc[idx].reset_index(drop=True)
    local_scores = scores
    for group_value, local_idx_obj in eval_groups.groupby(eval_groups).groups.items():
        local_idx = np.asarray(list(local_idx_obj), dtype=int)
        if len(local_idx) < min_eval_slice or eval_labels.iloc[local_idx].nunique() < 2:
            continue
        auc = _macro_ovr_auc(eval_labels.iloc[local_idx].reset_index(drop=True), local_scores[local_idx])
        values.append(auc)
        rows.append({"slice": str(group_value), "n": int(len(local_idx)), "auc": float(auc)})
    if not values:
        return None, rows
    return float(np.mean(values)), rows


def _policy_key(policy: dict[str, str]) -> str:
    if not policy:
        return "<base>"
    return ";".join(f"{key}={value}" for key, value in sorted(policy.items()))


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="COUGHVID")
    parser.add_argument("--slice-column", default="symptom_resp")
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--min-train-slice", type=int, default=100)
    parser.add_argument("--min-eval-slice", type=int, default=60)
    parser.add_argument("--switch-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/slice_balanced_gate_policy")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "source_only",
            "ce",
            "kd",
            "candidate_a",
            "candidate_b",
            "candidate_c",
            "candidate_f_artifact_env_irm_ramp",
        ],
    )
    args = parser.parse_args()

    config = TARGETS[args.target]
    manifest = _prepare_manifest(config["manifest"])
    base, method_scores = _method_predictions(args.target, args.methods)
    merged = base.merge(manifest, on="recording_id", how="left", suffixes=("", "_manifest"))
    if args.slice_column not in merged.columns:
        raise ValueError(f"{args.slice_column} missing for target {args.target}")
    labels = merged["true_label"].reset_index(drop=True)
    groups = merged[args.slice_column].fillna("missing").astype(str).reset_index(drop=True)

    n = len(labels)
    train_n = int(round(n * args.train_frac))
    rng = np.random.default_rng(args.seed)
    rows = []
    slice_delta_rows = []
    policy_counts: Counter[str] = Counter()

    for repeat in range(args.n_repeats):
        perm = rng.permutation(n)
        train_idx = np.sort(perm[:train_n])
        eval_idx = np.sort(perm[train_n:])
        if labels.iloc[eval_idx].nunique() < 2:
            continue
        policy = _fit_policy(
            labels,
            groups,
            method_scores,
            train_idx,
            args.min_train_slice,
            args.base_method,
            args.switch_margin,
        )
        policy_counts[_policy_key(policy)] += 1
        mixed_scores = _apply_policy(groups, method_scores, args.base_method, policy, eval_idx)
        base_scores = method_scores[args.base_method][eval_idx]
        try:
            overall_mixed = _macro_ovr_auc(labels.iloc[eval_idx].reset_index(drop=True), mixed_scores)
            overall_base = _macro_ovr_auc(labels.iloc[eval_idx].reset_index(drop=True), base_scores)
        except ValueError:
            continue
        slice_mixed, mixed_slice_rows = _slice_macro_auc(labels, mixed_scores, groups, eval_idx, args.min_eval_slice)
        slice_base, base_slice_rows = _slice_macro_auc(labels, base_scores, groups, eval_idx, args.min_eval_slice)
        if slice_mixed is None or slice_base is None:
            continue
        rows.append(
            {
                "repeat": repeat,
                "eval_n": int(len(eval_idx)),
                "policy_size": int(len(policy)),
                "overall_base_auc": overall_base,
                "overall_mixed_auc": overall_mixed,
                "overall_delta_vs_base": overall_mixed - overall_base,
                "slice_macro_base_auc": slice_base,
                "slice_macro_mixed_auc": slice_mixed,
                "slice_macro_delta_vs_base": slice_mixed - slice_base,
                "policy": _policy_key(policy),
            }
        )
        base_by_slice = {row["slice"]: row for row in base_slice_rows}
        for mixed_row in mixed_slice_rows:
            base_row = base_by_slice.get(mixed_row["slice"])
            if base_row is None:
                continue
            slice_delta_rows.append(
                {
                    "repeat": repeat,
                    "slice": mixed_row["slice"],
                    "n": mixed_row["n"],
                    "base_auc": base_row["auc"],
                    "mixed_auc": mixed_row["auc"],
                    "delta_vs_base": mixed_row["auc"] - base_row["auc"],
                    "policy": _policy_key(policy),
                }
            )

    result = pd.DataFrame(rows)
    slice_result = pd.DataFrame(slice_delta_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "slice_balanced_gate_policy.csv", index=False)
    slice_result.to_csv(args.out / "slice_balanced_gate_policy_slices.csv", index=False)
    if slice_result.empty:
        slice_summary = []
    else:
        slice_summary = (
            slice_result.groupby("slice")
            .agg(
                repeats=("repeat", "count"),
                mean_n=("n", "mean"),
                mean_delta_vs_base=("delta_vs_base", "mean"),
                median_delta_vs_base=("delta_vs_base", "median"),
                ci95_low=("delta_vs_base", lambda x: float(np.quantile(x, 0.025))),
                ci95_high=("delta_vs_base", lambda x: float(np.quantile(x, 0.975))),
            )
            .reset_index()
        )
        slice_summary.to_csv(args.out / "slice_balanced_gate_policy_slice_summary.csv", index=False)
        slice_summary = slice_summary.to_dict(orient="records")

    def _delta_summary(column: str) -> dict[str, float | int | None]:
        if result.empty:
            return {
                "n": 0,
                "mean": None,
                "median": None,
                "ci95_low": None,
                "ci95_high": None,
                "p_delta_le_0": None,
                "p_delta_lt_3pt": None,
            }
        values = result[column].to_numpy(dtype=float)
        return {
            "n": int(len(values)),
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "ci95_low": float(np.quantile(values, 0.025)),
            "ci95_high": float(np.quantile(values, 0.975)),
            "p_delta_le_0": float(np.mean(values <= 0.0)),
            "p_delta_lt_3pt": float(np.mean(values < 0.03)),
        }

    summary = {
        "target": args.target,
        "slice_column": args.slice_column,
        "base_method": args.base_method,
        "train_frac": args.train_frac,
        "min_train_slice": args.min_train_slice,
        "min_eval_slice": args.min_eval_slice,
        "switch_margin": args.switch_margin,
        "n_repeats_requested": args.n_repeats,
        "n_repeats_valid": int(len(result)),
        "overall_delta": _delta_summary("overall_delta_vs_base"),
        "slice_macro_delta": _delta_summary("slice_macro_delta_vs_base"),
        "slice_summary": slice_summary,
        "policy_counts": [{"policy": key, "count": count} for key, count in policy_counts.most_common()],
    }
    (args.out / "slice_balanced_gate_policy_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    policy_df = pd.DataFrame(summary["policy_counts"])
    slice_df = pd.DataFrame(slice_summary)
    lines = [
        "# Slice-Balanced Gate Policy Audit",
        "",
        "This audit repeats the held-out transfer-gate evaluation but scores both the usual sample-weighted overall AUROC and a symptom-slice macro AUROC that gives each valid symptom slice equal weight.",
        "",
        "## Overall Delta",
        "",
        "```json",
        json.dumps(summary["overall_delta"], indent=2),
        "```",
        "",
        "## Slice-Macro Delta",
        "",
        "```json",
        json.dumps(summary["slice_macro_delta"], indent=2),
        "```",
        "",
        "## Per-Slice Summary",
        "",
        _to_md(slice_df),
        "",
        "## Policy Counts",
        "",
        _to_md(policy_df.head(20)),
        "",
    ]
    report = args.out / "SLICE_BALANCED_GATE_POLICY.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
