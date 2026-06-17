from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from audit_crossfit_metadata_slice_policy import _apply_policy, _fit_policy
from audit_metadata_slice_oracle import TARGETS, _method_predictions, _prepare_manifest
from audit_prediction_ensemble_frontier import _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _leaf_groups(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing leaf columns: {missing}")
    parts = []
    for column in columns:
        values = df[column].fillna("missing").astype(str).str.strip()
        values = values.replace("", "missing")
        parts.append(column + "=" + values)
    return pd.concat(parts, axis=1).agg("|".join, axis=1)


def _slice_macro_delta(
    labels: pd.Series,
    groups: pd.Series,
    base_scores: np.ndarray,
    mixed_scores: np.ndarray,
    idx: np.ndarray,
    min_eval_slice: int,
) -> tuple[float | None, list[dict[str, object]]]:
    local_groups = groups.iloc[idx].reset_index(drop=True)
    local_labels = labels.iloc[idx].reset_index(drop=True)
    rows = []
    deltas = []
    for group_value, local_idx_obj in local_groups.groupby(local_groups).groups.items():
        local_idx = np.asarray(list(local_idx_obj), dtype=int)
        if len(local_idx) < min_eval_slice or local_labels.iloc[local_idx].nunique() < 2:
            continue
        base_auc = _macro_ovr_auc(local_labels.iloc[local_idx].reset_index(drop=True), base_scores[local_idx])
        mixed_auc = _macro_ovr_auc(local_labels.iloc[local_idx].reset_index(drop=True), mixed_scores[local_idx])
        delta = mixed_auc - base_auc
        deltas.append(delta)
        rows.append(
            {
                "slice": str(group_value),
                "n": int(len(local_idx)),
                "base_auc": float(base_auc),
                "mixed_auc": float(mixed_auc),
                "delta_vs_base": float(delta),
            }
        )
    if not deltas:
        return None, rows
    return float(np.mean(deltas)), rows


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_delta_le_0": None,
            "p_delta_lt_3pt": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "p_delta_le_0": float(np.mean(values <= 0.0)),
        "p_delta_lt_3pt": float(np.mean(values < 0.03)),
    }


def _policy_key(policy: dict[str, str]) -> str:
    if not policy:
        return "<base>"
    return "; ".join(f"{key}->{value}" for key, value in sorted(policy.items()))


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
    parser.add_argument("--leaf-columns", nargs="+", default=["symptom_resp", "age_bin"])
    parser.add_argument("--primary-slice-column", default="symptom_resp")
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--min-train-leaf", type=int, default=100)
    parser.add_argument("--min-eval-primary-slice", type=int, default=60)
    parser.add_argument("--switch-margin", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/leaf_metadata_gate_policy")
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
    leaf_groups = _leaf_groups(merged, args.leaf_columns).reset_index(drop=True)
    primary_groups = merged[args.primary_slice_column].fillna("missing").astype(str).reset_index(drop=True)
    labels = merged["true_label"].reset_index(drop=True)
    n = len(labels)
    train_n = int(round(n * args.train_frac))
    rng = np.random.default_rng(args.seed)

    rows = []
    primary_slice_rows = []
    policy_counts: Counter[str] = Counter()
    leaf_switch_counts: Counter[str] = Counter()
    for repeat in range(args.n_repeats):
        perm = rng.permutation(n)
        train_idx = np.sort(perm[:train_n])
        eval_idx = np.sort(perm[train_n:])
        if labels.iloc[eval_idx].nunique() < 2:
            continue
        policy = _fit_policy(
            labels,
            leaf_groups,
            method_scores,
            train_idx,
            args.min_train_leaf,
            args.base_method,
            args.switch_margin,
        )
        policy_counts[_policy_key(policy)] += 1
        for leaf, method in policy.items():
            leaf_switch_counts[f"{leaf}->{method}"] += 1
        mixed_scores = _apply_policy(leaf_groups, method_scores, args.base_method, policy, eval_idx)
        base_scores = method_scores[args.base_method][eval_idx]
        try:
            eval_labels = labels.iloc[eval_idx].reset_index(drop=True)
            overall_base = _macro_ovr_auc(eval_labels, base_scores)
            overall_mixed = _macro_ovr_auc(eval_labels, mixed_scores)
        except ValueError:
            continue
        primary_macro, primary_rows = _slice_macro_delta(
            labels,
            primary_groups,
            base_scores,
            mixed_scores,
            eval_idx,
            args.min_eval_primary_slice,
        )
        if primary_macro is None:
            continue
        rows.append(
            {
                "repeat": repeat,
                "eval_n": int(len(eval_idx)),
                "policy_size": int(len(policy)),
                "overall_base_auc": overall_base,
                "overall_mixed_auc": overall_mixed,
                "overall_delta_vs_base": overall_mixed - overall_base,
                "primary_slice_macro_delta_vs_base": primary_macro,
                "policy": _policy_key(policy),
            }
        )
        for item in primary_rows:
            primary_slice_rows.append({"repeat": repeat, **item, "policy": _policy_key(policy)})

    result = pd.DataFrame(rows)
    primary_result = pd.DataFrame(primary_slice_rows)
    margin_tag = f"margin{args.switch_margin:g}".replace(".", "p").replace("-", "neg")
    train_tag = f"train{args.train_frac:g}".replace(".", "p")
    tag = (
        f"{args.target}_{'_'.join(args.leaf_columns)}_{train_tag}_minLeaf{args.min_train_leaf}_{margin_tag}"
    ).replace("=", "-")
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / f"leaf_gate_{tag}.csv", index=False)
    primary_result.to_csv(args.out / f"leaf_gate_{tag}_primary_slices.csv", index=False)

    if primary_result.empty:
        primary_slice_summary = []
    else:
        primary_slice_summary_df = (
            primary_result.groupby("slice")
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
        primary_slice_summary_df.to_csv(args.out / f"leaf_gate_{tag}_primary_slice_summary.csv", index=False)
        primary_slice_summary = primary_slice_summary_df.to_dict(orient="records")

    summary = {
        "target": args.target,
        "leaf_columns": args.leaf_columns,
        "primary_slice_column": args.primary_slice_column,
        "base_method": args.base_method,
        "train_frac": args.train_frac,
        "min_train_leaf": args.min_train_leaf,
        "min_eval_primary_slice": args.min_eval_primary_slice,
        "switch_margin": args.switch_margin,
        "n_repeats_requested": args.n_repeats,
        "n_repeats_valid": int(len(result)),
        "overall_delta": _summary(result["overall_delta_vs_base"].to_numpy(dtype=float)) if not result.empty else _summary(np.asarray([])),
        "primary_slice_macro_delta": _summary(result["primary_slice_macro_delta_vs_base"].to_numpy(dtype=float))
        if not result.empty
        else _summary(np.asarray([])),
        "primary_slice_summary": primary_slice_summary,
        "top_leaf_switch_counts": [
            {"leaf_switch": key, "count": count} for key, count in leaf_switch_counts.most_common(30)
        ],
        "top_policy_counts": [{"policy": key, "count": count} for key, count in policy_counts.most_common(20)],
    }
    (args.out / f"leaf_gate_{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Leaf Metadata Gate Policy",
        "",
        f"- Target: `{args.target}`",
        f"- Leaf columns: `{', '.join(args.leaf_columns)}`",
        f"- Primary slice column: `{args.primary_slice_column}`",
        "",
        "## Overall Delta",
        "",
        "```json",
        json.dumps(summary["overall_delta"], indent=2),
        "```",
        "",
        "## Primary Slice-Macro Delta",
        "",
        "```json",
        json.dumps(summary["primary_slice_macro_delta"], indent=2),
        "```",
        "",
        "## Primary Slice Summary",
        "",
        _to_md(pd.DataFrame(primary_slice_summary)),
        "",
        "## Top Leaf Switches",
        "",
        _to_md(pd.DataFrame(summary["top_leaf_switch_counts"]).head(20)),
        "",
    ]
    report = args.out / f"LEAF_GATE_{tag}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
