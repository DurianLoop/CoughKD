from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata_slice_oracle import TARGETS, _method_predictions, _prepare_manifest
from audit_prediction_ensemble_frontier import _macro_ovr_auc
from audit_target_calibrated_stacking import _features
from audit_target_calibrated_transfer_controller import SLICE_CONFIG, _strategy_scores


ROOT = Path(__file__).resolve().parents[1]


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


def _score_auc(labels: pd.Series, scores: np.ndarray, idx: np.ndarray) -> float:
    return _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), scores[idx])


def _classify_failure(
    *,
    selected_strategy: str,
    selected_delta: float,
    oracle_strategy: str,
    oracle_delta: float,
    oracle_regret: float,
    oracle_tolerance: float,
    weak_delta: float,
) -> str:
    if oracle_regret <= oracle_tolerance:
        return "selected_near_oracle"
    if oracle_delta < weak_delta:
        return "all_strategies_weak"
    if selected_delta < 0.0 and oracle_strategy == "source_only":
        return "source_only_would_avoid_negative_transfer"
    if selected_strategy != oracle_strategy:
        return "wrong_strategy_family"
    return "calibration_refit_underperformed"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["COUGHVID", "TosCOVID"])
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=160)
    parser.add_argument("--outer-train-frac", type=float, default=0.5)
    parser.add_argument("--inner-train-frac", type=float, default=0.7)
    parser.add_argument("--stack-c", type=float, default=1.0)
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument("--oracle-tolerance", type=float, default=0.002)
    parser.add_argument("--weak-delta", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/transfer_controller_oracle_errors")
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
            "tcd_conf035",
            "tcd_very_strong",
        ],
    )
    args = parser.parse_args()

    strategies = ["source_only", "calib_best_single", "logistic_stacking", "slice_gate"]
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, object]] = []
    outer_strategy_rows: list[dict[str, object]] = []

    for target in args.targets:
        config = TARGETS[target]
        manifest = _prepare_manifest(config["manifest"])
        try:
            base, method_scores = _method_predictions(target, args.methods)
        except ValueError:
            continue
        methods = [method for method in args.methods if method in method_scores]
        if args.base_method not in methods:
            continue

        merged = base.merge(manifest, on="recording_id", how="left", suffixes=("", "_manifest"))
        labels = merged["true_label"].reset_index(drop=True)
        x_all = _features(method_scores, methods)

        slice_config = SLICE_CONFIG.get(target)
        groups = None
        min_train_slice = 0
        switch_margin = 0.0
        if slice_config and slice_config["slice_column"] in merged.columns:
            groups = merged[slice_config["slice_column"]].fillna("missing").astype(str).reset_index(drop=True)
            min_train_slice = int(slice_config["min_train_slice"])
            switch_margin = float(slice_config["switch_margin"])

        n = len(labels)
        outer_train_n = int(round(n * args.outer_train_frac))
        for repeat in range(args.n_repeats):
            outer_perm = rng.permutation(n)
            outer_train_idx = np.sort(outer_perm[:outer_train_n])
            outer_eval_idx = np.sort(outer_perm[outer_train_n:])
            if labels.iloc[outer_eval_idx].nunique() < 2:
                continue

            inner_train_n = int(round(len(outer_train_idx) * args.inner_train_frac))
            inner_perm = rng.permutation(len(outer_train_idx))
            inner_train_idx = np.sort(outer_train_idx[inner_perm[:inner_train_n]])
            inner_val_idx = np.sort(outer_train_idx[inner_perm[inner_train_n:]])
            if labels.iloc[inner_val_idx].nunique() < 2:
                continue

            try:
                inner_source_auc = _score_auc(labels, method_scores[args.base_method], inner_val_idx)
                source_auc = _score_auc(labels, method_scores[args.base_method], outer_eval_idx)
            except ValueError:
                continue

            inner_scored: list[dict[str, object]] = []
            for strategy in strategies:
                out = _strategy_scores(
                    strategy=strategy,
                    labels=labels,
                    method_scores=method_scores,
                    methods=methods,
                    base_method=args.base_method,
                    train_idx=inner_train_idx,
                    eval_idx=inner_val_idx,
                    x_all=x_all,
                    stack_c=args.stack_c,
                    groups=groups,
                    min_train_slice=min_train_slice,
                    switch_margin=switch_margin,
                )
                if out is None:
                    continue
                scores, _details = out
                try:
                    auc = _macro_ovr_auc(labels.iloc[inner_val_idx].reset_index(drop=True), scores)
                except ValueError:
                    continue
                inner_scored.append(
                    {
                        "strategy": strategy,
                        "inner_auc": auc,
                        "inner_delta_vs_source": auc - inner_source_auc,
                    }
                )
            if not inner_scored:
                continue

            selected = max(inner_scored, key=lambda row: float(row["inner_auc"]))
            if selected["strategy"] != "source_only" and float(selected["inner_delta_vs_source"]) < args.selection_margin:
                selected = {
                    "strategy": "source_only",
                    "inner_auc": inner_source_auc,
                    "inner_delta_vs_source": 0.0,
                }
            selected_strategy = str(selected["strategy"])

            outer_scored: list[dict[str, object]] = []
            for strategy in strategies:
                out = _strategy_scores(
                    strategy=strategy,
                    labels=labels,
                    method_scores=method_scores,
                    methods=methods,
                    base_method=args.base_method,
                    train_idx=outer_train_idx,
                    eval_idx=outer_eval_idx,
                    x_all=x_all,
                    stack_c=args.stack_c,
                    groups=groups,
                    min_train_slice=min_train_slice,
                    switch_margin=switch_margin,
                )
                if out is None:
                    continue
                scores, details = out
                try:
                    auc = _macro_ovr_auc(labels.iloc[outer_eval_idx].reset_index(drop=True), scores)
                except ValueError:
                    continue
                delta = auc - source_auc
                row = {
                    "target": target,
                    "repeat": repeat,
                    "strategy": strategy,
                    "outer_auc": auc,
                    "outer_delta_vs_source": delta,
                    "details": json.dumps(details, sort_keys=True),
                }
                outer_scored.append(row)
                outer_strategy_rows.append(row)
            if not outer_scored:
                continue

            oracle = max(outer_scored, key=lambda row: float(row["outer_auc"]))
            selected_outer = next((row for row in outer_scored if row["strategy"] == selected_strategy), None)
            if selected_outer is None:
                continue

            selected_auc = float(selected_outer["outer_auc"])
            selected_delta = float(selected_outer["outer_delta_vs_source"])
            oracle_auc = float(oracle["outer_auc"])
            oracle_delta = float(oracle["outer_delta_vs_source"])
            oracle_regret = oracle_auc - selected_auc
            failure_category = _classify_failure(
                selected_strategy=selected_strategy,
                selected_delta=selected_delta,
                oracle_strategy=str(oracle["strategy"]),
                oracle_delta=oracle_delta,
                oracle_regret=oracle_regret,
                oracle_tolerance=args.oracle_tolerance,
                weak_delta=args.weak_delta,
            )
            rows.append(
                {
                    "target": target,
                    "repeat": repeat,
                    "outer_train_n": int(len(outer_train_idx)),
                    "outer_eval_n": int(len(outer_eval_idx)),
                    "selected_strategy": selected_strategy,
                    "selected_inner_delta_vs_source": float(selected["inner_delta_vs_source"]),
                    "source_auc": source_auc,
                    "selected_auc": selected_auc,
                    "selected_delta_vs_source": selected_delta,
                    "oracle_strategy": str(oracle["strategy"]),
                    "oracle_auc": oracle_auc,
                    "oracle_delta_vs_source": oracle_delta,
                    "oracle_regret": oracle_regret,
                    "failure_category": failure_category,
                    "available_strategies": len(outer_scored),
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    outer_strategy_result = pd.DataFrame(outer_strategy_rows)
    result.to_csv(args.out / "transfer_controller_oracle_errors.csv", index=False)
    outer_strategy_result.to_csv(args.out / "transfer_controller_outer_strategies.csv", index=False)

    if result.empty:
        summary_df = pd.DataFrame()
        category_counts = pd.DataFrame()
        oracle_counts = pd.DataFrame()
    else:
        summary_df = (
            result.groupby("target")
            .agg(
                n_repeats=("repeat", "count"),
                mean_selected_delta=("selected_delta_vs_source", "mean"),
                mean_oracle_delta=("oracle_delta_vs_source", "mean"),
                mean_oracle_regret=("oracle_regret", "mean"),
                median_oracle_regret=("oracle_regret", "median"),
                p_near_oracle=("oracle_regret", lambda x: float(np.mean(np.asarray(x) <= args.oracle_tolerance))),
                p_regret_gt_1pt=("oracle_regret", lambda x: float(np.mean(np.asarray(x) > 0.01))),
                p_selected_negative=("selected_delta_vs_source", lambda x: float(np.mean(np.asarray(x) < 0.0))),
                p_oracle_ge_1pt=("oracle_delta_vs_source", lambda x: float(np.mean(np.asarray(x) >= 0.01))),
            )
            .reset_index()
        )
        category_counts = result.groupby(["target", "failure_category"]).size().reset_index(name="count")
        oracle_counts = result.groupby(["target", "oracle_strategy"]).size().reset_index(name="count")

    summary_df.to_csv(args.out / "transfer_controller_oracle_error_summary.csv", index=False)
    category_counts.to_csv(args.out / "transfer_controller_failure_categories.csv", index=False)
    oracle_counts.to_csv(args.out / "transfer_controller_oracle_strategy_counts.csv", index=False)
    payload = {
        "targets": sorted(result["target"].unique().tolist()) if not result.empty else [],
        "n_rows": int(len(result)),
        "oracle_tolerance": args.oracle_tolerance,
        "weak_delta": args.weak_delta,
        "summary": summary_df.to_dict(orient="records") if not summary_df.empty else [],
        "failure_categories": category_counts.to_dict(orient="records") if not category_counts.empty else [],
        "oracle_strategy_counts": oracle_counts.to_dict(orient="records") if not oracle_counts.empty else [],
    }
    (args.out / "transfer_controller_oracle_error_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Transfer Controller Oracle-Error Audit",
        "",
        "This diagnostic evaluates every available strategy on the same held-out outer split, then compares the target-calibrated selector against the held-out oracle best strategy.",
        "",
        "## Summary",
        "",
        _to_md(summary_df),
        "",
        "## Failure Categories",
        "",
        _to_md(category_counts),
        "",
        "## Oracle Strategy Counts",
        "",
        _to_md(oracle_counts),
        "",
        f"Oracle tolerance: `{args.oracle_tolerance}`",
        f"Weak-strategy delta threshold: `{args.weak_delta}`",
        "",
    ]
    report = args.out / "TRANSFER_CONTROLLER_ORACLE_ERRORS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
