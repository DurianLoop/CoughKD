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
from audit_target_calibrated_stacking import _features, _fit_stack, _predict_full_probs


ROOT = Path(__file__).resolve().parents[1]


SLICE_CONFIG = {
    "COUGHVID": {"slice_column": "symptom_resp", "min_train_slice": 80, "switch_margin": 0.02},
    "TosCOVID": {"slice_column": "age", "min_train_slice": 80, "switch_margin": 0.02},
    "UKCOVID": {"slice_column": "symptom_cough_any", "min_train_slice": 200, "switch_margin": 0.02},
}


def _eval_scores(labels: pd.Series, scores: np.ndarray, idx: np.ndarray) -> float:
    return _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), scores[idx])


def _best_single(
    labels: pd.Series,
    method_scores: dict[str, np.ndarray],
    methods: list[str],
    train_idx: np.ndarray,
) -> str | None:
    best_method = None
    best_auc = -np.inf
    for method in methods:
        try:
            auc = _eval_scores(labels, method_scores[method], train_idx)
        except ValueError:
            continue
        if auc > best_auc:
            best_auc = auc
            best_method = method
    return best_method


def _strategy_scores(
    *,
    strategy: str,
    labels: pd.Series,
    method_scores: dict[str, np.ndarray],
    methods: list[str],
    base_method: str,
    train_idx: np.ndarray,
    eval_idx: np.ndarray,
    x_all: np.ndarray,
    stack_c: float,
    groups: pd.Series | None,
    min_train_slice: int,
    switch_margin: float,
) -> tuple[np.ndarray, dict[str, object]] | None:
    details: dict[str, object] = {}
    if strategy == "source_only":
        return method_scores[base_method][eval_idx], details
    if strategy == "calib_best_single":
        method = _best_single(labels, method_scores, methods, train_idx)
        if method is None:
            return None
        details["chosen_method"] = method
        return method_scores[method][eval_idx], details
    if strategy == "logistic_stacking":
        model = _fit_stack(x_all[train_idx], labels.iloc[train_idx].reset_index(drop=True), stack_c)
        if model is None:
            return None
        details["c_value"] = stack_c
        return _predict_full_probs(model, x_all[eval_idx]), details
    if strategy == "slice_gate":
        if groups is None:
            return None
        policy = _fit_policy(
            labels,
            groups,
            method_scores,
            train_idx,
            min_train_slice,
            base_method,
            switch_margin,
        )
        details["policy_size"] = len(policy)
        details["policy"] = policy
        return _apply_policy(groups, method_scores, base_method, policy, eval_idx), details
    raise ValueError(f"Unknown strategy: {strategy}")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["COUGHVID", "TosCOVID"])
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=120)
    parser.add_argument("--outer-train-frac", type=float, default=0.5)
    parser.add_argument("--inner-train-frac", type=float, default=0.7)
    parser.add_argument("--stack-c", type=float, default=1.0)
    parser.add_argument("--selection-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/target_calibrated_transfer_controller")
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
    rows = []
    strategy_rows = []
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

            inner_source_auc = _eval_scores(labels, method_scores[args.base_method], inner_val_idx)
            scored_strategies = []
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
                scores, details = out
                try:
                    auc = _macro_ovr_auc(labels.iloc[inner_val_idx].reset_index(drop=True), scores)
                except ValueError:
                    continue
                scored_strategies.append(
                    {
                        "strategy": strategy,
                        "inner_auc": auc,
                        "inner_delta_vs_source": auc - inner_source_auc,
                        "details": details,
                    }
                )
            if not scored_strategies:
                continue
            selected = max(scored_strategies, key=lambda row: row["inner_auc"])
            if selected["strategy"] != "source_only" and selected["inner_delta_vs_source"] < args.selection_margin:
                selected = {
                    "strategy": "source_only",
                    "inner_auc": inner_source_auc,
                    "inner_delta_vs_source": 0.0,
                    "details": {},
                }
            selected_strategy = selected["strategy"]

            final_out = _strategy_scores(
                strategy=selected_strategy,
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
            if final_out is None:
                continue
            final_scores, final_details = final_out
            y_eval = labels.iloc[outer_eval_idx].reset_index(drop=True)
            source_scores_eval = method_scores[args.base_method][outer_eval_idx]
            try:
                source_auc = _macro_ovr_auc(y_eval, source_scores_eval)
                controller_auc = _macro_ovr_auc(y_eval, final_scores)
            except ValueError:
                continue
            for strategy_row in scored_strategies:
                strategy_rows.append(
                    {
                        "target": target,
                        "repeat": repeat,
                        "strategy": strategy_row["strategy"],
                        "inner_auc": strategy_row["inner_auc"],
                        "inner_delta_vs_source": strategy_row["inner_delta_vs_source"],
                    }
                )
            rows.append(
                {
                    "target": target,
                    "repeat": repeat,
                    "outer_train_n": int(len(outer_train_idx)),
                    "outer_eval_n": int(len(outer_eval_idx)),
                    "selected_strategy": selected_strategy,
                    "selected_inner_delta_vs_source": selected["inner_delta_vs_source"],
                    "source_auc": source_auc,
                    "controller_auc": controller_auc,
                    "delta_vs_source": controller_auc - source_auc,
                    "final_details": json.dumps(final_details, sort_keys=True),
                }
            )

    result = pd.DataFrame(rows)
    strategy_result = pd.DataFrame(strategy_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "target_calibrated_transfer_controller.csv", index=False)
    strategy_result.to_csv(args.out / "target_calibrated_transfer_controller_inner_strategies.csv", index=False)
    if result.empty:
        summary_df = pd.DataFrame()
    else:
        summary_df = (
            result.groupby("target")
            .agg(
                n_repeats=("repeat", "count"),
                mean_delta_vs_source=("delta_vs_source", "mean"),
                median_delta_vs_source=("delta_vs_source", "median"),
                ci95_low=("delta_vs_source", lambda x: float(np.quantile(x, 0.025))),
                ci95_high=("delta_vs_source", lambda x: float(np.quantile(x, 0.975))),
                p_delta_le_0=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) <= 0.0))),
                p_delta_lt_3pt=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) < 0.03))),
            )
            .reset_index()
        )
    summary_df.to_csv(args.out / "target_calibrated_transfer_controller_summary.csv", index=False)
    selection_counts = (
        result.groupby(["target", "selected_strategy"]).size().reset_index(name="count")
        if not result.empty
        else pd.DataFrame()
    )
    selection_counts.to_csv(args.out / "target_calibrated_transfer_controller_selection_counts.csv", index=False)
    large = summary_df[summary_df["target"].isin(["COUGHVID", "TosCOVID"])] if not summary_df.empty else summary_df
    payload = {
        "targets": sorted(result["target"].unique().tolist()) if not result.empty else [],
        "n_rows": int(len(result)),
        "selection_margin": args.selection_margin,
        "summary": summary_df.to_dict(orient="records"),
        "selection_counts": selection_counts.to_dict(orient="records") if not selection_counts.empty else [],
        "clears_3pt_on_any_large_target": bool((large["mean_delta_vs_source"] >= 0.03).any())
        if not large.empty
        else False,
        "clears_3pt_on_all_large_targets": bool(large["mean_delta_vs_source"].ge(0.03).all())
        if not large.empty and set(large["target"]) >= {"COUGHVID", "TosCOVID"}
        else False,
    }
    (args.out / "target_calibrated_transfer_controller_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Target-Calibrated Transfer Controller",
        "",
        "This nested audit lets a target calibration split choose among source-only, calibration-best single model, logistic stacking, and a target metadata slice gate when available.",
        "",
        "## Summary",
        "",
        _to_md(summary_df),
        "",
        "## Selection Counts",
        "",
        _to_md(selection_counts),
        "",
        f"Clears 3-point gate on any large target: `{payload['clears_3pt_on_any_large_target']}`",
        f"Clears 3-point gate on all large targets: `{payload['clears_3pt_on_all_large_targets']}`",
        "",
    ]
    report = args.out / "TARGET_CALIBRATED_TRANSFER_CONTROLLER.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
