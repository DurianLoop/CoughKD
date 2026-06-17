from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata_slice_oracle import TARGETS, _method_predictions, _prepare_manifest
from audit_prediction_ensemble_frontier import _macro_ovr_auc
from audit_target_calibrated_stacking import _features
from audit_target_calibrated_transfer_controller import SLICE_CONFIG, _strategy_scores, _to_md


ROOT = Path(__file__).resolve().parents[1]


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


def _bootstrap_delta_lcb(
    y_val: pd.Series,
    source_scores: np.ndarray,
    candidate_scores: np.ndarray,
    rng: np.random.Generator,
    n_bootstrap: int,
    alpha: float,
) -> dict[str, float | int | None]:
    deltas = []
    n = len(y_val)
    for _ in range(n_bootstrap):
        pos = rng.integers(0, n, size=n)
        y_boot = y_val.iloc[pos].reset_index(drop=True)
        if y_boot.nunique() < 2:
            continue
        try:
            candidate_auc = _macro_ovr_auc(y_boot, candidate_scores[pos])
            source_auc = _macro_ovr_auc(y_boot, source_scores[pos])
        except ValueError:
            continue
        deltas.append(candidate_auc - source_auc)
    if not deltas:
        return {"mean_delta": None, "lcb_delta": None, "valid_bootstrap": 0}
    arr = np.asarray(deltas, dtype=float)
    return {
        "mean_delta": float(np.mean(arr)),
        "lcb_delta": float(np.quantile(arr, alpha)),
        "valid_bootstrap": int(len(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["COUGHVID", "TosCOVID"])
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=80)
    parser.add_argument("--outer-train-frac", type=float, default=0.5)
    parser.add_argument("--inner-train-frac", type=float, default=0.7)
    parser.add_argument("--stack-c", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--lcb-floor", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/bootstrap_lcb_transfer_controller")
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
    skipped = defaultdict(int)
    for target in args.targets:
        config = TARGETS[target]
        manifest = _prepare_manifest(config["manifest"])
        try:
            base, method_scores = _method_predictions(target, args.methods)
        except ValueError:
            skipped[(target, "missing_predictions")] += 1
            continue
        methods = [method for method in args.methods if method in method_scores]
        if args.base_method not in methods:
            skipped[(target, "missing_base_method")] += 1
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
                skipped[(target, "outer_eval_single_class")] += 1
                continue
            inner_train_n = int(round(len(outer_train_idx) * args.inner_train_frac))
            inner_perm = rng.permutation(len(outer_train_idx))
            inner_train_idx = np.sort(outer_train_idx[inner_perm[:inner_train_n]])
            inner_val_idx = np.sort(outer_train_idx[inner_perm[inner_train_n:]])
            y_val = labels.iloc[inner_val_idx].reset_index(drop=True)
            if y_val.nunique() < 2:
                skipped[(target, "inner_val_single_class")] += 1
                continue

            source_scores_val = method_scores[args.base_method][inner_val_idx]
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
                scores, _details = out
                try:
                    inner_auc = _macro_ovr_auc(y_val, scores)
                    inner_source_auc = _macro_ovr_auc(y_val, source_scores_val)
                except ValueError:
                    continue
                if strategy == "source_only":
                    boot = {"mean_delta": 0.0, "lcb_delta": 0.0, "valid_bootstrap": args.n_bootstrap}
                else:
                    boot = _bootstrap_delta_lcb(
                        y_val,
                        source_scores_val,
                        scores,
                        rng,
                        args.n_bootstrap,
                        args.alpha,
                    )
                    if boot["lcb_delta"] is None:
                        continue
                scored = {
                    "strategy": strategy,
                    "inner_auc": inner_auc,
                    "inner_delta_vs_source": inner_auc - inner_source_auc,
                    "bootstrap_mean_delta": boot["mean_delta"],
                    "bootstrap_lcb_delta": boot["lcb_delta"],
                    "valid_bootstrap": boot["valid_bootstrap"],
                }
                scored_strategies.append(scored)
                strategy_rows.append({"target": target, "repeat": repeat, **scored})
            if not scored_strategies:
                continue

            selected = max(scored_strategies, key=lambda row: row["bootstrap_lcb_delta"])
            if selected["strategy"] != "source_only" and selected["bootstrap_lcb_delta"] < args.lcb_floor:
                selected = {
                    "strategy": "source_only",
                    "inner_delta_vs_source": 0.0,
                    "bootstrap_mean_delta": 0.0,
                    "bootstrap_lcb_delta": 0.0,
                    "valid_bootstrap": args.n_bootstrap,
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
            rows.append(
                {
                    "target": target,
                    "repeat": repeat,
                    "outer_train_n": int(len(outer_train_idx)),
                    "outer_eval_n": int(len(outer_eval_idx)),
                    "selected_strategy": selected_strategy,
                    "selected_inner_delta_vs_source": selected["inner_delta_vs_source"],
                    "selected_bootstrap_mean_delta": selected["bootstrap_mean_delta"],
                    "selected_bootstrap_lcb_delta": selected["bootstrap_lcb_delta"],
                    "selected_valid_bootstrap": selected["valid_bootstrap"],
                    "source_auc": source_auc,
                    "controller_auc": controller_auc,
                    "delta_vs_source": controller_auc - source_auc,
                    "final_details": json.dumps(final_details, sort_keys=True),
                }
            )

    result = pd.DataFrame(rows)
    strategy_result = pd.DataFrame(strategy_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "bootstrap_lcb_transfer_controller.csv", index=False)
    strategy_result.to_csv(args.out / "bootstrap_lcb_transfer_controller_inner_strategies.csv", index=False)
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
    selection_counts = (
        result.groupby(["target", "selected_strategy"]).size().reset_index(name="count")
        if not result.empty
        else pd.DataFrame()
    )
    summary_df.to_csv(args.out / "bootstrap_lcb_transfer_controller_summary.csv", index=False)
    selection_counts.to_csv(args.out / "bootstrap_lcb_transfer_controller_selection_counts.csv", index=False)
    large = summary_df[summary_df["target"].isin(["COUGHVID", "TosCOVID"])] if not summary_df.empty else summary_df
    payload = {
        "targets": sorted(result["target"].unique().tolist()) if not result.empty else [],
        "n_rows": int(len(result)),
        "n_bootstrap": args.n_bootstrap,
        "alpha": args.alpha,
        "lcb_floor": args.lcb_floor,
        "summary": summary_df.to_dict(orient="records"),
        "selection_counts": selection_counts.to_dict(orient="records") if not selection_counts.empty else [],
        "skipped": [{"target": key[0], "reason": key[1], "count": value} for key, value in skipped.items()],
        "clears_3pt_on_any_large_target": bool((large["mean_delta_vs_source"] >= 0.03).any())
        if not large.empty
        else False,
        "clears_3pt_on_all_large_targets": bool(large["mean_delta_vs_source"].ge(0.03).all())
        if not large.empty and set(large["target"]) >= {"COUGHVID", "TosCOVID"}
        else False,
    }
    (args.out / "bootstrap_lcb_transfer_controller_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Bootstrap-LCB Transfer Controller",
        "",
        "This nested audit selects a target-calibrated transfer strategy by the bootstrap lower confidence bound of inner-validation AUROC delta versus source-only.",
        "",
        f"Bootstrap samples per repeat: `{args.n_bootstrap}`",
        f"LCB alpha: `{args.alpha}`",
        f"LCB floor for switching: `{args.lcb_floor}`",
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
    report = args.out / "BOOTSTRAP_LCB_TRANSFER_CONTROLLER.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
