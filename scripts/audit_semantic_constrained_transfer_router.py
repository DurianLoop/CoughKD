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


DEFAULT_SEMANTIC_SLICE_COLUMNS = {
    "symptom_resp",
    "symptom_fever",
    "symptom_cough_any",
    "symptom_fatigue",
    "symptom_headache",
    "symptom_onset",
    "symptom_none",
    "cough_detected_bin",
    "source_status",
    "source_subset",
    "source_batch",
}


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


def _choose_strategy(target: str, semantic_slice_columns: set[str]) -> tuple[str, str]:
    slice_config = SLICE_CONFIG.get(target)
    if slice_config is None:
        return "logistic_stacking", "no_slice_config"
    slice_column = str(slice_config["slice_column"])
    if slice_column in semantic_slice_columns:
        return "slice_gate", f"semantic_slice:{slice_column}"
    return "logistic_stacking", f"nonsemantic_slice:{slice_column}"


def _random_split_indices(
    n: int,
    train_frac: float,
    rng: np.random.Generator,
    split_groups: pd.Series | None = None,
    allowed_idx: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if allowed_idx is None:
        allowed_idx = np.arange(n)
    allowed_idx = np.asarray(allowed_idx)
    if split_groups is None:
        perm = rng.permutation(allowed_idx)
        train_n = int(round(len(perm) * train_frac))
        return np.sort(perm[:train_n]), np.sort(perm[train_n:])

    groups = split_groups.astype(str)
    unique_groups = np.asarray(sorted(groups.iloc[allowed_idx].unique()))
    group_perm = rng.permutation(unique_groups)
    train_group_n = int(round(len(group_perm) * train_frac))
    train_groups = set(group_perm[:train_group_n])
    is_train = groups.isin(train_groups).to_numpy()
    train_idx = allowed_idx[is_train[allowed_idx]]
    eval_idx = allowed_idx[~is_train[allowed_idx]]
    return np.sort(train_idx), np.sort(eval_idx)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["COUGHVID", "TosCOVID"])
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=160)
    parser.add_argument("--outer-train-frac", type=float, default=0.5)
    parser.add_argument("--inner-train-frac", type=float, default=0.7)
    parser.add_argument("--stack-c", type=float, default=1.0)
    parser.add_argument(
        "--inner-guard-margin",
        type=float,
        default=None,
        help="If set, deploy the semantic strategy only when its inner-validation delta over source reaches this margin.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/semantic_constrained_transfer_router")
    parser.add_argument("--semantic-slice-columns", nargs="+", default=sorted(DEFAULT_SEMANTIC_SLICE_COLUMNS))
    parser.add_argument(
        "--group-split-column",
        default="",
        help="Optional manifest column, e.g. subject_id, used to keep target calibration/evaluation splits group-disjoint.",
    )
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

    rng = np.random.default_rng(args.seed)
    semantic_slice_columns = set(args.semantic_slice_columns)
    rows: list[dict[str, object]] = []

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
        split_groups = None
        if args.group_split_column:
            if args.group_split_column not in merged.columns:
                raise SystemExit(f"group split column not found for {target}: {args.group_split_column}")
            split_groups = merged[args.group_split_column].fillna("").astype(str).reset_index(drop=True)

        slice_config = SLICE_CONFIG.get(target)
        groups = None
        min_train_slice = 0
        switch_margin = 0.0
        slice_column = None
        if slice_config and slice_config["slice_column"] in merged.columns:
            slice_column = str(slice_config["slice_column"])
            groups = merged[slice_column].fillna("missing").astype(str).reset_index(drop=True)
            min_train_slice = int(slice_config["min_train_slice"])
            switch_margin = float(slice_config["switch_margin"])

        selected_strategy, selection_reason = _choose_strategy(target, semantic_slice_columns)
        if selected_strategy == "slice_gate" and groups is None:
            selected_strategy = "logistic_stacking"
            selection_reason = "slice_unavailable_fallback_logistic"
        base_selection_reason = selection_reason

        n = len(labels)
        for repeat in range(args.n_repeats):
            outer_train_idx, outer_eval_idx = _random_split_indices(
                n=n,
                train_frac=args.outer_train_frac,
                rng=rng,
                split_groups=split_groups,
            )
            if labels.iloc[outer_eval_idx].nunique() < 2:
                continue

            inner_train_idx, inner_val_idx = _random_split_indices(
                n=n,
                train_frac=args.inner_train_frac,
                rng=rng,
                split_groups=split_groups,
                allowed_idx=outer_train_idx,
            )
            if labels.iloc[inner_val_idx].nunique() < 2:
                continue

            deploy_strategy = selected_strategy
            deploy_reason = base_selection_reason
            inner_delta_vs_source = None
            if args.inner_guard_margin is not None and selected_strategy != "source_only":
                inner_out = _strategy_scores(
                    strategy=selected_strategy,
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
                if inner_out is None:
                    deploy_strategy = "source_only"
                    deploy_reason = f"{base_selection_reason}|guard_no_inner_score"
                    inner_delta_vs_source = None
                else:
                    inner_scores, _inner_details = inner_out
                    try:
                        inner_y = labels.iloc[inner_val_idx].reset_index(drop=True)
                        inner_source_auc = _macro_ovr_auc(inner_y, method_scores[args.base_method][inner_val_idx])
                        inner_strategy_auc = _macro_ovr_auc(inner_y, inner_scores)
                        inner_delta_vs_source = inner_strategy_auc - inner_source_auc
                    except ValueError:
                        deploy_strategy = "source_only"
                        deploy_reason = f"{base_selection_reason}|guard_no_inner_auc"
                    else:
                        if inner_delta_vs_source < args.inner_guard_margin:
                            deploy_strategy = "source_only"
                            deploy_reason = f"{base_selection_reason}|guard_fallback"

            out = _strategy_scores(
                strategy=deploy_strategy,
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
                y_eval = labels.iloc[outer_eval_idx].reset_index(drop=True)
                source_auc = _macro_ovr_auc(y_eval, method_scores[args.base_method][outer_eval_idx])
                router_auc = _macro_ovr_auc(y_eval, scores)
            except ValueError:
                continue
            rows.append(
                {
                    "target": target,
                    "repeat": repeat,
                    "outer_train_n": int(len(outer_train_idx)),
                    "outer_eval_n": int(len(outer_eval_idx)),
                    "group_split_column": args.group_split_column,
                    "semantic_strategy": selected_strategy,
                    "selected_strategy": deploy_strategy,
                    "selection_reason": deploy_reason,
                    "inner_guard_margin": args.inner_guard_margin,
                    "inner_delta_vs_source": inner_delta_vs_source,
                    "slice_column": slice_column,
                    "source_auc": source_auc,
                    "router_auc": router_auc,
                    "delta_vs_source": router_auc - source_auc,
                    "details": json.dumps(details, sort_keys=True),
                }
            )

    args.out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    result.to_csv(args.out / "semantic_constrained_transfer_router.csv", index=False)
    if result.empty:
        summary_df = pd.DataFrame()
        selection_counts = pd.DataFrame()
    else:
        summary_df = (
            result.groupby("target")
            .agg(
                n_repeats=("repeat", "count"),
                mean_delta_vs_source=("delta_vs_source", "mean"),
                median_delta_vs_source=("delta_vs_source", "median"),
                ci95_low=("delta_vs_source", lambda x: float(np.quantile(x, 0.025))),
                ci95_high=("delta_vs_source", lambda x: float(np.quantile(x, 0.975))),
                p_delta_lt_0=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) < 0.0))),
                p_delta_lt_1pt=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) < 0.01))),
                p_delta_lt_2pt=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) < 0.02))),
                p_delta_lt_3pt=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) < 0.03))),
            )
            .reset_index()
        )
        selection_counts = (
            result.groupby(["target", "selected_strategy", "selection_reason"])
            .size()
            .reset_index(name="count")
        )

    summary_df.to_csv(args.out / "semantic_constrained_transfer_router_summary.csv", index=False)
    selection_counts.to_csv(args.out / "semantic_constrained_transfer_router_selection_counts.csv", index=False)
    payload = {
        "targets": sorted(result["target"].unique().tolist()) if not result.empty else [],
        "n_rows": int(len(result)),
        "semantic_slice_columns": sorted(semantic_slice_columns),
        "inner_guard_margin": args.inner_guard_margin,
        "group_split_column": args.group_split_column,
        "summary": summary_df.to_dict(orient="records") if not summary_df.empty else [],
        "selection_counts": selection_counts.to_dict(orient="records") if not selection_counts.empty else [],
    }
    (args.out / "semantic_constrained_transfer_router_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Semantic-Constrained Transfer Router",
        "",
        "This audit uses target metadata semantics to restrict transfer strategy choice: symptom/acquisition slice metadata enables slice gating; demographic-only slice metadata routes to logistic stacking.",
        "",
        "## Summary",
        "",
        _to_md(summary_df),
        "",
        "## Selection Counts",
        "",
        _to_md(selection_counts),
        "",
        "Semantic slice columns:",
        "",
        ", ".join(sorted(semantic_slice_columns)),
        "",
        f"Inner guard margin: `{args.inner_guard_margin}`",
        "",
        f"Group split column: `{args.group_split_column}`",
        "",
    ]
    report = args.out / "SEMANTIC_CONSTRAINED_TRANSFER_ROUTER.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
