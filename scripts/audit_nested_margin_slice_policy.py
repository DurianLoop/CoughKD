from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_crossfit_metadata_slice_policy import _apply_policy, _fit_policy
from audit_metadata_slice_oracle import TARGETS, _method_predictions, _prepare_manifest
from audit_prediction_ensemble_frontier import _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _evaluate_policy(
    labels: pd.Series,
    groups: pd.Series,
    method_scores: dict[str, np.ndarray],
    base_method: str,
    policy: dict[str, str],
    eval_idx: np.ndarray,
) -> tuple[float, float, float]:
    mixed_scores = _apply_policy(groups, method_scores, base_method, policy, eval_idx)
    base_scores = method_scores[base_method][eval_idx]
    eval_labels = labels.iloc[eval_idx].reset_index(drop=True)
    mixed_auc = _macro_ovr_auc(eval_labels, mixed_scores)
    base_auc = _macro_ovr_auc(eval_labels, base_scores)
    return mixed_auc, base_auc, mixed_auc - base_auc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="COUGHVID")
    parser.add_argument("--slice-column", default="symptom_resp")
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--outer-train-frac", type=float, default=0.7)
    parser.add_argument("--inner-train-frac", type=float, default=0.7)
    parser.add_argument("--min-train-slice", type=int, default=100)
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.005, 0.01, 0.02, 0.03])
    parser.add_argument("--out", type=Path, default=ROOT / "runs/nested_margin_slice_policy")
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
    labels = merged["true_label"].reset_index(drop=True)
    groups = merged[args.slice_column].fillna("missing").astype(str).reset_index(drop=True)
    n = len(labels)
    outer_train_n = int(round(n * args.outer_train_frac))
    rng = np.random.default_rng(args.seed)

    rows = []
    margin_rows = []
    for repeat in range(args.n_repeats):
        outer_perm = rng.permutation(n)
        outer_train_idx = np.sort(outer_perm[:outer_train_n])
        outer_eval_idx = np.sort(outer_perm[outer_train_n:])
        if labels.iloc[outer_eval_idx].nunique() < 2:
            continue

        inner_n = int(round(len(outer_train_idx) * args.inner_train_frac))
        inner_perm_local = rng.permutation(len(outer_train_idx))
        inner_train_idx = np.sort(outer_train_idx[inner_perm_local[:inner_n]])
        inner_val_idx = np.sort(outer_train_idx[inner_perm_local[inner_n:]])
        if labels.iloc[inner_val_idx].nunique() < 2:
            continue

        best_margin = None
        best_inner_delta = -np.inf
        for margin in args.margins:
            policy = _fit_policy(
                labels,
                groups,
                method_scores,
                inner_train_idx,
                args.min_train_slice,
                args.base_method,
                margin,
            )
            try:
                _, _, inner_delta = _evaluate_policy(
                    labels, groups, method_scores, args.base_method, policy, inner_val_idx
                )
            except ValueError:
                continue
            margin_rows.append(
                {
                    "repeat": repeat,
                    "margin": margin,
                    "inner_delta": inner_delta,
                    "policy_size": len(policy),
                }
            )
            if inner_delta > best_inner_delta:
                best_inner_delta = inner_delta
                best_margin = margin
        if best_margin is None:
            continue

        final_policy = _fit_policy(
            labels,
            groups,
            method_scores,
            outer_train_idx,
            args.min_train_slice,
            args.base_method,
            best_margin,
        )
        try:
            mixed_auc, base_auc, outer_delta = _evaluate_policy(
                labels, groups, method_scores, args.base_method, final_policy, outer_eval_idx
            )
        except ValueError:
            continue
        rows.append(
            {
                "repeat": repeat,
                "target": args.target,
                "slice_column": args.slice_column,
                "outer_train_n": len(outer_train_idx),
                "outer_eval_n": len(outer_eval_idx),
                "best_margin": best_margin,
                "best_inner_delta": best_inner_delta,
                "final_policy_size": len(final_policy),
                "mixed_auc": mixed_auc,
                "base_auc": base_auc,
                "delta_vs_base": outer_delta,
            }
        )

    args.out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(rows)
    margin_df = pd.DataFrame(margin_rows)
    tag = f"{args.target}_{args.slice_column}_outer{args.outer_train_frac:g}".replace(".", "p")
    result.to_csv(args.out / f"nested_margin_{tag}.csv", index=False)
    margin_df.to_csv(args.out / f"nested_margin_{tag}_inner_margins.csv", index=False)
    deltas = result["delta_vs_base"].to_numpy(dtype=float)
    summary = {
        "target": args.target,
        "slice_column": args.slice_column,
        "outer_train_frac": args.outer_train_frac,
        "inner_train_frac": args.inner_train_frac,
        "margins": args.margins,
        "n_repeats_requested": args.n_repeats,
        "n_repeats_valid": int(len(result)),
        "mean_delta_vs_base": float(np.mean(deltas)) if len(deltas) else None,
        "median_delta_vs_base": float(np.median(deltas)) if len(deltas) else None,
        "ci95_low": float(np.quantile(deltas, 0.025)) if len(deltas) else None,
        "ci95_high": float(np.quantile(deltas, 0.975)) if len(deltas) else None,
        "p_delta_le_0": float(np.mean(deltas <= 0.0)) if len(deltas) else None,
        "p_delta_lt_3pt": float(np.mean(deltas < 0.03)) if len(deltas) else None,
        "selected_margin_counts": result["best_margin"].value_counts().sort_index().to_dict()
        if not result.empty
        else {},
    }
    (args.out / f"nested_margin_{tag}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Nested Margin Slice Policy",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    (args.out / f"NESTED_MARGIN_{tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / f"NESTED_MARGIN_{tag}.md")


if __name__ == "__main__":
    main()
