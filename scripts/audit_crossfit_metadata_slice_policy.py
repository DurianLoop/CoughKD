from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata_slice_oracle import TARGETS, _method_predictions, _prepare_manifest
from audit_prediction_ensemble_frontier import _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _fit_policy(
    labels: pd.Series,
    groups: pd.Series,
    method_scores: dict[str, np.ndarray],
    train_idx: np.ndarray,
    min_train_slice: int,
    base_method: str,
    switch_margin: float,
) -> dict[str, str]:
    policy = {}
    train_groups = groups.iloc[train_idx].reset_index(drop=True)
    train_labels = labels.iloc[train_idx].reset_index(drop=True)
    local_indices = np.arange(len(train_idx))
    for group_value, local_group_idx_obj in train_groups.groupby(train_groups).groups.items():
        local_group_idx = np.asarray(list(local_group_idx_obj), dtype=int)
        if len(local_group_idx) < min_train_slice or train_labels.iloc[local_group_idx].nunique() < 2:
            continue
        global_group_idx = train_idx[local_group_idx]
        try:
            base_auc = _macro_ovr_auc(
                train_labels.iloc[local_group_idx].reset_index(drop=True),
                method_scores[base_method][global_group_idx],
            )
        except ValueError:
            continue
        best_method = None
        best_auc = -np.inf
        for method, scores in method_scores.items():
            try:
                auc = _macro_ovr_auc(train_labels.iloc[local_group_idx].reset_index(drop=True), scores[global_group_idx])
            except ValueError:
                continue
            if auc > best_auc:
                best_auc = auc
                best_method = method
        if best_method is not None and best_auc - base_auc >= switch_margin:
            policy[str(group_value)] = best_method
    return policy


def _apply_policy(
    groups: pd.Series,
    method_scores: dict[str, np.ndarray],
    base_method: str,
    policy: dict[str, str],
    eval_idx: np.ndarray,
) -> np.ndarray:
    mixed = method_scores[base_method][eval_idx].copy()
    eval_groups = groups.iloc[eval_idx].astype(str).to_numpy()
    for local_i, group_value in enumerate(eval_groups):
        method = policy.get(str(group_value), base_method)
        mixed[local_i] = method_scores[method][eval_idx[local_i]]
    return mixed


def _sample_train_indices_by_slice(groups: pd.Series, train_per_slice: int, rng: np.random.Generator) -> np.ndarray:
    train_parts = []
    for _, idx_obj in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx_obj), dtype=int)
        if len(idx) <= train_per_slice:
            train_parts.append(idx)
            continue
        train_parts.append(rng.choice(idx, size=train_per_slice, replace=False))
    if not train_parts:
        return np.asarray([], dtype=int)
    return np.sort(np.concatenate(train_parts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="COUGHVID")
    parser.add_argument("--slice-column", default="symptom_resp")
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=200)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--train-per-slice", type=int, default=None)
    parser.add_argument("--min-train-slice", type=int, default=80)
    parser.add_argument("--switch-margin", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/crossfit_metadata_slice_policy")
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
    train_tag = (
        f"perSlice{args.train_per_slice}"
        if args.train_per_slice is not None
        else f"train{args.train_frac:g}".replace(".", "p")
    )
    margin_tag = f"margin{args.switch_margin:g}".replace(".", "p").replace("-", "neg")
    run_tag = f"{args.target}_{args.slice_column}_{train_tag}_min{args.min_train_slice}_{margin_tag}"

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
    policies = []
    for repeat in range(args.n_repeats):
        if args.train_per_slice is not None:
            train_idx = _sample_train_indices_by_slice(groups, args.train_per_slice, rng)
            train_mask = np.zeros(n, dtype=bool)
            train_mask[train_idx] = True
            eval_idx = np.flatnonzero(~train_mask)
        else:
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
        mixed_scores = _apply_policy(groups, method_scores, args.base_method, policy, eval_idx)
        base_scores = method_scores[args.base_method][eval_idx]
        try:
            mixed_auc = _macro_ovr_auc(labels.iloc[eval_idx].reset_index(drop=True), mixed_scores)
            base_auc = _macro_ovr_auc(labels.iloc[eval_idx].reset_index(drop=True), base_scores)
        except ValueError:
            continue
        rows.append(
            {
                "repeat": repeat,
                "target": args.target,
                "slice_column": args.slice_column,
                "train_n": train_n,
                "actual_train_n": int(len(train_idx)),
                "eval_n": len(eval_idx),
                "policy_size": len(policy),
                "mixed_auc": mixed_auc,
                "base_auc": base_auc,
                "delta_vs_base": mixed_auc - base_auc,
            }
        )
        policies.append({"repeat": repeat, "policy": policy})
    result = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / f"crossfit_{run_tag}.csv", index=False)
    (args.out / f"crossfit_{run_tag}_policies.json").write_text(
        json.dumps(policies, indent=2), encoding="utf-8"
    )
    deltas = result["delta_vs_base"].to_numpy(dtype=float)
    summary = {
        "target": args.target,
        "slice_column": args.slice_column,
        "base_method": args.base_method,
        "switch_margin": args.switch_margin,
        "n_repeats_requested": args.n_repeats,
        "n_repeats_valid": int(len(result)),
        "train_frac": args.train_frac,
        "train_per_slice": args.train_per_slice,
        "mean_delta_vs_base": float(np.mean(deltas)) if len(deltas) else None,
        "median_delta_vs_base": float(np.median(deltas)) if len(deltas) else None,
        "ci95_low": float(np.quantile(deltas, 0.025)) if len(deltas) else None,
        "ci95_high": float(np.quantile(deltas, 0.975)) if len(deltas) else None,
        "p_delta_le_0": float(np.mean(deltas <= 0.0)) if len(deltas) else None,
        "p_delta_lt_3pt": float(np.mean(deltas < 0.03)) if len(deltas) else None,
    }
    (args.out / f"crossfit_{run_tag}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Crossfit Metadata Slice Policy",
        "",
        f"- Target: `{args.target}`",
        f"- Slice column: `{args.slice_column}`",
        f"- Base method: `{args.base_method}`",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(summary, indent=2),
        "```",
        "",
    ]
    (args.out / f"CROSSFIT_{run_tag}.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / f"CROSSFIT_{run_tag}.md")


if __name__ == "__main__":
    main()
