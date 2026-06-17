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


def _evaluate_policy(labels, groups, method_scores, base_method, policy, eval_idx):
    mixed_scores = _apply_policy(groups, method_scores, base_method, policy, eval_idx)
    base_scores = method_scores[base_method][eval_idx]
    eval_labels = labels.iloc[eval_idx].reset_index(drop=True)
    mixed_auc = _macro_ovr_auc(eval_labels, mixed_scores)
    base_auc = _macro_ovr_auc(eval_labels, base_scores)
    return mixed_auc, base_auc, mixed_auc - base_auc


def _crossfit_with_groups(
    labels: pd.Series,
    groups: pd.Series,
    method_scores: dict[str, np.ndarray],
    base_method: str,
    train_frac: float,
    min_train_slice: int,
    switch_margin: float,
    n_repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(labels)
    train_n = int(round(n * train_frac))
    rows = []
    for repeat in range(n_repeats):
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
            min_train_slice,
            base_method,
            switch_margin,
        )
        try:
            mixed_auc, base_auc, delta = _evaluate_policy(
                labels, groups, method_scores, base_method, policy, eval_idx
            )
        except ValueError:
            continue
        rows.append(
            {
                "repeat": repeat,
                "policy_size": len(policy),
                "mixed_auc": mixed_auc,
                "base_auc": base_auc,
                "delta_vs_base": delta,
            }
        )
    return pd.DataFrame(rows)


def _summary(df: pd.DataFrame) -> dict[str, float | int | None]:
    deltas = df["delta_vs_base"].to_numpy(dtype=float) if not df.empty else np.asarray([])
    return {
        "n_repeats_valid": int(len(deltas)),
        "mean_delta_vs_base": float(np.mean(deltas)) if len(deltas) else None,
        "median_delta_vs_base": float(np.median(deltas)) if len(deltas) else None,
        "ci95_low": float(np.quantile(deltas, 0.025)) if len(deltas) else None,
        "ci95_high": float(np.quantile(deltas, 0.975)) if len(deltas) else None,
        "p_delta_le_0": float(np.mean(deltas <= 0.0)) if len(deltas) else None,
        "p_delta_lt_3pt": float(np.mean(deltas < 0.03)) if len(deltas) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="COUGHVID")
    parser.add_argument("--slice-column", default="symptom_resp")
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--train-frac", type=float, default=0.7)
    parser.add_argument("--min-train-slice", type=int, default=100)
    parser.add_argument("--switch-margin", type=float, default=0.02)
    parser.add_argument("--n-repeats", type=int, default=500)
    parser.add_argument("--n-controls", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/random_slice_negative_control")
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
    real_groups = merged[args.slice_column].fillna("missing").astype(str).reset_index(drop=True)
    if real_groups.nunique() != 2:
        raise ValueError(f"Expected binary slice for matched random control, got {real_groups.nunique()}")

    args.out.mkdir(parents=True, exist_ok=True)
    real_df = _crossfit_with_groups(
        labels,
        real_groups,
        method_scores,
        args.base_method,
        args.train_frac,
        args.min_train_slice,
        args.switch_margin,
        args.n_repeats,
        args.seed,
    )
    real_df.to_csv(args.out / "real_slice_crossfit.csv", index=False)

    counts = real_groups.value_counts().sort_values()
    small_n = int(counts.iloc[0])
    rng = np.random.default_rng(args.seed + 1000)
    control_summaries = []
    for control_id in range(args.n_controls):
        group_values = np.array(["control_major"] * len(real_groups), dtype=object)
        small_idx = rng.choice(np.arange(len(real_groups)), size=small_n, replace=False)
        group_values[small_idx] = "control_minor"
        control_groups = pd.Series(group_values)
        control_df = _crossfit_with_groups(
            labels,
            control_groups,
            method_scores,
            args.base_method,
            args.train_frac,
            args.min_train_slice,
            args.switch_margin,
            args.n_repeats,
            args.seed + 10_000 + control_id,
        )
        control_df.to_csv(args.out / f"control_{control_id:03d}_crossfit.csv", index=False)
        summary = _summary(control_df)
        summary["control_id"] = control_id
        control_summaries.append(summary)

    controls = pd.DataFrame(control_summaries)
    controls.to_csv(args.out / "random_control_summaries.csv", index=False)
    real_summary = _summary(real_df)
    control_means = controls["mean_delta_vs_base"].to_numpy(dtype=float)
    payload = {
        "target": args.target,
        "slice_column": args.slice_column,
        "train_frac": args.train_frac,
        "switch_margin": args.switch_margin,
        "real_summary": real_summary,
        "n_controls": args.n_controls,
        "control_mean_delta_mean": float(np.mean(control_means)) if len(control_means) else None,
        "control_mean_delta_ci95_low": float(np.quantile(control_means, 0.025)) if len(control_means) else None,
        "control_mean_delta_ci95_high": float(np.quantile(control_means, 0.975)) if len(control_means) else None,
        "p_control_mean_ge_real_mean": float(np.mean(control_means >= real_summary["mean_delta_vs_base"]))
        if len(control_means)
        else None,
    }
    (args.out / "random_slice_negative_control_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Random Slice Negative Control",
        "",
        "```json",
        json.dumps(payload, indent=2),
        "```",
        "",
    ]
    (args.out / "RANDOM_SLICE_NEGATIVE_CONTROL.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / "RANDOM_SLICE_NEGATIVE_CONTROL.md")


if __name__ == "__main__":
    main()
