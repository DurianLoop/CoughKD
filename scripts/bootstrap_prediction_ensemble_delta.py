from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audit_prediction_ensemble_frontier import _aligned_average, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _select_specs(target: str, candidate: str):
    specs = [spec for spec in _discover() if spec.target == target]
    if candidate == "source_only_seed_ensemble":
        return [spec for spec in specs if spec.method == "source_only"]
    if candidate == "source_ce_kd_uniform_ensemble":
        return [spec for spec in specs if spec.method in {"source_only", "ce", "kd"}]
    if candidate == "all_available_uniform_ensemble":
        return [
            spec
            for spec in specs
            if spec.method in {"source_only", "ce", "kd"} or spec.method.startswith("candidate_")
        ]
    suffix = "_seed_ensemble"
    if candidate.endswith(suffix):
        method = candidate[: -len(suffix)]
        return [spec for spec in specs if spec.method == method]
    raise ValueError(f"unsupported candidate: {candidate}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--baseline", default="source_only_seed_ensemble")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/prediction_ensemble_frontier")
    args = parser.parse_args()

    cand_specs = _select_specs(args.target, args.candidate)
    base_specs = _select_specs(args.target, args.baseline)
    labels, cand_scores, n = _aligned_average(cand_specs)
    base_labels, base_scores, base_n = _aligned_average(base_specs)
    if n != base_n or not labels.equals(base_labels):
        raise ValueError("candidate and baseline are not aligned")

    point_candidate = _macro_ovr_auc(labels, cand_scores)
    point_baseline = _macro_ovr_auc(labels, base_scores)
    point_delta = point_candidate - point_baseline

    rng = np.random.default_rng(args.seed)
    deltas = []
    y = labels.reset_index(drop=True)
    for _ in range(args.n_boot):
        idx = rng.integers(0, n, size=n)
        sampled_labels = y.iloc[idx].reset_index(drop=True)
        try:
            cand_auc = _macro_ovr_auc(sampled_labels, cand_scores[idx])
            base_auc = _macro_ovr_auc(sampled_labels, base_scores[idx])
        except ValueError:
            continue
        if np.isfinite(cand_auc) and np.isfinite(base_auc):
            deltas.append(cand_auc - base_auc)

    deltas_arr = np.asarray(deltas, dtype=float)
    summary = {
        "target": args.target,
        "candidate": args.candidate,
        "baseline": args.baseline,
        "n_examples": n,
        "n_boot_requested": args.n_boot,
        "n_boot_valid": int(len(deltas_arr)),
        "point_candidate": point_candidate,
        "point_baseline": point_baseline,
        "point_delta": point_delta,
        "ci95_low": float(np.quantile(deltas_arr, 0.025)) if len(deltas_arr) else None,
        "ci95_high": float(np.quantile(deltas_arr, 0.975)) if len(deltas_arr) else None,
        "p_delta_le_0": float(np.mean(deltas_arr <= 0.0)) if len(deltas_arr) else None,
        "p_delta_lt_3pt": float(np.mean(deltas_arr < 0.03)) if len(deltas_arr) else None,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    out_path = args.out / f"bootstrap_{args.target.lower()}_{args.candidate}.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
