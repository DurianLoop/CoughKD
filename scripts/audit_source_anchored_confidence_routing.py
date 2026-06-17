from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata_slice_oracle import TARGETS, _method_predictions
from audit_prediction_ensemble_frontier import PROB_COLS, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _entropy(probs: np.ndarray) -> np.ndarray:
    return -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1) / np.log(probs.shape[1])


def _confidence(probs: np.ndarray) -> np.ndarray:
    return np.max(probs, axis=1)


def _route_pair(
    source_scores: np.ndarray,
    candidate_scores: np.ndarray,
    rule: str,
    margin: float,
    min_candidate_conf: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_entropy = _entropy(source_scores)
    candidate_entropy = _entropy(candidate_scores)
    source_conf = _confidence(source_scores)
    candidate_conf = _confidence(candidate_scores)
    if rule == "candidate_lower_entropy":
        use_candidate = candidate_entropy + margin < source_entropy
    elif rule == "candidate_higher_confidence":
        use_candidate = candidate_conf > source_conf + margin
    elif rule == "candidate_entropy_or_confidence":
        use_candidate = (candidate_entropy + margin < source_entropy) | (candidate_conf > source_conf + margin)
    elif rule == "candidate_entropy_and_confidence":
        use_candidate = (candidate_entropy + margin < source_entropy) & (candidate_conf > source_conf + margin)
    else:
        raise ValueError(f"Unknown rule: {rule}")
    use_candidate = use_candidate & (candidate_conf >= min_candidate_conf)
    routed = source_scores.copy()
    routed[use_candidate] = candidate_scores[use_candidate]
    return routed, use_candidate


def _all_targets() -> list[str]:
    return sorted(set(TARGETS).intersection({spec.target for spec in _discover()}))


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
    parser.add_argument("--out", type=Path, default=ROOT / "runs/source_anchored_confidence_routing")
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
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
    parser.add_argument(
        "--rules",
        nargs="+",
        default=[
            "candidate_lower_entropy",
            "candidate_higher_confidence",
            "candidate_entropy_or_confidence",
            "candidate_entropy_and_confidence",
        ],
    )
    parser.add_argument("--margins", nargs="+", type=float, default=[0.0, 0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--min-candidate-confs", nargs="+", type=float, default=[0.0, 0.3, 0.4, 0.5])
    args = parser.parse_args()

    targets = args.targets or _all_targets()
    rows = []
    for target in targets:
        available_methods = [args.base_method, *args.methods]
        try:
            base, method_scores = _method_predictions(target, available_methods)
        except ValueError:
            continue
        labels = base["true_label"].reset_index(drop=True)
        source_scores = method_scores[args.base_method]
        source_auc = _macro_ovr_auc(labels, source_scores)
        for method in args.methods:
            if method not in method_scores:
                continue
            candidate_scores = method_scores[method]
            candidate_auc = _macro_ovr_auc(labels, candidate_scores)
            for rule in args.rules:
                for margin in args.margins:
                    for min_conf in args.min_candidate_confs:
                        routed_scores, use_candidate = _route_pair(
                            source_scores,
                            candidate_scores,
                            rule,
                            margin,
                            min_conf,
                        )
                        routed_auc = _macro_ovr_auc(labels, routed_scores)
                        rows.append(
                            {
                                "target": target,
                                "method": method,
                                "rule": rule,
                                "margin": margin,
                                "min_candidate_conf": min_conf,
                                "n_examples": int(len(labels)),
                                "candidate_coverage": float(np.mean(use_candidate)),
                                "source_auc": source_auc,
                                "candidate_auc": candidate_auc,
                                "routed_auc": routed_auc,
                                "delta_vs_source": routed_auc - source_auc,
                                "delta_vs_candidate": routed_auc - candidate_auc,
                            }
                        )

    result = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "source_anchored_confidence_routing.csv", index=False)
    if result.empty:
        best = pd.DataFrame()
    else:
        best = (
            result.sort_values(["target", "delta_vs_source"], ascending=[True, False])
            .groupby("target", as_index=False)
            .head(10)
        )
    large = result[result["target"].isin(["COUGHVID", "TosCOVID"])] if not result.empty else result
    best_large = (
        large.sort_values(["target", "delta_vs_source"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(5)
        if not large.empty
        else pd.DataFrame()
    )
    summary = {
        "targets": sorted(result["target"].unique().tolist()) if not result.empty else [],
        "n_rows": int(len(result)),
        "best_by_target": best.to_dict(orient="records"),
        "best_large_targets": best_large.to_dict(orient="records"),
        "clears_3pt_on_any_large_target": bool((best_large["delta_vs_source"] >= 0.03).any())
        if not best_large.empty
        else False,
        "clears_3pt_on_all_large_targets": bool(
            best_large.groupby("target")["delta_vs_source"].max().ge(0.03).all()
        )
        if not best_large.empty and set(best_large["target"]) >= {"COUGHVID", "TosCOVID"}
        else False,
    }
    (args.out / "source_anchored_confidence_routing_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Source-Anchored Confidence Routing",
        "",
        "This prediction-level audit keeps source-only as the default and switches individual target examples to a candidate method only when the candidate is more confident or lower-entropy by a fixed margin.",
        "",
        "## Best Large-Target Rows",
        "",
        _to_md(best_large),
        "",
        "## Best Rows By Target",
        "",
        _to_md(best),
        "",
        f"Clears 3-point gate on any large target: `{summary['clears_3pt_on_any_large_target']}`",
        f"Clears 3-point gate on all large targets: `{summary['clears_3pt_on_all_large_targets']}`",
        "",
    ]
    report = args.out / "SOURCE_ANCHORED_CONFIDENCE_ROUTING.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
