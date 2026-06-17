from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_prediction_ensemble_frontier import LABELS, PROB_COLS, _aligned_average, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _source_prior(manifest_path: Path) -> np.ndarray:
    manifest = pd.read_csv(manifest_path)
    if "split" in manifest.columns and (manifest["split"] == "train").any():
        manifest = manifest[manifest["split"] == "train"]
    counts = manifest["label"].value_counts()
    prior = np.asarray([counts.get(label, 0) for label in LABELS], dtype=float)
    if prior.sum() == 0:
        return np.ones(len(LABELS), dtype=float) / len(LABELS)
    return prior / prior.sum()


def _entropy(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-9, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / np.log(scores.shape[1])


def _features(scores: np.ndarray, source_prior: np.ndarray) -> dict[str, float]:
    mean_probs = scores.mean(axis=0)
    max_probs = scores.max(axis=1)
    ent = _entropy(scores)
    prior_l1 = float(np.abs(mean_probs - source_prior).sum())
    prior_l2 = float(np.sqrt(np.square(mean_probs - source_prior).sum()))
    return {
        "mean_entropy": float(ent.mean()),
        "std_entropy": float(ent.std()),
        "mean_max_prob": float(max_probs.mean()),
        "std_max_prob": float(max_probs.std()),
        "pred_prior_max": float(mean_probs.max()),
        "pred_prior_l1_to_source": prior_l1,
        "pred_prior_l2_to_source": prior_l2,
        "mean_prob_covid_positive": float(mean_probs[LABELS.index("covid_positive")]),
        "mean_prob_healthy": float(mean_probs[LABELS.index("healthy")]),
        "mean_prob_respiratory_illness": float(mean_probs[LABELS.index("respiratory_illness")]),
    }


def _safe_best(rows: list[dict[str, Any]], key: str, reverse: bool = False) -> dict[str, Any]:
    return sorted(rows, key=lambda row: row[key], reverse=reverse)[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "manifests/coswara_cough.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/unlabeled_prediction_geometry_selector")
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
            "candidate_d_active",
            "candidate_e_tga",
            "candidate_f_artifact_env_irm_ramp",
        ],
    )
    args = parser.parse_args()

    prior = _source_prior(args.source_manifest)
    specs = _discover()
    rows: list[dict[str, Any]] = []
    for target in sorted({spec.target for spec in specs}):
        for method in args.methods:
            method_specs = [spec for spec in specs if spec.target == target and spec.method == method]
            if not method_specs:
                continue
            labels, scores, n = _aligned_average(method_specs)
            row = {
                "target": target,
                "method": method,
                "n_runs": len(method_specs),
                "n_examples": n,
                "macro_ovr_auroc": _macro_ovr_auc(labels, scores),
                "seeds": ",".join(sorted({spec.seed for spec in method_specs})),
            }
            row.update(_features(scores, prior))
            rows.append(row)

    metrics = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.out / "prediction_geometry_rows.csv", index=False)

    selection_rows: list[dict[str, Any]] = []
    for target, target_df in metrics.groupby("target", sort=True):
        target_rows = target_df.to_dict(orient="records")
        source_rows = [row for row in target_rows if row["method"] == "source_only"]
        if not source_rows:
            continue
        source = source_rows[0]
        oracle = _safe_best(target_rows, "macro_ovr_auroc", reverse=True)
        selectors = {
            "min_entropy": _safe_best(target_rows, "mean_entropy"),
            "max_confidence": _safe_best(target_rows, "mean_max_prob", reverse=True),
            "min_source_prior_l2": _safe_best(target_rows, "pred_prior_l2_to_source"),
            "min_prior_collapse": _safe_best(target_rows, "pred_prior_max"),
        }
        stable_rows = [row for row in target_rows if row["method"] in {"source_only", "ce", "kd"}]
        if stable_rows:
            selectors["stable_min_entropy"] = _safe_best(stable_rows, "mean_entropy")
            selectors["stable_max_confidence"] = _safe_best(stable_rows, "mean_max_prob", reverse=True)
            selectors["stable_min_source_prior_l2"] = _safe_best(stable_rows, "pred_prior_l2_to_source")
        for selector, choice in selectors.items():
            selection_rows.append(
                {
                    "target": target,
                    "selector": selector,
                    "chosen_method": choice["method"],
                    "chosen_macro_ovr_auroc": choice["macro_ovr_auroc"],
                    "source_only_macro_ovr_auroc": source["macro_ovr_auroc"],
                    "oracle_method": oracle["method"],
                    "oracle_macro_ovr_auroc": oracle["macro_ovr_auroc"],
                    "delta_vs_source_only": choice["macro_ovr_auroc"] - source["macro_ovr_auroc"],
                    "oracle_gap": oracle["macro_ovr_auroc"] - choice["macro_ovr_auroc"],
                }
            )

    selections = pd.DataFrame(selection_rows)
    selections.to_csv(args.out / "unlabeled_selector_rows.csv", index=False)
    summary_rows = []
    for selector, selector_df in selections.groupby("selector", sort=True):
        summary_rows.append(
            {
                "selector": selector,
                "n_targets": int(len(selector_df)),
                "mean_delta_vs_source_only": float(selector_df["delta_vs_source_only"].mean()),
                "worst_delta_vs_source_only": float(selector_df["delta_vs_source_only"].min()),
                "mean_oracle_gap": float(selector_df["oracle_gap"].mean()),
                "clears_3pt_on_any_large_target": bool(
                    (
                        selector_df[selector_df["target"].isin(["COUGHVID", "TosCOVID"])]["delta_vs_source_only"]
                        >= 0.03
                    ).any()
                ),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values(["mean_delta_vs_source_only"], ascending=False)
    summary.to_csv(args.out / "unlabeled_selector_summary.csv", index=False)
    payload = {
        "source_manifest": str(args.source_manifest),
        "source_prior": {label: float(prior[idx]) for idx, label in enumerate(LABELS)},
        "best_selector": summary.iloc[0].to_dict() if not summary.empty else None,
        "clears_3pt_on_any_large_target": bool(summary["clears_3pt_on_any_large_target"].any()) if not summary.empty else False,
    }
    (args.out / "unlabeled_selector_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Unlabeled Prediction Geometry Selector",
        "",
        "This audit asks whether target-label-free prediction geometry can select a better method than source-only.",
        "",
        "## Selector Summary",
        "",
        _to_md(summary),
        "",
        "## Per-Target Selections",
        "",
        _to_md(selections.sort_values(["target", "selector"])),
        "",
    ]
    (args.out / "UNLABELED_SELECTOR_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / "UNLABELED_SELECTOR_REPORT.md")


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


if __name__ == "__main__":
    main()
