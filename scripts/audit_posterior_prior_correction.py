from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from audit_prediction_ensemble_frontier import LABELS, PROB_COLS, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _source_prior(manifest_path: Path) -> np.ndarray:
    manifest = pd.read_csv(manifest_path)
    if "split" in manifest.columns and (manifest["split"] == "train").any():
        manifest = manifest[manifest["split"] == "train"]
    counts = manifest["label"].value_counts()
    prior = np.asarray([counts.get(label, 0) for label in LABELS], dtype=float)
    return prior / prior.sum()


def _normalize(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(scores, 1e-12, None)
    return scores / scores.sum(axis=1, keepdims=True)


def _temperature(scores: np.ndarray, temp: float) -> np.ndarray:
    return _normalize(np.power(np.clip(scores, 1e-12, 1.0), 1.0 / temp))


def _target_balance(scores: np.ndarray, power: float) -> np.ndarray:
    mean = np.clip(scores.mean(axis=0), 1e-6, None)
    weights = np.power(1.0 / mean, power)
    return _normalize(scores * weights)


def _source_prior_pull(scores: np.ndarray, source_prior: np.ndarray, power: float) -> np.ndarray:
    mean = np.clip(scores.mean(axis=0), 1e-6, None)
    weights = np.power(source_prior / mean, power)
    return _normalize(scores * weights)


def _covid_binary_projection(scores: np.ndarray, pos_mass: float) -> np.ndarray:
    covid_idx = LABELS.index("covid_positive")
    healthy_idx = LABELS.index("healthy")
    respiratory_idx = LABELS.index("respiratory_illness")
    out = scores.copy()
    non_covid = out[:, healthy_idx] + out[:, respiratory_idx]
    out[:, covid_idx] = pos_mass * out[:, covid_idx]
    out[:, healthy_idx] = (1.0 - pos_mass) * non_covid * 0.75
    out[:, respiratory_idx] = (1.0 - pos_mass) * non_covid * 0.25
    return _normalize(out)


def _transforms(source_prior: np.ndarray) -> dict[str, Callable[[np.ndarray], np.ndarray]]:
    transforms: dict[str, Callable[[np.ndarray], np.ndarray]] = {"identity": lambda x: x}
    for temp in [0.5, 0.75, 1.25, 1.5, 2.0, 3.0]:
        transforms[f"temperature_{temp:g}"] = lambda x, temp=temp: _temperature(x, temp)
    for power in [0.25, 0.5, 0.75, 1.0]:
        transforms[f"target_balance_pow{power:g}"] = lambda x, power=power: _target_balance(x, power)
        transforms[f"source_prior_pull_pow{power:g}"] = (
            lambda x, power=power: _source_prior_pull(x, source_prior, power)
        )
    for pos_mass in [0.2, 0.3, 0.4, 0.5]:
        transforms[f"covid_binary_projection_{pos_mass:g}"] = (
            lambda x, pos_mass=pos_mass: _covid_binary_projection(x, pos_mass)
        )
    return transforms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "manifests/coswara_cough.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/posterior_prior_correction")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["source_only", "ce", "kd", "candidate_f_artifact_env_irm_ramp"],
    )
    args = parser.parse_args()
    source_prior = _source_prior(args.source_manifest)
    transforms = _transforms(source_prior)
    rows: list[dict[str, object]] = []
    for spec in _discover():
        if spec.method not in args.methods:
            continue
        df = pd.read_csv(spec.path).sort_values("recording_id").reset_index(drop=True)
        labels = df["true_label"]
        scores = df[PROB_COLS].to_numpy(dtype=float)
        base_auc = _macro_ovr_auc(labels, scores)
        for name, fn in transforms.items():
            corrected = fn(scores)
            auc = _macro_ovr_auc(labels, corrected)
            rows.append(
                {
                    "target": spec.target,
                    "method": spec.method,
                    "seed": spec.seed,
                    "transform": name,
                    "n_examples": len(df),
                    "macro_ovr_auroc": auc,
                    "delta_vs_identity": auc - base_auc,
                    "identity_macro_ovr_auroc": base_auc,
                }
            )
    result = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "posterior_prior_correction_rows.csv", index=False)
    best = (
        result.sort_values(["target", "method", "seed", "macro_ovr_auroc"], ascending=[True, True, True, False])
        .groupby(["target", "method", "seed"], as_index=False)
        .head(1)
        .sort_values(["target", "delta_vs_identity"], ascending=[True, False])
    )
    best.to_csv(args.out / "posterior_prior_correction_best.csv", index=False)
    large = best[best["target"].isin(["COUGHVID", "TosCOVID"])]
    summary = {
        "source_prior": {label: float(source_prior[idx]) for idx, label in enumerate(LABELS)},
        "max_delta_any": float(best["delta_vs_identity"].max()) if not best.empty else None,
        "max_delta_large_targets": float(large["delta_vs_identity"].max()) if not large.empty else None,
        "clears_3pt_large_target": bool((large["delta_vs_identity"] >= 0.03).any()) if not large.empty else False,
        "best_rows": best.head(20).to_dict(orient="records"),
    }
    (args.out / "posterior_prior_correction_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Posterior Prior-Correction Audit",
        "",
        "This label-revealed audit tests whether target-unlabeled posterior transforms can rescue external AUROC.",
        "",
        "## Best Transform Per Target/Method/Seed",
        "",
        _to_md(best.head(40)),
        "",
        f"Clears 3-point gate on COUGHVID/TosCOVID: `{summary['clears_3pt_large_target']}`",
        "",
    ]
    (args.out / "POSTERIOR_PRIOR_CORRECTION.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / "POSTERIOR_PRIOR_CORRECTION.md")


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
