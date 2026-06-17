from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_prediction_ensemble_frontier import LABELS, PROB_COLS, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _entropy(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-12, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / np.log(scores.shape[1])


def _margin(scores: np.ndarray) -> np.ndarray:
    sorted_scores = np.sort(scores, axis=1)
    return sorted_scores[:, -1] - sorted_scores[:, -2]


def _selector_scores(scores: np.ndarray) -> dict[str, np.ndarray]:
    max_prob = scores.max(axis=1)
    return {
        "max_prob": max_prob,
        "low_entropy": -_entropy(scores),
        "margin": _margin(scores),
    }


def _bootstrap_delta(labels: pd.Series, scores: np.ndarray, baseline_scores: np.ndarray, idx: np.ndarray, n_boot: int, seed: int) -> dict[str, float | int | None]:
    rng = np.random.default_rng(seed)
    if len(idx) < 20:
        return {"n_boot_valid": 0, "ci95_low": None, "ci95_high": None, "p_delta_le_0": None}
    sub_labels = labels.iloc[idx].reset_index(drop=True)
    sub_scores = scores[idx]
    sub_base = baseline_scores[idx]
    deltas = []
    n = len(idx)
    for _ in range(n_boot):
        boot_idx = rng.integers(0, n, size=n)
        boot_labels = sub_labels.iloc[boot_idx].reset_index(drop=True)
        try:
            auc = _macro_ovr_auc(boot_labels, sub_scores[boot_idx])
            base_auc = _macro_ovr_auc(boot_labels, sub_base[boot_idx])
        except ValueError:
            continue
        if np.isfinite(auc) and np.isfinite(base_auc):
            deltas.append(auc - base_auc)
    arr = np.asarray(deltas, dtype=float)
    return {
        "n_boot_valid": int(len(arr)),
        "ci95_low": float(np.quantile(arr, 0.025)) if len(arr) else None,
        "ci95_high": float(np.quantile(arr, 0.975)) if len(arr) else None,
        "p_delta_le_0": float(np.mean(arr <= 0.0)) if len(arr) else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/selective_coverage_frontier")
    parser.add_argument("--coverages", nargs="+", type=float, default=[0.5, 0.7, 0.8, 0.9])
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
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

    rows: list[dict[str, object]] = []
    bootstrap_rows: list[dict[str, object]] = []
    specs = _discover()
    for target in sorted({spec.target for spec in specs}):
        target_specs = [spec for spec in specs if spec.target == target]
        source_specs = [spec for spec in target_specs if spec.method == "source_only"]
        if not source_specs:
            continue
        # Use the best available source-only seed ensemble as the full-coverage baseline.
        source_frames = []
        for spec in source_specs:
            df = pd.read_csv(spec.path).sort_values("recording_id").reset_index(drop=True)
            source_frames.append(df[PROB_COLS].to_numpy(dtype=float))
        source_df = pd.read_csv(source_specs[0].path).sort_values("recording_id").reset_index(drop=True)
        labels = source_df["true_label"]
        source_scores = np.mean(source_frames, axis=0)
        source_full_auc = _macro_ovr_auc(labels, source_scores)

        for method in args.methods:
            method_specs = [spec for spec in target_specs if spec.method == method]
            if not method_specs:
                continue
            frames = []
            for spec in method_specs:
                df = pd.read_csv(spec.path).sort_values("recording_id").reset_index(drop=True)
                if not source_df["recording_id"].equals(df["recording_id"]):
                    raise ValueError(f"recording alignment mismatch for {spec.path}")
                frames.append(df[PROB_COLS].to_numpy(dtype=float))
            scores = np.mean(frames, axis=0)
            full_auc = _macro_ovr_auc(labels, scores)
            for selector_name, selector_values in _selector_scores(scores).items():
                order = np.argsort(-selector_values)
                for coverage in args.coverages:
                    keep_n = max(10, int(round(len(order) * coverage)))
                    idx = np.sort(order[:keep_n])
                    if labels.iloc[idx].nunique() < 2:
                        continue
                    try:
                        auc = _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), scores[idx])
                        source_subset_auc = _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), source_scores[idx])
                    except ValueError:
                        continue
                    row = {
                        "target": target,
                        "method": method,
                        "selector": selector_name,
                        "coverage": coverage,
                        "n_examples": int(len(idx)),
                        "n_runs": len(method_specs),
                        "selective_macro_ovr_auroc": auc,
                        "method_full_macro_ovr_auroc": full_auc,
                        "source_full_macro_ovr_auroc": source_full_auc,
                        "source_same_subset_macro_ovr_auroc": source_subset_auc,
                        "delta_vs_method_full": auc - full_auc,
                        "delta_vs_source_full": auc - source_full_auc,
                        "delta_vs_source_same_subset": auc - source_subset_auc,
                    }
                    rows.append(row)
    result = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "selective_coverage_rows.csv", index=False)
    large = result[result["target"].isin(["COUGHVID", "TosCOVID"])]
    best = (
        result.sort_values(["target", "delta_vs_source_full"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(8)
    )
    best.to_csv(args.out / "selective_coverage_best.csv", index=False)
    # Bootstrap only the best large-target rows vs source-only on the same retained subset.
    for _, row in large.sort_values("delta_vs_source_same_subset", ascending=False).head(8).iterrows():
        target = str(row["target"])
        method = str(row["method"])
        selector = str(row["selector"])
        coverage = float(row["coverage"])
        target_specs = [spec for spec in specs if spec.target == target]
        source_specs = [spec for spec in target_specs if spec.method == "source_only"]
        method_specs = [spec for spec in target_specs if spec.method == method]
        source_df = pd.read_csv(source_specs[0].path).sort_values("recording_id").reset_index(drop=True)
        labels = source_df["true_label"]
        source_scores = np.mean(
            [
                pd.read_csv(spec.path).sort_values("recording_id").reset_index(drop=True)[PROB_COLS].to_numpy(dtype=float)
                for spec in source_specs
            ],
            axis=0,
        )
        scores = np.mean(
            [
                pd.read_csv(spec.path).sort_values("recording_id").reset_index(drop=True)[PROB_COLS].to_numpy(dtype=float)
                for spec in method_specs
            ],
            axis=0,
        )
        selector_values = _selector_scores(scores)[selector]
        order = np.argsort(-selector_values)
        keep_n = max(10, int(round(len(order) * coverage)))
        idx = np.sort(order[:keep_n])
        boot = _bootstrap_delta(labels, scores, source_scores, idx, args.n_boot, args.seed)
        boot.update(row.to_dict())
        bootstrap_rows.append(boot)
    boot_df = pd.DataFrame(bootstrap_rows)
    boot_df.to_csv(args.out / "selective_coverage_bootstrap_large.csv", index=False)
    summary = {
        "clears_3pt_large_vs_source_full": bool((large["delta_vs_source_full"] >= 0.03).any()) if not large.empty else False,
        "clears_3pt_large_vs_source_same_subset": bool((large["delta_vs_source_same_subset"] >= 0.03).any()) if not large.empty else False,
        "best_large_vs_source_full": large.sort_values("delta_vs_source_full", ascending=False).head(5).to_dict(orient="records"),
        "best_large_vs_source_same_subset": large.sort_values("delta_vs_source_same_subset", ascending=False).head(5).to_dict(orient="records"),
    }
    (args.out / "selective_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Selective Coverage Frontier",
        "",
        "This audit tests no-label confidence/entropy/margin selection at fixed coverage.",
        "",
        "## Best Rows By Target",
        "",
        _to_md(best),
        "",
        "## Large-Target Bootstrap Vs Source Same Subset",
        "",
        _to_md(boot_df),
        "",
        f"Clears 3-point gate vs source full on COUGHVID/TosCOVID: `{summary['clears_3pt_large_vs_source_full']}`",
        f"Clears 3-point gate vs source same subset on COUGHVID/TosCOVID: `{summary['clears_3pt_large_vs_source_same_subset']}`",
        "",
    ]
    (args.out / "SELECTIVE_COVERAGE_FRONTIER.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / "SELECTIVE_COVERAGE_FRONTIER.md")


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
