from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
PROB_COLS = [
    "prob_covid_positive",
    "prob_covid_recovered",
    "prob_exposed",
    "prob_healthy",
    "prob_respiratory_illness",
]
LABELS = [
    "covid_positive",
    "covid_recovered",
    "exposed",
    "healthy",
    "respiratory_illness",
]


@dataclass(frozen=True)
class RunSpec:
    target: str
    method: str
    seed: str
    path: Path


def _parse_run(path: Path) -> RunSpec | None:
    name = path.name
    patterns = [
        ("COUGHVID", r"^external_coughvid_test_(?P<method>.+?)(?:_seed(?P<seed>\d+)|_baseline)?$"),
        ("TosCOVID", r"^external_toscovid2021_test_(?P<method>.+?)_seed(?P<seed>\d+)$"),
        ("UKCOVID", r"^external_ukcovid_open_(?P<method>.+?)_seed(?P<seed>\d+)$"),
        ("Virufy", r"^external_virufy_(?P<method>.+?)_seed(?P<seed>\d+)$"),
        ("Virufyseg", r"^external_virufyseg_(?P<method>.+?)_seed(?P<seed>\d+)$"),
    ]
    for target, pattern in patterns:
        match = re.match(pattern, name)
        if not match:
            continue
        method = match.group("method")
        seed = match.groupdict().get("seed") or "7"
        if method.startswith("stage3c_"):
            method = method.removeprefix("stage3c_")
        if method.startswith("stage3b_"):
            method = method.removeprefix("stage3b_")
        return RunSpec(target=target, method=method, seed=seed, path=path / "predictions.csv")
    return None


def _macro_ovr_auc(labels: pd.Series, scores: np.ndarray) -> float:
    y = np.column_stack([(labels.to_numpy() == label).astype(int) for label in LABELS])
    valid = [i for i in range(len(LABELS)) if len(np.unique(y[:, i])) == 2]
    if not valid:
        return float("nan")
    return float(roc_auc_score(y[:, valid], scores[:, valid], average="macro"))


def _load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"recording_id", "true_label", *PROB_COLS}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    return df[["recording_id", "true_label", *PROB_COLS]].copy()


def _aligned_average(specs: Iterable[RunSpec]) -> tuple[pd.Series, np.ndarray, int]:
    specs = list(specs)
    base: pd.DataFrame | None = None
    arrays: list[np.ndarray] = []
    for spec in specs:
        df = _load_prediction(spec.path)
        df = df.sort_values("recording_id").reset_index(drop=True)
        if base is None:
            base = df[["recording_id", "true_label"]]
        else:
            if not base["recording_id"].equals(df["recording_id"]):
                raise ValueError(f"recording_id mismatch while aligning {spec.path}")
            if not base["true_label"].equals(df["true_label"]):
                raise ValueError(f"true_label mismatch while aligning {spec.path}")
        arrays.append(df[PROB_COLS].to_numpy(dtype=float))
    if base is None:
        raise ValueError("no specs to average")
    return base["true_label"], np.mean(arrays, axis=0), len(base)


def _discover() -> list[RunSpec]:
    specs: list[RunSpec] = []
    for run_dir in (ROOT / "runs").iterdir():
        if not run_dir.is_dir():
            continue
        spec = _parse_run(run_dir)
        if spec and spec.path.is_file():
            specs.append(spec)
    return specs


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
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
    parser.add_argument("--out", type=Path, default=ROOT / "runs/prediction_ensemble_frontier")
    args = parser.parse_args()
    specs = _discover()
    rows: list[dict[str, object]] = []
    for target in sorted({spec.target for spec in specs}):
        target_specs = [spec for spec in specs if spec.target == target]
        source_specs = [spec for spec in target_specs if spec.method == "source_only"]
        baseline_auc = None
        if source_specs:
            labels, scores, n = _aligned_average(source_specs)
            baseline_auc = _macro_ovr_auc(labels, scores)
            rows.append(
                {
                    "target": target,
                    "candidate": "source_only_seed_ensemble",
                    "n_runs": len(source_specs),
                    "n_examples": n,
                    "macro_ovr_auroc": baseline_auc,
                    "delta_vs_source_only": 0.0,
                    "methods": "source_only",
                    "seeds": ",".join(sorted({spec.seed for spec in source_specs})),
                }
            )
        for method in sorted({spec.method for spec in target_specs}):
            method_specs = [spec for spec in target_specs if spec.method == method]
            labels, scores, n = _aligned_average(method_specs)
            auc = _macro_ovr_auc(labels, scores)
            rows.append(
                {
                    "target": target,
                    "candidate": f"{method}_seed_ensemble",
                    "n_runs": len(method_specs),
                    "n_examples": n,
                    "macro_ovr_auroc": auc,
                    "delta_vs_source_only": None if baseline_auc is None else auc - baseline_auc,
                    "methods": method,
                    "seeds": ",".join(sorted({spec.seed for spec in method_specs})),
                }
            )
        stable_methods = {"source_only", "ce", "kd"}
        stable_specs = [spec for spec in target_specs if spec.method in stable_methods]
        if len(stable_specs) >= 2:
            labels, scores, n = _aligned_average(stable_specs)
            auc = _macro_ovr_auc(labels, scores)
            rows.append(
                {
                    "target": target,
                    "candidate": "source_ce_kd_uniform_ensemble",
                    "n_runs": len(stable_specs),
                    "n_examples": n,
                    "macro_ovr_auroc": auc,
                    "delta_vs_source_only": None if baseline_auc is None else auc - baseline_auc,
                    "methods": ",".join(sorted(stable_methods)),
                    "seeds": ",".join(sorted({spec.seed for spec in stable_specs})),
                }
            )
        all_seeded_specs = [spec for spec in target_specs if spec.method in stable_methods or spec.method.startswith("candidate_")]
        if len(all_seeded_specs) >= 2:
            labels, scores, n = _aligned_average(all_seeded_specs)
            auc = _macro_ovr_auc(labels, scores)
            rows.append(
                {
                    "target": target,
                    "candidate": "all_available_uniform_ensemble",
                    "n_runs": len(all_seeded_specs),
                    "n_examples": n,
                    "macro_ovr_auroc": auc,
                    "delta_vs_source_only": None if baseline_auc is None else auc - baseline_auc,
                    "methods": ",".join(sorted({spec.method for spec in all_seeded_specs})),
                    "seeds": ",".join(sorted({spec.seed for spec in all_seeded_specs})),
                }
            )
    result = pd.DataFrame(rows).sort_values(["target", "macro_ovr_auroc"], ascending=[True, False])
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "prediction_ensemble_frontier.csv", index=False)
    best = (
        result.dropna(subset=["delta_vs_source_only"])
        .sort_values(["target", "delta_vs_source_only"], ascending=[True, False])
        .groupby("target", as_index=False)
        .head(5)
    )
    summary = {
        "targets": sorted(result["target"].unique().tolist()) if not result.empty else [],
        "best_by_target": best.to_dict(orient="records"),
        "clears_3pt_gate": bool((best["delta_vs_source_only"] >= 0.03).any()) if not best.empty else False,
    }
    (args.out / "prediction_ensemble_frontier_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    lines = [
        "# Prediction Ensemble Frontier",
        "",
        "This is a label-revealed audit of whether existing predictions contain a simple, no-retraining ensemble with a 3 point gain.",
        "",
        "## Best Deltas vs Source-Only Seed Ensemble",
        "",
        _markdown_table(best),
        "",
        f"Clears 3-point gate: `{summary['clears_3pt_gate']}`",
        "",
    ]
    (args.out / "PREDICTION_ENSEMBLE_FRONTIER.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / "PREDICTION_ENSEMBLE_FRONTIER.md")


if __name__ == "__main__":
    main()
