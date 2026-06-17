from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from audit_metadata_slice_oracle import TARGETS, _method_predictions
from audit_prediction_ensemble_frontier import LABELS, PROB_COLS, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _available_targets() -> list[str]:
    return sorted(set(TARGETS).intersection({spec.target for spec in _discover()}))


def _features(method_scores: dict[str, np.ndarray], methods: list[str]) -> np.ndarray:
    parts = []
    for method in methods:
        probs = np.clip(method_scores[method], 1e-6, 1.0)
        parts.append(np.log(probs))
    return np.concatenate(parts, axis=1)


def _predict_full_probs(model: LogisticRegression, x: np.ndarray) -> np.ndarray:
    pred = model.predict_proba(x)
    out = np.zeros((len(x), len(LABELS)), dtype=float)
    for local_i, label in enumerate(model.classes_):
        out[:, LABELS.index(label)] = pred[:, local_i]
    row_sum = out.sum(axis=1, keepdims=True)
    return np.divide(out, np.clip(row_sum, 1e-12, None))


def _fit_stack(x_train: np.ndarray, y_train: pd.Series, c_value: float) -> LogisticRegression | None:
    if y_train.nunique() < 2:
        return None
    model = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=2000,
        solver="lbfgs",
    )
    model.fit(x_train, y_train.to_numpy())
    return model


def _summary(values: np.ndarray) -> dict[str, float | int | None]:
    if len(values) == 0:
        return {
            "n": 0,
            "mean": None,
            "median": None,
            "ci95_low": None,
            "ci95_high": None,
            "p_delta_le_0": None,
            "p_delta_lt_3pt": None,
        }
    return {
        "n": int(len(values)),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "ci95_low": float(np.quantile(values, 0.025)),
        "ci95_high": float(np.quantile(values, 0.975)),
        "p_delta_le_0": float(np.mean(values <= 0.0)),
        "p_delta_lt_3pt": float(np.mean(values < 0.03)),
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=None)
    parser.add_argument("--base-method", default="source_only")
    parser.add_argument("--n-repeats", type=int, default=300)
    parser.add_argument("--train-fracs", nargs="+", type=float, default=[0.1, 0.3, 0.5])
    parser.add_argument("--c-values", nargs="+", type=float, default=[0.03, 0.1, 0.3, 1.0])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/target_calibrated_stacking")
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
    rows = []
    for target in args.targets or _available_targets():
        try:
            base, method_scores_all = _method_predictions(target, args.methods)
        except ValueError:
            continue
        methods = [method for method in args.methods if method in method_scores_all]
        if args.base_method not in methods or len(methods) < 2:
            continue
        labels = base["true_label"].reset_index(drop=True)
        x_all = _features(method_scores_all, methods)
        n = len(labels)
        for train_frac in args.train_fracs:
            train_n = int(round(n * train_frac))
            if train_n < 10 or train_n >= n:
                continue
            for repeat in range(args.n_repeats):
                perm = rng.permutation(n)
                train_idx = np.sort(perm[:train_n])
                eval_idx = np.sort(perm[train_n:])
                y_train = labels.iloc[train_idx].reset_index(drop=True)
                y_eval = labels.iloc[eval_idx].reset_index(drop=True)
                if y_train.nunique() < 2 or y_eval.nunique() < 2:
                    continue
                source_scores_eval = method_scores_all[args.base_method][eval_idx]
                try:
                    source_auc = _macro_ovr_auc(y_eval, source_scores_eval)
                except ValueError:
                    continue

                # Calibration-picked best single method is a strong non-stacking control.
                best_method = None
                best_calib_auc = -np.inf
                for method in methods:
                    try:
                        calib_auc = _macro_ovr_auc(y_train, method_scores_all[method][train_idx])
                    except ValueError:
                        continue
                    if calib_auc > best_calib_auc:
                        best_calib_auc = calib_auc
                        best_method = method
                if best_method is not None:
                    eval_auc = _macro_ovr_auc(y_eval, method_scores_all[best_method][eval_idx])
                    rows.append(
                        {
                            "target": target,
                            "train_frac": train_frac,
                            "repeat": repeat,
                            "candidate": "calib_best_single",
                            "c_value": None,
                            "chosen_method": best_method,
                            "n_train": int(len(train_idx)),
                            "n_eval": int(len(eval_idx)),
                            "source_auc": source_auc,
                            "candidate_auc": eval_auc,
                            "delta_vs_source": eval_auc - source_auc,
                        }
                    )

                for c_value in args.c_values:
                    model = _fit_stack(x_all[train_idx], y_train, c_value)
                    if model is None:
                        continue
                    stacked_scores = _predict_full_probs(model, x_all[eval_idx])
                    try:
                        stacked_auc = _macro_ovr_auc(y_eval, stacked_scores)
                    except ValueError:
                        continue
                    rows.append(
                        {
                            "target": target,
                            "train_frac": train_frac,
                            "repeat": repeat,
                            "candidate": "logistic_stacking",
                            "c_value": c_value,
                            "chosen_method": "",
                            "n_train": int(len(train_idx)),
                            "n_eval": int(len(eval_idx)),
                            "source_auc": source_auc,
                            "candidate_auc": stacked_auc,
                            "delta_vs_source": stacked_auc - source_auc,
                        }
                    )

    result = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "target_calibrated_stacking.csv", index=False)
    if result.empty:
        summary_rows = pd.DataFrame()
    else:
        summary_rows = (
            result.groupby(["target", "train_frac", "candidate", "c_value"], dropna=False)
            .agg(
                n_repeats=("repeat", "count"),
                mean_delta_vs_source=("delta_vs_source", "mean"),
                median_delta_vs_source=("delta_vs_source", "median"),
                ci95_low=("delta_vs_source", lambda x: float(np.quantile(x, 0.025))),
                ci95_high=("delta_vs_source", lambda x: float(np.quantile(x, 0.975))),
                p_delta_le_0=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) <= 0.0))),
                p_delta_lt_3pt=("delta_vs_source", lambda x: float(np.mean(np.asarray(x) < 0.03))),
            )
            .reset_index()
            .sort_values(["target", "mean_delta_vs_source"], ascending=[True, False])
        )
    summary_rows.to_csv(args.out / "target_calibrated_stacking_summary.csv", index=False)
    best = summary_rows.groupby("target", as_index=False).head(5) if not summary_rows.empty else summary_rows
    large = best[best["target"].isin(["COUGHVID", "TosCOVID"])] if not best.empty else best
    payload = {
        "targets": sorted(result["target"].unique().tolist()) if not result.empty else [],
        "n_rows": int(len(result)),
        "best_by_target": best.to_dict(orient="records"),
        "clears_3pt_on_any_large_target": bool((large["mean_delta_vs_source"] >= 0.03).any())
        if not large.empty
        else False,
        "clears_3pt_on_all_large_targets": bool(
            large.groupby("target")["mean_delta_vs_source"].max().ge(0.03).all()
        )
        if not large.empty and set(large["target"]) >= {"COUGHVID", "TosCOVID"}
        else False,
    }
    (args.out / "target_calibrated_stacking_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Target-Calibrated Stacking Audit",
        "",
        "This audit tests whether a small labeled target calibration split can learn a non-metadata global model combination from existing prediction CSVs.",
        "",
        "## Best Rows By Target",
        "",
        _to_md(best),
        "",
        f"Clears 3-point gate on any large target: `{payload['clears_3pt_on_any_large_target']}`",
        f"Clears 3-point gate on all large targets: `{payload['clears_3pt_on_all_large_targets']}`",
        "",
    ]
    report = args.out / "TARGET_CALIBRATED_STACKING.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
