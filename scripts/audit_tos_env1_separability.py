from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.metrics import average_precision, roc_auc


FEATURE_NAMES = [
    "duration_sec",
    "log_rms",
    "peak",
    "clipping_fraction",
    "silence_ratio",
    "zcr",
    "spectral_centroid",
    "spectral_bandwidth",
    "low_band_ratio",
    "mid_band_ratio",
    "high_band_ratio",
    "active_ratio",
]


PREDICTION_RUNS = {
    "source_only": ROOT / "runs/external_toscovid2021_test_source_only_seed7/predictions.csv",
    "candidate_e_tga": ROOT / "runs/external_toscovid2021_test_candidate_e_tga_seed7/predictions.csv",
    "candidate_f_artifact_irm": ROOT / "runs/external_toscovid2021_test_candidate_f_artifact_env_irm_ramp_seed7/predictions.csv",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def _binary_label(label: str) -> int:
    return 1 if label == "covid_positive" else 0


def _matrix(rows: list[dict[str, str]], names: list[str] = FEATURE_NAMES) -> Any:
    import numpy as np

    return np.asarray([[_safe_float(row.get(name, "")) for name in names] for row in rows], dtype=np.float32)


def _assign_envs(source_rows: list[dict[str, str]], target_rows: list[dict[str, str]], seed: int) -> tuple[dict[str, int], dict[str, int]]:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    source_x = scaler.fit_transform(_matrix(source_rows))
    model = KMeans(n_clusters=2, random_state=seed, n_init=20)
    source_env = model.fit_predict(source_x)
    target_env = model.predict(scaler.transform(_matrix(target_rows)))
    return (
        {row["recording_id"]: int(env) for row, env in zip(source_rows, source_env)},
        {row["recording_id"]: int(env) for row, env in zip(target_rows, target_env)},
    )


def _score_summary(labels: list[int], scores: list[float]) -> dict[str, float]:
    if len(set(labels)) < 2:
        return {"n": len(labels), "positive": sum(labels), "auroc": math.nan, "auprc": math.nan}
    return {
        "n": len(labels),
        "positive": sum(labels),
        "negative": len(labels) - sum(labels),
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
        "mean_pos_score": sum(score for label, score in zip(labels, scores) if label) / max(1, sum(labels)),
        "mean_neg_score": sum(score for label, score in zip(labels, scores) if not label) / max(1, len(labels) - sum(labels)),
    }


def _univariate_feature_auc(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    labels = [_binary_label(row["label"]) for row in rows]
    out = []
    for name in FEATURE_NAMES:
        scores = [_safe_float(row.get(name, "")) for row in rows]
        if len(set(labels)) < 2:
            auc = math.nan
            direction = "n/a"
        else:
            auc = roc_auc(labels, scores)
            direction = "positive_high"
            if auc < 0.5:
                auc = 1.0 - auc
                direction = "positive_low"
        out.append({"feature": name, "best_oriented_auc": auc, "direction": direction})
    out.sort(key=lambda item: float(item["best_oriented_auc"]) if not math.isnan(float(item["best_oriented_auc"])) else -1.0, reverse=True)
    return out


def _cv_feature_probe(rows: list[dict[str, str]], seed: int) -> dict[str, Any]:
    import numpy as np
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    labels = np.asarray([_binary_label(row["label"]) for row in rows], dtype=np.int64)
    if len(rows) < 50 or len(set(labels.tolist())) < 2 or min(Counter(labels.tolist()).values()) < 5:
        return {"enabled": False, "reason": f"insufficient rows/classes: n={len(rows)} pos={int(labels.sum())}"}
    x = _matrix(rows)
    folds = min(5, min(Counter(labels.tolist()).values()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.zeros(labels.shape[0], dtype=np.float32)
    for train_idx, test_idx in cv.split(x, labels):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
        clf.fit(x[train_idx], labels[train_idx])
        scores[test_idx] = clf.predict_proba(x[test_idx])[:, 1]
    return {
        "enabled": True,
        "folds": folds,
        "auroc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "mean_positive_score": float(scores[labels == 1].mean()),
        "mean_negative_score": float(scores[labels == 0].mean()),
    }


def _prediction_scores(path: Path) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    out = {}
    for row in rows:
        pc = _safe_float(row.get("prob_covid_positive", ""))
        ph = _safe_float(row.get("prob_healthy", ""))
        out[row["recording_id"]] = {
            "prob_covid_positive": pc,
            "healthy_inverse": -ph,
            "covid_vs_healthy_margin": pc - ph,
            "covid_vs_healthy_projected": pc / max(1e-12, pc + ph),
        }
    return out


def _binary_prediction_scores(path: Path) -> dict[str, dict[str, float]]:
    rows = _read_csv(path)
    return {row["recording_id"]: {"binary_score": _safe_float(row["score_covid_positive"])} for row in rows}


def _model_slice_auc(rows: list[dict[str, str]], score_maps: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    labels = [_binary_label(row["label"]) for row in rows]
    out = {}
    for method, score_map in score_maps.items():
        method_out = {}
        for score_name in sorted(next(iter(score_map.values())).keys()):
            paired = [(label, score_map.get(row["recording_id"], {}).get(score_name)) for label, row in zip(labels, rows)]
            paired = [(label, score) for label, score in paired if score is not None]
            if not paired:
                continue
            y = [int(label) for label, _ in paired]
            s = [float(score) for _, score in paired]
            method_out[score_name] = _score_summary(y, s)
        out[method] = method_out
    return out


def _slice_report(rows: list[dict[str, str]], seed: int, score_maps: dict[str, dict[str, dict[str, float]]]) -> dict[str, Any]:
    labels = [_binary_label(row["label"]) for row in rows]
    return {
        "n": len(rows),
        "label_counts": dict(Counter(row["label"] for row in rows)),
        "positive_rate": sum(labels) / max(1, len(labels)),
        "top_univariate_features": _univariate_feature_auc(rows)[:8],
        "feature_cv_probe": _cv_feature_probe(rows, seed),
        "model_scores": _model_slice_auc(rows, score_maps),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Tos Env1 Separability Audit",
        "",
        "## Env Distributions",
        "",
        "```json",
        json.dumps(summary["env_distributions"], indent=2),
        "```",
        "",
        "## Tos Slices",
        "",
    ]
    for env, item in summary["tos_by_env"].items():
        lines.extend(
            [
                f"### Tos Env {env}",
                "",
                f"- n: `{item['n']}`",
                f"- label counts: `{item['label_counts']}`",
                f"- artifact feature CV probe: `{item['feature_cv_probe']}`",
                "",
                "Top artifact features:",
                "",
                "| Feature | Best AUC | Direction |",
                "|---|---:|---|",
            ]
        )
        for feature in item["top_univariate_features"]:
            lines.append(f"| {feature['feature']} | {feature['best_oriented_auc']:.6f} | {feature['direction']} |")
        lines.extend(["", "Model score AUROCs:", "", "```json", json.dumps(item["model_scores"], indent=2), "```", ""])
    lines.extend(["## Source Env Slices", ""])
    for env, item in summary["source_by_env"].items():
        lines.extend(
            [
                f"### Source Env {env}",
                "",
                f"- n: `{item['n']}`",
                f"- label counts: `{item['label_counts']}`",
                f"- artifact feature CV probe: `{item['feature_cv_probe']}`",
                "",
            ]
        )
    lines.extend(["## Decision Hint", "", summary["decision_hint"], ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-features", type=Path, default=ROOT / "runs/artifact_environment_audit_seed7/source_artifact_features.csv")
    parser.add_argument("--tos-features", type=Path, default=ROOT / "runs/artifact_environment_audit_seed7/tos_artifact_features.csv")
    parser.add_argument("--binary-predictions", type=Path, default=ROOT / "runs/binary_covid_student_seed7/tos_binary_predictions.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/tos_env1_separability_audit_seed7")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    source_rows = _read_csv(args.source_features)
    tos_rows = _read_csv(args.tos_features)
    source_envs, tos_envs = _assign_envs(source_rows, tos_rows, args.seed)
    for row in source_rows:
        row["artifact_env"] = str(source_envs[row["recording_id"]])
    for row in tos_rows:
        row["artifact_env"] = str(tos_envs[row["recording_id"]])
    score_maps = {name: _prediction_scores(path) for name, path in PREDICTION_RUNS.items()}
    if args.binary_predictions.is_file():
        score_maps["binary_covid"] = _binary_prediction_scores(args.binary_predictions)
    source_by_env = {
        env: _slice_report([row for row in source_rows if row["artifact_env"] == env], args.seed, {})
        for env in sorted({row["artifact_env"] for row in source_rows})
    }
    tos_by_env = {
        env: _slice_report([row for row in tos_rows if row["artifact_env"] == env], args.seed, score_maps)
        for env in sorted({row["artifact_env"] for row in tos_rows})
    }
    env1 = tos_by_env.get("1", {})
    env1_probe = env1.get("feature_cv_probe", {})
    if env1_probe.get("enabled") and float(env1_probe.get("auroc", 0.0)) >= 0.60:
        decision_hint = "Tos env1 labels are separable in target artifact space; target-unlabeled structure may be useful but risks label-confound claims."
    else:
        decision_hint = "Tos env1 labels are weakly separable in artifact space; artifact-only interventions are unlikely to solve Tos."
    summary = {
        "env_distributions": {
            "source": dict(Counter(str(value) for value in source_envs.values())),
            "tos": dict(Counter(str(value) for value in tos_envs.values())),
        },
        "source_by_env": source_by_env,
        "tos_by_env": tos_by_env,
        "decision_hint": decision_hint,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tos_env1_separability_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "TOS_ENV1_SEPARABILITY_AUDIT.md", summary)
    print(str(args.out / "TOS_ENV1_SEPARABILITY_AUDIT.md"), flush=True)


if __name__ == "__main__":
    main()
