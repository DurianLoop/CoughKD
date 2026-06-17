from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_prediction_ensemble_frontier import PROB_COLS, _macro_ovr_auc
from audit_target_calibrated_stacking import _features, _fit_stack, _predict_full_probs


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "toscovid_official_split_semantic_router"

TRAIN_MANIFEST = ROOT / "manifests" / "toscovid2021_train_external.csv"
TEST_MANIFEST = ROOT / "manifests" / "toscovid2021_test_external.csv"

METHOD_SEEDS = {
    "source_only": ["7", "11", "23"],
    "ce": ["7", "11", "23"],
    "kd": ["7", "11", "23"],
    "tcd_conf035": ["7", "11", "23"],
    "tcd_very_strong": ["7", "11", "23"],
    "candidate_a": ["7", "11", "23"],
    "candidate_b": ["7", "11", "23"],
    "candidate_c": ["7", "11", "23"],
    "candidate_f_artifact_env_irm_ramp": ["7"],
}
METHODS = list(METHOD_SEEDS)
TIER_METHODS = {
    "stable_smoke": ["source_only", "ce", "kd"],
    "router_full": METHODS,
}


def _prediction_path(split: str, method: str, seed: str) -> Path:
    return RUNS / f"external_toscovid2021_{split}_{method}_seed{seed}" / "predictions.csv"


def _load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"recording_id", "true_label", *PROB_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[["recording_id", "true_label", *PROB_COLS]].sort_values("recording_id").reset_index(drop=True)


def _load_split_predictions(split: str) -> tuple[pd.DataFrame | None, dict[str, np.ndarray], list[dict[str, Any]]]:
    base: pd.DataFrame | None = None
    scores: dict[str, np.ndarray] = {}
    missing: list[dict[str, Any]] = []
    for method, seeds in METHOD_SEEDS.items():
        arrays = []
        for seed in seeds:
            path = _prediction_path(split, method, seed)
            if not path.is_file():
                missing.append(
                    {
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "path": str(path.relative_to(ROOT)),
                    }
                )
                continue
            df = _load_prediction(path)
            if base is None:
                base = df[["recording_id", "true_label"]].copy()
            elif not base.equals(df[["recording_id", "true_label"]]):
                raise ValueError(f"prediction alignment mismatch for {path}")
            arrays.append(df[PROB_COLS].to_numpy(dtype=float))
        if arrays:
            scores[method] = np.mean(arrays, axis=0)
    return base, scores, missing


def _missing_for_methods(split: str, methods: list[str]) -> list[dict[str, Any]]:
    rows = []
    for method in methods:
        for seed in METHOD_SEEDS[method]:
            path = _prediction_path(split, method, seed)
            if not path.is_file():
                rows.append(
                    {
                        "split": split,
                        "method": method,
                        "seed": seed,
                        "path": str(path.relative_to(ROOT)),
                    }
                )
    return rows


def _manifest_counts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False}
    df = pd.read_csv(path).fillna("")
    return {
        "exists": True,
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
        "covid_positive": int((df["label"] == "covid_positive").sum()) if "label" in df.columns else None,
        "healthy": int((df["label"] == "healthy").sum()) if "label" in df.columns else None,
        "audio_found": int(df["path"].astype(str).ne("").sum()) if "path" in df.columns else None,
    }


def _score_best_single(
    train_labels: pd.Series,
    test_labels: pd.Series,
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    methods: list[str],
) -> dict[str, Any]:
    best_method = None
    best_train_auc = -np.inf
    for method in methods:
        auc = _macro_ovr_auc(train_labels, train_scores[method])
        if auc > best_train_auc:
            best_method = method
            best_train_auc = auc
    if best_method is None:
        raise ValueError("no best method selected")
    test_auc = _macro_ovr_auc(test_labels, test_scores[best_method])
    return {
        "candidate": "official_train_best_single",
        "selected_method": best_method,
        "train_auc": best_train_auc,
        "test_auc": test_auc,
    }


def _score_logistic_stack(
    train_labels: pd.Series,
    test_labels: pd.Series,
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
    methods: list[str],
    c_value: float,
) -> dict[str, Any]:
    x_train = _features(train_scores, methods)
    x_test = _features(test_scores, methods)
    model = _fit_stack(x_train, train_labels, c_value)
    if model is None:
        raise ValueError("could not fit official train logistic stack")
    pred = _predict_full_probs(model, x_test)
    return {
        "candidate": "official_train_semantic_router",
        "selected_strategy": "logistic_stacking",
        "selection_reason": "TosCOVID age metadata is demographic/nonsemantic, so semantic router uses stacking rather than slice gating.",
        "c_value": c_value,
        "methods": ",".join(methods),
        "test_auc": _macro_ovr_auc(test_labels, pred),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _result_rows_for_methods(
    *,
    tier: str,
    methods: list[str],
    train_base: pd.DataFrame,
    test_base: pd.DataFrame,
    train_scores: dict[str, np.ndarray],
    test_scores: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    train_labels = train_base["true_label"].reset_index(drop=True)
    test_labels = test_base["true_label"].reset_index(drop=True)
    source_auc = _macro_ovr_auc(test_labels, test_scores["source_only"])
    best_single = _score_best_single(train_labels, test_labels, train_scores, test_scores, methods)
    semantic = _score_logistic_stack(train_labels, test_labels, train_scores, test_scores, methods, c_value=1.0)
    rows = []
    for row in [best_single, semantic]:
        rows.append(
            {
                "tier": tier,
                **row,
                "source_auc": source_auc,
                "delta_vs_source": row["test_auc"] - source_auc,
            }
        )
    return rows


def build_summary() -> dict[str, Any]:
    train_base, train_scores, train_missing = _load_split_predictions("train")
    test_base, test_scores, test_missing = _load_split_predictions("test")
    missing = train_missing + test_missing
    tier_status = {}
    for tier, methods in TIER_METHODS.items():
        tier_missing = _missing_for_methods("train", methods) + _missing_for_methods("test", methods)
        tier_status[tier] = {
            "methods": methods,
            "required_prediction_files": int(sum(len(METHOD_SEEDS[method]) for method in methods) * 2),
            "missing_prediction_files": len(tier_missing),
            "missing_predictions": tier_missing,
            "ready": len(tier_missing) == 0,
        }
    methods_ready = all(method in train_scores and method in test_scores for method in METHODS)
    manifests = {
        "train": _manifest_counts(TRAIN_MANIFEST),
        "test": _manifest_counts(TEST_MANIFEST),
    }

    partial_result_rows: list[dict[str, Any]] = []
    if train_base is not None and test_base is not None:
        for tier, status in tier_status.items():
            if status["ready"] and tier != "router_full":
                partial_result_rows.extend(
                    _result_rows_for_methods(
                        tier=tier,
                        methods=status["methods"],
                        train_base=train_base,
                        test_base=test_base,
                        train_scores=train_scores,
                        test_scores=test_scores,
                    )
                )

    if missing or train_base is None or test_base is None or not methods_ready:
        return {
            "verdict": "NOT_READY_MISSING_OFFICIAL_SPLIT_PREDICTIONS",
            "claim_role": "paper_credibility_upgrade_not_new_dataset_or_third_target",
            "manifests": manifests,
            "methods": METHODS,
            "method_seeds": METHOD_SEEDS,
            "tiers": tier_status,
            "available_train_methods": sorted(train_scores),
            "available_test_methods": sorted(test_scores),
            "missing_predictions": missing,
            "partial_result_rows": partial_result_rows,
            "partial_clears_1pt": bool(any(row["delta_vs_source"] >= 0.01 for row in partial_result_rows)),
            "partial_clears_3pt": bool(any(row["delta_vs_source"] >= 0.03 for row in partial_result_rows)),
            "next_command_full": (
                "D:\\conda\\envs\\CoughKD\\python.exe -B scripts\\evaluate_external_model_set.py "
                "--manifest manifests\\toscovid2021_train_external.csv --target-tag toscovid2021_train "
                "--root D:\\CoughKD --device auto --batch-size 16 --skip-existing"
            ),
            "next_command_stable_smoke": (
                "D:\\conda\\envs\\CoughKD\\python.exe -B scripts\\evaluate_external_model_set.py "
                "--manifest manifests\\toscovid2021_train_external.csv --target-tag toscovid2021_train "
                "--root D:\\CoughKD --device auto --batch-size 16 --skip-existing "
                "--methods source_only ce kd"
            ),
            "approval_boundary": "Requires inference on 4,803 existing TosCOVID train audio rows; no training and no new data download.",
        }

    result_rows = _result_rows_for_methods(
        tier="router_full",
        methods=METHODS,
        train_base=train_base,
        test_base=test_base,
        train_scores=train_scores,
        test_scores=test_scores,
    )

    return {
        "verdict": "READY_OFFICIAL_SPLIT_RESULT_AVAILABLE",
        "claim_role": "paper_credibility_upgrade_not_new_dataset_or_third_target",
        "manifests": manifests,
        "methods": METHODS,
        "method_seeds": METHOD_SEEDS,
        "tiers": tier_status,
        "source_auc": result_rows[0]["source_auc"] if result_rows else None,
        "result_rows": result_rows,
        "clears_1pt": bool(any(row["delta_vs_source"] >= 0.01 for row in result_rows)),
        "clears_3pt": bool(any(row["delta_vs_source"] >= 0.03 for row in result_rows)),
    }


def write_report(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toscovid_official_split_semantic_router.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# TosCOVID Official-Split Semantic Router",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Claim role: `{summary['claim_role']}`",
        "",
        "## Manifest Summary",
        "",
        _table(
            [
                {"split": split, **counts}
                for split, counts in summary["manifests"].items()
            ]
        ),
        "",
    ]
    if summary["verdict"] != "READY_OFFICIAL_SPLIT_RESULT_AVAILABLE":
        lines.extend(
            [
                "## Execution Tiers",
                "",
                _table(
                    [
                        {
                            "tier": tier,
                            "methods": ",".join(status["methods"]),
                            "required_prediction_files": status["required_prediction_files"],
                            "missing_prediction_files": status["missing_prediction_files"],
                            "ready": status["ready"],
                        }
                        for tier, status in summary["tiers"].items()
                    ]
                ),
                "",
                "## Missing Predictions",
                "",
                _table(summary["missing_predictions"]),
                "",
                "## Partial Official-Split Results",
                "",
                _table(summary.get("partial_result_rows", [])),
                "",
                f"Partial clears 1-point gate: `{summary.get('partial_clears_1pt', False)}`",
                "",
                f"Partial clears 3-point gate: `{summary.get('partial_clears_3pt', False)}`",
                "",
                "## Next Commands",
                "",
                f"Stable smoke: `{summary['next_command_stable_smoke']}`",
                "",
                f"Full router: `{summary['next_command_full']}`",
                "",
                "## Approval Boundary",
                "",
                summary["approval_boundary"],
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Official-Split Results",
                "",
                _table(summary["result_rows"]),
                "",
                f"Clears 1-point gate: `{summary['clears_1pt']}`",
                "",
                f"Clears 3-point gate: `{summary['clears_3pt']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Decision Boundary",
            "",
            "- This uses only existing TosCOVID 2021 local audio.",
            "- It is not a third-target claim.",
            "- Its purpose is to replace random TosCOVID calibration/evaluation splits with official train calibration and official test evaluation.",
            "",
        ]
    )
    (OUT / "TOSCOVID_OFFICIAL_SPLIT_SEMANTIC_ROUTER.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    summary = build_summary()
    write_report(summary)
    print(OUT / "TOSCOVID_OFFICIAL_SPLIT_SEMANTIC_ROUTER.md")
    print(summary["verdict"])


if __name__ == "__main__":
    main()
