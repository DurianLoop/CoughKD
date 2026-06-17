from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_prediction_ensemble_frontier import PROB_COLS, _macro_ovr_auc
from audit_target_calibrated_stacking import _features, _fit_stack, _predict_full_probs
from audit_toscovid_official_split_semantic_router import METHOD_SEEDS, TIER_METHODS


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "toscovid_official_calibration_budget"
TEST_TAG = "toscovid2021_test"
TEST_MANIFEST = ROOT / "manifests" / "toscovid2021_test_external.csv"


CALIBRATION_MANIFESTS = {
    "calib10": ROOT / "manifests" / "toscovid2021_train_calib10_external.csv",
    "calib20": ROOT / "manifests" / "toscovid2021_train_calib20_external.csv",
    "calib30": ROOT / "manifests" / "toscovid2021_train_calib30_external.csv",
    "calib50": ROOT / "manifests" / "toscovid2021_train_calib50_external.csv",
    "full_train": ROOT / "manifests" / "toscovid2021_train_external.csv",
}
CALIBRATION_TARGET_TAGS = {
    "calib10": "toscovid2021_train_calib10",
    "calib20": "toscovid2021_train_calib20",
    "calib30": "toscovid2021_train_calib30",
    "calib50": "toscovid2021_train_calib50",
    "full_train": "toscovid2021_train",
}


def _prediction_path(target_tag: str, method: str, seed: str) -> Path:
    return RUNS / f"external_{target_tag}_{method}_seed{seed}" / "predictions.csv"


def _load_prediction(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"recording_id", "true_label", *PROB_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")
    return df[["recording_id", "true_label", *PROB_COLS]].sort_values("recording_id").reset_index(drop=True)


def _load_scores(target_tag: str, methods: list[str]) -> tuple[pd.DataFrame | None, dict[str, np.ndarray], list[dict[str, Any]]]:
    base: pd.DataFrame | None = None
    scores: dict[str, np.ndarray] = {}
    missing: list[dict[str, Any]] = []
    for method in methods:
        arrays = []
        for seed in METHOD_SEEDS[method]:
            path = _prediction_path(target_tag, method, seed)
            if not path.is_file():
                missing.append(
                    {
                        "target_tag": target_tag,
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


def _manifest_counts(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "rows": 0, "subjects": 0}
    df = pd.read_csv(path).fillna("")
    return {
        "exists": True,
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "covid_positive_rows": int((df["label"] == "covid_positive").sum()),
        "healthy_rows": int((df["label"] == "healthy").sum()),
    }


def _compute_result(
    *,
    tier: str,
    methods: list[str],
    calibration_tag: str,
    calib_base: pd.DataFrame,
    calib_scores: dict[str, np.ndarray],
    test_base: pd.DataFrame,
    test_scores: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    calib_labels = calib_base["true_label"].reset_index(drop=True)
    test_labels = test_base["true_label"].reset_index(drop=True)
    source_auc = _macro_ovr_auc(test_labels, test_scores["source_only"])
    rows: list[dict[str, Any]] = []

    best_method = None
    best_auc = -np.inf
    for method in methods:
        auc = _macro_ovr_auc(calib_labels, calib_scores[method])
        if auc > best_auc:
            best_method = method
            best_auc = auc
    if best_method is not None:
        test_auc = _macro_ovr_auc(test_labels, test_scores[best_method])
        rows.append(
            {
                "calibration": calibration_tag,
                "tier": tier,
                "candidate": "official_calib_best_single",
                "selected_method": best_method,
                "calib_auc": best_auc,
                "test_auc": test_auc,
                "source_auc": source_auc,
                "delta_vs_source": test_auc - source_auc,
            }
        )

    model = _fit_stack(_features(calib_scores, methods), calib_labels, c_value=1.0)
    if model is not None:
        stack_scores = _predict_full_probs(model, _features(test_scores, methods))
        test_auc = _macro_ovr_auc(test_labels, stack_scores)
        rows.append(
            {
                "calibration": calibration_tag,
                "tier": tier,
                "candidate": "official_calib_semantic_router",
                "selected_method": "logistic_stacking",
                "calib_auc": "",
                "test_auc": test_auc,
                "source_auc": source_auc,
                "delta_vs_source": test_auc - source_auc,
            }
        )
    return rows


def _command(calibration: str, tier: str) -> str:
    target_tag = CALIBRATION_TARGET_TAGS[calibration]
    manifest = CALIBRATION_MANIFESTS[calibration].relative_to(ROOT)
    methods = ""
    if tier == "stable_smoke":
        methods = " --methods source_only ce kd"
    return (
        "D:\\conda\\envs\\CoughKD\\python.exe -B scripts\\evaluate_external_model_set.py "
        f"--manifest {manifest} --target-tag {target_tag} --root D:\\CoughKD "
        f"--device auto --batch-size 16 --skip-existing{methods}"
    )


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


def _recommendation(result_rows: list[dict[str, Any]], tier_rows: list[dict[str, Any]], commands: list[dict[str, str]]) -> dict[str, str]:
    if not result_rows:
        return {
            "state": "AWAIT_CALIB10_STABLE_SMOKE",
            "next_action": "Run the calib10 stable_smoke command first; it needs 9 calibration prediction files over 500 audio rows.",
            "rationale": "No official-train calibration subset has predictions yet, and test predictions are already present.",
        }

    best = max(result_rows, key=lambda row: float(row["delta_vs_source"]))
    best_delta = float(best["delta_vs_source"])
    best_calibration = str(best["calibration"])
    best_tier = str(best["tier"])
    if best_delta >= 0.03:
        return {
            "state": "PROMOTE_TO_FULL_ROUTER",
            "next_action": "Run router_full for the same or next larger calibration tier, then verify the full method set.",
            "rationale": f"Best official-test delta is {100.0 * best_delta:.2f} pp from {best_calibration}/{best_tier}, clearing the 3 pp escalation threshold.",
        }
    if best_delta >= 0.01:
        order = ["calib10", "calib20", "calib30", "calib50", "full_train"]
        try:
            next_calibration = order[min(order.index(best_calibration) + 1, len(order) - 1)]
        except ValueError:
            next_calibration = "calib20"
        return {
            "state": "PROMOTE_TO_NEXT_STABLE_SUBSET",
            "next_action": f"Run {next_calibration} stable_smoke before spending on router_full.",
            "rationale": f"Best official-test delta is {100.0 * best_delta:.2f} pp, enough to justify a larger low-cost probe but not a full-router claim.",
        }
    return {
        "state": "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE",
        "next_action": "Do not spend full-router compute on this TosCOVID official-split enhancement; return to literature/innovation search.",
        "rationale": f"Best available official-test delta is only {100.0 * best_delta:.2f} pp, below the 1 pp continuation threshold.",
    }


def build_summary() -> dict[str, Any]:
    test_base_by_tier: dict[str, pd.DataFrame | None] = {}
    test_scores_by_tier: dict[str, dict[str, np.ndarray]] = {}
    tier_rows = []
    result_rows = []
    missing_rows = []
    commands = []

    for tier, methods in TIER_METHODS.items():
        test_base, test_scores, test_missing = _load_scores(TEST_TAG, methods)
        test_base_by_tier[tier] = test_base
        test_scores_by_tier[tier] = test_scores
        for calibration, manifest in CALIBRATION_MANIFESTS.items():
            target_tag = CALIBRATION_TARGET_TAGS[calibration]
            calib_base, calib_scores, calib_missing = _load_scores(target_tag, methods)
            missing = test_missing + calib_missing
            ready = (
                manifest.is_file()
                and test_base is not None
                and calib_base is not None
                and not missing
                and all(method in test_scores and method in calib_scores for method in methods)
            )
            counts = _manifest_counts(manifest)
            tier_rows.append(
                {
                    "calibration": calibration,
                    "tier": tier,
                    "manifest_rows": counts["rows"],
                    "manifest_subjects": counts["subjects"],
                    "required_prediction_files": int(sum(len(METHOD_SEEDS[method]) for method in methods) * 2),
                    "missing_calibration_files": len(calib_missing),
                    "missing_test_files": len(test_missing),
                    "missing_prediction_files": len(missing),
                    "ready": ready,
                }
            )
            if ready:
                result_rows.extend(
                    _compute_result(
                        tier=tier,
                        methods=methods,
                        calibration_tag=calibration,
                        calib_base=calib_base,
                        calib_scores=calib_scores,
                        test_base=test_base,
                        test_scores=test_scores,
                    )
                )
            else:
                missing_rows.extend(missing)
                commands.append(
                    {
                        "calibration": calibration,
                        "tier": tier,
                        "command": _command(calibration, tier),
                    }
                )

    commands = list({(row["calibration"], row["tier"]): row for row in commands}.values())
    verdict = "READY_WITH_RESULTS" if result_rows else "NOT_READY_MISSING_CALIBRATION_PREDICTIONS"
    recommendation = _recommendation(result_rows, tier_rows, commands)
    return {
        "verdict": verdict,
        "claim_role": "low_cost_official_train_calibration_probe_not_final_claim",
        "recommendation": recommendation,
        "tiers": tier_rows,
        "results": result_rows,
        "missing_predictions": missing_rows,
        "commands": commands,
        "clears_1pt": bool(any(float(row["delta_vs_source"]) >= 0.01 for row in result_rows)),
        "clears_3pt": bool(any(float(row["delta_vs_source"]) >= 0.03 for row in result_rows)),
    }


def write_report(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toscovid_official_calibration_budget.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TosCOVID Official Calibration Budget",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Claim role: `{summary['claim_role']}`",
        f"- Clears 1-point gate: `{summary['clears_1pt']}`",
        f"- Clears 3-point gate: `{summary['clears_3pt']}`",
        f"- Recommendation: `{summary['recommendation']['state']}`",
        f"- Next action: {summary['recommendation']['next_action']}",
        f"- Rationale: {summary['recommendation']['rationale']}",
        "",
        "## Tier Readiness",
        "",
        _table(summary["tiers"]),
        "",
        "## Results",
        "",
        _table(summary["results"]),
        "",
        "## Next Commands",
        "",
        _table(summary["commands"]),
        "",
        "## Decision Boundary",
        "",
        "- These are low-cost probes using official train calibration subsets and official test evaluation.",
        "- Passing a subset probe can justify running the larger official-train calibration route.",
        "- Subset probes should not replace the full official-train result in the final paper claim.",
        "",
    ]
    (OUT / "TOSCOVID_OFFICIAL_CALIBRATION_BUDGET.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = build_summary()
    write_report(summary)
    print(OUT / "TOSCOVID_OFFICIAL_CALIBRATION_BUDGET.md")
    print(summary["verdict"])


if __name__ == "__main__":
    main()
