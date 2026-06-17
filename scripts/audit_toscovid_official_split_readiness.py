from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "toscovid_official_split_readiness"

FULL_MANIFEST = ROOT / "manifests" / "toscovid2021_full_external.csv"
TRAIN_MANIFEST = ROOT / "manifests" / "toscovid2021_train_external.csv"
TEST_MANIFEST = ROOT / "manifests" / "toscovid2021_test_external.csv"

CORE_METHODS = [
    "source_only",
    "ce",
    "kd",
    "tcd_conf035",
    "tcd_very_strong",
    "candidate_a",
    "candidate_b",
    "candidate_c",
    "candidate_f_artifact_env_irm_ramp",
]
SEEDS = ["7", "11", "23"]
METHOD_SEEDS = {
    method: SEEDS for method in CORE_METHODS if method != "candidate_f_artifact_env_irm_ramp"
}
METHOD_SEEDS["candidate_f_artifact_env_irm_ramp"] = ["7"]


def _run_dir(method: str, seed: str) -> Path:
    return RUNS / f"external_toscovid2021_test_{method}_seed{seed}"


def _read_predictions(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path, usecols=["recording_id"])
    return set(df["recording_id"].astype(str))


def _split_summary(df: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for split, part in df.groupby("split", dropna=False):
        rows.append(
            {
                "split": str(split),
                "rows": int(len(part)),
                "subjects": int(part["subject_id"].nunique()),
                "covid_positive": int((part["label"] == "covid_positive").sum()),
                "healthy": int((part["label"] == "healthy").sum()),
                "audio_found": int(part["path"].fillna("").astype(str).ne("").sum()),
            }
        )
    return sorted(rows, key=lambda row: row["split"])


def _coverage_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    split_ids = {
        split: set(part["recording_id"].astype(str))
        for split, part in df.groupby("split", dropna=False)
    }
    rows = []
    for method in CORE_METHODS:
        for seed in METHOD_SEEDS[method]:
            pred_path = _run_dir(method, seed) / "predictions.csv"
            pred_ids = _read_predictions(pred_path)
            row: dict[str, Any] = {
                "method": method,
                "seed": seed,
                "prediction_file": str(pred_path.relative_to(ROOT)),
                "file_exists": pred_path.is_file(),
                "prediction_rows": len(pred_ids),
            }
            for split, ids in split_ids.items():
                covered = len(ids & pred_ids)
                row[f"{split}_covered"] = covered
                row[f"{split}_coverage"] = covered / len(ids) if ids else 0.0
            rows.append(row)
    return rows


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{100.0 * value:.1f}%" if 0 <= value <= 1 else f"{value:.6f}"
    return str(value)


def _table(rows: list[dict[str, Any]], columns: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    cols = columns or list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def build_summary() -> dict[str, Any]:
    if not FULL_MANIFEST.is_file():
        raise SystemExit(f"missing full manifest: {FULL_MANIFEST}")
    df = pd.read_csv(FULL_MANIFEST).fillna("")
    split_rows = _split_summary(df)
    coverage = _coverage_rows(df)
    train_ready = all(row.get("train_coverage", 0.0) >= 0.999 for row in coverage)
    test_ready = all(row.get("test_coverage", 0.0) >= 0.999 for row in coverage)
    test_manifest_rows = int(pd.read_csv(TEST_MANIFEST).shape[0]) if TEST_MANIFEST.is_file() else 0

    verdict = "READY_FOR_OFFICIAL_SPLIT_REANALYSIS" if train_ready and test_ready else "NOT_READY_MISSING_OFFICIAL_TRAIN_PREDICTIONS"
    next_action = (
        "Run inference for the core methods on the official TosCOVID train split, then use train only for "
        "router calibration/model selection and official test only for evaluation."
        if not train_ready
        else "Recompute the semantic-router TosCOVID result with official train calibration and official test evaluation."
    )
    return {
        "verdict": verdict,
        "claim_role": "paper_credibility_upgrade_not_new_dataset_or_third_target",
        "full_manifest": str(FULL_MANIFEST.relative_to(ROOT)),
        "train_manifest": str(TRAIN_MANIFEST.relative_to(ROOT)),
        "test_manifest": str(TEST_MANIFEST.relative_to(ROOT)),
        "train_manifest_rows": int(pd.read_csv(TRAIN_MANIFEST).shape[0]) if TRAIN_MANIFEST.is_file() else 0,
        "test_manifest_rows": test_manifest_rows,
        "split_summary": split_rows,
        "core_methods": CORE_METHODS,
        "method_seeds": METHOD_SEEDS,
        "coverage": coverage,
        "train_predictions_ready": train_ready,
        "test_predictions_ready": test_ready,
        "next_action": next_action,
        "compute_boundary": "No training is required for this audit. The next step needs inference on 4,803 TosCOVID train audio rows for the selected existing checkpoints/methods, so ask before running it.",
    }


def write_report(summary: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toscovid_official_split_readiness.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    cov_cols = [
        "method",
        "seed",
        "file_exists",
        "prediction_rows",
        "train_covered",
        "train_coverage",
        "test_covered",
        "test_coverage",
    ]
    lines = [
        "# TosCOVID Official Split Readiness",
        "",
        f"- Verdict: `{summary['verdict']}`",
        f"- Claim role: `{summary['claim_role']}`",
        f"- Full manifest: `{summary['full_manifest']}`",
        f"- Train-only manifest rows: `{summary['train_manifest_rows']}`",
        f"- Test-only manifest rows already used: `{summary['test_manifest_rows']}`",
        f"- Next action: {summary['next_action']}",
        f"- Compute boundary: {summary['compute_boundary']}",
        "",
        "## Official Split Manifest",
        "",
        _table(summary["split_summary"]),
        "",
        "## Prediction Coverage",
        "",
        _table(summary["coverage"], cov_cols),
        "",
        "## Decision Boundary",
        "",
        "- This route uses the existing TosCOVID 2021 dataset only; it is not a third-target upgrade.",
        "- The value is reviewer-facing rigor: calibration/model selection can be separated from final official test evaluation.",
        "- Current predictions cover the official test split but not the official train split, so the experiment should wait for explicit approval before any inference pass.",
        "",
    ]
    (OUT / "TOSCOVID_OFFICIAL_SPLIT_READINESS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    summary = build_summary()
    write_report(summary)
    print(OUT / "TOSCOVID_OFFICIAL_SPLIT_READINESS.md")
    print(summary["verdict"])


if __name__ == "__main__":
    main()
