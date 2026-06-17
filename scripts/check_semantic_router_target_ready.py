from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from audit_metadata_slice_oracle import TARGETS, _prepare_manifest
from audit_prediction_ensemble_frontier import _discover
from audit_semantic_constrained_transfer_router import DEFAULT_SEMANTIC_SLICE_COLUMNS
from audit_target_calibrated_transfer_controller import SLICE_CONFIG


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_METHODS = [
    "source_only",
    "ce",
    "kd",
    "candidate_a",
    "candidate_b",
    "candidate_c",
    "candidate_f_artifact_env_irm_ramp",
    "tcd_conf035",
    "tcd_very_strong",
]


def _to_md_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _audio_hint(target: str) -> dict[str, object] | None:
    if target != "UKCOVID":
        return None
    path = ROOT / "runs" / "ukcovid_open_metadata_audit" / "ukcovid_audio_ready_summary.json"
    if not path.is_file():
        return {"status": "missing_audio_ready_summary", "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    manifest_audio = payload.get("manifest_audio_status", {})
    return {
        "status": payload.get("verdict"),
        "archive_complete": payload.get("archive_complete"),
        "unique_audio_found": manifest_audio.get("unique_audio_found"),
        "unique_audio_filenames": manifest_audio.get("unique_audio_filenames"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="UKCOVID")
    parser.add_argument("--out", type=Path, default=ROOT / "runs" / "semantic_router_third_target_readiness")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--semantic-slice-columns", nargs="+", default=sorted(DEFAULT_SEMANTIC_SLICE_COLUMNS))
    args = parser.parse_args()

    target = args.target
    semantic_slice_columns = set(args.semantic_slice_columns)
    checks: list[dict[str, object]] = []

    target_config = TARGETS.get(target)
    checks.append({"check": "target_registered", "pass": target_config is not None, "detail": target})

    manifest = None
    prepared = None
    if target_config is not None:
        manifest = Path(target_config["manifest"])
        checks.append({"check": "manifest_exists", "pass": manifest.is_file(), "detail": str(manifest)})
        if manifest.is_file():
            prepared = _prepare_manifest(manifest)
            checks.append({"check": "manifest_rows", "pass": len(prepared) > 0, "detail": len(prepared)})
            checks.append(
                {
                    "check": "manifest_subjects",
                    "pass": "subject_id" in prepared.columns and prepared["subject_id"].nunique() > 0,
                    "detail": int(prepared["subject_id"].nunique()) if "subject_id" in prepared.columns else "missing",
                }
            )

    slice_config = SLICE_CONFIG.get(target)
    checks.append({"check": "slice_config_registered", "pass": slice_config is not None, "detail": slice_config or {}})
    slice_column = str(slice_config["slice_column"]) if slice_config else None
    if slice_column and prepared is not None:
        exists = slice_column in prepared.columns
        unique = int(prepared[slice_column].nunique(dropna=False)) if exists else 0
        checks.append({"check": "slice_column_exists", "pass": exists, "detail": slice_column})
        checks.append({"check": "slice_column_has_groups", "pass": unique >= 2, "detail": unique})
        checks.append(
            {
                "check": "slice_column_semantic",
                "pass": slice_column in semantic_slice_columns,
                "detail": slice_column,
            }
        )

    specs = [spec for spec in _discover() if spec.target == target and spec.method in args.methods]
    method_counts = Counter(spec.method for spec in specs)
    present_methods = sorted(method_counts)
    missing_methods = [method for method in args.methods if method not in method_counts]
    checks.append({"check": "prediction_runs_present", "pass": bool(specs), "detail": len(specs)})
    checks.append({"check": "source_only_present", "pass": "source_only" in method_counts, "detail": method_counts.get("source_only", 0)})
    checks.append(
        {
            "check": "minimum_strategy_pool",
            "pass": "source_only" in method_counts and len(present_methods) >= 2,
            "detail": ",".join(present_methods) if present_methods else "none",
        }
    )

    audio = _audio_hint(target)
    if audio is not None:
        checks.append({"check": "audio_ready", "pass": audio.get("status") == "READY_FOR_EVALUATION", "detail": audio})

    blocking = [row["check"] for row in checks if not row["pass"]]
    ready = not blocking
    if not ready and target == "UKCOVID" and "prediction_runs_present" in blocking and "audio_ready" in blocking:
        verdict = "NOT_READY_AUDIO_AND_PREDICTIONS_MISSING"
    elif not ready:
        verdict = "NOT_READY_" + "_".join(str(item).upper() for item in blocking[:3])
    else:
        verdict = "READY_FOR_SEMANTIC_ROUTER_AUDIT"

    payload = {
        "target": target,
        "verdict": verdict,
        "ready": ready,
        "checks": checks,
        "present_methods": present_methods,
        "method_counts": dict(method_counts),
        "missing_methods": missing_methods,
        "semantic_slice_columns": sorted(semantic_slice_columns),
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"{target.lower()}_semantic_router_readiness.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        f"# Semantic Router Target Readiness: {target}",
        "",
        f"- Verdict: `{verdict}`",
        f"- Ready: `{ready}`",
        "",
        "## Checks",
        "",
        _to_md_table(checks),
        "",
        "## Prediction Methods",
        "",
        f"- Present: `{', '.join(present_methods) if present_methods else 'none'}`",
        f"- Missing: `{', '.join(missing_methods) if missing_methods else 'none'}`",
        "",
        "## Interpretation",
        "",
    ]
    if ready:
        lines.append("This target has the registry, manifest, slice column, and prediction runs needed for semantic-router auditing.")
    else:
        lines.append("This target is not ready for semantic-router auditing. Fix the failed checks above before running the 1000-repeat audit.")
    (args.out / f"{target.upper()}_SEMANTIC_ROUTER_READINESS.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print(json.dumps({"target": target, "verdict": verdict, "ready": ready, "blocking": blocking}, indent=2))
    print(args.out / f"{target.upper()}_SEMANTIC_ROUTER_READINESS.md")


if __name__ == "__main__":
    main()
