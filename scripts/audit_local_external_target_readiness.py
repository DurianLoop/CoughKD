from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from audit_prediction_ensemble_frontier import _discover


ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent


def _manifest_rows(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _safe_nunique(df: pd.DataFrame, column: str) -> int | None:
    if column not in df.columns:
        return None
    return int(df[column].nunique(dropna=True))


def _audio_status(df: pd.DataFrame) -> dict[str, Any]:
    if "path" not in df.columns:
        return {
            "audio_found": None,
            "audio_total": None,
            "audio_found_fraction": None,
            "audio_root": "",
        }
    candidates = [WORKSPACE_ROOT, ROOT]
    raw_paths = [str(value) for value in df["path"].fillna("")]
    root_counts = {
        str(base): sum(1 for value in raw_paths if (base / value).is_file())
        for base in candidates
    }
    audio_root, found = max(root_counts.items(), key=lambda item: item[1])
    total = len(raw_paths)
    return {
        "audio_found": int(found),
        "audio_total": int(total),
        "audio_found_fraction": None if total == 0 else float(found / total),
        "audio_root": audio_root if found else "",
    }


def _symptom_status(df: pd.DataFrame) -> dict[str, Any]:
    if "symptoms" not in df.columns:
        return {
            "has_symptoms": False,
            "structured_key_value_symptoms": False,
            "nonempty_symptom_rows": 0,
            "example_symptoms": "",
        }
    symptoms = df["symptoms"].fillna("").astype(str).str.strip()
    nonempty = symptoms[symptoms != ""]
    examples = [value for value in nonempty.head(3).tolist() if value]
    structured = bool(nonempty.str.contains("=", regex=False).any())
    return {
        "has_symptoms": True,
        "structured_key_value_symptoms": structured,
        "nonempty_symptom_rows": int(len(nonempty)),
        "example_symptoms": " | ".join(examples),
    }


def _prediction_status() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for spec in _discover():
        target = out.setdefault(spec.target, {"methods": set(), "runs": 0})
        target["methods"].add(spec.method)
        target["runs"] += 1
    return {
        target: {"methods": sorted(value["methods"]), "runs": int(value["runs"])}
        for target, value in sorted(out.items())
    }


def _target_name(path: Path) -> str:
    stem = path.stem
    for suffix in ["_external", "_test_external", "_metadata_only", "_test_metadata_only"]:
        stem = stem.replace(suffix, "")
    return stem


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No rows._"
    cols = [
        "manifest",
        "target_guess",
        "rows",
        "subjects",
        "labels",
        "audio_found",
        "audio_total",
        "audio_found_fraction",
        "audio_root",
        "has_symptoms",
        "structured_key_value_symptoms",
        "prediction_target",
        "prediction_runs",
        "prediction_methods",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append("nan" if pd.isna(value) else f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, default=ROOT / "manifests")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/local_external_target_readiness")
    args = parser.parse_args()

    predictions = _prediction_status()
    prediction_name_map = {
        "coughvid": "COUGHVID",
        "toscovid2021": "TosCOVID",
        "toscovid2021_test": "TosCOVID",
        "toscovid": "TosCOVID",
        "virufy": "Virufy",
        "virufy_segmented": "Virufyseg",
    }
    rows = []
    for path in sorted(args.manifest_dir.glob("*.csv")):
        df = _manifest_rows(path)
        target_guess = _target_name(path)
        pred_target = prediction_name_map.get(target_guess)
        pred = predictions.get(pred_target or "", {"methods": [], "runs": 0})
        audio = _audio_status(df)
        symptoms = _symptom_status(df)
        label_counts = df["label"].value_counts().to_dict() if "label" in df.columns else {}
        rows.append(
            {
                "manifest": str(path.relative_to(ROOT)),
                "target_guess": target_guess,
                "rows": int(len(df)),
                "subjects": _safe_nunique(df, "subject_id"),
                "labels": json.dumps(label_counts, sort_keys=True),
                **audio,
                **symptoms,
                "prediction_target": pred_target or "",
                "prediction_runs": int(pred["runs"]),
                "prediction_methods": ",".join(pred["methods"]),
            }
        )

    result = pd.DataFrame(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "local_external_target_readiness.csv", index=False)
    payload = {
        "n_manifests": int(len(result)),
        "prediction_targets": predictions,
        "ready_with_structured_symptoms_predictions": result[
            result["structured_key_value_symptoms"] & (result["prediction_runs"] > 0)
        ].to_dict(orient="records"),
        "ready_with_audio_symptoms_predictions": result[
            (result["audio_found_fraction"].fillna(0.0) > 0.95)
            & result["has_symptoms"]
            & (result["prediction_runs"] > 0)
        ].to_dict(orient="records"),
        "audio_blocked_with_structured_symptoms": result[
            (result["audio_found_fraction"].fillna(0.0) < 0.95) & result["structured_key_value_symptoms"]
        ].to_dict(orient="records"),
    }
    (args.out / "local_external_target_readiness_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# Local External Target Readiness",
        "",
        "This audit summarizes which local manifests can currently support metadata-slice transfer-gate experiments without new downloads.",
        "",
        "## Manifest Table",
        "",
        _to_md(result),
        "",
        "## Interpretation",
        "",
        f"- Ready with structured symptoms and prediction runs: `{len(payload['ready_with_structured_symptoms_predictions'])}`",
        f"- Ready with audio, symptoms, and prediction runs: `{len(payload['ready_with_audio_symptoms_predictions'])}`",
        f"- Audio-blocked but structured symptoms available: `{len(payload['audio_blocked_with_structured_symptoms'])}`",
        "",
    ]
    report = args.out / "LOCAL_EXTERNAL_TARGET_READINESS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
