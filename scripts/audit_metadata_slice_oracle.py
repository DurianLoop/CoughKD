from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_prediction_ensemble_frontier import PROB_COLS, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


TARGETS = {
    "COUGHVID": {
        "manifest": ROOT / "manifests/coughvid_external.csv",
        "columns": ["sex", "age_bin", "source_status", "cough_detected_bin", "symptom_resp", "symptom_fever"],
    },
    "TosCOVID": {
        "manifest": ROOT / "manifests/toscovid2021_test_external.csv",
        "columns": ["sex", "age", "device", "source_status", "source_subset", "source_batch"],
    },
    "UKCOVID": {
        "manifest": ROOT / "manifests/ukcovid_open_test_external.csv",
        "columns": [
            "sex",
            "age",
            "source_status",
            "symptom_cough_any",
            "symptom_fatigue",
            "symptom_headache",
            "symptom_onset",
            "symptom_none",
        ],
    },
    "Virufyseg": {
        "manifest": ROOT / "manifests/virufy_segmented_external.csv",
        "columns": ["sex", "age_bin", "source_status", "num_clips_bin"],
    },
}


def _age_bin(value: object) -> str:
    try:
        age = float(value)
    except Exception:
        return str(value) if str(value) else "missing"
    if age < 30:
        return "<30"
    if age < 45:
        return "30-44"
    if age < 60:
        return "45-59"
    return "60+"


def _cough_detected_bin(value: object) -> str:
    try:
        score = float(value)
    except Exception:
        return "missing"
    if score < 0.8:
        return "<0.8"
    if score < 0.95:
        return "0.8-0.95"
    return ">=0.95"


def _symptom_flag(text: object, key: str) -> str:
    raw = str(text).lower()
    if f"{key}=true" in raw or f"{key}=1" in raw:
        return "true"
    if f"{key}=false" in raw or f"{key}=0" in raw:
        return "false"
    return "missing"


def _any_symptom_flag(text: object, keys: list[str]) -> str:
    values = [_symptom_flag(text, key) for key in keys]
    if "true" in values:
        return "true"
    if all(value == "false" for value in values):
        return "false"
    return "missing"


def _prepare_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "age" in df.columns:
        df["age_bin"] = df["age"].map(_age_bin)
    if "cough_detected" in df.columns:
        df["cough_detected_bin"] = df["cough_detected"].map(_cough_detected_bin)
    if "symptoms" in df.columns:
        df["symptom_resp"] = df["symptoms"].map(lambda x: _symptom_flag(x, "respiratory_condition"))
        df["symptom_fever"] = df["symptoms"].map(lambda x: _symptom_flag(x, "fever_muscle_pain"))
        df["symptom_cough_any"] = df["symptoms"].map(
            lambda x: _any_symptom_flag(x, ["symptom_cough_any", "symptom_new_continuous_cough"])
        )
        for key in [
            "symptom_fatigue",
            "symptom_headache",
            "symptom_onset",
            "symptom_none",
            "symptom_change_to_sense_of_smell_or_taste",
            "symptom_runny_or_blocked_nose",
            "symptom_shortness_of_breath",
        ]:
            df[key] = df["symptoms"].map(lambda x, key=key: _symptom_flag(x, key))
    if "subject_id" in df.columns:
        counts = df.groupby("subject_id")["recording_id"].transform("count")
        df["num_clips_bin"] = pd.cut(
            counts,
            bins=[0, 1, 2, 5, 10, 10**9],
            labels=["1", "2", "3-5", "6-10", "10+"],
            include_lowest=True,
        ).astype(str)
    return df


def _method_predictions(target: str, methods: list[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    specs = [spec for spec in _discover() if spec.target == target and spec.method in methods]
    by_method: dict[str, list[Path]] = {}
    for spec in specs:
        by_method.setdefault(spec.method, []).append(spec.path)
    base: pd.DataFrame | None = None
    predictions: dict[str, np.ndarray] = {}
    for method, paths in sorted(by_method.items()):
        frames = []
        for path in sorted(paths):
            df = pd.read_csv(path).sort_values("recording_id").reset_index(drop=True)
            if base is None:
                base = df[["recording_id", "true_label"]].copy()
            else:
                if not base["recording_id"].equals(df["recording_id"]):
                    raise ValueError(f"recording mismatch for {path}")
            frames.append(df[PROB_COLS].to_numpy(dtype=float))
        predictions[method] = np.mean(frames, axis=0)
    if base is None:
        raise ValueError(f"No predictions found for {target}")
    return base, predictions


def _slice_oracle(
    labels: pd.Series,
    base_scores: np.ndarray,
    method_scores: dict[str, np.ndarray],
    groups: pd.Series,
    min_slice_size: int,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    mixed = base_scores.copy()
    rows = []
    for group_value, idx_obj in groups.groupby(groups).groups.items():
        idx = np.asarray(list(idx_obj), dtype=int)
        if len(idx) < min_slice_size or labels.iloc[idx].nunique() < 2:
            continue
        best_method = None
        best_auc = -np.inf
        for method, scores in method_scores.items():
            try:
                auc = _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), scores[idx])
            except ValueError:
                continue
            if auc > best_auc:
                best_auc = auc
                best_method = method
        if best_method is None:
            continue
        mixed[idx] = method_scores[best_method][idx]
        rows.append({"slice": str(group_value), "n": int(len(idx)), "best_method": best_method, "slice_auc": float(best_auc)})
    return mixed, rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/metadata_slice_oracle")
    parser.add_argument("--min-slice-size", type=int, default=120)
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
    rows = []
    slice_rows = []
    for target, config in TARGETS.items():
        manifest = _prepare_manifest(config["manifest"])
        base, method_scores = _method_predictions(target, args.methods)
        merged = base.merge(manifest, on="recording_id", how="left", suffixes=("", "_manifest"))
        labels = merged["true_label"]
        source_scores = method_scores["source_only"]
        source_auc = _macro_ovr_auc(labels, source_scores)
        oracle_method = max(method_scores, key=lambda method: _macro_ovr_auc(labels, method_scores[method]))
        oracle_single_auc = _macro_ovr_auc(labels, method_scores[oracle_method])
        for column in config["columns"]:
            if column not in merged.columns:
                continue
            groups = merged[column].fillna("missing").astype(str)
            if groups.nunique() < 2:
                continue
            mixed, selected = _slice_oracle(labels, source_scores, method_scores, groups, args.min_slice_size)
            mixed_auc = _macro_ovr_auc(labels, mixed)
            rows.append(
                {
                    "target": target,
                    "slice_column": column,
                    "n_slices_total": int(groups.nunique()),
                    "n_slices_used": len(selected),
                    "source_auc": source_auc,
                    "best_single_method": oracle_method,
                    "best_single_auc": oracle_single_auc,
                    "slice_oracle_auc": mixed_auc,
                    "delta_vs_source": mixed_auc - source_auc,
                    "delta_vs_best_single": mixed_auc - oracle_single_auc,
                }
            )
            for item in selected:
                slice_rows.append({"target": target, "slice_column": column, **item})
    result = pd.DataFrame(rows).sort_values(["target", "delta_vs_source"], ascending=[True, False])
    slices = pd.DataFrame(slice_rows)
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "metadata_slice_oracle_summary.csv", index=False)
    slices.to_csv(args.out / "metadata_slice_oracle_selected_slices.csv", index=False)
    large = result[result["target"].isin(["COUGHVID", "TosCOVID"])]
    payload = {
        "min_slice_size": args.min_slice_size,
        "clears_3pt_large_target": bool((large["delta_vs_source"] >= 0.03).any()) if not large.empty else False,
        "best_large_rows": large.head(10).to_dict(orient="records"),
        "best_all_rows": result.head(20).to_dict(orient="records"),
    }
    (args.out / "metadata_slice_oracle_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Metadata Slice Oracle",
        "",
        "This label-revealed upper bound asks whether metadata-conditioned method selection has enough signal to pursue.",
        "",
        "## Best Rows",
        "",
        _to_md(result.head(30)),
        "",
        f"Clears 3-point gate on COUGHVID/TosCOVID: `{payload['clears_3pt_large_target']}`",
        "",
    ]
    (args.out / "METADATA_SLICE_ORACLE.md").write_text("\n".join(lines), encoding="utf-8")
    print(args.out / "METADATA_SLICE_ORACLE.md")


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
