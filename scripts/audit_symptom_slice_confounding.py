from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from audit_metadata_slice_oracle import _prepare_manifest
from audit_prediction_ensemble_frontier import LABELS, PROB_COLS, _discover, _macro_ovr_auc


ROOT = Path(__file__).resolve().parents[1]


def _load_method_scores(target: str, method: str) -> tuple[pd.DataFrame, np.ndarray]:
    specs = [spec for spec in _discover() if spec.target == target and spec.method == method]
    if not specs:
        raise ValueError(f"No predictions for target={target} method={method}")
    frames = []
    base = None
    for spec in specs:
        df = pd.read_csv(spec.path).sort_values("recording_id").reset_index(drop=True)
        if base is None:
            base = df[["recording_id", "true_label"]].copy()
        elif not base["recording_id"].equals(df["recording_id"]):
            raise ValueError(f"recording mismatch for {spec.path}")
        frames.append(df[PROB_COLS].to_numpy(dtype=float))
    assert base is not None
    return base, np.mean(frames, axis=0)


def _binary_feature_ovr(labels: pd.Series, feature: np.ndarray) -> dict[str, float]:
    out = {}
    y = labels.to_numpy()
    values = []
    for label in LABELS:
        binary = (y == label).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        auc = float(roc_auc_score(binary, feature))
        out[label] = auc
        values.append(auc)
    out["macro_ovr_auroc"] = float(np.mean(values)) if values else float("nan")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/coughvid_external.csv")
    parser.add_argument("--target", default="COUGHVID")
    parser.add_argument("--slice-column", default="symptom_resp")
    parser.add_argument("--source-method", default="source_only")
    parser.add_argument("--method-a", default="candidate_b")
    parser.add_argument("--method-b", default="candidate_f_artifact_env_irm_ramp")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/symptom_slice_confounding")
    args = parser.parse_args()

    manifest = _prepare_manifest(args.manifest)
    base, source_scores = _load_method_scores(args.target, args.source_method)
    _, a_scores = _load_method_scores(args.target, args.method_a)
    _, b_scores = _load_method_scores(args.target, args.method_b)
    merged = base.merge(manifest, on="recording_id", how="left", suffixes=("", "_manifest"))
    labels = merged["true_label"].reset_index(drop=True)
    groups = merged[args.slice_column].fillna("missing").astype(str).reset_index(drop=True)
    if groups.nunique() != 2:
        raise ValueError(f"Expected binary slice, got {groups.nunique()}")

    # Gate used by current candidate: false -> method_a, true -> method_b.
    gate_scores = source_scores.copy()
    true_mask = groups == "true"
    false_mask = groups == "false"
    gate_scores[false_mask.to_numpy()] = a_scores[false_mask.to_numpy()]
    gate_scores[true_mask.to_numpy()] = b_scores[true_mask.to_numpy()]

    source_auc = _macro_ovr_auc(labels, source_scores)
    a_auc = _macro_ovr_auc(labels, a_scores)
    b_auc = _macro_ovr_auc(labels, b_scores)
    gate_auc = _macro_ovr_auc(labels, gate_scores)

    slice_rows = []
    for group_value in sorted(groups.unique()):
        idx = np.flatnonzero((groups == group_value).to_numpy())
        label_counts = labels.iloc[idx].value_counts().to_dict()
        label_prior = {label: float(label_counts.get(label, 0) / len(idx)) for label in LABELS}
        source_slice_auc = _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), source_scores[idx])
        a_slice_auc = _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), a_scores[idx])
        b_slice_auc = _macro_ovr_auc(labels.iloc[idx].reset_index(drop=True), b_scores[idx])
        gate_method = args.method_b if group_value == "true" else args.method_a
        gate_slice_auc = b_slice_auc if group_value == "true" else a_slice_auc
        slice_rows.append(
            {
                "slice": group_value,
                "n": int(len(idx)),
                "label_counts": label_counts,
                "label_prior": label_prior,
                "source_slice_auc": source_slice_auc,
                f"{args.method_a}_slice_auc": a_slice_auc,
                f"{args.method_b}_slice_auc": b_slice_auc,
                "gate_method": gate_method,
                "gate_slice_auc": gate_slice_auc,
                "gate_delta_vs_source_slice": gate_slice_auc - source_slice_auc,
            }
        )

    feature_true = (groups == "true").astype(float).to_numpy()
    feature_false = (groups == "false").astype(float).to_numpy()
    symptom_true_ovr = _binary_feature_ovr(labels, feature_true)
    symptom_false_ovr = _binary_feature_ovr(labels, feature_false)

    args.out.mkdir(parents=True, exist_ok=True)
    summary = {
        "target": args.target,
        "slice_column": args.slice_column,
        "source_method": args.source_method,
        "gate_false_method": args.method_a,
        "gate_true_method": args.method_b,
        "overall": {
            "source_auc": source_auc,
            args.method_a: a_auc,
            args.method_b: b_auc,
            "gate_auc": gate_auc,
            "gate_delta_vs_source": gate_auc - source_auc,
        },
        "slice_rows": slice_rows,
        "symptom_true_ovr": symptom_true_ovr,
        "symptom_false_ovr": symptom_false_ovr,
    }
    (args.out / "symptom_slice_confounding_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {
                "slice": row["slice"],
                "n": row["n"],
                "source_slice_auc": row["source_slice_auc"],
                "gate_method": row["gate_method"],
                "gate_slice_auc": row["gate_slice_auc"],
                "gate_delta_vs_source_slice": row["gate_delta_vs_source_slice"],
                **{f"prior_{label}": row["label_prior"][label] for label in LABELS},
            }
            for row in slice_rows
        ]
    ).to_csv(args.out / "symptom_slice_confounding_slices.csv", index=False)
    lines = [
        "# Symptom Slice Confounding Audit",
        "",
        "## Overall",
        "",
        "```json",
        json.dumps(summary["overall"], indent=2),
        "```",
        "",
        "## Symptom Metadata OVR",
        "",
        "Symptom true as score:",
        "",
        "```json",
        json.dumps(symptom_true_ovr, indent=2),
        "```",
        "",
        "Symptom false as score:",
        "",
        "```json",
        json.dumps(symptom_false_ovr, indent=2),
        "```",
        "",
        "## Slices",
        "",
        _to_md(pd.read_csv(args.out / "symptom_slice_confounding_slices.csv")),
        "",
    ]
    report = args.out / "SYMPTOM_SLICE_CONFOUNDING.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


def _to_md(df: pd.DataFrame) -> str:
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
