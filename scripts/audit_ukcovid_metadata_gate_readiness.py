from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


ROOT = Path(__file__).resolve().parents[1]


def _symptom_keys(symptoms: pd.Series) -> list[str]:
    keys = set()
    for raw in symptoms.fillna("").astype(str):
        for part in raw.split(";"):
            if "=" not in part:
                continue
            key, _ = part.split("=", 1)
            if key.startswith("symptom_"):
                keys.add(key)
    return sorted(keys)


def _symptom_feature(symptoms: pd.Series, key: str) -> np.ndarray:
    pattern = re.compile(rf"(?:^|;){re.escape(key)}=([^;]+)")
    out = []
    for raw in symptoms.fillna("").astype(str):
        match = pattern.search(raw)
        if not match:
            out.append(0.0)
            continue
        value = match.group(1).strip().lower()
        try:
            out.append(1.0 if float(value) != 0.0 else 0.0)
        except ValueError:
            out.append(0.0 if value in {"false", "none", "nan", ""} else 1.0)
    return np.asarray(out, dtype=float)


def _binary_auc(labels: pd.Series, score: np.ndarray) -> float | None:
    y = (labels.to_numpy() == "covid_positive").astype(int)
    if len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return None
    return float(roc_auc_score(y, score))


def _slice_prior(labels: pd.Series, mask: np.ndarray) -> dict[str, float | int]:
    n = int(mask.sum())
    if n == 0:
        return {"n": 0, "covid_positive": 0.0, "healthy": 0.0}
    sub = labels.iloc[np.flatnonzero(mask)]
    counts = sub.value_counts()
    return {
        "n": n,
        "covid_positive": float(counts.get("covid_positive", 0) / n),
        "healthy": float(counts.get("healthy", 0) / n),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "manifests/ukcovid_open_test_external.csv")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/ukcovid_metadata_gate_readiness")
    parser.add_argument("--min-slice-subjects", type=int, default=200)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest)
    labels = df["label"]
    subjects = df["subject_id"]
    rows = []
    for key in _symptom_keys(df["symptoms"]):
        feature = _symptom_feature(df["symptoms"], key)
        true_mask = feature > 0.5
        false_mask = ~true_mask
        true_subjects = int(subjects[true_mask].nunique())
        false_subjects = int(subjects[false_mask].nunique())
        rows.append(
            {
                "feature": key,
                "kind": "symptom",
                "auc_covid_positive": _binary_auc(labels, feature),
                "true_clips": int(true_mask.sum()),
                "false_clips": int(false_mask.sum()),
                "true_subjects": true_subjects,
                "false_subjects": false_subjects,
                "min_subjects": min(true_subjects, false_subjects),
                "ready_min_subjects": min(true_subjects, false_subjects) >= args.min_slice_subjects,
                "true_covid_prior": _slice_prior(labels, true_mask)["covid_positive"],
                "false_covid_prior": _slice_prior(labels, false_mask)["covid_positive"],
                "prior_gap_true_minus_false": _slice_prior(labels, true_mask)["covid_positive"]
                - _slice_prior(labels, false_mask)["covid_positive"],
            }
        )

    for column in ["age", "sex"]:
        for value, group in df.groupby(column, dropna=False):
            mask = (df[column].fillna("missing").astype(str) == str(value)).to_numpy()
            rows.append(
                {
                    "feature": f"{column}={value}",
                    "kind": column,
                    "auc_covid_positive": _binary_auc(labels, mask.astype(float)),
                    "true_clips": int(mask.sum()),
                    "false_clips": int((~mask).sum()),
                    "true_subjects": int(subjects[mask].nunique()),
                    "false_subjects": int(subjects[~mask].nunique()),
                    "min_subjects": min(int(subjects[mask].nunique()), int(subjects[~mask].nunique())),
                    "ready_min_subjects": min(int(subjects[mask].nunique()), int(subjects[~mask].nunique()))
                    >= args.min_slice_subjects,
                    "true_covid_prior": _slice_prior(labels, mask)["covid_positive"],
                    "false_covid_prior": _slice_prior(labels, ~mask)["covid_positive"],
                    "prior_gap_true_minus_false": _slice_prior(labels, mask)["covid_positive"]
                    - _slice_prior(labels, ~mask)["covid_positive"],
                }
            )

    result = pd.DataFrame(rows)
    result = result.sort_values(["ready_min_subjects", "auc_covid_positive"], ascending=[False, False])
    args.out.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out / "ukcovid_metadata_gate_readiness.csv", index=False)
    ready = result[result["ready_min_subjects"]]
    payload = {
        "manifest": str(args.manifest),
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()),
        "label_counts": df["label"].value_counts().to_dict(),
        "min_slice_subjects": args.min_slice_subjects,
        "n_ready_features": int(len(ready)),
        "best_ready_features": ready.head(20).to_dict(orient="records"),
    }
    (args.out / "ukcovid_metadata_gate_readiness_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    lines = [
        "# UKCOVID Metadata Gate Readiness",
        "",
        "This audit uses metadata only. It checks whether UKCOVID can support symptom/demographic slice-gate validation once audio is available.",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps({k: v for k, v in payload.items() if k != "best_ready_features"}, indent=2),
        "```",
        "",
        "## Best Ready Features",
        "",
        _to_md(ready.head(20)),
        "",
    ]
    report = args.out / "UKCOVID_METADATA_GATE_READINESS.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


def _to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No ready rows._"
    cols = [
        "feature",
        "kind",
        "auc_covid_positive",
        "true_subjects",
        "false_subjects",
        "true_covid_prior",
        "false_covid_prior",
        "prior_gap_true_minus_false",
    ]
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
