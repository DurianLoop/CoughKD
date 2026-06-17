from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "manifests" / "toscovid2021_train_external.csv"
OUT_DIR = ROOT / "runs" / "toscovid_official_calibration_subsets"


def _subject_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for subject_id, group in df.groupby("subject_id", sort=True):
        labels = sorted(group["label"].astype(str).unique())
        if len(labels) != 1:
            raise ValueError(f"subject has multiple labels: {subject_id} -> {labels}")
        rows.append(
            {
                "subject_id": subject_id,
                "label": labels[0],
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _select_subjects(subjects: pd.DataFrame, frac: float, rng: np.random.Generator) -> set[str]:
    selected: list[str] = []
    for _, group in subjects.groupby("label", sort=True):
        ids = group["subject_id"].astype(str).to_numpy()
        n = max(1, int(round(len(ids) * frac)))
        chosen = rng.choice(ids, size=min(n, len(ids)), replace=False)
        selected.extend(chosen.tolist())
    return set(selected)


def _write_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--fractions", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.5])
    args = parser.parse_args()

    manifest = args.manifest if args.manifest.is_absolute() else ROOT / args.manifest
    df = pd.read_csv(manifest).fillna("")
    subjects = _subject_table(df)
    rng = np.random.default_rng(args.seed)
    rows: list[dict[str, Any]] = []

    for frac in args.fractions:
        tag = f"calib{int(round(frac * 100)):02d}"
        selected_subjects = _select_subjects(subjects, frac, rng)
        subset = df[df["subject_id"].astype(str).isin(selected_subjects)].copy()
        out_path = ROOT / "manifests" / f"toscovid2021_train_{tag}_external.csv"
        subset.to_csv(out_path, index=False)
        rows.append(
            {
                "tag": tag,
                "fraction": frac,
                "manifest": str(out_path.relative_to(ROOT)),
                "rows": int(len(subset)),
                "subjects": int(subset["subject_id"].nunique()),
                "covid_positive_rows": int((subset["label"] == "covid_positive").sum()),
                "healthy_rows": int((subset["label"] == "healthy").sum()),
                "covid_positive_subjects": int(subjects[subjects["subject_id"].isin(selected_subjects)].query("label == 'covid_positive'").shape[0]),
                "healthy_subjects": int(subjects[subjects["subject_id"].isin(selected_subjects)].query("label == 'healthy'").shape[0]),
            }
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "source_manifest": str(manifest.relative_to(ROOT)),
        "seed": args.seed,
        "fractions": args.fractions,
        "subsets": rows,
        "decision_boundary": "Subject-stratified calibration subsets are for low-cost official-train calibration probes; final paper claims should distinguish subset probes from full official-train calibration.",
    }
    (OUT_DIR / "toscovid_official_calibration_subsets.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TosCOVID Official Calibration Subsets",
        "",
        f"- Source manifest: `{payload['source_manifest']}`",
        f"- Seed: `{args.seed}`",
        "",
        "## Subsets",
        "",
        _write_table(rows),
        "",
        "## Decision Boundary",
        "",
        payload["decision_boundary"],
        "",
    ]
    (OUT_DIR / "TOSCOVID_OFFICIAL_CALIBRATION_SUBSETS.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(OUT_DIR / "TOSCOVID_OFFICIAL_CALIBRATION_SUBSETS.md")


if __name__ == "__main__":
    main()
