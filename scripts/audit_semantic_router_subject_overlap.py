from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_metadata_slice_oracle import TARGETS, _prepare_manifest


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "semantic_router_subject_overlap"


def _to_md_table(rows: list[dict[str, object]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col, "")
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _audit_target(target: str, n_repeats: int, outer_train_frac: float, seed: int) -> dict[str, object]:
    manifest = _prepare_manifest(TARGETS[target]["manifest"])
    if "subject_id" not in manifest.columns:
        subjects = manifest["recording_id"].astype(str).to_numpy()
        subject_source = "recording_id_fallback"
    else:
        subjects = manifest["subject_id"].fillna("").astype(str).to_numpy()
        subject_source = "subject_id"
    n = len(subjects)
    unique_subjects = int(pd.Series(subjects).nunique())
    subject_counts = pd.Series(subjects).value_counts()
    multi_subjects = int((subject_counts > 1).sum())
    max_records_per_subject = int(subject_counts.max()) if len(subject_counts) else 0

    rng = np.random.default_rng(seed)
    train_n = int(round(n * outer_train_frac))
    overlaps = []
    train_subjects_n = []
    eval_subjects_n = []
    for _ in range(n_repeats):
        perm = rng.permutation(n)
        train_idx = perm[:train_n]
        eval_idx = perm[train_n:]
        train_subjects = set(subjects[train_idx])
        eval_subjects = set(subjects[eval_idx])
        overlap = train_subjects & eval_subjects
        overlaps.append(len(overlap))
        train_subjects_n.append(len(train_subjects))
        eval_subjects_n.append(len(eval_subjects))

    overlap_arr = np.asarray(overlaps, dtype=float)
    train_subjects_arr = np.asarray(train_subjects_n, dtype=float)
    eval_subjects_arr = np.asarray(eval_subjects_n, dtype=float)
    return {
        "target": target,
        "rows": n,
        "unique_subjects": unique_subjects,
        "subject_source": subject_source,
        "multi_record_subjects": multi_subjects,
        "max_records_per_subject": max_records_per_subject,
        "row_level_split_overlap_mean_subjects": float(overlap_arr.mean()),
        "row_level_split_overlap_q95_subjects": float(np.quantile(overlap_arr, 0.95)),
        "row_level_split_overlap_any_rate": float(np.mean(overlap_arr > 0)),
        "mean_train_subjects": float(train_subjects_arr.mean()),
        "mean_eval_subjects": float(eval_subjects_arr.mean()),
        "risk": "LOW" if multi_subjects == 0 else "HIGH",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", nargs="+", default=["COUGHVID", "TosCOVID", "UKCOVID"])
    parser.add_argument("--n-repeats", type=int, default=1000)
    parser.add_argument("--outer-train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20270609)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()

    rows = [_audit_target(target, args.n_repeats, args.outer_train_frac, args.seed) for target in args.targets]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "semantic_router_subject_overlap.json").write_text(
        json.dumps({"rows": rows, "n_repeats": args.n_repeats, "outer_train_frac": args.outer_train_frac}, indent=2),
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(args.out / "semantic_router_subject_overlap.csv", index=False)

    lines = [
        "# Semantic-router Subject-Overlap Audit",
        "",
        "This audit checks whether the current row-level target calibration/evaluation split can place recordings from the same subject on both sides of the split.",
        "",
        "## Summary",
        "",
        _to_md_table(rows),
        "",
        "## Interpretation",
        "",
    ]
    risky = [row["target"] for row in rows if row["risk"] == "HIGH"]
    if risky:
        lines.append(
            "Row-level semantic-router resampling has potential subject-overlap risk for: "
            + ", ".join(f"`{target}`" for target in risky)
            + ". Treat current results as credible but not subject-disjoint until a grouped resampling audit is run."
        )
    else:
        lines.append("No target has repeated subjects under the current manifest subject identifiers.")
    lines.append("")
    (args.out / "SEMANTIC_ROUTER_SUBJECT_OVERLAP.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "risky_targets": risky}, indent=2))


if __name__ == "__main__":
    main()
