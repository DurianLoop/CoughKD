from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.metrics import multiclass_ovr_auroc


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prob_cols(rows: list[dict[str, str]]) -> list[str]:
    return [key for key in rows[0] if key.startswith("prob_")]


def _load_ensemble(paths: list[Path]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    classes: list[str] | None = None
    for path in paths:
        rows = _read_rows(path)
        if not rows:
            continue
        cols = _prob_cols(rows)
        current_classes = [col.removeprefix("prob_") for col in cols]
        if classes is None:
            classes = current_classes
        elif classes != current_classes:
            raise ValueError(f"class mismatch in {path}: {current_classes} != {classes}")
        for row in rows:
            grouped[row["subject_id"]].append(row)
    if classes is None:
        raise ValueError("no prediction rows found")

    ensemble: dict[str, dict[str, Any]] = {}
    for subject_id, rows in grouped.items():
        labels = {row["true_label"] for row in rows}
        if len(labels) != 1:
            raise ValueError(f"subject {subject_id} has inconsistent labels: {sorted(labels)}")
        probs = []
        for class_name in classes:
            col = f"prob_{class_name}"
            probs.append(sum(float(row[col]) for row in rows) / len(rows))
        ensemble[subject_id] = {
            "true_label": next(iter(labels)),
            "probs": probs,
            "n_rows": len(rows),
        }
    return classes, ensemble


def _macro_auc(subject_ids: list[str], rows: dict[str, dict[str, Any]], classes: list[str]) -> float:
    labels = [str(rows[subject_id]["true_label"]) for subject_id in subject_ids]
    scores = [list(rows[subject_id]["probs"]) for subject_id in subject_ids]
    return float(multiclass_ovr_auroc(labels, scores, classes)["macro_ovr_auroc"])


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return float(ordered[idx])


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row[col]
            values.append(f"{value:.6f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", type=Path, required=True)
    parser.add_argument("--target", required=True)
    parser.add_argument("--baseline-name", default="source_only_seed_ensemble")
    parser.add_argument("--candidate-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    base_classes, baseline = _load_ensemble(args.baseline)
    cand_classes, candidate = _load_ensemble(args.candidate)
    if base_classes != cand_classes:
        raise ValueError(f"class mismatch: {base_classes} != {cand_classes}")
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        raise SystemExit("no overlapping subject ids")

    point_base = _macro_auc(ids, baseline, base_classes)
    point_cand = _macro_auc(ids, candidate, base_classes)

    rng = random.Random(args.seed)
    deltas: list[float] = []
    for _ in range(args.bootstrap):
        sample_ids = [rng.choice(ids) for _ in ids]
        try:
            deltas.append(
                _macro_auc(sample_ids, candidate, base_classes)
                - _macro_auc(sample_ids, baseline, base_classes)
            )
        except ValueError:
            continue

    result = {
        "target": args.target,
        "baseline": args.baseline_name,
        "candidate": args.candidate_name,
        "n_subjects": len(ids),
        "classes": base_classes,
        "n_boot_requested": args.bootstrap,
        "n_boot_valid": len(deltas),
        "baseline_macro_ovr_auroc": point_base,
        "candidate_macro_ovr_auroc": point_cand,
        "point_delta": point_cand - point_base,
        "ci95_low": _quantile(deltas, 0.025),
        "ci95_high": _quantile(deltas, 0.975),
        "p_delta_le_0": sum(1 for value in deltas if value <= 0.0) / max(1, len(deltas)),
        "p_delta_lt_0_03": sum(1 for value in deltas if value < 0.03) / max(1, len(deltas)),
        "baseline_paths": [str(path) for path in args.baseline],
        "candidate_paths": [str(path) for path in args.candidate],
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "subject_ensemble_delta.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    rows = [
        {
            "target": result["target"],
            "candidate": result["candidate"],
            "n_subjects": result["n_subjects"],
            "baseline_auc": result["baseline_macro_ovr_auroc"],
            "candidate_auc": result["candidate_macro_ovr_auroc"],
            "delta": result["point_delta"],
            "ci95_low": result["ci95_low"],
            "ci95_high": result["ci95_high"],
            "p_delta_le_0": result["p_delta_le_0"],
            "p_delta_lt_0_03": result["p_delta_lt_0_03"],
        }
    ]
    lines = [
        f"# Subject-Ensemble Delta: {args.target}",
        "",
        _table(rows),
        "",
        "## Interpretation",
        "",
        "- The unit of resampling is `subject_id`, not clip.",
        "- This is auxiliary stress evidence when the subject count is small; it is not a standalone large-target validation gate.",
        "",
    ]
    (args.out / "SUBJECT_ENSEMBLE_DELTA.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(args.out / "SUBJECT_ENSEMBLE_DELTA.md")


if __name__ == "__main__":
    main()
