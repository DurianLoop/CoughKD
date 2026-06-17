from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.metrics import multiclass_ovr_auroc


def _read_predictions(path: Path, id_field: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {row[id_field]: row for row in rows}


def _classes(rows: list[dict[str, str]]) -> list[str]:
    prob_cols = [key for key in rows[0] if key.startswith("prob_")]
    return [key.removeprefix("prob_") for key in prob_cols]


def _scores(rows: list[dict[str, str]], classes: list[str]) -> tuple[list[str], list[list[float]]]:
    labels = [row["true_label"] for row in rows]
    scores = [[float(row[f"prob_{class_name}"]) for class_name in classes] for row in rows]
    return labels, scores


def _macro_auc(rows: list[dict[str, str]], classes: list[str]) -> float:
    labels, scores = _scores(rows, classes)
    return float(multiclass_ovr_auroc(labels, scores, classes)["macro_ovr_auroc"])


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round(q * (len(values) - 1))))
    return float(values[idx])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--id-field", default="recording_id")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    baseline = _read_predictions(args.baseline, args.id_field)
    candidate = _read_predictions(args.candidate, args.id_field)
    ids = sorted(set(baseline) & set(candidate))
    if not ids:
        raise SystemExit("no overlapping prediction ids")
    base_rows = [baseline[item] for item in ids]
    cand_rows = [candidate[item] for item in ids]
    classes = _classes(base_rows)
    point_base = _macro_auc(base_rows, classes)
    point_cand = _macro_auc(cand_rows, classes)
    rng = random.Random(args.seed)
    deltas: list[float] = []
    for _ in range(args.bootstrap):
        sample_ids = [rng.choice(ids) for _ in ids]
        sample_base = [baseline[item] for item in sample_ids]
        sample_cand = [candidate[item] for item in sample_ids]
        try:
            deltas.append(_macro_auc(sample_cand, classes) - _macro_auc(sample_base, classes))
        except ValueError:
            continue
    result: dict[str, Any] = {
        "baseline": str(args.baseline),
        "candidate": str(args.candidate),
        "id_field": args.id_field,
        "n": len(ids),
        "classes": classes,
        "baseline_macro_ovr_auroc": point_base,
        "candidate_macro_ovr_auroc": point_cand,
        "point_delta": point_cand - point_base,
        "bootstrap": {
            "requested": args.bootstrap,
            "valid": len(deltas),
            "mean_delta": sum(deltas) / max(1, len(deltas)),
            "ci95_low": _quantile(deltas, 0.025),
            "ci95_high": _quantile(deltas, 0.975),
            "p_delta_gt_0": sum(1 for value in deltas if value > 0) / max(1, len(deltas)),
            "p_delta_gt_0_03": sum(1 for value in deltas if value > 0.03) / max(1, len(deltas)),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
