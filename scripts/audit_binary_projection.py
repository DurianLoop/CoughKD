from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.metrics import average_precision, roc_auc


DEFAULT_RUNS = {
    "source_only": ROOT / "runs/external_toscovid2021_test_source_only_seed7/predictions.csv",
    "candidate_e_tga": ROOT / "runs/external_toscovid2021_test_candidate_e_tga_seed7/predictions.csv",
    "candidate_f_artifact_irm": ROOT / "runs/external_toscovid2021_test_candidate_f_artifact_env_irm_ramp_seed7/predictions.csv",
}


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _audit(path: Path) -> dict[str, float]:
    rows = _read_rows(path)
    labels = [1 if row["true_label"] == "covid_positive" else 0 for row in rows]
    covid = [float(row["prob_covid_positive"]) for row in rows]
    healthy = [float(row["prob_healthy"]) for row in rows]
    projected = [pos / max(1e-12, pos + neg) for pos, neg in zip(covid, healthy)]
    margin = [pos - neg for pos, neg in zip(covid, healthy)]
    positives = sum(labels)
    negatives = len(labels) - positives
    return {
        "n": len(rows),
        "positive": positives,
        "negative": negatives,
        "raw_covid_auc": roc_auc(labels, covid),
        "healthy_inverse_auc": roc_auc([1 - value for value in labels], healthy),
        "projected_auc": roc_auc(labels, projected),
        "margin_auc": roc_auc(labels, margin),
        "projected_ap": average_precision(labels, projected),
        "mean_projected_positive": sum(score for score, label in zip(projected, labels) if label) / max(1, positives),
        "mean_projected_negative": sum(score for score, label in zip(projected, labels) if not label) / max(1, negatives),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/target_failure_slice_audit_seed7/binary_projection_audit.json")
    args = parser.parse_args()

    result = {name: _audit(path) for name, path in DEFAULT_RUNS.items()}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
