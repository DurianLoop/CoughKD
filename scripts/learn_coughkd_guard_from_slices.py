"""Leave-one-slice-out stress test for CoughKD-Guard scoring.

The score must use target-unlabeled signals only at selection time. This script
uses labelled outcomes from other COUGHVID slices to fit a tiny linear guard,
then evaluates selection on the held-out slice. This is a proxy for learning a
guard from historical target shifts; it is not a final claim.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "runs" / "coughvid_slice_guard" / "slice_signals.csv"
OUT = ROOT / "runs" / "coughvid_slice_guard_loso"
REFERENCE = "source_only"
FEATURES = [
    "target_covid_prob",
    "pred_healthy_rate",
    "target_confidence",
    "target_entropy",
    "pred_covid_rate",
    "agree_source_only",
    "l1_to_source_only",
    "confidence_gap_source_only",
    "entropy_gap_source_only",
]


def _read_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with IN.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"slice", "method"}:
                    parsed[key] = value
                elif value == "":
                    parsed[key] = None
                else:
                    parsed[key] = float(value)
            if parsed["method"] != REFERENCE and parsed.get("macro_delta") is not None:
                rows.append(parsed)
    return rows


def _zscore_params(rows: list[dict[str, Any]], feature: str) -> tuple[float, float]:
    values = [float(row[feature]) for row in rows if row.get(feature) is not None and not math.isnan(float(row[feature]))]
    mu = mean(values)
    sd = math.sqrt(max(mean([(value - mu) ** 2 for value in values]), 1e-12))
    return mu, sd


def _score(row: dict[str, Any], weights: dict[str, float], params: dict[str, tuple[float, float]]) -> float:
    total = 0.0
    for feature, weight in weights.items():
        value = float(row[feature])
        mu, sd = params[feature]
        total += weight * ((value - mu) / sd)
    return total


def _corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return 0.0
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _candidate_weights() -> list[dict[str, float]]:
    # Keep this intentionally tiny: the guard is a proxy model-selection rule,
    # not a large learner tuned on target labels.
    values = [-1.0, 0.0, 1.0]
    candidates: list[dict[str, float]] = []
    for combo in itertools.product(values, repeat=len(FEATURES)):
        if all(value == 0 for value in combo):
            continue
        weights = dict(zip(FEATURES, combo))
        l1 = sum(abs(value) for value in combo)
        candidates.append({key: value / l1 for key, value in weights.items()})
    return candidates


def _fit_weights(train_rows: list[dict[str, Any]]) -> tuple[dict[str, float], dict[str, tuple[float, float]], float]:
    params = {feature: _zscore_params(train_rows, feature) for feature in FEATURES}
    best_weights: dict[str, float] | None = None
    best_score = -999.0
    for weights in _candidate_weights():
        scores = [_score(row, weights, params) for row in train_rows]
        deltas = [float(row["macro_delta"]) for row in train_rows]
        corr = _corr(scores, deltas)
        # Penalize selecting negative-transfer methods on training slices.
        penalty = 0.0
        for slice_name in sorted({row["slice"] for row in train_rows}):
            slice_rows = [row for row in train_rows if row["slice"] == slice_name]
            selected = max(slice_rows, key=lambda row: _score(row, weights, params))
            if float(selected["macro_delta"]) < 0:
                penalty += 0.25
        objective = corr - penalty
        if objective > best_score:
            best_score = objective
            best_weights = weights
    assert best_weights is not None
    return best_weights, params, best_score


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    rows = _read_rows()
    slices = sorted({row["slice"] for row in rows})
    decisions: list[dict[str, Any]] = []
    for heldout in slices:
        train_rows = [row for row in rows if row["slice"] != heldout]
        test_rows = [row for row in rows if row["slice"] == heldout]
        weights, params, train_objective = _fit_weights(train_rows)
        selected = max(test_rows, key=lambda row: _score(row, weights, params))
        best = max(test_rows, key=lambda row: float(row["macro_delta"]))
        kd = next(row for row in test_rows if row["method"] == "kd")
        decisions.append(
            {
                "heldout_slice": heldout,
                "selected": selected["method"],
                "selected_macro_delta": selected["macro_delta"],
                "best": best["method"],
                "best_macro_delta": best["macro_delta"],
                "kd_macro_delta": kd["macro_delta"],
                "selected_negative_transfer": float(selected["macro_delta"]) < 0,
                "train_objective": train_objective,
                **{f"w_{key}": value for key, value in weights.items()},
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "loso_guard_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    summary = {
        "slices": len(decisions),
        "negative_selected": sum(1 for row in decisions if row["selected_negative_transfer"]),
        "mean_selected_macro_delta": mean(float(row["selected_macro_delta"]) for row in decisions),
        "mean_best_macro_delta": mean(float(row["best_macro_delta"]) for row in decisions),
        "mean_kd_macro_delta": mean(float(row["kd_macro_delta"]) for row in decisions),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Leave-One-Slice-Out CoughKD-Guard Audit",
        "",
        "This is a proxy stress test. It learns a simple linear guard from all but one COUGHVID slice and selects a method on the held-out slice using target-unlabeled signals only.",
        "",
        "## Summary",
        "",
        f"- Slices: `{summary['slices']}`",
        f"- Negative selected: `{summary['negative_selected']}/{summary['slices']}`",
        f"- Mean selected macro delta: `{_fmt(summary['mean_selected_macro_delta'])}`",
        f"- Mean best post-hoc macro delta: `{_fmt(summary['mean_best_macro_delta'])}`",
        f"- Mean vanilla KD macro delta: `{_fmt(summary['mean_kd_macro_delta'])}`",
        "",
        "## Decisions",
        "",
        "| Held-out slice | Selected | Selected delta | Best post-hoc | KD delta | Negative? |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in decisions:
        lines.append(
            f"| {row['heldout_slice']} | {row['selected']} | {_fmt(row['selected_macro_delta'])} | {row['best']} ({_fmt(row['best_macro_delta'])}) | {_fmt(row['kd_macro_delta'])} | {row['selected_negative_transfer']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If this learned guard still collapses to one conservative method or selects negative transfer often, CoughKD-Guard is not yet a strong method claim.",
            "- If it consistently beats vanilla KD and avoids negative transfer, it remains worth validating on an independent external target.",
        ]
    )
    (OUT / "LOSO_GUARD_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "LOSO_GUARD_AUDIT.md")


if __name__ == "__main__":
    main()
