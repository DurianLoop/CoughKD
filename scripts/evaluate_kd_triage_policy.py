"""Evaluate simple target-unlabeled KD-Triage policies on completed runs.

The script uses only signals that can be computed without target labels for
selection, then reports the external-label outcome for audit. With the current
single-target setup this is exploratory; it becomes claim-relevant only after
adding another external target such as DiCOVA.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SIGNALS = ROOT / "runs" / "kd_failure_signal_audit" / "signals.csv"
OUT = ROOT / "runs" / "kd_triage_policy_audit"
REFERENCE = "source_only"


def _read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"method", "kind"}:
                    parsed[key] = value
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def _zscore(rows: list[dict[str, Any]], key: str, value: float) -> float:
    vals = [float(row[key]) for row in rows]
    mu = mean(vals)
    var = mean([(item - mu) ** 2 for item in vals])
    sd = math.sqrt(max(var, 1e-12))
    return (value - mu) / sd


def _policy_scores(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    for row in rows:
        if row["method"] == REFERENCE:
            continue
        # Hypothesis from the first signal audit:
        # target_covid_prob and pred_healthy_rate aligned with macro delta, while
        # high entropy and low target confidence indicated weaker transfer.
        score = (
            0.40 * _zscore(rows, "target_covid_prob", row["target_covid_prob"])
            + 0.25 * _zscore(rows, "pred_healthy_rate", row["pred_healthy_rate"])
            + 0.20 * _zscore(rows, "target_confidence", row["target_confidence"])
            - 0.15 * _zscore(rows, "target_entropy", row["target_entropy"])
        )
        scored.append({**row, "triage_score": score})
    return sorted(scored, key=lambda item: float(item["triage_score"]), reverse=True)


def _best(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    return max(rows, key=lambda item: float(item[key]))


def _fmt(value: float) -> str:
    return f"{value:.6f}"


def main() -> None:
    rows = _read_rows(SIGNALS)
    ref = next(row for row in rows if row["method"] == REFERENCE)
    scored = _policy_scores(rows)
    selected = scored[0]
    best_macro = _best(rows, "macro_delta")
    best_covid = _best(rows, "covid_delta")
    always_kd = next(row for row in rows if row["method"] == "kd")

    negative_transfer = [row for row in rows if row["method"] != REFERENCE and float(row["macro_delta"]) < 0]
    policy_negative = float(selected["macro_delta"]) < 0
    report = {
        "reference": REFERENCE,
        "selected_by_triage": selected["method"],
        "selected_macro_delta": selected["macro_delta"],
        "selected_covid_delta": selected["covid_delta"],
        "best_macro_method": best_macro["method"],
        "best_macro_delta": best_macro["macro_delta"],
        "best_covid_method": best_covid["method"],
        "best_covid_delta": best_covid["covid_delta"],
        "always_kd_macro_delta": always_kd["macro_delta"],
        "always_kd_covid_delta": always_kd["covid_delta"],
        "candidate_negative_transfer_rate": len(negative_transfer) / max(1, len(rows) - 1),
        "policy_negative_transfer": policy_negative,
        "ranked_methods": [
            {
                "method": item["method"],
                "score": item["triage_score"],
                "macro_delta": item["macro_delta"],
                "covid_delta": item["covid_delta"],
                "target_covid_prob": item["target_covid_prob"],
                "pred_healthy_rate": item["pred_healthy_rate"],
                "target_confidence": item["target_confidence"],
                "target_entropy": item["target_entropy"],
            }
            for item in scored
        ],
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "triage_policy_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (OUT / "triage_policy_rank.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(report["ranked_methods"][0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(report["ranked_methods"])

    lines = [
        "# KD-Triage Policy Audit",
        "",
        "This is an exploratory single-target audit using COUGHVID only.",
        "The policy uses no target labels for selection, but the outcome is evaluated with target labels.",
        "",
        f"- Reference: `{REFERENCE}`",
        f"- Selected by triage: `{selected['method']}`",
        f"- Selected macro delta: `{_fmt(float(selected['macro_delta']))}`",
        f"- Selected COVID delta: `{_fmt(float(selected['covid_delta']))}`",
        f"- Best macro method after seeing labels: `{best_macro['method']}` (`{_fmt(float(best_macro['macro_delta']))}`)",
        f"- Best COVID method after seeing labels: `{best_covid['method']}` (`{_fmt(float(best_covid['covid_delta']))}`)",
        f"- Always vanilla KD macro delta: `{_fmt(float(always_kd['macro_delta']))}`",
        f"- Candidate negative-transfer rate: `{len(negative_transfer)}/{max(1, len(rows) - 1)}`",
        f"- Triage selected negative transfer: `{policy_negative}`",
        "",
        "## Ranked Methods",
        "",
        "| Rank | Method | Triage score | Macro delta | COVID delta | Target COVID prob | Pred healthy | Confidence | Entropy |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for idx, item in enumerate(scored, start=1):
        lines.append(
            f"| {idx} | {item['method']} | {_fmt(float(item['triage_score']))} | {_fmt(float(item['macro_delta']))} | {_fmt(float(item['covid_delta']))} | {_fmt(float(item['target_covid_prob']))} | {_fmt(float(item['pred_healthy_rate']))} | {_fmt(float(item['target_confidence']))} | {_fmt(float(item['target_entropy']))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- If the top-ranked method also has positive external delta, KD-Triage may be worth validating on DiCOVA.",
            "- If it selects a negative-transfer method, the current unlabeled signals are not reliable enough and the direction should be revised.",
            "- This audit is not claimable until it is tested across at least two external targets or many target shifts.",
        ]
    )
    (OUT / "KD_TRIAGE_POLICY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "KD_TRIAGE_POLICY_AUDIT.md")


if __name__ == "__main__":
    main()
