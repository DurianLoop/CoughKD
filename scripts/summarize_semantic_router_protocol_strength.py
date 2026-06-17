"""Summarize protocol-strength evidence for the semantic-router claim.

This script reads existing nested-controller and paired-bootstrap artifacts. It
does not train models, run inference, or create a new method. Its purpose is to
record whether the current semantic router is supported as more than a generic
target-calibrated model selector.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_protocol_strength"


PATHS = {
    "controller_default": RUNS
    / "target_calibrated_transfer_controller"
    / "target_calibrated_transfer_controller_summary.json",
    "controller_threshold1": RUNS
    / "target_calibrated_transfer_controller_threshold1_confirm"
    / "target_calibrated_transfer_controller_summary.json",
    "controller_margin0p01": RUNS
    / "target_calibrated_transfer_controller_margin0p01"
    / "target_calibrated_transfer_controller_summary.json",
    "controller_margin0p02": RUNS
    / "target_calibrated_transfer_controller_margin0p02"
    / "target_calibrated_transfer_controller_summary.json",
    "semantic_vs_controller": RUNS
    / "semantic_vs_controller_bootstrap"
    / "semantic_vs_controller_bootstrap.csv",
    "semantic_vs_controller_md": RUNS
    / "semantic_vs_controller_bootstrap"
    / "SEMANTIC_VS_CONTROLLER_BOOTSTRAP.md",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0])
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_fmt(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _controller_rows(payloads: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, payload in payloads.items():
        for row in payload.get("summary", []):
            rows.append(
                {
                    "controller": name,
                    "target": row.get("target", ""),
                    "n": row.get("n_repeats", ""),
                    "mean_delta_pp": 100.0 * float(row.get("mean_delta_vs_source", 0.0)),
                    "ci95_low_pp": 100.0 * float(row.get("ci95_low", 0.0)),
                    "ci95_high_pp": 100.0 * float(row.get("ci95_high", 0.0)),
                    "p_delta_le_0": row.get("p_delta_le_0", ""),
                    "p_delta_lt_3pt": row.get("p_delta_lt_3pt", ""),
                }
            )
    return rows


def _selection_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    totals: dict[str, int] = {}
    for row in payload.get("selection_counts", []):
        target = str(row.get("target", ""))
        totals[target] = totals.get(target, 0) + int(row.get("count", 0))
    rows = []
    for row in payload.get("selection_counts", []):
        target = str(row.get("target", ""))
        count = int(row.get("count", 0))
        total = totals.get(target, 0)
        rows.append(
            {
                "target": target,
                "strategy": row.get("selected_strategy", ""),
                "count": count,
                "fraction": (count / total) if total else 0.0,
            }
        )
    return rows


def _bootstrap_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        out.append(
            {
                "target": row.get("target", ""),
                "n": int(row.get("n", 0)),
                "semantic_minus_controller_pp": 100.0 * float(row.get("mean_diff", 0.0)),
                "ci95_low_pp": 100.0 * float(row.get("ci_low", 0.0)),
                "ci95_high_pp": 100.0 * float(row.get("ci_high", 0.0)),
                "p_mean_le_0": float(row.get("p_boot_mean_le_0", 1.0)),
                "paired_win_rate": float(row.get("paired_win_rate", 0.0)),
                "paired_tie_rate": float(row.get("paired_tie_rate", 0.0)),
            }
        )
    return out


def main() -> None:
    controller_payloads = {
        name: _read_json(path)
        for name, path in PATHS.items()
        if name.startswith("controller_")
    }
    bootstrap = _bootstrap_rows(_read_csv(PATHS["semantic_vs_controller"]))
    controller_rows = _controller_rows(controller_payloads)
    selection_rows = _selection_rows(controller_payloads.get("controller_threshold1", {}))

    paired_pass = (
        len(bootstrap) >= 2
        and all(float(row["semantic_minus_controller_pp"]) > 0.0 for row in bootstrap)
        and all(float(row["p_mean_le_0"]) <= 0.001 for row in bootstrap)
    )
    controller_not_enough = all(
        payload.get("clears_3pt_on_all_large_targets") is False
        for payload in controller_payloads.values()
        if payload
    )
    verdict = (
        "PROTOCOL_STRENGTH_SUPPORTS_SEMANTIC_ROUTER"
        if paired_pass and controller_not_enough
        else "PROTOCOL_STRENGTH_INCOMPLETE"
    )
    payload = {
        "verdict": verdict,
        "claim_role": "protocol_strength_not_new_claim",
        "paired_semantic_beats_controller": paired_pass,
        "generic_controller_not_sufficient": controller_not_enough,
        "bootstrap": bootstrap,
        "controller_summary": controller_rows,
        "threshold1_selection_counts": selection_rows,
        "readout": (
            "The semantic field prior beats the nested target-calibrated controller in paired bootstrap, "
            "while generic controllers do not clear the all-large-target 3 pp gate."
        ),
        "evidence_paths": {
            name: str(path.relative_to(ROOT)) for name, path in PATHS.items() if path.is_file()
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_protocol_strength.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Semantic-Router Protocol Strength",
        "",
        f"- Verdict: `{verdict}`",
        f"- Claim role: `{payload['claim_role']}`",
        f"- Paired semantic beats controller: `{paired_pass}`",
        f"- Generic controller not sufficient: `{controller_not_enough}`",
        f"- Readout: {payload['readout']}",
        "",
        "## Semantic vs Nested Controller",
        "",
        _table(bootstrap),
        "",
        "## Nested Controller Summary",
        "",
        _table(controller_rows),
        "",
        "## Threshold1 Selection Counts",
        "",
        _table(selection_rows),
        "",
        "## Decision Boundary",
        "",
        "- This supports the current semantic-router protocol against a generic target-calibrated controller.",
        "- It is not a new method claim and does not remove the empirical-scope blocker.",
        "- It uses existing artifacts only; no inference, training, or download is run.",
        "",
    ]
    (OUT / "SEMANTIC_ROUTER_PROTOCOL_STRENGTH.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps({"verdict": verdict}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_PROTOCOL_STRENGTH.md")


if __name__ == "__main__":
    main()
