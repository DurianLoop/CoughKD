"""Decide the next branch after the TosCOVID calib10 stable probe.

This script is intentionally read-only with respect to predictions: it consumes
the calibration-budget gate output and writes a decision artifact. It does not
train, run inference, or download data.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "toscovid_calib10_result_decision"

CALIBRATION_BUDGET = (
    RUNS
    / "toscovid_official_calibration_budget"
    / "toscovid_official_calibration_budget.json"
)
APPROVAL_PACKET = (
    RUNS
    / "toscovid_calib10_stable_probe_approval"
    / "toscovid_calib10_stable_probe_approval.json"
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _calib10_tier(budget: dict[str, Any]) -> dict[str, Any]:
    for row in budget.get("tiers", []):
        if row.get("calibration") == "calib10" and row.get("tier") == "stable_smoke":
            return row
    return {}


def _calib10_results(budget: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in budget.get("results", []):
        if row.get("calibration") == "calib10" and row.get("tier") == "stable_smoke":
            rows.append(row)
    return rows


def _decision(best_delta: float) -> dict[str, Any]:
    if best_delta >= 0.03:
        return {
            "state": "PROMOTE_TO_FULL_ROUTER",
            "verdict": "CALIB10_STABLE_CLEARS_3PT",
            "next_action": "Ask before running router_full for calib10 or the next larger stable subset.",
            "rationale": f"Best calib10 stable official-test delta is {best_delta * 100.0:.2f} pp, clearing the 3 pp escalation threshold.",
            "approval_required": True,
            "large_external_data": False,
        }
    if best_delta >= 0.01:
        return {
            "state": "PROMOTE_TO_NEXT_STABLE_SUBSET",
            "verdict": "CALIB10_STABLE_CLEARS_1PT",
            "next_action": "Ask before running calib20 stable_smoke.",
            "rationale": f"Best calib10 stable official-test delta is {best_delta * 100.0:.2f} pp, enough for the next low-cost subset but not enough for automatic full-router escalation.",
            "approval_required": True,
            "large_external_data": False,
        }
    return {
        "state": "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE",
        "verdict": "CALIB10_STABLE_FAILS_1PT",
        "next_action": "Stop this official-split enhancement route and restart the literature/innovation loop under the local-only constraint.",
        "rationale": f"Best calib10 stable official-test delta is {best_delta * 100.0:.2f} pp, below the 1 pp continuation threshold.",
        "approval_required": False,
        "large_external_data": False,
    }


def build_payload() -> dict[str, Any]:
    budget = _read_json(CALIBRATION_BUDGET)
    approval = _read_json(APPROVAL_PACKET)
    tier = _calib10_tier(budget)
    results = _calib10_results(budget)

    if not budget:
        decision = {
            "state": "REFRESH_CALIBRATION_BUDGET",
            "verdict": "MISSING_CALIBRATION_BUDGET",
            "next_action": "Run scripts/audit_toscovid_official_calibration_budget.py.",
            "rationale": "The calibration-budget artifact is missing.",
            "approval_required": False,
            "large_external_data": False,
        }
        best = {}
    elif not results:
        missing = int(tier.get("missing_prediction_files", 0) or 0)
        decision = {
            "state": "AWAIT_CALIB10_STABLE_SMOKE",
            "verdict": "AWAITING_APPROVED_INFERENCE",
            "next_action": "Run the approved calib10 stable smoke only if the user explicitly approves it.",
            "rationale": f"Calib10 stable has no result rows yet and is missing {missing} prediction files.",
            "approval_required": True,
            "large_external_data": False,
        }
        best = {}
    else:
        best = max(results, key=lambda row: float(row["delta_vs_source"]))
        decision = _decision(float(best["delta_vs_source"]))

    return {
        "verdict": decision["verdict"],
        "state": decision["state"],
        "decision": decision,
        "best_result": best,
        "results": results,
        "tier": tier,
        "approval_packet_verdict": approval.get("verdict", "missing"),
        "approved_command": approval.get(
            "approved_noninteractive_driver",
            "scripts\\run_toscovid_official_calibration_probe_windows.cmd calib10 stable yes",
        ),
        "decision_rule": "below 1 pp: stop route; 1-3 pp: ask for next stable subset; at least 3 pp: ask before full-router escalation",
        "evidence_paths": {
            "calibration_budget": str(CALIBRATION_BUDGET.relative_to(ROOT)),
            "approval_packet": str(APPROVAL_PACKET.relative_to(ROOT)),
        },
    }


def write_report(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toscovid_calib10_result_decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# TosCOVID Calib10 Result Decision",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- State: `{payload['state']}`",
        f"- Approval required: `{payload['decision']['approval_required']}`",
        f"- Large external data: `{payload['decision']['large_external_data']}`",
        f"- Approval packet verdict: `{payload['approval_packet_verdict']}`",
        f"- Decision rule: {payload['decision_rule']}",
        f"- Next action: {payload['decision']['next_action']}",
        f"- Rationale: {payload['decision']['rationale']}",
        "",
        "## Best Result",
        "",
        _table([payload["best_result"]] if payload["best_result"] else []),
        "",
        "## Calib10 Stable Results",
        "",
        _table(payload["results"]),
        "",
        "## Tier Readiness",
        "",
        _table([payload["tier"]] if payload["tier"] else []),
        "",
        "## Approved Command",
        "",
        "```cmd",
        payload["approved_command"],
        "```",
        "",
        "## Boundary",
        "",
        "- This is a branch decision artifact, not a paper claim.",
        "- It reads existing gate outputs only and does not train, infer, or download data.",
        "- Any next compute step still requires explicit user approval.",
        "",
    ]
    (OUT / "TOSCOVID_CALIB10_RESULT_DECISION.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload()
    write_report(payload)
    print(json.dumps({"verdict": payload["verdict"], "state": payload["state"]}, indent=2))
    print(OUT / "TOSCOVID_CALIB10_RESULT_DECISION.md")


if __name__ == "__main__":
    main()
