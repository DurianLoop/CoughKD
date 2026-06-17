"""Self-test the TosCOVID calib10 branch decision thresholds.

This verifies the control logic around the pre-registered thresholds:

- below 1 pp: stop the official-split enhancement route;
- 1 pp to below 3 pp: ask before the next stable subset;
- at least 3 pp: ask before full-router escalation.

It does not read audio, predictions, manifests, or model checkpoints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from audit_toscovid_calib10_result_decision import _decision


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "toscovid_calib10_decision_selftest"


CASES = [
    {
        "case": "negative_delta",
        "delta": -0.002,
        "expected_state": "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE",
        "expected_verdict": "CALIB10_STABLE_FAILS_1PT",
    },
    {
        "case": "just_below_1pp",
        "delta": 0.0099,
        "expected_state": "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE",
        "expected_verdict": "CALIB10_STABLE_FAILS_1PT",
    },
    {
        "case": "exactly_1pp",
        "delta": 0.01,
        "expected_state": "PROMOTE_TO_NEXT_STABLE_SUBSET",
        "expected_verdict": "CALIB10_STABLE_CLEARS_1PT",
    },
    {
        "case": "middle_between_1pp_and_3pp",
        "delta": 0.02,
        "expected_state": "PROMOTE_TO_NEXT_STABLE_SUBSET",
        "expected_verdict": "CALIB10_STABLE_CLEARS_1PT",
    },
    {
        "case": "just_below_3pp",
        "delta": 0.0299,
        "expected_state": "PROMOTE_TO_NEXT_STABLE_SUBSET",
        "expected_verdict": "CALIB10_STABLE_CLEARS_1PT",
    },
    {
        "case": "exactly_3pp",
        "delta": 0.03,
        "expected_state": "PROMOTE_TO_FULL_ROUTER",
        "expected_verdict": "CALIB10_STABLE_CLEARS_3PT",
    },
    {
        "case": "above_3pp",
        "delta": 0.045,
        "expected_state": "PROMOTE_TO_FULL_ROUTER",
        "expected_verdict": "CALIB10_STABLE_CLEARS_3PT",
    },
]


def _table(rows: list[dict[str, Any]]) -> str:
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


def build_payload() -> dict[str, Any]:
    rows = []
    for case in CASES:
        decision = _decision(float(case["delta"]))
        passed = (
            decision["state"] == case["expected_state"]
            and decision["verdict"] == case["expected_verdict"]
            and decision["large_external_data"] is False
            and (
                decision["approval_required"] is False
                if case["expected_state"] == "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE"
                else decision["approval_required"] is True
            )
        )
        rows.append(
            {
                "case": case["case"],
                "delta": case["delta"],
                "expected_state": case["expected_state"],
                "observed_state": decision["state"],
                "expected_verdict": case["expected_verdict"],
                "observed_verdict": decision["verdict"],
                "approval_required": decision["approval_required"],
                "large_external_data": decision["large_external_data"],
                "pass": passed,
            }
        )
    return {
        "verdict": "PASS" if all(row["pass"] for row in rows) else "BLOCK",
        "cases": rows,
        "decision_rule": "below 1 pp stop; 1-3 pp next stable subset; at least 3 pp full-router escalation request",
        "scope": "logic-only self-test; no inference, training, download, manifest read, or prediction read",
    }


def write_report(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "toscovid_calib10_decision_selftest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# TosCOVID Calib10 Decision Self-Test",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Decision rule: {payload['decision_rule']}",
        f"- Scope: {payload['scope']}",
        "",
        "## Cases",
        "",
        _table(payload["cases"]),
        "",
    ]
    (OUT / "TOSCOVID_CALIB10_DECISION_SELFTEST.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    payload = build_payload()
    write_report(payload)
    print(json.dumps({"verdict": payload["verdict"], "cases": len(payload["cases"])}, indent=2))
    print(OUT / "TOSCOVID_CALIB10_DECISION_SELFTEST.md")


if __name__ == "__main__":
    main()
