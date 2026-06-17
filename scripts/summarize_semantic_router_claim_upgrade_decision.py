"""Pre-register claim-upgrade actions from the UKCOVID third-target gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_claim_upgrade_decision"


PATHS = {
    "third_target_success": RUNS
    / "semantic_router_third_target_success"
    / "semantic_router_third_target_success.json",
    "next_action": RUNS / "semantic_router_next_action" / "semantic_router_next_action.json",
    "claim_dossier": RUNS / "semantic_router_claim_dossier" / "semantic_router_claim_dossier.json",
    "submission_readiness": RUNS
    / "semantic_router_submission_readiness"
    / "semantic_router_submission_readiness.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _decision(third_target_verdict: str) -> dict[str, Any]:
    if third_target_verdict == "THIRD_TARGET_SUPPORTS_CLAIM":
        return {
            "decision": "UPGRADE_ALLOWED_AFTER_INTEGRATION",
            "claim_level": "third-target-supported semantic-router claim",
            "paper_action": (
                "Integrate UKCOVID main and control results into the main paper, supplement, "
                "claim dossier, and submission-readiness gate before describing the claim as third-target supported."
            ),
            "forbidden_until_done": [
                "Do not call the paper final-ready until the main text, supplement, and readiness artifacts are refreshed.",
                "Do not omit split-tail risk or subject-grouped UKCOVID settings.",
            ],
        }
    if third_target_verdict == "THIRD_TARGET_DOES_NOT_SUPPORT_CLAIM":
        return {
            "decision": "DO_NOT_UPGRADE_PIVOT_OR_DOWNGRADE",
            "claim_level": "two-target credible candidate only, or pivot to a new innovation loop",
            "paper_action": (
                "Do not upgrade the semantic-router claim. Either keep it as a two-target limitation-bounded finding, "
                "or start a new literature-and-experiment loop for a different innovation."
            ),
            "forbidden_until_done": [
                "Do not report the semantic router as generalized to UKCOVID.",
                "Do not tune thresholds post hoc against UKCOVID without a new pre-registration note.",
            ],
        }
    return {
        "decision": "NO_UPGRADE_YET",
        "claim_level": "credible two-target candidate, not final-ready",
        "paper_action": (
            "Keep the current conservative claim wording. The next material action is to obtain UKCOVID audio, "
            "run the pre-registered subject-grouped audits, and then rerun this decision gate."
        ),
        "forbidden_until_done": [
            "Do not claim third-target support.",
            "Do not mark the ICASSP objective complete.",
            "Do not remove the empirical-scope limitation.",
        ],
    }


def _table(rows: list[dict[str, str]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(col, "") for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    loaded = {name: _read_json(path) for name, path in PATHS.items()}
    third_target_verdict = str(loaded["third_target_success"].get("verdict", "missing"))
    next_action_state = str(loaded["next_action"].get("state", "missing"))
    claim_dossier_verdict = str(loaded["claim_dossier"].get("verdict", "missing"))
    readiness_verdict = str(loaded["submission_readiness"].get("verdict", "missing"))
    decision = _decision(third_target_verdict)

    payload = {
        **decision,
        "third_target_verdict": third_target_verdict,
        "next_action_state": next_action_state,
        "claim_dossier_verdict": claim_dossier_verdict,
        "submission_readiness_verdict": readiness_verdict,
        "evidence_paths": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_claim_upgrade_decision.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    gate_rows = [
        {"gate": "third_target_success", "verdict": third_target_verdict},
        {"gate": "next_action", "verdict": next_action_state},
        {"gate": "claim_dossier", "verdict": claim_dossier_verdict},
        {"gate": "submission_readiness", "verdict": readiness_verdict},
    ]
    lines = [
        "# Semantic-Router Claim Upgrade Decision",
        "",
        f"- Decision: `{payload['decision']}`",
        f"- Claim level: `{payload['claim_level']}`",
        f"- Paper action: {payload['paper_action']}",
        "",
        "## Gate Inputs",
        "",
        _table(gate_rows),
        "",
        "## Forbidden Until Done",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["forbidden_until_done"])
    lines.append("")
    (OUT / "SEMANTIC_ROUTER_CLAIM_UPGRADE_DECISION.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: payload[k] for k in ["decision", "claim_level", "third_target_verdict"]}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_CLAIM_UPGRADE_DECISION.md")


if __name__ == "__main__":
    main()
