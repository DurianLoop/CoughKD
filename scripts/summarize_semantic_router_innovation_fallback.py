"""Summarize the next innovation-loop action after the UKCOVID gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_innovation_fallback"


PATHS = {
    "third_target_success": RUNS
    / "semantic_router_third_target_success"
    / "semantic_router_third_target_success.json",
    "claim_upgrade_decision": RUNS
    / "semantic_router_claim_upgrade_decision"
    / "semantic_router_claim_upgrade_decision.json",
    "next_action": RUNS / "semantic_router_next_action" / "semantic_router_next_action.json",
    "low_cost_rescout": RUNS
    / "semantic_router_third_target_execution_pack"
    / "LOW_COST_THIRD_TARGET_RESCOUT_20260609.md",
    "external_actions": RUNS / "next_external_state_actions" / "NEXT_EXTERNAL_STATE_ACTIONS.md",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _choose_state(third_target_verdict: str, claim_decision: str) -> dict[str, Any]:
    if third_target_verdict == "THIRD_TARGET_SUPPORTS_CLAIM":
        return {
            "state": "INTEGRATE_SUCCESSFUL_THIRD_TARGET",
            "active_route": "semantic_router_claim_integration",
            "next_action": "Integrate UKCOVID results into the main paper, supplement, claim dossier, and final readiness gates.",
            "fallback_triggered": False,
            "forbidden": [
                "Do not start a new innovation loop before integrating and auditing the successful third-target result.",
            ],
        }
    if third_target_verdict == "THIRD_TARGET_DOES_NOT_SUPPORT_CLAIM":
        return {
            "state": "FALLBACK_INNOVATION_LOOP_TRIGGERED",
            "active_route": "new_literature_and_experiment_loop",
            "next_action": (
                "Do not tune the semantic router post hoc. Start a new scoped innovation loop: "
                "first try HeAR/Cambridge if access is approved; otherwise reframe to a non-COVID cough/task-shift route such as CODA TB."
            ),
            "fallback_triggered": True,
            "forbidden": [
                "Do not claim the semantic router generalizes to UKCOVID.",
                "Do not change UKCOVID thresholds after seeing the negative result without a new pre-registration note.",
                "Do not promote small Virufy/Virufyseg signals to the main claim.",
            ],
        }
    return {
        "state": "WAIT_FOR_THIRD_TARGET",
        "active_route": "ukcovid_pre_registered_validation",
        "next_action": "Obtain UKCOVID audio, run the pre-registered subject-grouped third-target audits, then rerun this fallback gate.",
        "fallback_triggered": False,
        "forbidden": [
            "Do not start a replacement innovation loop before the current UKCOVID gate is actually evaluated or explicitly abandoned.",
            "Do not mark the ICASSP objective complete.",
        ],
    }


def main() -> None:
    third_target = _read_json(PATHS["third_target_success"])
    claim_upgrade = _read_json(PATHS["claim_upgrade_decision"])
    next_action = _read_json(PATHS["next_action"])
    low_cost = _read_text(PATHS["low_cost_rescout"])
    external = _read_text(PATHS["external_actions"])

    third_target_verdict = str(third_target.get("verdict", "missing"))
    claim_decision = str(claim_upgrade.get("decision", "missing"))
    state = _choose_state(third_target_verdict, claim_decision)

    route_evidence = {
        "ukcovid_best_public_third_target": "No low-cost public replacement was found" in low_cost,
        "hear_route_recorded": "HeAR PyTorch frozen embedding gate" in external,
        "opera_route_recorded": "OPERA respiratory foundation gate" in external,
        "small_virufy_not_main_claim": "Use as auxiliary evidence only" in external,
    }
    payload = {
        **state,
        "third_target_verdict": third_target_verdict,
        "claim_upgrade_decision": claim_decision,
        "next_action_state": next_action.get("state", "missing"),
        "route_evidence": route_evidence,
        "evidence_paths": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_innovation_fallback.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# Semantic-Router Innovation Fallback",
        "",
        f"- State: `{payload['state']}`",
        f"- Active route: `{payload['active_route']}`",
        f"- Fallback triggered: `{payload['fallback_triggered']}`",
        f"- Third-target verdict: `{third_target_verdict}`",
        f"- Claim-upgrade decision: `{claim_decision}`",
        f"- Next action: {payload['next_action']}",
        "",
        "## Route Evidence",
        "",
        "| item | present |",
        "| --- | --- |",
    ]
    for key, present in route_evidence.items():
        lines.append(f"| {key} | {present} |")
    lines.extend(["", "## Forbidden", ""])
    lines.extend(f"- {item}" for item in payload["forbidden"])
    lines.append("")
    (OUT / "SEMANTIC_ROUTER_INNOVATION_FALLBACK.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({k: payload[k] for k in ["state", "active_route", "fallback_triggered"]}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_INNOVATION_FALLBACK.md")


if __name__ == "__main__":
    main()
