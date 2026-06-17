"""Summarize the next branch in the local-only ICASSP innovation loop.

This controller is a process guard. It reads existing gates and decides what the
next scientific branch would be after approval, success, or failure. It does
not train, run inference, download data, or edit the paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_branch_controller"


PATHS = {
    "goal_completion": RUNS
    / "semantic_router_goal_completion"
    / "semantic_router_goal_completion.json",
    "claim_dossier": RUNS / "semantic_router_claim_dossier" / "semantic_router_claim_dossier.json",
    "next_action": RUNS / "semantic_router_next_action" / "semantic_router_next_action.json",
    "calibration_budget": RUNS
    / "toscovid_official_calibration_budget"
    / "toscovid_official_calibration_budget.json",
    "calib10_result_decision": RUNS
    / "toscovid_calib10_result_decision"
    / "toscovid_calib10_result_decision.json",
    "local_only_queue": RUNS
    / "semantic_router_local_only_innovation_queue"
    / "semantic_router_local_only_innovation_queue.json",
    "no_inference_frontier": RUNS
    / "semantic_router_no_inference_frontier"
    / "semantic_router_no_inference_frontier.json",
}


LITERATURE_RISKS = [
    {
        "source": "DHAuDS, arXiv:2511.18421",
        "url": "https://arxiv.org/abs/2511.18421",
        "risk": "Audio TTA and domain-shift benchmarking are active, so generic test-time adaptation is crowded.",
        "impact": "Do not restart with a plain audio TTA claim unless a cough-specific mechanism and local evidence clear the gate.",
    },
    {
        "source": "Audio TTA under background noise, arXiv:2507.15523",
        "url": "https://arxiv.org/abs/2507.15523",
        "risk": "Audio classification TTA under noise already compares TTT, TENT, and CoNMix-style methods.",
        "impact": "Plain confidence/entropy adaptation is not a strong novelty route.",
    },
    {
        "source": "EMO-TTA, arXiv:2509.25495",
        "url": "https://arxiv.org/abs/2509.25495",
        "risk": "Training-free statistical audio-language adaptation is also adjacent.",
        "impact": "Prediction-distribution correction needs more than a posterior transform to become claimable.",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _find_command(calibration_budget: dict[str, Any], calibration: str, tier: str) -> str:
    for row in calibration_budget.get("commands", []):
        if row.get("calibration") == calibration and row.get("tier") == tier:
            return str(row.get("command", ""))
    return ""


def _branch(
    calibration_budget: dict[str, Any],
    next_action: dict[str, Any],
    calib10_decision: dict[str, Any],
) -> dict[str, Any]:
    recommendation = calibration_budget.get("recommendation", {})
    state = str(
        calib10_decision.get("state")
        or recommendation.get("state")
        or next_action.get("state")
        or "missing"
    )
    decision = calib10_decision.get("decision", {})

    if state == "AWAIT_CALIB10_STABLE_SMOKE":
        return {
            "branch": state,
            "decision": "await_user_approval",
            "next_step": "Run the already prepared TosCOVID calib10 stable smoke only if explicitly approved.",
            "approval_required": True,
            "large_external_data": False,
            "command": next_action.get("command", ""),
            "failure_policy": decision.get(
                "rationale",
                "If the approved smoke later falls below 1 pp, stop this official-split enhancement route and restart literature search under the local-only constraint.",
            ),
        }

    if state == "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE":
        return {
            "branch": state,
            "decision": "restart_literature_loop_local_only",
            "next_step": "Do not tune the semantic router post hoc. Start a new local-only literature hypothesis search and require a pre-registered prediction-only or tiny-approved-inference gate.",
            "approval_required": False,
            "large_external_data": False,
            "command": "",
            "failure_policy": decision.get(
                "rationale",
                "Prediction-only alternatives remain negative frontier or mechanism support unless a new validated mechanism clears both large targets.",
            ),
        }

    if state == "PROMOTE_TO_NEXT_STABLE_SUBSET":
        rationale = str(recommendation.get("rationale", ""))
        calibration = "calib20"
        for candidate in ["calib20", "calib30", "calib50", "full_train"]:
            if candidate in str(recommendation.get("next_action", "")):
                calibration = candidate
                break
        return {
            "branch": state,
            "decision": "ask_before_next_low_cost_subset",
            "next_step": str(decision.get("next_action") or recommendation.get("next_action", "")),
            "approval_required": True,
            "large_external_data": False,
            "command": _find_command(calibration_budget, calibration, "stable_smoke"),
            "rationale": str(decision.get("rationale") or rationale),
        }

    if state == "PROMOTE_TO_FULL_ROUTER":
        return {
            "branch": state,
            "decision": "ask_before_full_router",
            "next_step": str(decision.get("next_action") or recommendation.get("next_action", "")),
            "approval_required": True,
            "large_external_data": False,
            "command": _find_command(calibration_budget, "calib10", "router_full"),
            "rationale": str(decision.get("rationale") or recommendation.get("rationale", "")),
        }

    return {
        "branch": state,
        "decision": "refresh_gates",
        "next_step": "Refresh next-action, calibration-budget, local-only queue, and no-inference frontier before choosing another branch.",
        "approval_required": False,
        "large_external_data": False,
        "command": "",
    }


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


def main() -> None:
    goal = _read_json(PATHS["goal_completion"])
    claim = _read_json(PATHS["claim_dossier"])
    next_action = _read_json(PATHS["next_action"])
    calibration_budget = _read_json(PATHS["calibration_budget"])
    calib10_decision = _read_json(PATHS["calib10_result_decision"])
    local_queue = _read_json(PATHS["local_only_queue"])
    no_inference = _read_json(PATHS["no_inference_frontier"])
    branch = _branch(calibration_budget, next_action, calib10_decision)

    forbidden = [
        "Do not download new datasets in the current local-only loop.",
        "Do not train or run inference without explicit approval.",
        "Do not promote prediction-only alternatives rejected by the no-inference frontier.",
        "Do not mark the full ICASSP objective complete while goal completion is NOT_COMPLETE.",
    ]
    payload = {
        "verdict": "BRANCH_CONTROLLER_ACTIVE",
        "goal_verdict": goal.get("verdict", "missing"),
        "claim_dossier_verdict": claim.get("verdict", "missing"),
        "local_only_queue_verdict": local_queue.get("verdict", "missing"),
        "no_inference_frontier_verdict": no_inference.get("verdict", "missing"),
        "calibration_recommendation": calibration_budget.get("recommendation", {}),
        "calib10_result_decision": calib10_decision,
        "branch": branch,
        "literature_risks": LITERATURE_RISKS,
        "forbidden": forbidden,
        "evidence_paths": {
            name: str(path.relative_to(ROOT)) for name, path in PATHS.items() if path.is_file()
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_branch_controller.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    branch_rows = [{k: v for k, v in branch.items() if k != "failure_policy"}]
    lines = [
        "# Semantic-Router Branch Controller",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Goal verdict: `{payload['goal_verdict']}`",
        f"- Claim dossier verdict: `{payload['claim_dossier_verdict']}`",
        f"- Local-only queue verdict: `{payload['local_only_queue_verdict']}`",
        f"- No-inference frontier verdict: `{payload['no_inference_frontier_verdict']}`",
        "",
        "## Active Branch",
        "",
        _table(branch_rows),
        "",
    ]
    if branch.get("failure_policy"):
        lines.extend(["## Failure Policy", "", str(branch["failure_policy"]), ""])
    lines.extend(
        [
            "## Literature Risk Anchors",
            "",
            _table(LITERATURE_RISKS),
            "",
            "## Forbidden",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in forbidden)
    lines.append("")
    (OUT / "SEMANTIC_ROUTER_BRANCH_CONTROLLER.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(json.dumps({k: payload[k] for k in ["verdict", "goal_verdict"]}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_BRANCH_CONTROLLER.md")


if __name__ == "__main__":
    main()
