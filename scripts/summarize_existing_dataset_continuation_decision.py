"""Summarize the existing-dataset-only continuation decision.

This is a process guard for the ICASSP innovation loop. It does not train,
download, or run inference. It records whether the next useful step can be done
with the current local datasets and which tempting routes should remain closed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_existing_dataset_decision"

PATHS = {
    "goal_completion": RUNS
    / "semantic_router_goal_completion"
    / "semantic_router_goal_completion.json",
    "claim_dossier": RUNS / "semantic_router_claim_dossier" / "semantic_router_claim_dossier.json",
    "local_only_queue": RUNS
    / "semantic_router_local_only_innovation_queue"
    / "semantic_router_local_only_innovation_queue.json",
    "branch_controller": RUNS
    / "semantic_router_branch_controller"
    / "semantic_router_branch_controller.json",
    "local_dataset_frontier": RUNS
    / "semantic_router_local_dataset_frontier"
    / "semantic_router_local_dataset_frontier.json",
    "no_inference_frontier": RUNS
    / "semantic_router_no_inference_frontier"
    / "semantic_router_no_inference_frontier.json",
    "calib10_decision": RUNS
    / "toscovid_calib10_result_decision"
    / "toscovid_calib10_result_decision.json",
    "submission_readiness": RUNS
    / "semantic_router_submission_readiness"
    / "semantic_router_submission_readiness.json",
    "novelty_source_ledger": RUNS
    / "semantic_router_novelty_matrix"
    / "SEMANTIC_ROUTER_SOURCE_LEDGER.md",
}


LITERATURE_SWEEP = [
    {
        "source": "CoughSense, arXiv:2606.02998",
        "collision_readout": "adjacent_not_direct",
        "why": "Symptom-conditioned and domain-adaptive cough modeling is crowded; no field-semantics router with all/no/inverted semantic controls.",
    },
    {
        "source": "GLoRIA, arXiv:2603.02464",
        "collision_readout": "adjacent_not_direct",
        "why": "Metadata gates audio adaptation parameters, but does not choose between slice-gated transfer and calibration stacking by metadata field safety.",
    },
    {
        "source": "PulmoVec, arXiv:2603.15688",
        "collision_readout": "adjacent_not_direct",
        "why": "Demographic metadata stacking is close to the TosCOVID stacking side and should be cited as crowded, not claimed as new.",
    },
    {
        "source": "AcuLa / BTS / RespiraMFM",
        "collision_readout": "adjacent_not_direct",
        "why": "Metadata-to-text and audio-language respiratory modeling are crowded; the remaining claim is transfer-strategy routing by target field semantics.",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    if cols is None:
        cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines)


def _current_decision(
    goal: dict[str, Any],
    claim: dict[str, Any],
    local_queue: dict[str, Any],
    branch: dict[str, Any],
    local_frontier: dict[str, Any],
    no_inference: dict[str, Any],
    calib10: dict[str, Any],
) -> dict[str, Any]:
    active_branch = branch.get("branch", {})
    state = str(active_branch.get("branch") or calib10.get("state") or local_queue.get("next_action_state") or "")
    approval_required = bool(
        active_branch.get("approval_required")
        or calib10.get("decision", {}).get("approval_required")
        or local_queue.get("requires_user_approval")
    )
    if state == "AWAIT_CALIB10_STABLE_SMOKE":
        decision = "ASK_BEFORE_CALIB10_INFERENCE"
        next_action = "Use the existing TosCOVID calib10 subset only if the user approves the prepared smoke inference."
    elif state == "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE":
        decision = "RESTART_LOCAL_ONLY_LITERATURE_LOOP"
        next_action = "Do not tune the failed route; search for a new hypothesis testable from existing artifacts."
    elif state in {"PROMOTE_TO_NEXT_STABLE_SUBSET", "PROMOTE_TO_FULL_ROUTER"}:
        decision = "ASK_BEFORE_ESCALATING_EXISTING_DATA_INFERENCE"
        next_action = str(active_branch.get("next_step") or calib10.get("decision", {}).get("next_action", ""))
    else:
        decision = "REFRESH_GATES"
        next_action = "Refresh local-only queue, branch controller, no-inference frontier, and readiness gates."

    closed_routes = [
        {
            "route": "new_external_dataset_download",
            "status": "closed_by_user_constraint",
            "reason": "The current loop should use existing local datasets.",
        },
        {
            "route": "training_or_inference_without_approval",
            "status": "closed_by_user_constraint",
            "reason": "The user asked to avoid high-cost actions and approval is required before inference/training.",
        },
        {
            "route": "prediction_only_replacement_claim",
            "status": no_inference.get("verdict", "missing"),
            "reason": no_inference.get(
                "claim_readout",
                "Prediction-only alternatives have not replaced the semantic-router claim.",
            ),
        },
        {
            "route": "local_third_target_claim",
            "status": local_frontier.get("verdict", "missing"),
            "reason": local_frontier.get(
                "next_action",
                "Local third-target evidence is not currently sufficient for a robust main claim.",
            ),
        },
    ]

    return {
        "verdict": "EXISTING_DATASET_DECISION_ACTIVE",
        "decision": decision,
        "next_action": next_action,
        "approval_required": approval_required,
        "large_external_data": False,
        "state": state or "missing",
        "goal_verdict": goal.get("verdict", "missing"),
        "claim_dossier_verdict": claim.get("verdict", "missing"),
        "local_only_queue_verdict": local_queue.get("verdict", "missing"),
        "local_frontier_verdict": local_frontier.get("verdict", "missing"),
        "no_inference_frontier_verdict": no_inference.get("verdict", "missing"),
        "calib10_verdict": calib10.get("verdict", "missing"),
        "closed_routes": closed_routes,
    }


def main() -> None:
    goal = _read_json(PATHS["goal_completion"])
    claim = _read_json(PATHS["claim_dossier"])
    local_queue = _read_json(PATHS["local_only_queue"])
    branch = _read_json(PATHS["branch_controller"])
    local_frontier = _read_json(PATHS["local_dataset_frontier"])
    no_inference = _read_json(PATHS["no_inference_frontier"])
    calib10 = _read_json(PATHS["calib10_decision"])
    readiness = _read_json(PATHS["submission_readiness"])
    ledger = _read_text(PATHS["novelty_source_ledger"])

    decision = _current_decision(
        goal,
        claim,
        local_queue,
        branch,
        local_frontier,
        no_inference,
        calib10,
    )
    ledger_hits = {
        "coughsense": "CoughSense" in ledger,
        "gloria": "GLoRIA" in ledger,
        "pulmovec": "PulmoVec" in ledger,
        "acula": "AcuLa" in ledger,
        "respiramfm": "RespiraMFM" in ledger,
    }
    payload = {
        **decision,
        "readiness_verdict": readiness.get("verdict", "missing"),
        "literature_sweep": LITERATURE_SWEEP,
        "ledger_hits": ledger_hits,
        "evidence_paths": {
            name: str(path.relative_to(ROOT)) for name, path in PATHS.items() if path.is_file()
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_existing_dataset_decision.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Existing-Dataset Continuation Decision",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Decision: `{payload['decision']}`",
        f"- State: `{payload['state']}`",
        f"- Approval required: `{payload['approval_required']}`",
        f"- Large external data: `{payload['large_external_data']}`",
        f"- Goal verdict: `{payload['goal_verdict']}`",
        f"- Claim dossier verdict: `{payload['claim_dossier_verdict']}`",
        f"- Readiness verdict: `{payload['readiness_verdict']}`",
        "",
        "## Next Action",
        "",
        str(payload["next_action"]),
        "",
        "## Closed Routes",
        "",
        _table(payload["closed_routes"], ["route", "status", "reason"]),
        "",
        "## Literature Sweep Readout",
        "",
        _table(LITERATURE_SWEEP, ["source", "collision_readout", "why"]),
        "",
        "## Ledger Hits",
        "",
        _table([{"source": k, "present": v} for k, v in ledger_hits.items()]),
        "",
        "## Interpretation",
        "",
        "The current scientific path remains the narrow field-semantics transfer-router claim. "
        "Under the existing-dataset constraint, the only active experiment is the approval-bound TosCOVID calib10 smoke. "
        "If that experiment is not approved or later fails below the preregistered 1 pp threshold, the loop should restart literature search instead of promoting prediction-only alternatives or tuning the failed route.",
        "",
    ]
    (OUT / "EXISTING_DATASET_CONTINUATION_DECISION.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "verdict": payload["verdict"],
                "decision": payload["decision"],
                "state": payload["state"],
                "approval_required": payload["approval_required"],
            },
            indent=2,
        )
    )
    print(OUT / "EXISTING_DATASET_CONTINUATION_DECISION.md")


if __name__ == "__main__":
    main()
