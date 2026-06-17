"""Summarize the local-dataset-only innovation queue for the ICASSP loop.

This gate is deliberately non-experimental: it does not train models, run
inference, or download data. It keeps the next innovation step aligned with the
current user constraint that the project should use existing local datasets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_local_only_innovation_queue"


PATHS = {
    "goal_completion": RUNS
    / "semantic_router_goal_completion"
    / "semantic_router_goal_completion.json",
    "claim_dossier": RUNS / "semantic_router_claim_dossier" / "semantic_router_claim_dossier.json",
    "next_action": RUNS / "semantic_router_next_action" / "semantic_router_next_action.json",
    "local_dataset_frontier": RUNS
    / "semantic_router_local_dataset_frontier"
    / "semantic_router_local_dataset_frontier.json",
    "toscovid_calibration_budget": RUNS
    / "toscovid_official_calibration_budget"
    / "toscovid_official_calibration_budget.json",
    "toscovid_calib10_approval": RUNS
    / "toscovid_calib10_stable_probe_approval"
    / "toscovid_calib10_stable_probe_approval.json",
    "calib10_result_decision": RUNS
    / "toscovid_calib10_result_decision"
    / "toscovid_calib10_result_decision.json",
    "failure_handoff": RUNS
    / "semantic_router_failure_handoff"
    / "semantic_router_failure_handoff.json",
    "novelty_source_ledger": RUNS
    / "semantic_router_novelty_matrix"
    / "SEMANTIC_ROUTER_SOURCE_LEDGER.md",
    "no_inference_frontier": RUNS
    / "semantic_router_no_inference_frontier"
    / "semantic_router_no_inference_frontier.json",
}


LITERATURE_REFRESH = [
    {
        "source": "AcuLa, arXiv:2512.04847",
        "close_to": "medical-audio semantic teacher, structured metadata to clinical text, UK COVID-19 and CoughVID",
        "decision": "cite_as_strong_adjacent_not_collision",
        "reason": "It aligns audio encoders with metadata-derived clinical language; the local claim is transfer-strategy routing by target metadata field safety.",
    },
    {
        "source": "CoughSense, arXiv:2606.02998",
        "close_to": "multi-dataset cough classification, symptom conditioning, domain adaptation",
        "decision": "do_not_pursue_as_novelty",
        "reason": "It occupies the obvious symptom-conditioned cough/foundation-model route, but not transfer-strategy routing by metadata field safety.",
    },
    {
        "source": "GLoRIA, arXiv:2603.02464",
        "close_to": "metadata-gated audio adaptation",
        "decision": "cite_as_adjacent_not_collision",
        "reason": "It gates ASR adaptation parameters with metadata; the local claim is a router between existing transfer strategies.",
    },
    {
        "source": "BTS, arXiv:2406.06786",
        "close_to": "metadata-aided respiratory sound classification",
        "decision": "cite_as_adjacent_not_collision",
        "reason": "It turns metadata into text/context for classification, not a field-semantics safety prior for transfer selection.",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _exists(path: Path) -> bool:
    return path.is_file()


def _queue(
    goal: dict[str, Any],
    claim: dict[str, Any],
    next_action: dict[str, Any],
    local_frontier: dict[str, Any],
    calibration: dict[str, Any],
    approval: dict[str, Any],
    calib10_decision: dict[str, Any],
) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    calib10_state = str(calib10_decision.get("state", ""))

    if next_action.get("state") == "AWAIT_CALIB10_STABLE_SMOKE" or calib10_state == "AWAIT_CALIB10_STABLE_SMOKE":
        queue.append(
            {
                "rank": 1,
                "candidate": "semantic_router_official_calibration_sufficiency",
                "status": "awaiting_user_approval",
                "uses_existing_datasets_only": True,
                "requires_training": False,
                "requires_inference": True,
                "requires_download": False,
                "approval_boundary": approval.get(
                    "approval_boundary",
                    "Runs inference only on 500 existing TosCOVID calib10 audio rows for 9 method/seed jobs; no training and no data download.",
                ),
                "validation_gate": calib10_decision.get(
                    "decision_rule",
                    "calib10 stable smoke; stop below 1 pp, expand stable subsets above 1 pp, consider full-router only above 3 pp",
                ),
                "claim_role": "reviewer-facing rigor upgrade for the current semantic-router claim, not a third-target replacement",
                "command": next_action.get("command", ""),
            }
        )
    elif calib10_state == "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE":
        queue.append(
            {
                "rank": 1,
                "candidate": "restart_literature_loop_local_only",
                "status": "triggered_by_calib10_failure",
                "uses_existing_datasets_only": True,
                "requires_training": False,
                "requires_inference": False,
                "requires_download": False,
                "validation_gate": calib10_decision.get("decision", {}).get("rationale", "calib10 stable failed below 1 pp"),
                "claim_role": "replacement search, not post-hoc tuning",
            }
        )
    elif calib10_state in {"PROMOTE_TO_NEXT_STABLE_SUBSET", "PROMOTE_TO_FULL_ROUTER"}:
        queue.append(
            {
                "rank": 1,
                "candidate": "semantic_router_official_calibration_sufficiency",
                "status": "passed_calib10_requires_next_approval",
                "uses_existing_datasets_only": True,
                "requires_training": False,
                "requires_inference": True,
                "requires_download": False,
                "validation_gate": calib10_decision.get("decision", {}).get("rationale", "calib10 stable passed"),
                "claim_role": "approval-bound reviewer-facing rigor upgrade",
            }
        )

    queue.append(
        {
            "rank": 2,
            "candidate": "semantic_router_current_two_target_claim",
            "status": "credible_not_final_ready",
            "uses_existing_datasets_only": True,
            "requires_training": False,
            "requires_inference": False,
            "requires_download": False,
            "validation_gate": "keep two-target effect, semantic controls, subject grouping, and moderate-label boundary passing",
            "claim_role": claim.get(
                "claim",
                "Target metadata semantics can act as a safety prior for moderate-label transfer routing under cough dataset shift.",
            ),
            "remaining_gap": "not complete until claim-upgrade gate allows final paper claim",
        }
    )

    if local_frontier.get("verdict") == "NO_LOCAL_THIRD_TARGET_READY":
        queue.append(
            {
                "rank": 3,
                "candidate": "local_micro_stress_only",
                "status": "auxiliary_only",
                "uses_existing_datasets_only": True,
                "requires_training": False,
                "requires_inference": False,
                "requires_download": False,
                "validation_gate": "Virufy/Virufyseg may support supplement stress evidence but cannot upgrade the main claim",
                "claim_role": "non-claim mechanism/stress evidence",
                "remaining_gap": "effective independent sample size is too small for a robust third target",
            }
        )

    if calibration.get("recommendation", {}).get("state") == "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE":
        queue.append(
            {
                "rank": 4,
                "candidate": "restart_literature_loop_local_only",
                "status": "trigger_if_calib_route_fails",
                "uses_existing_datasets_only": True,
                "requires_training": False,
                "requires_inference": False,
                "requires_download": False,
                "validation_gate": "new candidate must be testable from existing COUGHVID/TosCOVID/Virufy predictions or tiny approved inference only",
                "claim_role": "replacement search, not post-hoc tuning",
            }
        )

    if goal.get("verdict") == "NOT_COMPLETE":
        queue.append(
            {
                "rank": 99,
                "candidate": "do_not_mark_complete",
                "status": "guardrail",
                "uses_existing_datasets_only": True,
                "requires_training": False,
                "requires_inference": False,
                "requires_download": False,
                "validation_gate": "goal completion must remain NOT_COMPLETE until all blockers clear",
                "claim_role": "process guard",
            }
        )
    return queue


def _table(rows: list[dict[str, Any]], cols: list[str] | None = None) -> str:
    if not rows:
        return "_No rows._"
    if cols is None:
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
    local_frontier = _read_json(PATHS["local_dataset_frontier"])
    calibration = _read_json(PATHS["toscovid_calibration_budget"])
    approval = _read_json(PATHS["toscovid_calib10_approval"])
    calib10_decision = _read_json(PATHS["calib10_result_decision"])
    failure_handoff = _read_json(PATHS["failure_handoff"])
    no_inference_frontier = _read_json(PATHS["no_inference_frontier"])

    queue = _queue(goal, claim, next_action, local_frontier, calibration, approval, calib10_decision)
    if no_inference_frontier.get("verdict") == "NO_INFERENCE_FRONTIER_REVIEWED":
        queue.insert(
            3,
            {
                "rank": 3,
                "candidate": "prediction_only_alternative_frontier",
                "status": "reviewed_no_replacement_claim",
                "uses_existing_datasets_only": True,
                "requires_training": False,
                "requires_inference": False,
                "requires_download": False,
                "validation_gate": "prediction-only ensemble, stacking, prior correction, and oracle routes cannot replace the semantic-router claim",
                "claim_role": "negative frontier and mechanism-support boundary",
            },
        )
        for i, row in enumerate(queue, start=1):
            row["rank"] = i if row.get("candidate") != "do_not_mark_complete" else 99
    payload = {
        "verdict": "LOCAL_ONLY_QUEUE_ACTIVE",
        "user_constraint": "Use existing local datasets; ask before inference/training/downloads.",
        "goal_verdict": goal.get("verdict", "missing"),
        "claim_dossier_verdict": claim.get("verdict", "missing"),
        "next_action_state": next_action.get("state", "missing"),
        "requires_user_approval": bool(next_action.get("requires_user_approval")),
        "large_external_data": bool(next_action.get("large_external_data")),
        "local_frontier_verdict": local_frontier.get("verdict", "missing"),
        "calibration_recommendation": calibration.get("recommendation", {}),
        "calib10_result_decision": calib10_decision,
        "failure_handoff_verdict": failure_handoff.get("verdict", "missing"),
        "failure_handoff_state": failure_handoff.get("state", "missing"),
        "no_inference_frontier_verdict": no_inference_frontier.get("verdict", "missing"),
        "literature_refresh": LITERATURE_REFRESH,
        "queue": queue,
        "evidence_paths": {
            name: str(path.relative_to(ROOT)) for name, path in PATHS.items() if _exists(path)
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_local_only_innovation_queue.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lit_rows = [
        {
            "source": item["source"],
            "close_to": item["close_to"],
            "decision": item["decision"],
            "reason": item["reason"],
        }
        for item in LITERATURE_REFRESH
    ]
    lines = [
        "# Semantic-Router Local-Only Innovation Queue",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- User constraint: {payload['user_constraint']}",
        f"- Goal verdict: `{payload['goal_verdict']}`",
        f"- Claim dossier verdict: `{payload['claim_dossier_verdict']}`",
        f"- Next action state: `{payload['next_action_state']}`",
        f"- Requires user approval: `{payload['requires_user_approval']}`",
        f"- Large external data: `{payload['large_external_data']}`",
        f"- Local frontier verdict: `{payload['local_frontier_verdict']}`",
        f"- Calib10 result decision: `{calib10_decision.get('verdict', 'missing')}`",
        f"- Failure handoff: `{payload['failure_handoff_verdict']}` / `{payload['failure_handoff_state']}`",
        "",
        "## Candidate Queue",
        "",
        _table(
            queue,
            [
                "rank",
                "candidate",
                "status",
                "uses_existing_datasets_only",
                "requires_training",
                "requires_inference",
                "requires_download",
                "claim_role",
            ],
        ),
        "",
        "## Literature Refresh Readout",
        "",
        _table(lit_rows, ["source", "close_to", "decision", "reason"]),
        "",
        "## Decision Rules",
        "",
        "- Do not download new datasets for the current loop.",
        "- Do not train or run inference without explicit approval.",
        "- If calib10 stable smoke is approved and fails below 1 pp, stop the official-split enhancement route and restart the literature loop under the local-only constraint.",
        "- If it clears 1 pp, expand only to the next TosCOVID official calibration subset before considering any full-router spend.",
        "- If it clears 3 pp, treat that as evidence to justify full-router escalation, not as automatic final paper readiness.",
        "",
    ]
    (OUT / "SEMANTIC_ROUTER_LOCAL_ONLY_INNOVATION_QUEUE.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps({k: payload[k] for k in ["verdict", "next_action_state", "requires_user_approval"]}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_LOCAL_ONLY_INNOVATION_QUEUE.md")


if __name__ == "__main__":
    main()
