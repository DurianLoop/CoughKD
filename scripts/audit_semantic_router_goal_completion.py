"""Audit whether the long-running ICASSP 2027 objective is actually complete."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_goal_completion"


PATHS = {
    "claim_dossier": RUNS / "semantic_router_claim_dossier" / "semantic_router_claim_dossier.json",
    "submission_readiness": RUNS
    / "semantic_router_submission_readiness"
    / "semantic_router_submission_readiness.json",
    "claim_consistency": RUNS / "semantic_router_claim_consistency" / "semantic_router_claim_consistency.json",
    "claim_upgrade_decision": RUNS
    / "semantic_router_claim_upgrade_decision"
    / "semantic_router_claim_upgrade_decision.json",
    "third_target_success": RUNS
    / "semantic_router_third_target_success"
    / "semantic_router_third_target_success.json",
    "next_action": RUNS / "semantic_router_next_action" / "semantic_router_next_action.json",
    "innovation_fallback": RUNS
    / "semantic_router_innovation_fallback"
    / "semantic_router_innovation_fallback.json",
    "local_dataset_frontier": RUNS
    / "semantic_router_local_dataset_frontier"
    / "semantic_router_local_dataset_frontier.json",
    "local_existing_run_rescout": RUNS
    / "semantic_router_local_existing_run_rescout"
    / "semantic_router_local_existing_run_rescout.json",
    "local_only_innovation_queue": RUNS
    / "semantic_router_local_only_innovation_queue"
    / "semantic_router_local_only_innovation_queue.json",
    "no_inference_frontier": RUNS
    / "semantic_router_no_inference_frontier"
    / "semantic_router_no_inference_frontier.json",
    "existing_dataset_decision": RUNS
    / "semantic_router_existing_dataset_decision"
    / "semantic_router_existing_dataset_decision.json",
    "metadata_field_risk_profile": RUNS
    / "metadata_field_risk_profile"
    / "metadata_field_risk_profile.json",
    "protocol_strength": RUNS
    / "semantic_router_protocol_strength"
    / "semantic_router_protocol_strength.json",
    "branch_controller": RUNS
    / "semantic_router_branch_controller"
    / "semantic_router_branch_controller.json",
    "toscovid_official_split_readiness": RUNS
    / "toscovid_official_split_readiness"
    / "toscovid_official_split_readiness.json",
    "toscovid_official_split_semantic_router": RUNS
    / "toscovid_official_split_semantic_router"
    / "toscovid_official_split_semantic_router.json",
    "toscovid_official_calibration_budget": RUNS
    / "toscovid_official_calibration_budget"
    / "toscovid_official_calibration_budget.json",
    "calib10_decision_selftest": RUNS
    / "toscovid_calib10_decision_selftest"
    / "toscovid_calib10_decision_selftest.json",
    "failure_handoff": RUNS
    / "semantic_router_failure_handoff"
    / "semantic_router_failure_handoff.json",
    "official_status": RUNS / "semantic_router_submission_readiness" / "ICASSP_2027_OFFICIAL_STATUS.md",
    "icassp_page_rule": RUNS / "icassp_page_rule" / "icassp_page_rule.json",
    "novelty_source_ledger": RUNS / "semantic_router_novelty_matrix" / "SEMANTIC_ROUTER_SOURCE_LEDGER.md",
    "collision_watch": RUNS
    / "semantic_router_collision_watch"
    / "semantic_router_collision_watch.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _write_text_atomic(path: Path, text: str) -> None:
    if path.is_file() and path.read_text(encoding="utf-8", errors="replace") == text:
        return
    for _ in range(5):
        try:
            path.write_text(text, encoding="utf-8")
            return
        except PermissionError:
            time.sleep(0.2)
    path.write_text(text, encoding="utf-8")


def _claim_requirement(dossier: dict[str, Any], name: str) -> dict[str, str]:
    for row in dossier.get("requirements", []):
        if row.get("requirement") == name:
            return {
                "status": str(row.get("status", "MISSING")),
                "evidence": str(row.get("evidence", "")),
                "remaining_gap": str(row.get("remaining_gap", "")),
            }
    return {"status": "MISSING", "evidence": "", "remaining_gap": "requirement row missing from claim dossier"}


def _item(name: str, status: str, evidence: str, remaining_gap: str) -> dict[str, str]:
    return {
        "requirement": name,
        "status": status,
        "evidence": evidence,
        "remaining_gap": remaining_gap,
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
        lines.append("| " + " | ".join(row.get(col, "").replace("\n", " ") for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    claim_dossier = _read_json(PATHS["claim_dossier"])
    readiness = _read_json(PATHS["submission_readiness"])
    claim_consistency = _read_json(PATHS["claim_consistency"])
    claim_upgrade = _read_json(PATHS["claim_upgrade_decision"])
    third_target = _read_json(PATHS["third_target_success"])
    next_action = _read_json(PATHS["next_action"])
    innovation_fallback = _read_json(PATHS["innovation_fallback"])
    local_dataset_frontier = _read_json(PATHS["local_dataset_frontier"])
    local_existing_run_rescout = _read_json(PATHS["local_existing_run_rescout"])
    local_only_queue = _read_json(PATHS["local_only_innovation_queue"])
    no_inference_frontier = _read_json(PATHS["no_inference_frontier"])
    existing_dataset_decision = _read_json(PATHS["existing_dataset_decision"])
    metadata_risk = _read_json(PATHS["metadata_field_risk_profile"])
    protocol_strength = _read_json(PATHS["protocol_strength"])
    branch_controller = _read_json(PATHS["branch_controller"])
    toscovid_official_split = _read_json(PATHS["toscovid_official_split_readiness"])
    toscovid_official_router = _read_json(PATHS["toscovid_official_split_semantic_router"])
    toscovid_calibration_budget = _read_json(PATHS["toscovid_official_calibration_budget"])
    calib10_decision_selftest = _read_json(PATHS["calib10_decision_selftest"])
    failure_handoff = _read_json(PATHS["failure_handoff"])
    official_status = _read_text(PATHS["official_status"])
    page_rule = _read_json(PATHS["icassp_page_rule"])
    novelty_ledger = _read_text(PATHS["novelty_source_ledger"])
    collision_watch = _read_json(PATHS["collision_watch"])

    two_target = _claim_requirement(claim_dossier, "Two-target main effect")
    controls = _claim_requirement(claim_dossier, "Negative semantic controls")
    subject_grouped = _claim_requirement(claim_dossier, "Subject-disjoint resampling")
    novelty = _claim_requirement(claim_dossier, "Novelty boundary")
    moderate_label = _claim_requirement(claim_dossier, "Moderate-label boundary")
    mechanism_support = _claim_requirement(claim_dossier, "Mechanism support from existing runs")
    third_scope = _claim_requirement(claim_dossier, "Third-target empirical scope")
    local_frontier = _claim_requirement(claim_dossier, "Local-dataset-only frontier")
    toscovid_official_route = _claim_requirement(claim_dossier, "TosCOVID official-split route")
    format_status = _claim_requirement(claim_dossier, "ICASSP 2027 format status")

    novelty_current = (
        novelty.get("status") == "PASS"
        and "not absolute proof" in novelty.get("evidence", "")
        and "https://arxiv.org/abs/2512.04847" in novelty_ledger
        and "https://www.nature.com/articles/s41746-026-02445-4" in novelty_ledger
        and "https://arxiv.org/abs/2606.02998" in novelty_ledger
        and "https://arxiv.org/abs/2606.09966" in novelty_ledger
        and "https://arxiv.org/abs/2603.15688" in novelty_ledger
        and "https://arxiv.org/abs/2601.07969" in novelty_ledger
        and "https://www.sciencedirect.com/science/article/pii/S0736584526000566" in novelty_ledger
        and "https://openaccess.thecvf.com/content/CVPR2026/papers/Sultana_CoFiDA-M_Concept-Aware_Feature_Modulation_for_Cross-Domain_Adaptation_with_Image-Only_Inference_CVPR_2026_paper.pdf" in novelty_ledger
        and collision_watch.get("verdict") == "NO_DIRECT_COLLISION_FOUND_IN_LATEST_WATCH"
    )
    final_author_kit_verified = "final author kit/template was not located" not in official_status.lower()
    page_rule_ok = page_rule.get("verdict") == "PAGE_RULE_PASS"
    claim_upgrade_allowed = claim_upgrade.get("decision") == "UPGRADE_ALLOWED_AFTER_INTEGRATION"

    requirements = [
        _item(
            "literature_collision_audit",
            "PASS" if novelty_current else "INCOMPLETE",
            novelty.get("evidence", "missing novelty evidence"),
            (
                "Scoped source ledger is current enough for the candidate, but it is not absolute proof."
                if novelty_current
                else "Refresh novelty matrix/source ledger/collision watch before relying on the claim."
            ),
        ),
        _item(
            "claimable_innovation_boundary",
            "PASS" if claim_consistency.get("verdict") == "PASS" and claim_dossier.get("verdict") else "INCOMPLETE",
            f"claim_consistency={claim_consistency.get('verdict', 'missing')}; claim_dossier={claim_dossier.get('verdict', 'missing')}",
            "Keep explicit non-claims; do not broaden beyond metadata-semantics transfer routing.",
        ),
        _item(
            "two_target_effect_size",
            "PASS" if two_target.get("status") == "PASS" else "INCOMPLETE",
            two_target.get("evidence", ""),
            two_target.get("remaining_gap", ""),
        ),
        _item(
            "negative_controls",
            "PASS" if controls.get("status") == "PASS" else "INCOMPLETE",
            controls.get("evidence", ""),
            controls.get("remaining_gap", ""),
        ),
        _item(
            "subject_grouped_evidence",
            "PASS" if subject_grouped.get("status") == "PASS" else "INCOMPLETE",
            subject_grouped.get("evidence", ""),
            subject_grouped.get("remaining_gap", ""),
        ),
        _item(
            "moderate_label_boundary",
            "PASS" if moderate_label.get("status") == "PASS" else "INCOMPLETE",
            moderate_label.get("evidence", ""),
            moderate_label.get("remaining_gap", ""),
        ),
        _item(
            "mechanism_support_not_new_claim",
            "PASS"
            if mechanism_support.get("status") == "PASS"
            and local_existing_run_rescout.get("verdict") == "NO_NEW_LOCAL_CLAIM_FROM_EXISTING_RUNS"
            else "INCOMPLETE",
            mechanism_support.get("evidence", ""),
            mechanism_support.get(
                "remaining_gap",
                "Keep random-slice, slice-balanced, and oracle evidence as mechanism support rather than a separate claim.",
            ),
        ),
        _item(
            "metadata_field_risk_profile",
            "PASS"
            if metadata_risk.get("verdict") == "METADATA_FIELD_RISK_PROFILE_ACTIVE"
            and metadata_risk.get("runs_inference") is False
            and metadata_risk.get("runs_training") is False
            and metadata_risk.get("uses_existing_datasets_only") is True
            and metadata_risk.get("demographic_fields_have_signal") is True
            and metadata_risk.get("symptom_safe_field_profiled") is True
            else "INCOMPLETE",
            (
                f"metadata-risk={metadata_risk.get('verdict', 'missing')}; "
                f"demographic_signal={metadata_risk.get('demographic_fields_have_signal', 'missing')}; "
                f"interpretation={metadata_risk.get('interpretation', 'missing')}"
            ),
            "Use this as mechanism-boundary evidence only; it is not a new main effect.",
        ),
        _item(
            "third_target_validation",
            "PASS" if third_target.get("verdict") == "THIRD_TARGET_SUPPORTS_CLAIM" else "BLOCK",
            f"third_target_success={third_target.get('verdict', 'missing')}; claim-upgrade={claim_upgrade.get('decision', 'missing')}",
            third_scope.get("remaining_gap", "No local third target is ready; keep local stress evidence auxiliary."),
        ),
        _item(
            "local_dataset_frontier",
            "PASS" if local_dataset_frontier.get("verdict") in {"NO_LOCAL_THIRD_TARGET_READY", "LOCAL_THIRD_TARGET_READY"} and local_frontier.get("status") == "PASS" else "INCOMPLETE",
            f"local-frontier={local_dataset_frontier.get('verdict', 'missing')}; {local_frontier.get('evidence', '')}",
            local_frontier.get("remaining_gap", "Refresh local dataset frontier after any new local target is onboarded."),
        ),
        _item(
            "local_only_innovation_queue",
            "PASS"
            if local_only_queue.get("verdict") == "LOCAL_ONLY_QUEUE_ACTIVE"
            and local_only_queue.get("large_external_data") is False
            and bool(local_only_queue.get("queue"))
            else "INCOMPLETE",
            (
                f"local-only queue={local_only_queue.get('verdict', 'missing')}; "
                f"next-action={local_only_queue.get('next_action_state', 'missing')}; "
                f"approval={local_only_queue.get('requires_user_approval', 'missing')}"
            ),
            "Keep the next innovation step on existing datasets unless the user explicitly changes the constraint.",
        ),
        _item(
            "no_inference_alternative_frontier",
            "PASS"
            if no_inference_frontier.get("verdict") == "NO_INFERENCE_FRONTIER_REVIEWED"
            and no_inference_frontier.get("large_external_data") is False
            else "INCOMPLETE",
            (
                f"no-inference frontier={no_inference_frontier.get('verdict', 'missing')}; "
                f"readout={no_inference_frontier.get('claim_readout', 'missing')}"
            ),
            "Do not upgrade prediction-only alternatives unless a new validated route clears the claim gates.",
        ),
        _item(
            "existing_dataset_constraint",
            "PASS"
            if existing_dataset_decision.get("verdict") == "EXISTING_DATASET_DECISION_ACTIVE"
            and existing_dataset_decision.get("decision") == "ASK_BEFORE_CALIB10_INFERENCE"
            and existing_dataset_decision.get("large_external_data") is False
            and existing_dataset_decision.get("approval_required") is True
            and all(existing_dataset_decision.get("ledger_hits", {}).values())
            else "INCOMPLETE",
            (
                f"existing-dataset decision={existing_dataset_decision.get('decision', 'missing')}; "
                f"state={existing_dataset_decision.get('state', 'missing')}; "
                f"readiness={existing_dataset_decision.get('readiness_verdict', 'missing')}"
            ),
            "Keep the loop on existing datasets; ask before the prepared TosCOVID calib10 smoke inference.",
        ),
        _item(
            "protocol_strength_vs_nested_controller",
            "PASS"
            if protocol_strength.get("verdict") == "PROTOCOL_STRENGTH_SUPPORTS_SEMANTIC_ROUTER"
            and protocol_strength.get("paired_semantic_beats_controller") is True
            else "INCOMPLETE",
            (
                f"protocol-strength={protocol_strength.get('verdict', 'missing')}; "
                f"readout={protocol_strength.get('readout', 'missing')}"
            ),
            "This is protocol-strength evidence only; it does not remove the empirical-scope blocker.",
        ),
        _item(
            "branch_controller",
            "PASS"
            if branch_controller.get("verdict") == "BRANCH_CONTROLLER_ACTIVE"
            and branch_controller.get("branch", {}).get("large_external_data") is False
            else "INCOMPLETE",
            (
                f"branch-controller={branch_controller.get('verdict', 'missing')}; "
                f"branch={branch_controller.get('branch', {}).get('branch', 'missing')}; "
                f"decision={branch_controller.get('branch', {}).get('decision', 'missing')}"
            ),
            "Use the branch controller after approval, failure, or success instead of changing the claim post hoc.",
        ),
        _item(
            "calib10_decision_selftest",
            "PASS"
            if calib10_decision_selftest.get("verdict") == "PASS"
            and all(row.get("pass") is True for row in calib10_decision_selftest.get("cases", []))
            else "INCOMPLETE",
            (
                f"calib10 decision self-test={calib10_decision_selftest.get('verdict', 'missing')}; "
                f"cases={len(calib10_decision_selftest.get('cases', []))}"
            ),
            "Keep threshold self-tests passing before using calib10 results for branch decisions.",
        ),
        _item(
            "failure_handoff_preregistered",
            "PASS"
            if failure_handoff.get("verdict")
            in {"FAILURE_HANDOFF_PRE_REGISTERED", "FAILURE_HANDOFF_TRIGGERED"}
            and bool(failure_handoff.get("candidates"))
            else "INCOMPLETE",
            (
                f"failure-handoff={failure_handoff.get('verdict', 'missing')}; "
                f"state={failure_handoff.get('state', 'missing')}"
            ),
            "If calib10 fails, use this handoff rather than post-hoc tuning the failed route.",
        ),
        _item(
            "toscovid_official_split_route",
            "PASS"
            if toscovid_official_split.get("verdict")
            in {"NOT_READY_MISSING_OFFICIAL_TRAIN_PREDICTIONS", "READY_FOR_OFFICIAL_SPLIT_REANALYSIS"}
            and toscovid_official_router.get("verdict")
            in {"NOT_READY_MISSING_OFFICIAL_SPLIT_PREDICTIONS", "READY_OFFICIAL_SPLIT_RESULT_AVAILABLE"}
            and toscovid_calibration_budget.get("verdict")
            in {"NOT_READY_MISSING_CALIBRATION_PREDICTIONS", "READY_WITH_RESULTS"}
            and toscovid_official_route.get("status") == "PASS"
            else "INCOMPLETE",
            f"official-split={toscovid_official_split.get('verdict', 'missing')}; router={toscovid_official_router.get('verdict', 'missing')}; calibration-budget={toscovid_calibration_budget.get('verdict', 'missing')}; {toscovid_official_route.get('evidence', '')}",
            "Ask before running inference; calibration subsets support lower-cost probes before full 4,803-row TosCOVID train calibration.",
        ),
        _item(
            "claim_upgrade_allowed",
            "PASS" if claim_upgrade_allowed else "BLOCK",
            f"claim-upgrade decision is {claim_upgrade.get('decision', 'missing')}",
            "Rerun this gate only after third-target success and paper integration.",
        ),
        _item(
            "fallback_innovation_loop",
            "PASS" if innovation_fallback.get("state") in {"WAIT_FOR_THIRD_TARGET", "FALLBACK_INNOVATION_LOOP_TRIGGERED", "INTEGRATE_SUCCESSFUL_THIRD_TARGET"} else "INCOMPLETE",
            f"innovation-fallback state is {innovation_fallback.get('state', 'missing')}",
            "If no local target can support the claim, use the pre-registered fallback route rather than post-hoc tuning the current claim.",
        ),
        _item(
            "icassp2027_submission_format",
            "PASS" if final_author_kit_verified and format_status.get("status") == "PASS" and page_rule_ok else "WARN",
            (
                f"{format_status.get('evidence', '')} "
                f"page-rule={page_rule.get('verdict', 'missing')}; "
                f"page_count={page_rule.get('page_count', 'missing')}."
            ).strip(),
            (
                "Current 5-page PDF passes the recorded page-rule audit, but final ICASSP 2027 author kit/template must be verified."
                if page_rule_ok
                else format_status.get("remaining_gap", "Final ICASSP 2027 author kit/template must be verified.")
            ),
        ),
    ]

    blocking = [row for row in requirements if row["status"] == "BLOCK"]
    warnings = [row for row in requirements if row["status"] == "WARN"]
    incomplete = [row for row in requirements if row["status"] == "INCOMPLETE"]
    verdict = (
        "COMPLETE"
        if not blocking and not warnings and not incomplete
        else "NOT_COMPLETE"
    )

    payload = {
        "verdict": verdict,
        "requirements": requirements,
        "blocking": blocking,
        "warnings": warnings,
        "incomplete": incomplete,
        "next_action_state": next_action.get("state", "missing"),
        "next_action": next_action.get("next_action", ""),
        "evidence_paths": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        OUT / "semantic_router_goal_completion.json",
        json.dumps(payload, indent=2),
    )

    lines = [
        "# Semantic-Router Goal Completion Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Blocking requirements: `{len(blocking)}`",
        f"- Warnings: `{len(warnings)}`",
        f"- Incomplete requirements: `{len(incomplete)}`",
        f"- Next action state: `{payload['next_action_state']}`",
        f"- Next action: {payload['next_action']}",
        "",
        "## Requirement Audit",
        "",
        _table(requirements),
        "",
        "## Interpretation",
        "",
    ]
    if verdict == "COMPLETE":
        lines.append("The objective is fully evidenced by current artifacts. It is now eligible for final completion marking after human review.")
    else:
        lines.append("The objective is not complete. Current evidence supports a credible two-target candidate plus local micro-stress evidence, but no local third target is ready and final-format verification remains unresolved.")
    lines.append("")
    _write_text_atomic(OUT / "SEMANTIC_ROUTER_GOAL_COMPLETION.md", "\n".join(lines))
    print(json.dumps({"verdict": verdict, "blocking": len(blocking), "warnings": len(warnings), "incomplete": len(incomplete)}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_GOAL_COMPLETION.md")


if __name__ == "__main__":
    main()
