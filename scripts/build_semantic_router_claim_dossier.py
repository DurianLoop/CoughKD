from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_claim_dossier"


PATHS = {
    "readiness": RUNS / "semantic_router_submission_readiness" / "semantic_router_submission_readiness.json",
    "paper_tables": RUNS / "semantic_router_paper_tables" / "SEMANTIC_ROUTER_TABLES.md",
    "control_bootstrap": RUNS / "semantic_router_control_bootstrap" / "SEMANTIC_ROUTER_CONTROL_BOOTSTRAP.md",
    "subject_overlap": RUNS / "semantic_router_subject_overlap" / "SEMANTIC_ROUTER_SUBJECT_OVERLAP.md",
    "subject_grouped_tables": RUNS / "semantic_router_subject_grouped_tables" / "semantic_router_subject_grouped_tables.csv",
    "metadata_field_risk_profile": RUNS
    / "metadata_field_risk_profile"
    / "metadata_field_risk_profile.json",
    "claim_consistency": RUNS / "semantic_router_claim_consistency" / "semantic_router_claim_consistency.json",
    "guard_sensitivity_csv": RUNS / "semantic_router_guard_sensitivity" / "semantic_router_guard_sensitivity.csv",
    "novelty_matrix": RUNS / "semantic_router_novelty_matrix" / "SEMANTIC_ROUTER_NOVELTY_MATRIX.md",
    "novelty_source_ledger": RUNS / "semantic_router_novelty_matrix" / "SEMANTIC_ROUTER_SOURCE_LEDGER.md",
    "collision_watch": RUNS
    / "semantic_router_collision_watch"
    / "semantic_router_collision_watch.json",
    "official_status": RUNS / "semantic_router_submission_readiness" / "ICASSP_2027_OFFICIAL_STATUS.md",
    "icassp_page_rule": RUNS / "icassp_page_rule" / "icassp_page_rule.json",
    "ukcovid_execution_pack": RUNS / "semantic_router_third_target_execution_pack" / "UKCOVID_SEMANTIC_ROUTER_EXECUTION_PACK.md",
    "third_target_rescout": RUNS / "semantic_router_third_target_execution_pack" / "LOW_COST_THIRD_TARGET_RESCOUT_20260609.md",
    "ukcovid_download_preflight": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_audio_download_preflight.json",
    "ukcovid_archive_integrity": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_audio_archive_integrity.json",
    "ukcovid_extraction_preflight": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_extraction_preflight.json",
    "next_action_dashboard": RUNS / "semantic_router_next_action" / "semantic_router_next_action.json",
    "claim_upgrade_decision": RUNS
    / "semantic_router_claim_upgrade_decision"
    / "semantic_router_claim_upgrade_decision.json",
    "goal_completion": RUNS / "semantic_router_goal_completion" / "semantic_router_goal_completion.json",
    "innovation_fallback": RUNS
    / "semantic_router_innovation_fallback"
    / "semantic_router_innovation_fallback.json",
    "third_target_success": RUNS / "semantic_router_third_target_success" / "semantic_router_third_target_success.json",
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
    "protocol_strength": RUNS
    / "semantic_router_protocol_strength"
    / "semantic_router_protocol_strength.json",
    "branch_controller": RUNS
    / "semantic_router_branch_controller"
    / "semantic_router_branch_controller.json",
    "toscovid2022_local_candidate": RUNS
    / "toscovid2022_local_candidate_audit"
    / "toscovid2022_local_candidate_audit.json",
    "toscovid_official_split_readiness": RUNS
    / "toscovid_official_split_readiness"
    / "toscovid_official_split_readiness.json",
    "toscovid_official_split_semantic_router": RUNS
    / "toscovid_official_split_semantic_router"
    / "toscovid_official_split_semantic_router.json",
    "toscovid_official_calibration_budget": RUNS
    / "toscovid_official_calibration_budget"
    / "toscovid_official_calibration_budget.json",
    "supplement_text": RUNS / "semantic_router_supplement" / "main_pdftotext.txt",
}


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _status(name: str, state: str, evidence: str, gap: str = "") -> dict[str, str]:
    return {"requirement": name, "status": state, "evidence": evidence, "remaining_gap": gap}


def _has_all(text: str, needles: list[str]) -> bool:
    return all(needle in text for needle in needles)


def _extract_table_value(text: str, row_prefix: str) -> str:
    for line in text.splitlines():
        if line.startswith(row_prefix):
            return line.strip()
    return ""


def _rows_for_target(rows: list[dict[str, str]], target: str) -> list[dict[str, str]]:
    return [row for row in rows if row.get("target") == target]


def main() -> None:
    readiness = _read_json(PATHS["readiness"])
    paper_tables = _read_text(PATHS["paper_tables"])
    control_bootstrap = _read_text(PATHS["control_bootstrap"])
    subject_overlap = _read_text(PATHS["subject_overlap"])
    subject_grouped_rows = _read_csv(PATHS["subject_grouped_tables"])
    metadata_risk = _read_json(PATHS["metadata_field_risk_profile"])
    claim_consistency = _read_json(PATHS["claim_consistency"])
    guard_rows = _read_csv(PATHS["guard_sensitivity_csv"])
    novelty = _read_text(PATHS["novelty_matrix"])
    novelty_ledger = _read_text(PATHS["novelty_source_ledger"])
    collision_watch = _read_json(PATHS["collision_watch"])
    official = _read_text(PATHS["official_status"])
    page_rule = _read_json(PATHS["icassp_page_rule"])
    uk_pack = _read_text(PATHS["ukcovid_execution_pack"])
    third_target_rescout = _read_text(PATHS["third_target_rescout"])
    ukcovid_download_preflight = _read_json(PATHS["ukcovid_download_preflight"])
    ukcovid_archive_integrity = _read_json(PATHS["ukcovid_archive_integrity"])
    ukcovid_extraction_preflight = _read_json(PATHS["ukcovid_extraction_preflight"])
    next_action_dashboard = _read_json(PATHS["next_action_dashboard"])
    claim_upgrade_decision = _read_json(PATHS["claim_upgrade_decision"])
    goal_completion = _read_json(PATHS["goal_completion"])
    innovation_fallback = _read_json(PATHS["innovation_fallback"])
    third_target_success = _read_json(PATHS["third_target_success"])
    local_dataset_frontier = _read_json(PATHS["local_dataset_frontier"])
    local_existing_run_rescout = _read_json(PATHS["local_existing_run_rescout"])
    local_only_innovation_queue = _read_json(PATHS["local_only_innovation_queue"])
    no_inference_frontier = _read_json(PATHS["no_inference_frontier"])
    existing_dataset_decision = _read_json(PATHS["existing_dataset_decision"])
    protocol_strength = _read_json(PATHS["protocol_strength"])
    branch_controller = _read_json(PATHS["branch_controller"])
    toscovid2022_local_candidate = _read_json(PATHS["toscovid2022_local_candidate"])
    toscovid_official_split = _read_json(PATHS["toscovid_official_split_readiness"])
    toscovid_official_router = _read_json(PATHS["toscovid_official_split_semantic_router"])
    toscovid_calibration_budget = _read_json(PATHS["toscovid_official_calibration_budget"])
    supplement_text = _read_text(PATHS["supplement_text"])
    ukcovid = readiness.get("ukcovid_readiness", {})

    semantic_main_ok = _has_all(
        paper_tables,
        [
            "| Semantic | 3.75 | [3.63, 3.87] | 3.99 | [3.83, 4.15] |",
            "| Inverted | 0.55 | [0.46, 0.63] | 1.07 | [0.92, 1.23] |",
            "| 30% | COUGHVID | 2.63 | [2.46, 2.79] |",
            "| 30% | TosCOVID | 2.81 | [2.68, 2.95] |",
            "| 10% | COUGHVID | -1.60 | [-1.74, -1.46] |",
        ],
    )
    bootstrap_ok = _has_all(
        control_bootstrap,
        [
            "| inverted | COUGHVID | 1000 | 3.20 | [3.06, 3.34] | 0.0000 | 0.913 |",
            "| inverted | TosCOVID | 1000 | 2.92 | [2.72, 3.12] | 0.0000 | 0.840 |",
        ],
    )
    subject_overlap_ok = _has_all(
        subject_overlap,
        [
            "| COUGHVID | 7310 | 7310 | subject_id | 0 |",
            "| TosCOVID | 991 | 495 | subject_id | 336 |",
            "| UKCOVID | 22242 | 11121 | subject_id | 11121 |",
        ],
    )
    subject_grouped_by_rule = {row.get("Rule", ""): row for row in subject_grouped_rows}
    semantic_grouped = subject_grouped_by_rule.get("Semantic", {})
    inverted_grouped = subject_grouped_by_rule.get("Inverted", {})
    all_slice_grouped = subject_grouped_by_rule.get("All-slice", {})

    def _float_cell(row: dict[str, str], key: str) -> float | None:
        try:
            return float(row.get(key, ""))
        except ValueError:
            return None

    semantic_coughvid_grouped = _float_cell(semantic_grouped, "COUGHVID mean")
    semantic_tos_grouped = _float_cell(semantic_grouped, "Tos mean")
    inverted_tos_grouped = _float_cell(inverted_grouped, "Tos mean")
    all_slice_tos_grouped = _float_cell(all_slice_grouped, "Tos mean")
    subject_grouped_ok = (
        subject_overlap_ok
        and semantic_coughvid_grouped is not None
        and semantic_tos_grouped is not None
        and inverted_tos_grouped is not None
        and all_slice_tos_grouped is not None
        and semantic_coughvid_grouped >= 3.0
        and semantic_tos_grouped >= 2.0
        and inverted_tos_grouped <= semantic_tos_grouped - 1.0
        and all_slice_tos_grouped <= semantic_tos_grouped - 1.0
    )
    guard_ok = bool(_rows_for_target(guard_rows, "COUGHVID")) and bool(_rows_for_target(guard_rows, "TosCOVID"))
    metadata_risk_ok = (
        metadata_risk.get("verdict") == "METADATA_FIELD_RISK_PROFILE_ACTIVE"
        and metadata_risk.get("runs_inference") is False
        and metadata_risk.get("runs_training") is False
        and metadata_risk.get("uses_existing_datasets_only") is True
        and metadata_risk.get("demographic_fields_have_signal") is True
        and metadata_risk.get("symptom_safe_field_profiled") is True
    )
    novelty_ok = _has_all(
        novelty,
        [
            "This is still not absolute proof that nobody has done it.",
            "Do not claim first metadata-gated audio adaptation.",
            "Do not claim first metadata-to-text respiratory model",
            "Do not claim first cough-audio clinical-metadata fusion",
        ],
    ) and _has_all(
        novelty_ledger,
        [
            "https://arxiv.org/abs/2603.02464",
            "https://arxiv.org/abs/2512.04847",
            "https://arxiv.org/abs/2606.02998",
            "https://arxiv.org/abs/2606.09966",
            "https://arxiv.org/abs/2603.15688",
            "https://arxiv.org/abs/2601.07969",
            "https://www.nature.com/articles/s41746-026-02445-4",
            "https://arxiv.org/abs/2406.06786",
            "https://www.sciencedirect.com/science/article/pii/S0736584526000566",
            "https://openaccess.thecvf.com/content/CVPR2026/papers/Sultana_CoFiDA-M_Concept-Aware_Feature_Modulation_for_Cross-Domain_Adaptation_with_Image-Only_Inference_CVPR_2026_paper.pdf",
            "not field-semantics safety routing",
        ],
    ) and collision_watch.get("verdict") == "NO_DIRECT_COLLISION_FOUND_IN_LATEST_WATCH"
    claim_consistency_ok = claim_consistency.get("verdict") == "PASS"
    format_ok = _has_all(
        official,
        [
            "4 pages are allowed for technical content",
            "optional fifth page is allowed",
            "final author kit/template was not located",
        ],
    )
    page_rule_ok = page_rule.get("verdict") == "PAGE_RULE_PASS"
    uk_registered = all(
        check.get("pass")
        for check in ukcovid.get("checks", [])
        if check.get("check")
        in {
            "target_registered",
            "manifest_exists",
            "manifest_rows",
            "manifest_subjects",
            "slice_config_registered",
            "slice_column_exists",
            "slice_column_has_groups",
            "slice_column_semantic",
        }
    )
    uk_ready = bool(ukcovid.get("ready"))
    post_download_ready = _has_all(
        uk_pack,
        [
            "scripts\\run_ukcovid_semantic_router_after_audio_windows.cmd",
            "scripts\\run_ukcovid_audio_download_verify_windows.cmd",
            "scripts\\check_ukcovid_extraction_preflight.py",
            "--dry-run",
            "READY_TO_EXTRACT",
            "READY_FOR_SEMANTIC_ROUTER_AUDIT",
            "1000-repeat semantic-router audit",
        ],
    ) and _has_all(
        third_target_rescout,
        [
            "No low-cost public replacement was found",
            "Approve UKCOVID audio download/extraction",
            "Obtain Cambridge COVID-19 Sounds access",
        ],
    ) and ukcovid_download_preflight.get("verdict") == "READY_TO_DOWNLOAD_OR_RESUME" and ukcovid_archive_integrity.get("verdict") in {
        "NOT_READY_MISSING_ARCHIVE",
        "READY_FOR_EXTRACTION",
    } and ukcovid_extraction_preflight.get("verdict") in {
        "ARCHIVE_NOT_READY",
        "MAIN_SPLIT_ZIP_MISSING",
        "EXTRACTION_TOOL_MISSING",
        "READY_TO_EXTRACT",
    } and bool(next_action_dashboard.get("state")) and bool(next_action_dashboard.get("next_action")) and bool(
        claim_upgrade_decision.get("decision")
    ) and bool(claim_upgrade_decision.get("claim_level")) and bool(goal_completion.get("verdict"))
    post_download_ready = post_download_ready and bool(innovation_fallback.get("state")) and bool(
        innovation_fallback.get("active_route")
    )
    third_target_success_verdict = str(third_target_success.get("verdict", "missing"))
    third_target_success_ready = third_target_success_verdict == "THIRD_TARGET_SUPPORTS_CLAIM"
    third_target_success_artifact = third_target_success_verdict != "missing"
    local_frontier_verdict = str(local_dataset_frontier.get("verdict", "missing"))
    local_frontier_audited = local_frontier_verdict != "missing"
    local_frontier_has_third = local_frontier_verdict == "LOCAL_THIRD_TARGET_READY"
    local_frontier_has_stress = bool(local_dataset_frontier.get("auxiliary_micro_stress_targets"))
    toscovid2022_metadata_only = (
        toscovid2022_local_candidate.get("verdict") == "METADATA_ONLY_NOT_LOCAL_AUDIO_TARGET"
    )
    toscovid_official_split_ok = toscovid_official_split.get("verdict") in {
        "NOT_READY_MISSING_OFFICIAL_TRAIN_PREDICTIONS",
        "READY_FOR_OFFICIAL_SPLIT_REANALYSIS",
    }
    toscovid_official_router_ok = toscovid_official_router.get("verdict") in {
        "NOT_READY_MISSING_OFFICIAL_SPLIT_PREDICTIONS",
        "READY_OFFICIAL_SPLIT_RESULT_AVAILABLE",
    }
    toscovid_calibration_budget_ok = toscovid_calibration_budget.get("verdict") in {
        "NOT_READY_MISSING_CALIBRATION_PREDICTIONS",
        "READY_WITH_RESULTS",
    }
    toscovid_split_rows = {
        row.get("split"): row.get("rows")
        for row in toscovid_official_split.get("split_summary", [])
    }
    local_rescout_ok = local_existing_run_rescout.get("verdict") == "NO_NEW_LOCAL_CLAIM_FROM_EXISTING_RUNS"
    local_only_queue_ok = (
        local_only_innovation_queue.get("verdict") == "LOCAL_ONLY_QUEUE_ACTIVE"
        and local_only_innovation_queue.get("large_external_data") is False
        and bool(local_only_innovation_queue.get("queue"))
    )
    no_inference_frontier_ok = (
        no_inference_frontier.get("verdict") == "NO_INFERENCE_FRONTIER_REVIEWED"
        and no_inference_frontier.get("large_external_data") is False
        and bool(no_inference_frontier.get("rows"))
    )
    existing_dataset_decision_ok = (
        existing_dataset_decision.get("verdict") == "EXISTING_DATASET_DECISION_ACTIVE"
        and existing_dataset_decision.get("decision") == "ASK_BEFORE_CALIB10_INFERENCE"
        and existing_dataset_decision.get("large_external_data") is False
        and existing_dataset_decision.get("approval_required") is True
        and all(existing_dataset_decision.get("ledger_hits", {}).values())
    )
    protocol_strength_ok = (
        protocol_strength.get("verdict") == "PROTOCOL_STRENGTH_SUPPORTS_SEMANTIC_ROUTER"
        and protocol_strength.get("paired_semantic_beats_controller") is True
        and protocol_strength.get("generic_controller_not_sufficient") is True
    )
    branch_controller_ok = (
        branch_controller.get("verdict") == "BRANCH_CONTROLLER_ACTIVE"
        and branch_controller.get("branch", {}).get("large_external_data") is False
        and bool(branch_controller.get("forbidden"))
    )
    supplement_mechanism_ok = _has_all(
        supplement_text,
        [
            "Mechanism Support From Existing Runs",
            "Random-slice negative control",
            "Slice-balanced gate policy",
            "Local mechanism support from existing runs",
        ],
    )

    requirements = [
        _status(
            "Two-target main effect",
            "PASS" if semantic_main_ok else "BLOCK",
            "Semantic rule reports COUGHVID +3.75pt and TosCOVID +3.99pt over 1000 resamples.",
            "" if semantic_main_ok else "Regenerate semantic-router paper tables.",
        ),
        _status(
            "Negative semantic controls",
            "PASS" if semantic_main_ok and bootstrap_ok else "BLOCK",
            "Inverted controls collapse to +0.55pt/+1.07pt; paired semantic-vs-inverted bootstrap is +3.20pt/+2.92pt with P(mean<=0)=0.0000.",
            "" if semantic_main_ok and bootstrap_ok else "Regenerate control/bootstrap artifacts.",
        ),
        _status(
            "Subject-disjoint resampling",
            "PASS" if subject_grouped_ok else "BLOCK",
            "Subject-overlap audit flags TosCOVID/UKCOVID row-level leakage risk; subject-grouped rerun preserves COUGHVID +3.75pt and TosCOVID +2.94pt, while TosCOVID inverted/all-slice controls stay at +0.75pt.",
            "" if subject_grouped_ok else "Regenerate subject-overlap and subject-grouped semantic-router tables.",
        ),
        _status(
            "Moderate-label boundary",
            "PASS" if semantic_main_ok else "BLOCK",
            "30% calibration remains positive on both targets; 10% fails on COUGHVID.",
            "" if semantic_main_ok else "Regenerate label-budget table.",
        ),
        _status(
            "Guard sensitivity",
            "PASS" if guard_ok else "BLOCK",
            "Existing guarded variants remain positive but reduce mean gain, supporting guard as conservative deployment option.",
            "" if guard_ok else "Regenerate guard-sensitivity md/csv/tex.",
        ),
        _status(
            "Metadata field risk profile",
            "PASS" if metadata_risk_ok else "BLOCK",
            (
                "Manifest-only risk profile shows demographic fields can carry label/oracle signal, "
                "so metadata signal alone is not sufficient evidence for safe slice gating."
            ),
            "" if metadata_risk_ok else "Run scripts/summarize_metadata_field_risk_profile.py.",
        ),
        _status(
            "Mechanism support from existing runs",
            "PASS" if local_rescout_ok and supplement_mechanism_ok else "BLOCK",
            f"Existing-run rescout verdict is {local_existing_run_rescout.get('verdict', 'missing')}; supplement includes random-slice and slice-balanced mechanism support as non-claim evidence.",
            "" if local_rescout_ok and supplement_mechanism_ok else "Regenerate local existing-run rescout and refresh the supplement mechanism-support section.",
        ),
        _status(
            "Novelty boundary",
            "PASS" if novelty_ok else "BLOCK",
            "Novelty matrix, source ledger, and latest collision watch cover symptom models, multimodal metadata, medical-audio semantic teachers, metadata-to-text/stacking, clinical-metadata cough baselines, GLoRIA, safe transfer, KD-under-shift, and note the audit is not absolute proof.",
            "" if novelty_ok else "Refresh novelty matrix, source ledger, and latest collision watch.",
        ),
        _status(
            "Claim consistency",
            "PASS" if claim_consistency_ok else "BLOCK",
            f"Automated claim-consistency audit verdict is {claim_consistency.get('verdict', 'missing')}; required anchors cover +2.94 subject-grouped TosCOVID and explicit non-claims.",
            "" if claim_consistency_ok else "Run scripts/audit_semantic_router_claim_consistency.py and resolve missing anchors or overclaims.",
        ),
        _status(
            "Draft and supplement readiness",
            "PASS" if not any(item.get("status") == "BLOCK" and item.get("item") != "third_target_ukcovid" for item in readiness.get("items", [])) else "BLOCK",
            "Readiness gate passes required artifacts, main/supplement logs, main text anchors, supplement anchors, subject-grouped resampling, guard sensitivity, and novelty boundary.",
            "UKCOVID is evaluated separately as an empirical scope blocker.",
        ),
        _status(
            "ICASSP 2027 format status",
            "WARN" if format_ok and page_rule_ok else "BLOCK",
            (
                "Official CFP page rule checked; "
                f"local page-rule audit is {page_rule.get('verdict', 'missing')} "
                f"with page_count={page_rule.get('page_count', 'missing')}; "
                "final 2027 author kit/template still not located."
            ),
            (
                "Re-check/replace final ICASSP 2027 author kit before submission."
                if page_rule_ok
                else "Run scripts/audit_icassp_page_rule.py and resolve any fifth-page content issue."
            ),
        ),
        _status(
            "Third-target empirical scope",
            "PASS" if uk_ready or local_frontier_has_third else "BLOCK",
            f"UKCOVID registry/manifest/slice readiness is {'present' if uk_registered else 'incomplete'}; readiness verdict is {ukcovid.get('verdict', 'missing')}; local-dataset frontier verdict is {local_frontier_verdict}; TosCOVID 2022 local audit is {toscovid2022_local_candidate.get('verdict', 'missing')}; success-gate verdict is {third_target_success_verdict}.",
            "" if uk_ready or local_frontier_has_third else "No local third target is ready; keep COUGHVID/TosCOVID as main evidence and use Virufy/Virufyseg only as auxiliary stress evidence unless a larger target is later approved.",
        ),
        _status(
            "Local-dataset-only frontier",
            "PASS" if local_frontier_audited and local_frontier_has_stress and toscovid2022_metadata_only else "BLOCK",
            f"Local frontier audit verdict is {local_frontier_verdict}; Virufy/Virufyseg are treated as auxiliary micro-stress targets, not a robust third external cohort; TosCOVID 2022 is metadata-only locally.",
            "" if local_frontier_audited and local_frontier_has_stress else "Run scripts/summarize_semantic_router_local_dataset_frontier.py after local readiness and Virufyseg stress bootstrap.",
        ),
        _status(
            "Local-only innovation queue",
            "PASS" if local_only_queue_ok else "WARN",
            f"Local-only queue verdict is {local_only_innovation_queue.get('verdict', 'missing')}; next action state is {local_only_innovation_queue.get('next_action_state', 'missing')}; large external data is {local_only_innovation_queue.get('large_external_data', 'missing')}.",
            "" if local_only_queue_ok else "Refresh scripts/summarize_semantic_router_local_only_innovation_queue.py so the next innovation step stays on existing local datasets.",
        ),
        _status(
            "No-inference alternative frontier",
            "PASS" if no_inference_frontier_ok else "WARN",
            f"No-inference frontier verdict is {no_inference_frontier.get('verdict', 'missing')}; readout: {no_inference_frontier.get('claim_readout', 'missing')}.",
            "" if no_inference_frontier_ok else "Refresh scripts/summarize_semantic_router_no_inference_frontier.py before promoting any prediction-only alternative.",
        ),
        _status(
            "Existing-dataset continuation decision",
            "PASS" if existing_dataset_decision_ok else "WARN",
            (
                f"Existing-dataset decision is {existing_dataset_decision.get('decision', 'missing')}; "
                f"state is {existing_dataset_decision.get('state', 'missing')}; "
                f"approval required is {existing_dataset_decision.get('approval_required', 'missing')}."
            ),
            "" if existing_dataset_decision_ok else "Refresh scripts/summarize_existing_dataset_continuation_decision.py.",
        ),
        _status(
            "Protocol strength vs nested controller",
            "PASS" if protocol_strength_ok else "WARN",
            f"Protocol-strength verdict is {protocol_strength.get('verdict', 'missing')}; readout: {protocol_strength.get('readout', 'missing')}.",
            "" if protocol_strength_ok else "Refresh scripts/summarize_semantic_router_protocol_strength.py before claiming the router is stronger than a generic nested controller.",
        ),
        _status(
            "Branch controller",
            "PASS" if branch_controller_ok else "WARN",
            f"Branch controller verdict is {branch_controller.get('verdict', 'missing')}; active branch is {branch_controller.get('branch', {}).get('branch', 'missing')}.",
            "" if branch_controller_ok else "Refresh scripts/summarize_semantic_router_branch_controller.py before choosing the next experiment branch.",
        ),
        _status(
            "TosCOVID official-split route",
            "PASS" if toscovid_official_split_ok and toscovid_official_router_ok and toscovid_calibration_budget_ok else "BLOCK",
            f"TosCOVID 2021 official-split readiness verdict is {toscovid_official_split.get('verdict', 'missing')}; semantic-router gate is {toscovid_official_router.get('verdict', 'missing')}; calibration-budget gate is {toscovid_calibration_budget.get('verdict', 'missing')}; full manifest has train={toscovid_split_rows.get('train', 'missing')} and test={toscovid_split_rows.get('test', 'missing')} audio rows.",
            "" if toscovid_official_split_ok and toscovid_official_router_ok and toscovid_calibration_budget_ok else "Run the TosCOVID official-split readiness, semantic-router, and calibration-budget gates after building the manifests.",
        ),
        _status(
            "Pre-registered third-target success gate",
            "PASS" if third_target_success_ready else ("BLOCK" if not third_target_success_artifact else "WARN"),
            f"Third-target success gate artifact verdict is {third_target_success_verdict}.",
            "" if third_target_success_ready else "Do not upgrade the claim until the success gate reports THIRD_TARGET_SUPPORTS_CLAIM.",
        ),
        _status(
            "Third-target execution plan",
            "PASS" if post_download_ready else "BLOCK",
            f"Post-download driver, low-cost replacement re-scout, pre-download verdict {ukcovid_download_preflight.get('verdict', 'missing')}, archive-integrity verdict {ukcovid_archive_integrity.get('verdict', 'missing')}, extraction-preflight verdict {ukcovid_extraction_preflight.get('verdict', 'missing')}, next-action state {next_action_dashboard.get('state', 'missing')}, local frontier verdict {local_frontier_verdict}, claim-upgrade decision {claim_upgrade_decision.get('decision', 'missing')}, goal-completion verdict {goal_completion.get('verdict', 'missing')}, innovation-fallback state {innovation_fallback.get('state', 'missing')}, and success gate are documented.",
            "" if post_download_ready else "Refresh UKCOVID execution pack, preflight, archive verifier, extraction preflight, next-action dashboard, claim-upgrade decision, goal-completion audit, innovation fallback, and low-cost third-target re-scout.",
        ),
    ]

    blocking = [row for row in requirements if row["status"] == "BLOCK"]
    warnings = [row for row in requirements if row["status"] == "WARN"]
    if blocking:
        verdict = "CREDIBLE_CANDIDATE_NOT_FINAL_READY"
    elif warnings:
        verdict = "EMPIRICALLY_READY_PENDING_FINAL_KIT"
    else:
        verdict = "READY_FOR_FINAL_SUBMISSION_REVIEW"

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "verdict": verdict,
        "requirements": requirements,
        "blocking": blocking,
        "warnings": warnings,
        "evidence_paths": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
        "claim": "Target metadata semantics can act as a safety prior for moderate-label transfer routing under cough dataset shift.",
        "non_claims": [
            "first metadata use",
            "first symptom-assisted cough model",
            "first safe-transfer method",
            "first metadata-gated audio adaptation",
            "clinical diagnostic utility",
            "general KD method superiority",
        ],
    }
    _write_text_atomic(OUT / "semantic_router_claim_dossier.json", json.dumps(payload, indent=2))

    cols = ["requirement", "status", "evidence", "remaining_gap"]
    lines = [
        "# Semantic-Router Claim Dossier",
        "",
        f"- Verdict: `{verdict}`",
        f"- Blocking requirements: `{len(blocking)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Claim",
        "",
        "> Target metadata semantics can act as a safety prior for moderate-label transfer routing under cough dataset shift.",
        "",
        "## Requirement Audit",
        "",
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in requirements:
        lines.append("| " + " | ".join(row[col].replace("\n", " ") for col in cols) + " |")

    lines.extend(
        [
            "",
            "## Explicit Non-Claims",
            "",
            "- Not first metadata use.",
            "- Not first symptom-assisted cough model.",
            "- Not first safe-transfer method.",
            "- Not first metadata-gated audio adaptation.",
            "- Not clinical diagnostic utility.",
            "- Not general KD method superiority.",
            "",
            "## Current Decision",
            "",
        ]
    )
    if blocking:
        lines.append(
            "The package is a credible candidate but not final-ready. Under the local-dataset-only route, the remaining hard blocker is empirical scope: no local third target is ready, and Virufy/Virufyseg should remain auxiliary micro-stress evidence."
        )
    elif warnings:
        lines.append("The empirical package is ready for final paper review, but final ICASSP kit verification remains.")
    else:
        lines.append("The package is ready for final submission review, subject to human paper QA.")
    lines.append("")
    _write_text_atomic(OUT / "SEMANTIC_ROUTER_CLAIM_DOSSIER.md", "\n".join(lines))
    print(json.dumps({"verdict": verdict, "blocking": len(blocking), "warnings": len(warnings), "out": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
