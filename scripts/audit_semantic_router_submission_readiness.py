from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_submission_readiness"


REQUIRED_FILES = {
    "icassp_main_tex": ROOT / "paper_icassp2027" / "main.tex",
    "icassp_main_pdf": ROOT / "paper_icassp2027" / "main.pdf",
    "icassp_main_text": ROOT / "paper_icassp2027" / "main_pdftotext.txt",
    "icassp_page_rule_md": RUNS / "icassp_page_rule" / "ICASSP_PAGE_RULE.md",
    "icassp_page_rule_json": RUNS / "icassp_page_rule" / "icassp_page_rule.json",
    "semantic_tables_tex": RUNS / "semantic_router_paper_tables" / "semantic_router_tables.tex",
    "semantic_tables_md": RUNS / "semantic_router_paper_tables" / "SEMANTIC_ROUTER_TABLES.md",
    "semantic_guard_sensitivity_md": RUNS / "semantic_router_guard_sensitivity" / "SEMANTIC_ROUTER_GUARD_SENSITIVITY.md",
    "semantic_guard_sensitivity_csv": RUNS / "semantic_router_guard_sensitivity" / "semantic_router_guard_sensitivity.csv",
    "semantic_guard_sensitivity_tex": RUNS / "semantic_router_guard_sensitivity" / "semantic_router_guard_sensitivity.tex",
    "semantic_subject_overlap_md": RUNS / "semantic_router_subject_overlap" / "SEMANTIC_ROUTER_SUBJECT_OVERLAP.md",
    "semantic_subject_overlap_csv": RUNS / "semantic_router_subject_overlap" / "semantic_router_subject_overlap.csv",
    "semantic_subject_overlap_json": RUNS / "semantic_router_subject_overlap" / "semantic_router_subject_overlap.json",
    "semantic_subject_grouped_tables_md": RUNS / "semantic_router_subject_grouped_tables" / "SEMANTIC_ROUTER_SUBJECT_GROUPED_TABLES.md",
    "semantic_subject_grouped_tables_csv": RUNS / "semantic_router_subject_grouped_tables" / "semantic_router_subject_grouped_tables.csv",
    "semantic_subject_grouped_tables_tex": RUNS / "semantic_router_subject_grouped_tables" / "semantic_router_subject_grouped_tables.tex",
    "semantic_control_bootstrap": RUNS / "semantic_router_control_bootstrap" / "SEMANTIC_ROUTER_CONTROL_BOOTSTRAP.md",
    "semantic_vs_controller_bootstrap": RUNS / "semantic_vs_controller_bootstrap" / "SEMANTIC_VS_CONTROLLER_BOOTSTRAP.md",
    "metadata_field_risk_profile_md": RUNS
    / "metadata_field_risk_profile"
    / "METADATA_FIELD_RISK_PROFILE.md",
    "metadata_field_risk_profile_json": RUNS
    / "metadata_field_risk_profile"
    / "metadata_field_risk_profile.json",
    "novelty_matrix_md": RUNS / "semantic_router_novelty_matrix" / "SEMANTIC_ROUTER_NOVELTY_MATRIX.md",
    "novelty_matrix_tex": RUNS / "semantic_router_novelty_matrix" / "semantic_router_novelty_matrix.tex",
    "novelty_source_ledger": RUNS / "semantic_router_novelty_matrix" / "SEMANTIC_ROUTER_SOURCE_LEDGER.md",
    "collision_watch_md": RUNS
    / "semantic_router_collision_watch"
    / "SEMANTIC_ROUTER_COLLISION_WATCH.md",
    "collision_watch_json": RUNS
    / "semantic_router_collision_watch"
    / "semantic_router_collision_watch.json",
    "supplement_pdf": RUNS / "semantic_router_supplement" / "main.pdf",
    "supplement_text": RUNS / "semantic_router_supplement" / "main_pdftotext.txt",
    "ukcovid_execution_pack": RUNS / "semantic_router_third_target_execution_pack" / "UKCOVID_SEMANTIC_ROUTER_EXECUTION_PACK.md",
    "ukcovid_download_verify_driver": ROOT / "scripts" / "run_ukcovid_audio_download_verify_windows.cmd",
    "ukcovid_extraction_preflight_script": ROOT / "scripts" / "check_ukcovid_extraction_preflight.py",
    "third_target_rescout": RUNS / "semantic_router_third_target_execution_pack" / "LOW_COST_THIRD_TARGET_RESCOUT_20260609.md",
    "ukcovid_download_preflight_md": RUNS / "ukcovid_open_metadata_audit" / "UKCOVID_AUDIO_DOWNLOAD_PREFLIGHT.md",
    "ukcovid_download_preflight_json": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_audio_download_preflight.json",
    "ukcovid_archive_integrity_md": RUNS / "ukcovid_open_metadata_audit" / "UKCOVID_AUDIO_ARCHIVE_INTEGRITY.md",
    "ukcovid_archive_integrity_json": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_audio_archive_integrity.json",
    "ukcovid_extraction_preflight_md": RUNS / "ukcovid_open_metadata_audit" / "UKCOVID_EXTRACTION_PREFLIGHT.md",
    "ukcovid_extraction_preflight_json": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_extraction_preflight.json",
    "next_action_dashboard_md": RUNS / "semantic_router_next_action" / "SEMANTIC_ROUTER_NEXT_ACTION.md",
    "next_action_dashboard_json": RUNS / "semantic_router_next_action" / "semantic_router_next_action.json",
    "claim_upgrade_decision_md": RUNS
    / "semantic_router_claim_upgrade_decision"
    / "SEMANTIC_ROUTER_CLAIM_UPGRADE_DECISION.md",
    "claim_upgrade_decision_json": RUNS
    / "semantic_router_claim_upgrade_decision"
    / "semantic_router_claim_upgrade_decision.json",
    "goal_completion_md": RUNS / "semantic_router_goal_completion" / "SEMANTIC_ROUTER_GOAL_COMPLETION.md",
    "goal_completion_json": RUNS / "semantic_router_goal_completion" / "semantic_router_goal_completion.json",
    "ukcovid_readiness": RUNS / "semantic_router_third_target_readiness" / "ukcovid_semantic_router_readiness.json",
    "third_target_success_md": RUNS / "semantic_router_third_target_success" / "SEMANTIC_ROUTER_THIRD_TARGET_SUCCESS.md",
    "third_target_success_json": RUNS / "semantic_router_third_target_success" / "semantic_router_third_target_success.json",
    "local_dataset_frontier_md": RUNS
    / "semantic_router_local_dataset_frontier"
    / "SEMANTIC_ROUTER_LOCAL_DATASET_FRONTIER.md",
    "local_dataset_frontier_json": RUNS
    / "semantic_router_local_dataset_frontier"
    / "semantic_router_local_dataset_frontier.json",
    "local_virufyseg_stress_tex": RUNS
    / "semantic_router_local_dataset_frontier"
    / "virufyseg_subject_ensemble_stress.tex",
    "local_existing_run_rescout_md": RUNS
    / "semantic_router_local_existing_run_rescout"
    / "SEMANTIC_ROUTER_LOCAL_EXISTING_RUN_RESCOUT.md",
    "local_existing_run_rescout_json": RUNS
    / "semantic_router_local_existing_run_rescout"
    / "semantic_router_local_existing_run_rescout.json",
    "local_only_innovation_queue_md": RUNS
    / "semantic_router_local_only_innovation_queue"
    / "SEMANTIC_ROUTER_LOCAL_ONLY_INNOVATION_QUEUE.md",
    "local_only_innovation_queue_json": RUNS
    / "semantic_router_local_only_innovation_queue"
    / "semantic_router_local_only_innovation_queue.json",
    "no_inference_frontier_md": RUNS
    / "semantic_router_no_inference_frontier"
    / "SEMANTIC_ROUTER_NO_INFERENCE_FRONTIER.md",
    "no_inference_frontier_json": RUNS
    / "semantic_router_no_inference_frontier"
    / "semantic_router_no_inference_frontier.json",
    "existing_dataset_decision_md": RUNS
    / "semantic_router_existing_dataset_decision"
    / "EXISTING_DATASET_CONTINUATION_DECISION.md",
    "existing_dataset_decision_json": RUNS
    / "semantic_router_existing_dataset_decision"
    / "semantic_router_existing_dataset_decision.json",
    "protocol_strength_md": RUNS
    / "semantic_router_protocol_strength"
    / "SEMANTIC_ROUTER_PROTOCOL_STRENGTH.md",
    "protocol_strength_json": RUNS
    / "semantic_router_protocol_strength"
    / "semantic_router_protocol_strength.json",
    "branch_controller_md": RUNS
    / "semantic_router_branch_controller"
    / "SEMANTIC_ROUTER_BRANCH_CONTROLLER.md",
    "branch_controller_json": RUNS
    / "semantic_router_branch_controller"
    / "semantic_router_branch_controller.json",
    "failure_handoff_md": RUNS
    / "semantic_router_failure_handoff"
    / "SEMANTIC_ROUTER_FAILURE_HANDOFF.md",
    "failure_handoff_json": RUNS
    / "semantic_router_failure_handoff"
    / "semantic_router_failure_handoff.json",
    "toscovid2022_local_candidate_md": RUNS
    / "toscovid2022_local_candidate_audit"
    / "TOSCOVID2022_LOCAL_CANDIDATE_AUDIT.md",
    "toscovid2022_local_candidate_json": RUNS
    / "toscovid2022_local_candidate_audit"
    / "toscovid2022_local_candidate_audit.json",
    "toscovid2021_full_manifest": ROOT / "manifests" / "toscovid2021_full_external.csv",
    "toscovid2021_train_manifest": ROOT / "manifests" / "toscovid2021_train_external.csv",
    "toscovid_official_split_readiness_md": RUNS
    / "toscovid_official_split_readiness"
    / "TOSCOVID_OFFICIAL_SPLIT_READINESS.md",
    "toscovid_official_split_readiness_json": RUNS
    / "toscovid_official_split_readiness"
    / "toscovid_official_split_readiness.json",
    "toscovid_official_split_semantic_router_md": RUNS
    / "toscovid_official_split_semantic_router"
    / "TOSCOVID_OFFICIAL_SPLIT_SEMANTIC_ROUTER.md",
    "toscovid_official_split_semantic_router_json": RUNS
    / "toscovid_official_split_semantic_router"
    / "toscovid_official_split_semantic_router.json",
    "toscovid_official_train_inference_driver": ROOT
    / "scripts"
    / "run_toscovid_official_train_inference_windows.cmd",
    "toscovid_official_calibration_subsets_md": RUNS
    / "toscovid_official_calibration_subsets"
    / "TOSCOVID_OFFICIAL_CALIBRATION_SUBSETS.md",
    "toscovid_official_calibration_subsets_json": RUNS
    / "toscovid_official_calibration_subsets"
    / "toscovid_official_calibration_subsets.json",
    "toscovid_official_calibration_budget_md": RUNS
    / "toscovid_official_calibration_budget"
    / "TOSCOVID_OFFICIAL_CALIBRATION_BUDGET.md",
    "toscovid_official_calibration_budget_json": RUNS
    / "toscovid_official_calibration_budget"
    / "toscovid_official_calibration_budget.json",
    "toscovid_official_calibration_probe_driver": ROOT
    / "scripts"
    / "run_toscovid_official_calibration_probe_windows.cmd",
    "toscovid_calib10_probe_approval_md": RUNS
    / "toscovid_calib10_stable_probe_approval"
    / "TOSCOVID_CALIB10_STABLE_PROBE_APPROVAL.md",
    "toscovid_calib10_probe_approval_json": RUNS
    / "toscovid_calib10_stable_probe_approval"
    / "toscovid_calib10_stable_probe_approval.json",
    "toscovid_calib10_result_decision_md": RUNS
    / "toscovid_calib10_result_decision"
    / "TOSCOVID_CALIB10_RESULT_DECISION.md",
    "toscovid_calib10_result_decision_json": RUNS
    / "toscovid_calib10_result_decision"
    / "toscovid_calib10_result_decision.json",
    "toscovid_calib10_decision_selftest_md": RUNS
    / "toscovid_calib10_decision_selftest"
    / "TOSCOVID_CALIB10_DECISION_SELFTEST.md",
    "toscovid_calib10_decision_selftest_json": RUNS
    / "toscovid_calib10_decision_selftest"
    / "toscovid_calib10_decision_selftest.json",
    "icassp2027_official_status": OUT / "ICASSP_2027_OFFICIAL_STATUS.md",
    "claim_dossier_md": RUNS / "semantic_router_claim_dossier" / "SEMANTIC_ROUTER_CLAIM_DOSSIER.md",
    "claim_dossier_json": RUNS / "semantic_router_claim_dossier" / "semantic_router_claim_dossier.json",
    "claim_consistency_md": RUNS / "semantic_router_claim_consistency" / "SEMANTIC_ROUTER_CLAIM_CONSISTENCY.md",
    "claim_consistency_json": RUNS / "semantic_router_claim_consistency" / "semantic_router_claim_consistency.json",
}


MAIN_TEXT_NEEDLES = [
    "semantic-constrained router",
    "target metadata semantics can",
    "+3.75",
    "+3.99",
    "+2.94",
    "subject-grouped Tos",
    "inverted controls collapse",
    "30% target calibration remains useful",
]


SUPPLEMENT_NEEDLES = [
    "Semantic Controls and Label Budget",
    "Subject-Grouped Resampling",
    "Inner Guard Sensitivity",
    "Paired Bootstrap Against Semantic Controls",
    "Mechanism Support From Existing Runs",
    "Random-slice negative control",
    "Local mechanism support",
    "Novelty Boundary Matrix",
    "Local Micro-Stress Evidence",
    "no local third target is ready",
]


def _exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


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


def _claim_consistency_verdict(path: Path) -> str:
    payload = _read_json(path)
    return str(payload.get("verdict", "missing"))


def _log_scan(path: Path) -> dict[str, object]:
    text = _read_text(path)
    patterns = ["Undefined", "LaTeX Warning", "Package .*Warning", "Error", "Overfull"]
    hits = []
    for pattern in patterns:
        if re.search(pattern, text):
            hits.append(pattern)
    return {"path": str(path.relative_to(ROOT)), "clean": not hits, "hits": hits}


def _needle_status(path: Path, needles: list[str]) -> dict[str, object]:
    text = _read_text(path)
    found = {needle: (needle in text) for needle in needles}
    return {"path": str(path.relative_to(ROOT)), "all_found": all(found.values()), "found": found}


def _status_item(name: str, status: str, detail: str) -> dict[str, str]:
    return {"item": name, "status": status, "detail": detail}


def _markdown_table(rows: list[dict[str, str]]) -> str:
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
    file_status = {name: _exists(path) for name, path in REQUIRED_FILES.items()}
    main_text_status = _needle_status(REQUIRED_FILES["icassp_main_text"], MAIN_TEXT_NEEDLES)
    supplement_text_status = _needle_status(REQUIRED_FILES["supplement_text"], SUPPLEMENT_NEEDLES)
    main_log_status = _log_scan(ROOT / "paper_icassp2027" / "main.log")
    supplement_log_status = _log_scan(RUNS / "semantic_router_supplement" / "main.log")
    ukcovid = _read_json(REQUIRED_FILES["ukcovid_readiness"])
    claim_consistency_verdict = _claim_consistency_verdict(REQUIRED_FILES["claim_consistency_json"])
    page_rule = _read_json(REQUIRED_FILES["icassp_page_rule_json"])

    items: list[dict[str, str]] = []
    missing = [name for name, ok in file_status.items() if not ok]
    items.append(
        _status_item(
            "required_artifacts",
            "PASS" if not missing else "BLOCK",
            "all required artifacts present" if not missing else "missing: " + ", ".join(missing),
        )
    )
    items.append(
        _status_item(
            "main_pdf_compile_log",
            "PASS" if main_log_status["clean"] else "BLOCK",
            "clean" if main_log_status["clean"] else "hits: " + ", ".join(main_log_status["hits"]),
        )
    )
    items.append(
        _status_item(
            "supplement_compile_log",
            "PASS" if supplement_log_status["clean"] else "BLOCK",
            "clean" if supplement_log_status["clean"] else "hits: " + ", ".join(supplement_log_status["hits"]),
        )
    )
    items.append(
        _status_item(
            "main_claim_text",
            "PASS" if main_text_status["all_found"] else "BLOCK",
            "main PDF text contains semantic-router claim anchors",
        )
    )
    page_rule_ok = (
        file_status.get("icassp_page_rule_md")
        and file_status.get("icassp_page_rule_json")
        and page_rule.get("verdict") == "PAGE_RULE_PASS"
    )
    items.append(
        _status_item(
            "icassp_page_rule",
            "PASS" if page_rule_ok else "BLOCK",
            (
                f"page-rule verdict is {page_rule.get('verdict', 'missing')}; "
                f"page_count={page_rule.get('page_count', 'missing')}; "
                f"page5_forbidden_hits={len(page_rule.get('page5_forbidden_hits', []))}"
            ),
        )
    )
    items.append(
        _status_item(
            "supplement_text",
            "PASS" if supplement_text_status["all_found"] else "BLOCK",
            "supplement PDF text contains controls, guard sensitivity, bootstrap, novelty, and third-target anchors",
        )
    )
    guard_files_ok = all(
        file_status.get(name)
        for name in [
            "semantic_guard_sensitivity_md",
            "semantic_guard_sensitivity_csv",
            "semantic_guard_sensitivity_tex",
        ]
    )
    items.append(
        _status_item(
            "guard_sensitivity",
            "PASS" if guard_files_ok else "BLOCK",
            "guard-sensitivity md/csv/tex artifacts present",
        )
    )
    subject_grouped_ok = all(
        file_status.get(name)
        for name in [
            "semantic_subject_overlap_md",
            "semantic_subject_overlap_csv",
            "semantic_subject_overlap_json",
            "semantic_subject_grouped_tables_md",
            "semantic_subject_grouped_tables_csv",
            "semantic_subject_grouped_tables_tex",
        ]
    )
    items.append(
        _status_item(
            "subject_grouped_resampling",
            "PASS" if subject_grouped_ok else "BLOCK",
            "subject-overlap audit and subject-grouped control tables present",
        )
    )
    metadata_risk = _read_json(REQUIRED_FILES["metadata_field_risk_profile_json"])
    metadata_risk_ok = (
        file_status.get("metadata_field_risk_profile_md")
        and file_status.get("metadata_field_risk_profile_json")
        and metadata_risk.get("verdict") == "METADATA_FIELD_RISK_PROFILE_ACTIVE"
        and metadata_risk.get("runs_inference") is False
        and metadata_risk.get("runs_training") is False
        and metadata_risk.get("uses_existing_datasets_only") is True
        and metadata_risk.get("demographic_fields_have_signal") is True
        and metadata_risk.get("symptom_safe_field_profiled") is True
    )
    items.append(
        _status_item(
            "metadata_field_risk_profile",
            "PASS" if metadata_risk_ok else "BLOCK",
            (
                f"metadata-field risk profile verdict is "
                f"{metadata_risk.get('verdict', 'missing')}; "
                f"demographic_signal={metadata_risk.get('demographic_fields_have_signal', 'missing')}"
            ),
        )
    )
    dossier_ok = file_status.get("claim_dossier_md") and file_status.get("claim_dossier_json")
    items.append(
        _status_item(
            "claim_dossier",
            "PASS" if dossier_ok else "BLOCK",
            "claim dossier md/json artifacts present",
        )
    )
    claim_consistency_ok = (
        file_status.get("claim_consistency_md")
        and file_status.get("claim_consistency_json")
        and claim_consistency_verdict == "PASS"
    )
    items.append(
        _status_item(
            "claim_consistency",
            "PASS" if claim_consistency_ok else "BLOCK",
            f"claim-consistency audit verdict is {claim_consistency_verdict}",
        )
    )
    third_target_success_ok = file_status.get("third_target_success_md") and file_status.get("third_target_success_json")
    items.append(
        _status_item(
            "third_target_success_gate",
            "PASS" if third_target_success_ok else "BLOCK",
            "pre-registered third-target success-gate md/json artifacts present",
        )
    )
    third_target_rescout_ok = file_status.get("third_target_rescout")
    items.append(
        _status_item(
            "third_target_rescout",
            "PASS" if third_target_rescout_ok else "BLOCK",
            "low-cost public third-target replacement re-scout artifact present",
        )
    )
    local_frontier = _read_json(REQUIRED_FILES["local_dataset_frontier_json"])
    local_frontier_ok = (
        file_status.get("local_dataset_frontier_md")
        and file_status.get("local_dataset_frontier_json")
        and file_status.get("local_virufyseg_stress_tex")
        and local_frontier.get("verdict") in {"NO_LOCAL_THIRD_TARGET_READY", "LOCAL_THIRD_TARGET_READY"}
    )
    items.append(
        _status_item(
            "local_dataset_frontier",
            "PASS" if local_frontier_ok else "BLOCK",
            f"local dataset frontier verdict is {local_frontier.get('verdict', 'missing')}",
        )
    )
    local_queue = _read_json(REQUIRED_FILES["local_only_innovation_queue_json"])
    local_queue_ok = (
        file_status.get("local_only_innovation_queue_md")
        and file_status.get("local_only_innovation_queue_json")
        and local_queue.get("verdict") == "LOCAL_ONLY_QUEUE_ACTIVE"
        and local_queue.get("large_external_data") is False
        and bool(local_queue.get("queue"))
    )
    items.append(
        _status_item(
            "local_only_innovation_queue",
            "PASS" if local_queue_ok else "BLOCK",
            (
                f"local-only queue verdict is {local_queue.get('verdict', 'missing')}; "
                f"next action state is {local_queue.get('next_action_state', 'missing')}"
            ),
        )
    )
    no_inference_frontier = _read_json(REQUIRED_FILES["no_inference_frontier_json"])
    no_inference_frontier_ok = (
        file_status.get("no_inference_frontier_md")
        and file_status.get("no_inference_frontier_json")
        and no_inference_frontier.get("verdict") == "NO_INFERENCE_FRONTIER_REVIEWED"
        and no_inference_frontier.get("large_external_data") is False
    )
    items.append(
        _status_item(
            "no_inference_frontier",
            "PASS" if no_inference_frontier_ok else "BLOCK",
            (
                f"prediction-only frontier verdict is "
                f"{no_inference_frontier.get('verdict', 'missing')}"
            ),
        )
    )
    existing_dataset_decision = _read_json(REQUIRED_FILES["existing_dataset_decision_json"])
    existing_dataset_decision_ok = (
        file_status.get("existing_dataset_decision_md")
        and file_status.get("existing_dataset_decision_json")
        and existing_dataset_decision.get("verdict") == "EXISTING_DATASET_DECISION_ACTIVE"
        and existing_dataset_decision.get("decision") == "ASK_BEFORE_CALIB10_INFERENCE"
        and existing_dataset_decision.get("large_external_data") is False
        and existing_dataset_decision.get("approval_required") is True
        and all(existing_dataset_decision.get("ledger_hits", {}).values())
    )
    items.append(
        _status_item(
            "existing_dataset_decision",
            "PASS" if existing_dataset_decision_ok else "BLOCK",
            (
                f"existing-dataset decision is "
                f"{existing_dataset_decision.get('decision', 'missing')}; "
                f"state={existing_dataset_decision.get('state', 'missing')}; "
                f"approval_required={existing_dataset_decision.get('approval_required', 'missing')}"
            ),
        )
    )
    protocol_strength = _read_json(REQUIRED_FILES["protocol_strength_json"])
    protocol_strength_ok = (
        file_status.get("protocol_strength_md")
        and file_status.get("protocol_strength_json")
        and protocol_strength.get("verdict") == "PROTOCOL_STRENGTH_SUPPORTS_SEMANTIC_ROUTER"
        and protocol_strength.get("paired_semantic_beats_controller") is True
    )
    items.append(
        _status_item(
            "protocol_strength",
            "PASS" if protocol_strength_ok else "BLOCK",
            f"protocol-strength verdict is {protocol_strength.get('verdict', 'missing')}",
        )
    )
    branch_controller = _read_json(REQUIRED_FILES["branch_controller_json"])
    branch_controller_ok = (
        file_status.get("branch_controller_md")
        and file_status.get("branch_controller_json")
        and branch_controller.get("verdict") == "BRANCH_CONTROLLER_ACTIVE"
        and branch_controller.get("branch", {}).get("large_external_data") is False
    )
    items.append(
        _status_item(
            "branch_controller",
            "PASS" if branch_controller_ok else "BLOCK",
            (
                f"branch controller verdict is {branch_controller.get('verdict', 'missing')}; "
                f"branch is {branch_controller.get('branch', {}).get('branch', 'missing')}"
            ),
        )
    )
    failure_handoff = _read_json(REQUIRED_FILES["failure_handoff_json"])
    failure_handoff_ok = (
        file_status.get("failure_handoff_md")
        and file_status.get("failure_handoff_json")
        and failure_handoff.get("verdict")
        in {"FAILURE_HANDOFF_PRE_REGISTERED", "FAILURE_HANDOFF_TRIGGERED"}
        and failure_handoff.get("user_constraint")
        == "Use existing local datasets; ask before inference, training, or downloads."
        and bool(failure_handoff.get("candidates"))
    )
    items.append(
        _status_item(
            "failure_handoff",
            "PASS" if failure_handoff_ok else "BLOCK",
            (
                f"failure-handoff verdict is {failure_handoff.get('verdict', 'missing')}; "
                f"state is {failure_handoff.get('state', 'missing')}"
            ),
        )
    )
    local_rescout = _read_json(REQUIRED_FILES["local_existing_run_rescout_json"])
    local_rescout_ok = (
        file_status.get("local_existing_run_rescout_md")
        and file_status.get("local_existing_run_rescout_json")
        and local_rescout.get("verdict") == "NO_NEW_LOCAL_CLAIM_FROM_EXISTING_RUNS"
    )
    items.append(
        _status_item(
            "local_existing_run_rescout",
            "PASS" if local_rescout_ok else "BLOCK",
            f"existing-run rescout verdict is {local_rescout.get('verdict', 'missing')}",
        )
    )
    toscovid2022 = _read_json(REQUIRED_FILES["toscovid2022_local_candidate_json"])
    toscovid2022_ok = (
        file_status.get("toscovid2022_local_candidate_md")
        and file_status.get("toscovid2022_local_candidate_json")
        and toscovid2022.get("verdict") == "METADATA_ONLY_NOT_LOCAL_AUDIO_TARGET"
    )
    items.append(
        _status_item(
            "toscovid2022_local_candidate",
            "PASS" if toscovid2022_ok else "BLOCK",
            f"TosCOVID 2022 local audit verdict is {toscovid2022.get('verdict', 'missing')}",
        )
    )
    toscovid_official = _read_json(REQUIRED_FILES["toscovid_official_split_readiness_json"])
    toscovid_official_ok = (
        file_status.get("toscovid2021_full_manifest")
        and file_status.get("toscovid2021_train_manifest")
        and file_status.get("toscovid_official_split_readiness_md")
        and file_status.get("toscovid_official_split_readiness_json")
        and toscovid_official.get("verdict")
        in {"NOT_READY_MISSING_OFFICIAL_TRAIN_PREDICTIONS", "READY_FOR_OFFICIAL_SPLIT_REANALYSIS"}
    )
    items.append(
        _status_item(
            "toscovid_official_split_route",
            "PASS" if toscovid_official_ok else "BLOCK",
            f"TosCOVID 2021 official-split readiness verdict is {toscovid_official.get('verdict', 'missing')}; this is a credibility-upgrade route, not a third-target upgrade.",
        )
    )
    toscovid_official_router = _read_json(REQUIRED_FILES["toscovid_official_split_semantic_router_json"])
    toscovid_official_router_ok = (
        file_status.get("toscovid_official_split_semantic_router_md")
        and file_status.get("toscovid_official_split_semantic_router_json")
        and file_status.get("toscovid_official_train_inference_driver")
        and toscovid_official_router.get("verdict")
        in {
            "NOT_READY_MISSING_OFFICIAL_SPLIT_PREDICTIONS",
            "READY_OFFICIAL_SPLIT_RESULT_AVAILABLE",
        }
    )
    items.append(
        _status_item(
            "toscovid_official_split_semantic_router",
            "PASS" if toscovid_official_router_ok else "BLOCK",
            f"official-split semantic-router gate verdict is {toscovid_official_router.get('verdict', 'missing')}; missing train predictions require approval before inference.",
        )
    )
    toscovid_calib_budget = _read_json(REQUIRED_FILES["toscovid_official_calibration_budget_json"])
    toscovid_calib_budget_ok = (
        file_status.get("toscovid_official_calibration_subsets_md")
        and file_status.get("toscovid_official_calibration_subsets_json")
        and file_status.get("toscovid_official_calibration_budget_md")
        and file_status.get("toscovid_official_calibration_budget_json")
        and file_status.get("toscovid_official_calibration_probe_driver")
        and toscovid_calib_budget.get("verdict")
        in {"NOT_READY_MISSING_CALIBRATION_PREDICTIONS", "READY_WITH_RESULTS"}
    )
    items.append(
        _status_item(
            "toscovid_official_calibration_budget",
            "PASS" if toscovid_calib_budget_ok else "BLOCK",
            f"official calibration budget verdict is {toscovid_calib_budget.get('verdict', 'missing')}; low-cost subset probes are approval-bound.",
        )
    )
    calib10_approval = _read_json(REQUIRED_FILES["toscovid_calib10_probe_approval_json"])
    calib10_decision = _read_json(REQUIRED_FILES["toscovid_calib10_result_decision_json"])
    calib10_selftest = _read_json(REQUIRED_FILES["toscovid_calib10_decision_selftest_json"])
    calib10_approval_ok = (
        file_status.get("toscovid_calib10_probe_approval_md")
        and file_status.get("toscovid_calib10_probe_approval_json")
        and calib10_approval.get("verdict") == "READY_FOR_USER_APPROVAL"
        and calib10_approval.get("all_checkpoints_exist") is True
    )
    items.append(
        _status_item(
            "toscovid_calib10_probe_approval_packet",
            "PASS" if calib10_approval_ok else "BLOCK",
            f"calib10 stable approval packet verdict is {calib10_approval.get('verdict', 'missing')}",
        )
    )
    calib10_decision_ok = (
        file_status.get("toscovid_calib10_result_decision_md")
        and file_status.get("toscovid_calib10_result_decision_json")
        and calib10_decision.get("state")
        in {
            "AWAIT_CALIB10_STABLE_SMOKE",
            "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE",
            "PROMOTE_TO_NEXT_STABLE_SUBSET",
            "PROMOTE_TO_FULL_ROUTER",
        }
        and calib10_decision.get("decision", {}).get("large_external_data") is False
    )
    items.append(
        _status_item(
            "toscovid_calib10_result_decision",
            "PASS" if calib10_decision_ok else "BLOCK",
            (
                f"calib10 result decision verdict is {calib10_decision.get('verdict', 'missing')}; "
                f"state is {calib10_decision.get('state', 'missing')}; "
                f"approval_required={calib10_decision.get('decision', {}).get('approval_required', 'missing')}"
            ),
        )
    )
    calib10_selftest_ok = (
        file_status.get("toscovid_calib10_decision_selftest_md")
        and file_status.get("toscovid_calib10_decision_selftest_json")
        and calib10_selftest.get("verdict") == "PASS"
        and len(calib10_selftest.get("cases", [])) >= 7
        and all(row.get("pass") is True for row in calib10_selftest.get("cases", []))
    )
    items.append(
        _status_item(
            "toscovid_calib10_decision_selftest",
            "PASS" if calib10_selftest_ok else "BLOCK",
            (
                f"calib10 decision self-test verdict is {calib10_selftest.get('verdict', 'missing')}; "
                f"cases={len(calib10_selftest.get('cases', []))}"
            ),
        )
    )
    items.append(
        _status_item(
            "ukcovid_download_verify_driver",
            "PASS" if file_status.get("ukcovid_download_verify_driver") and file_status.get("ukcovid_extraction_preflight_script") else "BLOCK",
            "approval-time UKCOVID download/checksum driver and extraction preflight script present",
        )
    )
    preflight = _read_json(REQUIRED_FILES["ukcovid_download_preflight_json"])
    preflight_ok = (
        file_status.get("ukcovid_download_preflight_md")
        and file_status.get("ukcovid_download_preflight_json")
        and preflight.get("verdict") == "READY_TO_DOWNLOAD_OR_RESUME"
    )
    items.append(
        _status_item(
            "ukcovid_download_preflight",
            "PASS" if preflight_ok else "BLOCK",
            f"pre-download audio preflight verdict is {preflight.get('verdict', 'missing')}",
        )
    )
    archive_integrity = _read_json(REQUIRED_FILES["ukcovid_archive_integrity_json"])
    archive_integrity_ok = (
        file_status.get("ukcovid_archive_integrity_md")
        and file_status.get("ukcovid_archive_integrity_json")
        and archive_integrity.get("verdict") in {"NOT_READY_MISSING_ARCHIVE", "READY_FOR_EXTRACTION"}
    )
    items.append(
        _status_item(
            "ukcovid_archive_integrity_gate",
            "PASS" if archive_integrity_ok else "BLOCK",
            f"archive integrity verifier verdict is {archive_integrity.get('verdict', 'missing')}",
        )
    )
    extraction_preflight = _read_json(REQUIRED_FILES["ukcovid_extraction_preflight_json"])
    extraction_preflight_ok = (
        file_status.get("ukcovid_extraction_preflight_md")
        and file_status.get("ukcovid_extraction_preflight_json")
        and extraction_preflight.get("verdict")
        in {"ARCHIVE_NOT_READY", "MAIN_SPLIT_ZIP_MISSING", "EXTRACTION_TOOL_MISSING", "READY_TO_EXTRACT"}
    )
    items.append(
        _status_item(
            "ukcovid_extraction_preflight",
            "PASS" if extraction_preflight_ok else "BLOCK",
            f"extraction preflight verdict is {extraction_preflight.get('verdict', 'missing')}",
        )
    )
    next_action = _read_json(REQUIRED_FILES["next_action_dashboard_json"])
    next_action_ok = (
        file_status.get("next_action_dashboard_md")
        and file_status.get("next_action_dashboard_json")
        and bool(next_action.get("state"))
        and bool(next_action.get("next_action"))
    )
    items.append(
        _status_item(
            "next_action_dashboard",
            "PASS" if next_action_ok else "BLOCK",
            f"next action state is {next_action.get('state', 'missing')}",
        )
    )
    claim_upgrade = _read_json(REQUIRED_FILES["claim_upgrade_decision_json"])
    claim_upgrade_ok = (
        file_status.get("claim_upgrade_decision_md")
        and file_status.get("claim_upgrade_decision_json")
        and bool(claim_upgrade.get("decision"))
        and bool(claim_upgrade.get("claim_level"))
    )
    items.append(
        _status_item(
            "claim_upgrade_decision",
            "PASS" if claim_upgrade_ok else "BLOCK",
            f"claim-upgrade decision is {claim_upgrade.get('decision', 'missing')}",
        )
    )
    goal_completion = _read_json(REQUIRED_FILES["goal_completion_json"])
    goal_completion_ok = (
        file_status.get("goal_completion_md")
        and file_status.get("goal_completion_json")
        and bool(goal_completion.get("verdict"))
        and bool(goal_completion.get("requirements"))
    )
    items.append(
        _status_item(
            "goal_completion_audit",
            "PASS" if goal_completion_ok else "BLOCK",
            f"full-goal completion verdict is {goal_completion.get('verdict', 'missing')}",
        )
    )
    items.append(
        _status_item(
            "novelty_boundary",
            "PASS"
            if file_status.get("novelty_matrix_md")
            and file_status.get("novelty_matrix_tex")
            and file_status.get("novelty_source_ledger")
            else "BLOCK",
            "novelty matrix and source ledger available; still a scoped audit, not absolute proof",
        )
    )
    collision_watch = _read_json(REQUIRED_FILES["collision_watch_json"])
    collision_watch_ok = (
        file_status.get("collision_watch_md")
        and file_status.get("collision_watch_json")
        and collision_watch.get("verdict") == "NO_DIRECT_COLLISION_FOUND_IN_LATEST_WATCH"
    )
    items.append(
        _status_item(
            "collision_watch",
            "PASS" if collision_watch_ok else "BLOCK",
            f"latest collision-watch verdict is {collision_watch.get('verdict', 'missing')}",
        )
    )
    uk_ready = bool(ukcovid.get("ready"))
    uk_verdict = str(ukcovid.get("verdict", "missing"))
    items.append(
        _status_item(
            "third_target_ukcovid",
            "PASS" if uk_ready else "BLOCK",
            uk_verdict,
        )
    )
    items.append(
        _status_item(
            "icasSP2027_final_kit",
            "WARN",
            "2027 CFP page rule checked; final author kit/template still needs re-check before submission",
        )
    )

    blocking = [row for row in items if row["status"] == "BLOCK"]
    warnings = [row for row in items if row["status"] == "WARN"]
    if blocking:
        verdict = "NOT_READY_THIRD_TARGET_OR_ARTIFACTS_MISSING"
    elif warnings:
        verdict = "CONDITIONALLY_READY_PENDING_FINAL_KIT"
    else:
        verdict = "READY_FOR_SUBMISSION_PACKAGE_REVIEW"

    payload = {
        "verdict": verdict,
        "items": items,
        "file_status": file_status,
        "main_text_status": main_text_status,
        "supplement_text_status": supplement_text_status,
        "main_log_status": main_log_status,
        "supplement_log_status": supplement_log_status,
        "ukcovid_readiness": ukcovid,
        "icassp_page_rule": page_rule,
    }

    OUT.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(
        OUT / "semantic_router_submission_readiness.json",
        json.dumps(payload, indent=2),
    )

    lines = [
        "# Semantic-Router Submission Readiness",
        "",
        f"- Verdict: `{verdict}`",
        f"- Blocking items: `{len(blocking)}`",
        f"- Warnings: `{len(warnings)}`",
        "",
        "## Checklist",
        "",
        _markdown_table(items),
        "",
        "## Interpretation",
        "",
    ]
    if blocking:
        lines.append("The current package is a credible candidate, but it is not final-ready. Resolve the blocking items before claiming the full ICASSP objective is complete.")
    elif warnings:
        lines.append("The empirical package passes this audit, but final conference-format checks remain.")
    else:
        lines.append("The package passes this automated audit. A human final read is still required before submission.")
    _write_text_atomic(
        OUT / "SEMANTIC_ROUTER_SUBMISSION_READINESS.md",
        "\n".join(lines) + "\n",
    )

    print(json.dumps({"verdict": verdict, "blocking": blocking, "warnings": warnings}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_SUBMISSION_READINESS.md")


if __name__ == "__main__":
    main()
