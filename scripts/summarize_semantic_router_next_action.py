"""Summarize the next concrete action for the semantic-router ICASSP package."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_next_action"


PATHS = {
    "submission_readiness": RUNS
    / "semantic_router_submission_readiness"
    / "semantic_router_submission_readiness.json",
    "claim_dossier": RUNS / "semantic_router_claim_dossier" / "semantic_router_claim_dossier.json",
    "download_preflight": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_audio_download_preflight.json",
    "download_approval": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_download_approval_status.json",
    "archive_integrity": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_audio_archive_integrity.json",
    "extraction_preflight": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_extraction_preflight.json",
    "audio_ready": RUNS / "ukcovid_open_metadata_audit" / "ukcovid_audio_ready_summary.json",
    "target_readiness": RUNS
    / "semantic_router_third_target_readiness"
    / "ukcovid_semantic_router_readiness.json",
    "third_target_success": RUNS
    / "semantic_router_third_target_success"
    / "semantic_router_third_target_success.json",
    "local_dataset_frontier": RUNS
    / "semantic_router_local_dataset_frontier"
    / "semantic_router_local_dataset_frontier.json",
    "toscovid_calibration_budget": RUNS
    / "toscovid_official_calibration_budget"
    / "toscovid_official_calibration_budget.json",
    "toscovid_calib10_probe_approval": RUNS
    / "toscovid_calib10_stable_probe_approval"
    / "toscovid_calib10_stable_probe_approval.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _verdicts() -> dict[str, str]:
    return {name: str(_read_json(path).get("verdict", "missing")) for name, path in PATHS.items()}


def _action(
    verdicts: dict[str, str],
    download: dict[str, Any],
    extraction: dict[str, Any],
    approval: dict[str, Any],
    local_frontier: dict[str, Any],
    toscovid_calibration_budget: dict[str, Any],
) -> dict[str, Any]:
    if verdicts["submission_readiness"] in {
        "READY_FOR_SUBMISSION_PACKAGE_REVIEW",
        "CONDITIONALLY_READY_PENDING_FINAL_KIT",
    }:
        return {
            "state": "PACKAGE_GATE_PASSED",
            "next_action": "Run final human paper QA and re-check the official ICASSP 2027 author kit.",
            "requires_user_approval": False,
            "large_external_data": False,
        }

    if verdicts["third_target_success"] == "THIRD_TARGET_SUPPORTS_CLAIM":
        return {
            "state": "THIRD_TARGET_SUPPORTS_CLAIM",
            "next_action": "Refresh submission readiness, claim dossier, main paper, and supplement with the UKCOVID third-target result.",
            "requires_user_approval": False,
            "large_external_data": False,
        }

    calib_recommendation = toscovid_calibration_budget.get("recommendation", {})
    if local_frontier.get("verdict") == "NO_LOCAL_THIRD_TARGET_READY" and calib_recommendation.get("state"):
        command = ""
        if calib_recommendation.get("state") == "AWAIT_CALIB10_STABLE_SMOKE":
            command = "scripts\\run_toscovid_official_calibration_probe_windows.cmd calib10 stable yes"
        return {
            "state": str(calib_recommendation.get("state")),
            "next_action": str(calib_recommendation.get("next_action", "")),
            "rationale": str(calib_recommendation.get("rationale", "")),
            "command": command,
            "requires_user_approval": True,
            "large_external_data": False,
        }

    if local_frontier.get("verdict") == "NO_LOCAL_THIRD_TARGET_READY":
        return {
            "state": "LOCAL_DATASET_MODE_NO_THIRD_TARGET",
            "next_action": "Use existing COUGHVID/TosCOVID as the two main targets, add Virufy/Virufyseg only as auxiliary micro-stress evidence, and keep searching locally before any large download or heavy training.",
            "requires_user_approval": False,
            "large_external_data": False,
        }

    if local_frontier.get("verdict") == "LOCAL_THIRD_TARGET_READY":
        return {
            "state": "LOCAL_THIRD_TARGET_READY",
            "next_action": "Run the semantic-router audit on the local third-target candidate and refresh the claim dossier before changing the paper claim.",
            "requires_user_approval": False,
            "large_external_data": False,
        }

    if verdicts["target_readiness"] == "READY_FOR_SEMANTIC_ROUTER_AUDIT":
        return {
            "state": "READY_FOR_UKCOVID_SEMANTIC_ROUTER_AUDIT",
            "next_action": "Run scripts\\run_ukcovid_semantic_router_after_audio_windows.cmd to execute the pre-registered subject-grouped UKCOVID audits.",
            "requires_user_approval": False,
            "large_external_data": False,
        }

    if verdicts["audio_ready"] == "READY_FOR_EVALUATION":
        return {
            "state": "READY_FOR_UKCOVID_ONBOARDING",
            "next_action": "Run scripts\\run_ukcovid_semantic_router_after_audio_windows.cmd; it will rebuild the manifest, onboard predictions, and run the audit gates.",
            "requires_user_approval": False,
            "large_external_data": False,
        }

    if verdicts["extraction_preflight"] == "READY_TO_EXTRACT":
        command = extraction.get("extract_command", "")
        return {
            "state": "READY_TO_EXTRACT_UKCOVID_ARCHIVE",
            "next_action": "Extract the verified UKCOVID split archive, then rerun the post-audio driver.",
            "command": command,
            "requires_user_approval": False,
            "large_external_data": False,
        }

    if verdicts["archive_integrity"] == "READY_FOR_EXTRACTION":
        return {
            "state": "CHECK_EXTRACTION_TOOL",
            "next_action": "Run D:\\conda\\envs\\CoughKD\\python.exe -B scripts\\check_ukcovid_extraction_preflight.py and install or point to 7-Zip if needed.",
            "requires_user_approval": False,
            "large_external_data": False,
        }

    if verdicts["archive_integrity"] == "ARCHIVE_CHECKSUM_FAILED":
        return {
            "state": "REDOWNLOAD_FAILED_ARCHIVE_PARTS",
            "next_action": "Redownload only the failed UKCOVID archive parts, then rerun scripts\\verify_ukcovid_audio_archive.py.",
            "requires_user_approval": True,
            "large_external_data": True,
        }

    if verdicts["download_preflight"] == "READY_TO_DOWNLOAD_OR_RESUME":
        archive = download.get("archive", {})
        remaining_gib = archive.get("remaining_archive_gib")
        remaining = f"{remaining_gib:.2f} GiB" if isinstance(remaining_gib, (int, float)) else "unknown size"
        approval_status = approval.get("status", "missing")
        approval_suffix = (
            f" Current approval status is {approval_status}; no download has started."
            if approval_status in {"requested_timeout", "denied", "not_requested"}
            else ""
        )
        return {
            "state": "AWAITING_UKCOVID_DOWNLOAD_APPROVAL",
            "next_action": f"Ask the user to approve the resumable UKCOVID audio download ({remaining} remaining), then run scripts\\run_ukcovid_audio_download_verify_windows.cmd.{approval_suffix}",
            "requires_user_approval": True,
            "large_external_data": True,
        }

    return {
        "state": "REFRESH_PREFLIGHTS",
        "next_action": "Refresh UKCOVID preflight, archive verifier, extraction preflight, submission readiness, and claim dossier before choosing the next action.",
        "requires_user_approval": False,
        "large_external_data": False,
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
    verdicts = _verdicts()
    download = _read_json(PATHS["download_preflight"])
    approval = _read_json(PATHS["download_approval"])
    extraction = _read_json(PATHS["extraction_preflight"])
    local_frontier = _read_json(PATHS["local_dataset_frontier"])
    toscovid_calibration_budget = _read_json(PATHS["toscovid_calibration_budget"])
    action = _action(verdicts, download, extraction, approval, local_frontier, toscovid_calibration_budget)

    payload = {
        "state": action["state"],
        "next_action": action["next_action"],
        "rationale": action.get("rationale", ""),
        "requires_user_approval": bool(action.get("requires_user_approval")),
        "large_external_data": bool(action.get("large_external_data")),
        "command": action.get("command", ""),
        "verdicts": verdicts,
        "evidence_paths": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_next_action.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows = [{"gate": name, "verdict": verdict} for name, verdict in verdicts.items()]
    lines = [
        "# Semantic-Router Next Action Dashboard",
        "",
        f"- State: `{payload['state']}`",
        f"- Requires user approval: `{payload['requires_user_approval']}`",
        f"- Large external data: `{payload['large_external_data']}`",
        f"- Next action: {payload['next_action']}",
        f"- Rationale: {payload['rationale']}",
        "",
        "## Gate Verdicts",
        "",
        _table(rows),
        "",
    ]
    if payload["command"]:
        lines.extend(["## Command", "", "```cmd", payload["command"], "```", ""])
    (OUT / "SEMANTIC_ROUTER_NEXT_ACTION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(OUT / "SEMANTIC_ROUTER_NEXT_ACTION.md")


if __name__ == "__main__":
    main()
