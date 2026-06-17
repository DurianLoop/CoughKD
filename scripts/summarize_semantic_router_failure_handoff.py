"""Pre-register the local-only handoff if calib10 fails.

This is a process guard for the autonomous innovation loop. It reads existing
evidence and records what should happen if the current approval-bound calib10
stable probe fails below the continuation threshold. It does not train, infer,
or download data.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_failure_handoff"


PATHS = {
    "calib10_decision": RUNS
    / "toscovid_calib10_result_decision"
    / "toscovid_calib10_result_decision.json",
    "no_inference_frontier": RUNS
    / "semantic_router_no_inference_frontier"
    / "semantic_router_no_inference_frontier.json",
    "existing_gain_matrix": RUNS
    / "existing_external_gain_matrix"
    / "EXISTING_EXTERNAL_GAIN_MATRIX.md",
    "artifact_environment": RUNS
    / "artifact_environment_audit_seed7"
    / "ARTIFACT_ENVIRONMENT_AUDIT.md",
    "tos_env_separability": RUNS
    / "tos_env1_separability_audit_seed7"
    / "TOS_ENV1_SEPARABILITY_AUDIT.md",
    "novelty_source_ledger": RUNS
    / "semantic_router_novelty_matrix"
    / "SEMANTIC_ROUTER_SOURCE_LEDGER.md",
    "branch_controller": RUNS
    / "semantic_router_branch_controller"
    / "semantic_router_branch_controller.json",
}


LITERATURE_RISK_ANCHORS = [
    {
        "source": "RespiraMFM, arXiv:2606.09966",
        "risk": "Respiratory audio + symptoms/history + audio-text contrastive alignment is now a very current foundation-model direction.",
        "impact": "Do not pivot to a broad metadata/audio-language respiratory foundation claim.",
    },
    {
        "source": "AcuLa, arXiv:2512.04847",
        "risk": "Medical-audio semantic-teacher alignment from structured metadata is already occupied.",
        "impact": "Do not claim novelty for metadata-to-report or semantic-teacher alignment.",
    },
    {
        "source": "SLM-TTA, arXiv:2512.24739",
        "risk": "Generic audio/speech test-time adaptation is active and crowded.",
        "impact": "Do not restart with a plain confidence/entropy/TTA rule.",
    },
    {
        "source": "CoughSense, arXiv:2606.02998",
        "risk": "Cough-specific active-frame/Whisper and symptom/domain-adaptation ideas are adjacent.",
        "impact": "Do not claim novelty for active cough pooling or symptom-conditioned cough modeling.",
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


def _extract_delta(text: str, target: str, method: str) -> float | None:
    for line in text.splitlines():
        if not line.startswith(f"| {target} |"):
            continue
        if f"| {method} |" not in line:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) < 7:
            continue
        try:
            return float(cells[5])
        except ValueError:
            return None
    return None


def _state(calib10: dict[str, Any]) -> dict[str, Any]:
    state = str(calib10.get("state", "missing"))
    verdict = str(calib10.get("verdict", "missing"))
    if state == "STOP_OFFICIAL_SPLIT_ENHANCEMENT_ROUTE":
        return {
            "state": "LOCAL_ONLY_LITERATURE_RESTART_TRIGGERED",
            "verdict": "FAILURE_HANDOFF_TRIGGERED",
            "next_action": "Start a fresh local-only literature loop and require a new pre-registered validation design before running more compute.",
            "active": True,
        }
    return {
        "state": "WAIT_FOR_CALIB10_RESULT",
        "verdict": "FAILURE_HANDOFF_PRE_REGISTERED",
        "next_action": "Keep the handoff dormant until calib10 stable smoke is approved and evaluated.",
        "active": False,
        "calib10_state": state,
        "calib10_verdict": verdict,
    }


def _candidates(
    *,
    no_inference: dict[str, Any],
    gain_text: str,
    artifact_text: str,
    tos_env_text: str,
    novelty_text: str,
) -> list[dict[str, Any]]:
    candidate_f_coughvid = _extract_delta(
        gain_text, "COUGHVID", "candidate_f_artifact_env_irm_ramp"
    )
    candidate_f_tos = _extract_delta(
        gain_text, "Tos COVID-19", "candidate_f_artifact_env_irm_ramp"
    )
    ce_tos = _extract_delta(gain_text, "Tos COVID-19", "ce")
    artifact_ready = "Proceed: artifact pseudo-environments are detectable" in artifact_text
    tos_artifact_weak = "artifact-only interventions are unlikely to solve Tos" in tos_env_text
    no_inference_rejected = (
        no_inference.get("verdict") == "NO_INFERENCE_FRONTIER_REVIEWED"
        and "No prediction-only alternative" in str(no_inference.get("claim_readout", ""))
    )
    novelty_lower = novelty_text.lower()
    audio_language_crowded = all(
        needle in novelty_text
        for needle in [
            "AcuLa",
            "RespiraMFM",
        ]
    ) and "metadata-to-language representation alignment" in novelty_lower

    return [
        {
            "candidate": "official_calibration_sufficiency",
            "status": "active_pending_approval",
            "uses_existing_datasets_only": True,
            "requires_training": False,
            "requires_inference": True,
            "decision": "keep_as_current_next_experiment",
            "evidence": "calib10 stable result is still missing; this is the only active low-cost compute branch.",
            "validation_design": "approved calib10 stable smoke, then <1pp stop / 1-3pp expand stable subset / >=3pp ask before full-router.",
        },
        {
            "candidate": "prediction_only_replacement",
            "status": "rejected_before_failure_handoff",
            "uses_existing_datasets_only": True,
            "requires_training": False,
            "requires_inference": False,
            "decision": "do_not_promote",
            "evidence": f"no_inference_frontier_rejected={no_inference_rejected}",
            "validation_design": "Only reconsider if a new literature mechanism gives a non-generic cough-specific rule and both large targets clear >=1pp.",
        },
        {
            "candidate": "target_coverage_aware_artifact_alignment",
            "status": "diagnostic_only_not_ready",
            "uses_existing_datasets_only": True,
            "requires_training": True,
            "requires_inference": True,
            "decision": "do_not_train_yet",
            "evidence": (
                f"artifact_env_ready={artifact_ready}; "
                f"candidate_f_COUGHVID_delta={candidate_f_coughvid}; "
                f"candidate_f_Tos_delta={candidate_f_tos}; "
                f"Tos_artifact_weak={tos_artifact_weak}"
            ),
            "validation_design": "Requires a new preflight proving target artifact slices predict recoverable error on both large targets before any training request.",
        },
        {
            "candidate": "ce_or_plain_target_stack_revival",
            "status": "baseline_only",
            "uses_existing_datasets_only": True,
            "requires_training": False,
            "requires_inference": False,
            "decision": "do_not_claim_as_innovation",
            "evidence": f"Tos CE is the best existing plain method at delta={ce_tos}, but COUGHVID support is not symmetric and the family is ordinary calibration/stacking.",
            "validation_design": "Use as a baseline/control inside a stronger mechanism, not as the main claim.",
        },
        {
            "candidate": "audio_language_metadata_foundation_route",
            "status": "crowded_literature_route",
            "uses_existing_datasets_only": False,
            "requires_training": True,
            "requires_inference": True,
            "decision": "reject_under_local_only_constraint",
            "evidence": f"audio_language_crowded={audio_language_crowded}; RespiraMFM/AcuLa/BTS occupy broad metadata-audio-language modeling.",
            "validation_design": "Do not pursue unless the user approves new assets/compute and the claim is narrowed away from broad audio-language alignment.",
        },
        {
            "candidate": "guarded_failure_slice_audit_reframe",
            "status": "paper_framing_fallback",
            "uses_existing_datasets_only": True,
            "requires_training": False,
            "requires_inference": False,
            "decision": "fallback_if_positive_method_search_exhausts",
            "evidence": "Repeated local routes show COUGHVID-positive/TosCOVID-negative behavior; this can support a rigorous audit paper but not the requested positive 1-3pp method claim.",
            "validation_design": "Only use if the user accepts a weaker/failure-audit paper framing; it does not satisfy the current positive-innovation objective by itself.",
        },
    ]


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
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    calib10 = _read_json(PATHS["calib10_decision"])
    no_inference = _read_json(PATHS["no_inference_frontier"])
    gain_text = _read_text(PATHS["existing_gain_matrix"])
    artifact_text = _read_text(PATHS["artifact_environment"])
    tos_env_text = _read_text(PATHS["tos_env_separability"])
    novelty_text = _read_text(PATHS["novelty_source_ledger"])
    branch = _read_json(PATHS["branch_controller"])

    state = _state(calib10)
    candidates = _candidates(
        no_inference=no_inference,
        gain_text=gain_text,
        artifact_text=artifact_text,
        tos_env_text=tos_env_text,
        novelty_text=novelty_text,
    )
    payload = {
        **state,
        "branch_controller_state": branch.get("branch", {}).get("branch", "missing"),
        "user_constraint": "Use existing local datasets; ask before inference, training, or downloads.",
        "candidates": candidates,
        "literature_risk_anchors": LITERATURE_RISK_ANCHORS,
        "forbidden_after_failure": [
            "Do not tune thresholds on the failed route after seeing the calib10 result.",
            "Do not revive prediction-only alternatives already rejected by the no-inference frontier.",
            "Do not restart with broad audio-language or metadata-foundation claims crowded by RespiraMFM, AcuLa, BTS, and related work.",
            "Do not present Virufy/Virufyseg tiny-target gains as the main ICASSP claim.",
        ],
        "evidence_paths": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items() if path.is_file()},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_failure_handoff.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Semantic-Router Failure Handoff",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- State: `{payload['state']}`",
        f"- Active: `{payload['active']}`",
        f"- Branch controller state: `{payload['branch_controller_state']}`",
        f"- User constraint: {payload['user_constraint']}",
        f"- Next action: {payload['next_action']}",
        "",
        "## Candidate Handoff Queue",
        "",
        _table(
            candidates,
            [
                "candidate",
                "status",
                "uses_existing_datasets_only",
                "requires_training",
                "requires_inference",
                "decision",
                "evidence",
                "validation_design",
            ],
        ),
        "",
        "## Literature Risk Anchors",
        "",
        _table(LITERATURE_RISK_ANCHORS, ["source", "risk", "impact"]),
        "",
        "## Forbidden After Failure",
        "",
    ]
    lines.extend(f"- {item}" for item in payload["forbidden_after_failure"])
    lines.append("")
    (OUT / "SEMANTIC_ROUTER_FAILURE_HANDOFF.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print(json.dumps({"verdict": payload["verdict"], "state": payload["state"]}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_FAILURE_HANDOFF.md")


if __name__ == "__main__":
    main()
