"""Summarize prediction-only alternatives for the semantic-router loop.

The goal is to keep the innovation search honest when inference approval is not
available. This script reads existing audit artifacts only. It does not train,
run inference, download data, or use target labels to create a new deployable
method.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_no_inference_frontier"


PATHS = {
    "prediction_ensemble": RUNS
    / "prediction_ensemble_frontier"
    / "prediction_ensemble_frontier_summary.json",
    "target_calibrated_stacking": RUNS
    / "target_calibrated_stacking"
    / "target_calibrated_stacking_summary.json",
    "metadata_slice_oracle": RUNS
    / "metadata_slice_oracle"
    / "metadata_slice_oracle_summary.json",
    "posterior_prior_correction": RUNS
    / "posterior_prior_correction"
    / "posterior_prior_correction_summary.json",
    "transfer_gap_diagnostics": RUNS
    / "transfer_gap_diagnostics"
    / "transfer_gap_diagnostics.json",
    "local_only_queue": RUNS
    / "semantic_router_local_only_innovation_queue"
    / "semantic_router_local_only_innovation_queue.json",
}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _best_by_target(summary: dict[str, Any], target: str) -> dict[str, Any]:
    rows = [row for row in summary.get("best_by_target", []) if row.get("target") == target]
    if not rows:
        return {}
    key = "delta_vs_source_only" if "delta_vs_source_only" in rows[0] else "mean_delta_vs_source"
    return max(rows, key=lambda row: float(row.get(key, 0.0) or 0.0))


def _large_target_status(ensemble: dict[str, Any], stacking: dict[str, Any], prior: dict[str, Any]) -> dict[str, Any]:
    coughvid_ensemble = _best_by_target(ensemble, "COUGHVID")
    tos_ensemble = _best_by_target(ensemble, "TosCOVID")
    coughvid_stack = _best_by_target(stacking, "COUGHVID")
    tos_stack = _best_by_target(stacking, "TosCOVID")
    return {
        "prediction_ensemble_coughvid_delta": coughvid_ensemble.get("delta_vs_source_only"),
        "prediction_ensemble_toscovid_delta": tos_ensemble.get("delta_vs_source_only"),
        "target_calibrated_stacking_coughvid_delta": coughvid_stack.get("mean_delta_vs_source"),
        "target_calibrated_stacking_toscovid_delta": tos_stack.get("mean_delta_vs_source"),
        "posterior_prior_max_large_delta": prior.get("max_delta_large_targets"),
    }


def _rows(
    ensemble: dict[str, Any],
    stacking: dict[str, Any],
    oracle: dict[str, Any],
    prior: dict[str, Any],
    transfer: dict[str, Any],
) -> list[dict[str, Any]]:
    large = _large_target_status(ensemble, stacking, prior)
    coughvid_ensemble = large["prediction_ensemble_coughvid_delta"] or 0.0
    tos_ensemble = large["prediction_ensemble_toscovid_delta"] or 0.0
    coughvid_stack = large["target_calibrated_stacking_coughvid_delta"] or 0.0
    tos_stack = large["target_calibrated_stacking_toscovid_delta"] or 0.0
    prior_large = large["posterior_prior_max_large_delta"] or 0.0
    oracle_large = oracle.get("clears_3pt_large_target") is True
    transfer_rows = transfer.get("target_shift", [])
    transfer_has_large = bool(transfer_rows)

    return [
        {
            "candidate": "plain_prediction_ensemble",
            "evidence": f"COUGHVID {100 * coughvid_ensemble:.2f} pp; TosCOVID {100 * tos_ensemble:.2f} pp",
            "decision": "reject_as_main_claim",
            "reason": "No-retraining ensembles do not clear the 1-3 pp gate on both large targets; strong Virufy/Virufyseg gains are too small-sample for the main claim.",
            "next_if_needed": "Use as negative control or auxiliary stress evidence only.",
        },
        {
            "candidate": "global_target_calibrated_stacking",
            "evidence": f"COUGHVID {100 * coughvid_stack:.2f} pp; TosCOVID {100 * tos_stack:.2f} pp",
            "decision": "reject_as_replacement_claim",
            "reason": "It is positive on TosCOVID but weak/unstable on COUGHVID, and stacking is an established modeling family.",
            "next_if_needed": "Keep as the non-metadata baseline inside the semantic-router story.",
        },
        {
            "candidate": "metadata_slice_oracle",
            "evidence": f"large-target oracle clears 3 pp: {oracle_large}",
            "decision": "mechanism_support_only",
            "reason": "The oracle shows slice-level signal exists, but it is label-revealed and cannot be claimed as a deployable method.",
            "next_if_needed": "Use it to justify semantic controls and the field-semantics prior, not as a standalone innovation.",
        },
        {
            "candidate": "posterior_prior_correction",
            "evidence": f"max large-target delta {100 * prior_large:.2f} pp",
            "decision": "reject_as_main_claim",
            "reason": "Target-unlabeled posterior transforms are too small on COUGHVID/TosCOVID and collide with broad calibration/TTA ideas.",
            "next_if_needed": "Do not spend more local-only effort here unless a new literature mechanism appears.",
        },
        {
            "candidate": "transfer_gap_diagnostic_rule",
            "evidence": f"target-shift rows present: {transfer_has_large}",
            "decision": "not_ready_as_method",
            "reason": "Diagnostics are label-free and useful, but source-train logits or a validated rule are missing; plain confidence/entropy rules collide with uncertainty KD/TTA literature.",
            "next_if_needed": "Only revive after a new, non-generic cough-specific mechanism is found.",
        },
    ]


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
    ensemble = _read_json(PATHS["prediction_ensemble"])
    stacking = _read_json(PATHS["target_calibrated_stacking"])
    oracle = _read_json(PATHS["metadata_slice_oracle"])
    prior = _read_json(PATHS["posterior_prior_correction"])
    transfer = _read_json(PATHS["transfer_gap_diagnostics"])
    local_queue = _read_json(PATHS["local_only_queue"])

    rows = _rows(ensemble, stacking, oracle, prior, transfer)
    payload = {
        "verdict": "NO_INFERENCE_FRONTIER_REVIEWED",
        "claim_readout": "No prediction-only alternative should replace the current semantic-router claim.",
        "next_action_state": local_queue.get("next_action_state", "missing"),
        "requires_user_approval": bool(local_queue.get("requires_user_approval")),
        "large_external_data": False,
        "rows": rows,
        "evidence_paths": {
            name: str(path.relative_to(ROOT)) for name, path in PATHS.items() if path.is_file()
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_no_inference_frontier.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Semantic-Router No-Inference Frontier",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Claim readout: {payload['claim_readout']}",
        f"- Next action state: `{payload['next_action_state']}`",
        f"- Requires user approval: `{payload['requires_user_approval']}`",
        f"- Large external data: `{payload['large_external_data']}`",
        "",
        "## Prediction-Only Alternatives",
        "",
        _table(rows),
        "",
        "## Decision Boundary",
        "",
        "- This artifact reads existing prediction-only audits only.",
        "- It does not train models, run inference, download data, or create a new deployable method from label-revealed choices.",
        "- If calib10 stable smoke is not approved or later fails, restart the literature loop under the local-only constraint instead of promoting these weaker alternatives.",
        "",
    ]
    (OUT / "SEMANTIC_ROUTER_NO_INFERENCE_FRONTIER.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(json.dumps({k: payload[k] for k in ["verdict", "next_action_state", "requires_user_approval"]}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_NO_INFERENCE_FRONTIER.md")


if __name__ == "__main__":
    main()
