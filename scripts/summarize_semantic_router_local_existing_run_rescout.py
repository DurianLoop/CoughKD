from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT_DIR = RUNS / "semantic_router_local_existing_run_rescout"


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _pp(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{100.0 * x:+.2f} pp"


def _p(x: float | None) -> str:
    if x is None:
        return "NA"
    return f"{x:.3f}"


def _ci(low: float | None, high: float | None) -> str:
    return f"[{_pp(low)}, {_pp(high)}]"


def _subject_rows() -> list[dict[str, Any]]:
    paths = [
        RUNS / "subject_consistency_frontier" / "bootstrap_toscovid2021_test_ce_vs_source_only.json",
        RUNS / "subject_consistency_frontier" / "bootstrap_virufyseg_candidate_c_vs_source_only.json",
        RUNS / "subject_consistency_frontier" / "bootstrap_virufyseg_kd_vs_source_only.json",
    ]
    rows = []
    for path in paths:
        data = _load_json(path)
        rows.append(
            {
                "candidate": f"{data['target_tag']}:{'+'.join(data['candidate_methods'])}",
                "subjects": data["n_subjects"],
                "delta": data["point_delta"],
                "ci95_low": data["ci95_low"],
                "ci95_high": data["ci95_high"],
                "p_delta_le_0": data["p_delta_le_0"],
                "p_delta_lt_3pt": data["p_delta_lt_3pt"],
                "decision": "micro_stress_only" if data["target_tag"] == "virufyseg" else "fail_subject_stability",
            }
        )
    return rows


def build_summary() -> dict[str, Any]:
    controller = _load_json(
        RUNS
        / "target_calibrated_transfer_controller"
        / "target_calibrated_transfer_controller_summary.json"
    )
    oracle = _load_json(RUNS / "metadata_slice_oracle" / "metadata_slice_oracle_summary.json")
    slice_gate = _load_json(RUNS / "slice_balanced_gate_policy" / "slice_balanced_gate_policy_summary.json")
    random_control = _load_json(
        RUNS / "random_slice_negative_control" / "random_slice_negative_control_summary.json"
    )
    unlabeled = _load_json(
        RUNS / "unlabeled_prediction_geometry_selector" / "unlabeled_selector_summary.json"
    )
    selective = _load_json(RUNS / "selective_coverage_frontier" / "selective_coverage_summary.json")
    local_frontier = _load_json(
        RUNS
        / "semantic_router_local_dataset_frontier"
        / "semantic_router_local_dataset_frontier.json"
    )

    support_evidence = [
        {
            "name": "random_slice_negative_control",
            "role": "supports_semantic_slice_not_random",
            "delta": random_control["real_summary"]["mean_delta_vs_base"],
            "ci95_low": random_control["real_summary"]["ci95_low"],
            "ci95_high": random_control["real_summary"]["ci95_high"],
            "p_delta_le_0": random_control["real_summary"]["p_delta_le_0"],
            "p_control_mean_ge_real_mean": random_control["p_control_mean_ge_real_mean"],
            "claim_use": "COUGHVID symptom_resp gate is not reproduced by random partitions.",
        },
        {
            "name": "slice_balanced_gate_policy",
            "role": "supports_resp_symptom_mechanism",
            "delta": slice_gate["overall_delta"]["mean"],
            "ci95_low": slice_gate["overall_delta"]["ci95_low"],
            "ci95_high": slice_gate["overall_delta"]["ci95_high"],
            "p_delta_le_0": slice_gate["overall_delta"]["p_delta_le_0"],
            "p_delta_lt_3pt": slice_gate["overall_delta"]["p_delta_lt_3pt"],
            "claim_use": "COUGHVID respiratory-symptom slice carries most of the gain.",
        },
        {
            "name": "metadata_slice_oracle",
            "role": "upper_bound_only",
            "clears_3pt_large_target": oracle["clears_3pt_large_target"],
            "best_large_rows": oracle["best_large_rows"][:3],
            "claim_use": "Label-revealed upper bound motivates semantic routing but is not itself deployable.",
        },
    ]

    failed_or_insufficient = [
        {
            "name": "target_calibrated_transfer_controller",
            "decision": "fail_large_target_3pt_stability",
            "rows": controller["summary"],
            "reason": "Mean gains are below 3 pp or confidence intervals cross zero on both large targets.",
        },
        {
            "name": "unlabeled_prediction_geometry_selector",
            "decision": "fail_worst_target_safety",
            "best_selector": unlabeled["best_selector"],
            "reason": "Best unlabeled selector has negative worst-target delta and no large-target 3 pp gate.",
        },
        {
            "name": "selective_coverage_frontier",
            "decision": "fail_full_target_and_subset_bias",
            "clears_3pt_large_vs_source_full": selective["clears_3pt_large_vs_source_full"],
            "clears_3pt_large_vs_source_same_subset": selective["clears_3pt_large_vs_source_same_subset"],
            "best_large_vs_source_full": selective["best_large_vs_source_full"][:3],
            "reason": "Selective subsets improve some visible rows but do not clear the large-target full gate; same-subset controls expose selection effects.",
        },
        {
            "name": "subject_consistency_frontier",
            "decision": "fail_or_micro_stress_only",
            "rows": _subject_rows(),
            "reason": "TosCOVID subject ensemble is near zero; Virufyseg has positive point estimates but only 16 subjects and intervals cross zero.",
        },
        {
            "name": "local_dataset_frontier",
            "decision": local_frontier["verdict"],
            "next_action": local_frontier["next_action"],
            "reason": "Existing local data do not provide a non-duplicate robust third target.",
        },
    ]

    return {
        "verdict": "NO_NEW_LOCAL_CLAIM_FROM_EXISTING_RUNS",
        "recommended_claim_state": "KEEP_SEMANTIC_ROUTER_AS_MAIN_CANDIDATE_NO_UPGRADE",
        "support_evidence": support_evidence,
        "failed_or_insufficient_candidates": failed_or_insufficient,
        "next_low_cost_actions": [
            "Keep the semantic-router claim boundary narrow: target metadata field semantics as a safety prior for routing.",
            "Use random-slice and slice-balanced gate policy as support evidence, not as a separate novelty claim.",
            "Do not claim selective coverage, generic target-calibrated control, unlabeled geometry selection, or Virufyseg as a robust third-target result.",
            "Continue local-only scouting first; ask before any heavy training or new external data acquisition.",
        ],
    }


def write_report(summary: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "semantic_router_local_existing_run_rescout.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines: list[str] = []
    lines.append("# Semantic-Router Local Existing-Run Rescout")
    lines.append("")
    lines.append(f"- Verdict: `{summary['verdict']}`")
    lines.append(f"- Recommended claim state: `{summary['recommended_claim_state']}`")
    lines.append(
        "- Interpretation: existing local runs strengthen the current semantic-router story, "
        "but they do not expose a separate claimable innovation or a robust local third target."
    )
    lines.append("")
    lines.append("## Support Evidence")
    lines.append("")
    lines.append("| evidence | role | key result | claim use |")
    lines.append("| --- | --- | --- | --- |")
    for item in summary["support_evidence"]:
        if item["name"] == "metadata_slice_oracle":
            best = item["best_large_rows"][0]
            key = (
                f"best {best['target']}:{best['slice_column']} oracle "
                f"{_pp(best['delta_vs_source'])} vs source"
            )
        else:
            key = (
                f"delta {_pp(item['delta'])}, CI {_ci(item['ci95_low'], item['ci95_high'])}, "
                f"p<=0 {_p(item.get('p_delta_le_0'))}"
            )
        lines.append(f"| {item['name']} | {item['role']} | {key} | {item['claim_use']} |")
    lines.append("")
    lines.append("## Failed Or Insufficient Candidates")
    lines.append("")
    lines.append("| candidate | decision | key reason |")
    lines.append("| --- | --- | --- |")
    for item in summary["failed_or_insufficient_candidates"]:
        lines.append(f"| {item['name']} | `{item['decision']}` | {item['reason']} |")
    lines.append("")
    lines.append("## Subject Bootstrap Details")
    lines.append("")
    lines.append("| candidate | subjects | delta | CI95 | p<=0 | p<3pp | decision |")
    lines.append("| --- | ---: | ---: | --- | ---: | ---: | --- |")
    subject_item = next(
        item for item in summary["failed_or_insufficient_candidates"] if item["name"] == "subject_consistency_frontier"
    )
    for row in subject_item["rows"]:
        lines.append(
            f"| {row['candidate']} | {row['subjects']} | {_pp(row['delta'])} | "
            f"{_ci(row['ci95_low'], row['ci95_high'])} | {_p(row['p_delta_le_0'])} | "
            f"{_p(row['p_delta_lt_3pt'])} | `{row['decision']}` |"
        )
    lines.append("")
    lines.append("## Next Low-Cost Actions")
    lines.append("")
    for action in summary["next_low_cost_actions"]:
        lines.append(f"- {action}")
    lines.append("")

    (OUT_DIR / "SEMANTIC_ROUTER_LOCAL_EXISTING_RUN_RESCOUT.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main() -> None:
    summary = build_summary()
    write_report(summary)
    print(f"Wrote {OUT_DIR / 'SEMANTIC_ROUTER_LOCAL_EXISTING_RUN_RESCOUT.md'}")
    print(summary["verdict"])


if __name__ == "__main__":
    main()
