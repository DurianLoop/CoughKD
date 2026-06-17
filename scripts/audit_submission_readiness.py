"""Audit whether CoughKD-ShiftAudit is ready for an ICASSP-strength claim."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "submission_readiness"


REQUIRED_FILES = {
    "paper_pdf": ROOT / "paper" / "main.pdf",
    "paper_tex": ROOT / "paper" / "main.tex",
    "paper_tables_tex": RUNS / "kd_failure_analysis" / "paper_ready_tables" / "paper_ready_tables.tex",
    "main_table": RUNS / "kd_failure_analysis" / "SHIFT_AUDIT_MAIN_TABLE.md",
    "multitarget_table": RUNS / "kd_failure_analysis" / "SHIFT_AUDIT_MULTITARGET_TABLE.md",
    "claim_box": RUNS / "kd_failure_analysis" / "paper_ready_tables" / "claim_box.md",
    "calibration_efficiency": RUNS / "calibration_efficiency" / "summary.json",
    "kd_failure_summary": RUNS / "kd_failure_analysis" / "summary.json",
    "slice_audit": RUNS / "coughvid_slice_guard" / "COUGHVID_SLICE_GUARD_AUDIT.md",
    "bootstrap_sensitivity": RUNS / "coughvid_bootstrap_sensitivity" / "BOOTSTRAP_SENSITIVITY.md",
    "protocol_figure": RUNS / "kd_failure_analysis" / "figures" / "shift_audit_protocol.pdf",
    "macro_ece_figure": RUNS / "kd_failure_analysis" / "figures" / "external_macro_vs_ece.pdf",
    "domain_probe_figure": RUNS / "kd_failure_analysis" / "figures" / "external_macro_vs_domain_probe.pdf",
    "slice_gap_figure": RUNS / "kd_failure_analysis" / "figures" / "slice_delta_gap.pdf",
    "bootstrap_figure": RUNS / "kd_failure_analysis" / "figures" / "bootstrap_delta_distribution.pdf",
    "claim_boundary_audit": RUNS / "claim_boundary" / "CLAIM_BOUNDARY_AUDIT.md",
    "icassp_draft_quality": RUNS / "icassp_draft_quality" / "ICASSP_DRAFT_QUALITY.md",
    "ukcovid_metadata_audit": RUNS / "ukcovid_open_metadata_audit" / "UKCOVID_METADATA_AUDIT.md",
    "ukcovid_overlap_audit": RUNS / "overlap_audit" / "coswara_vs_ukcovid_open_test" / "OVERLAP_AUDIT.md",
    "ukcovid_audio_ready_check": RUNS / "ukcovid_open_metadata_audit" / "UKCOVID_AUDIO_READY_CHECK.md",
}


def _exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _target_strength(row: dict[str, str]) -> str:
    target = row.get("target", "")
    caveat = row.get("caveat", "").lower()
    n = int(float(row.get("n_examples") or 0))
    if target == "coughvid" and n >= 1000:
        return "main_external"
    if "tiny" in caveat or "segmented" in caveat or n < 200:
        return "stress_only"
    return "candidate_external"


def _multitarget_status() -> dict[str, Any]:
    rows = _read_csv(RUNS / "kd_failure_analysis" / "shift_audit_multitarget_table.csv")
    targets = []
    for row in rows:
        item = dict(row)
        item["strength"] = _target_strength(row)
        targets.append(item)
    main_count = sum(1 for row in targets if row["strength"] == "main_external")
    candidate_count = sum(1 for row in targets if row["strength"] in {"main_external", "candidate_external"})
    stress_count = sum(1 for row in targets if row["strength"] == "stress_only")
    return {
        "targets": targets,
        "main_or_candidate_external_count": candidate_count,
        "main_external_count": main_count,
        "stress_only_count": stress_count,
        "has_second_large_external": candidate_count >= 2,
    }


def _overlap_status() -> dict[str, Any]:
    summaries = []
    for path in sorted((RUNS / "overlap_audit").glob("*/overlap_summary.json")):
        data = _read_json(path)
        data["path"] = str(path.relative_to(ROOT))
        summaries.append(data)
    return {
        "num_overlap_audits": len(summaries),
        "audits": summaries,
        "all_existing_audits_clean": all(item.get("num_matches", 1) == 0 for item in summaries) if summaries else False,
    }


def _paper_status() -> dict[str, Any]:
    tex = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8") if (ROOT / "paper" / "main.tex").is_file() else ""
    return {
        "pdf_exists": _exists(ROOT / "paper" / "main.pdf"),
        "uses_shift_audit_title": "CoughKD-ShiftAudit" in tex,
        "contains_old_method_title": "Distilling Audio Foundation Models into Compact Cough Screening Networks" in tex,
    }


def _metric_status() -> dict[str, Any]:
    rows = _read_csv(RUNS / "kd_failure_analysis" / "shift_audit_main_table.csv")
    by_method = {row["method"]: row for row in rows}
    source = by_method.get("source_only", {})
    best = None
    if rows:
        best = max(rows, key=lambda row: float(row.get("external_macro_auroc") or "-inf"))
    return {
        "num_methods": len(rows),
        "source_external_macro": float(source.get("external_macro_auroc", "nan")) if source else None,
        "best_method": best.get("method") if best else None,
        "best_external_macro": float(best.get("external_macro_auroc", "nan")) if best else None,
        "best_delta_vs_source": float(best.get("delta_vs_source", "nan")) if best else None,
        "method_claim_supported": bool(best and float(best.get("delta_vs_source", 0.0)) >= 0.003),
    }


def _readiness_verdict(file_status: dict[str, bool], target_status: dict[str, Any], paper_status: dict[str, Any], metric_status: dict[str, Any]) -> tuple[str, list[str]]:
    blockers: list[str] = []
    missing = [name for name, ok in file_status.items() if not ok]
    if missing:
        blockers.append("Missing required evidence files: " + ", ".join(missing))
    if not paper_status["pdf_exists"]:
        blockers.append("Paper PDF has not been generated.")
    if paper_status["contains_old_method_title"]:
        blockers.append("Paper still contains old method-superiority title.")
    if not target_status["has_second_large_external"]:
        blockers.append("No second large independent/candidate external target yet.")
    if metric_status["method_claim_supported"]:
        blockers.append("A method-style claim may be possible, but it must be checked on a second target before use.")

    if missing or not paper_status["pdf_exists"]:
        return "NO_GO_INCOMPLETE_ARTIFACTS", blockers
    if not target_status["has_second_large_external"]:
        return "CONDITIONAL_GO_AUDIT_PAPER_NEEDS_SECOND_EXTERNAL", blockers
    return "GO_FOR_FULL_SHIFT_AUDIT_CLAIM_PENDING_FINAL_WRITING", blockers


def _write_report(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "readiness_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# CoughKD-ShiftAudit Submission Readiness",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Generated files checked: `{sum(payload['file_status'].values())}/{len(payload['file_status'])}`",
        f"- Main/candidate external targets: `{payload['target_status']['main_or_candidate_external_count']}`",
        f"- Stress-only targets: `{payload['target_status']['stress_only_count']}`",
        f"- Best COUGHVID method: `{payload['metric_status']['best_method']}`",
        f"- Best delta vs source: `{payload['metric_status']['best_delta_vs_source']}`",
        "",
        "## Blockers / Caveats",
        "",
    ]
    if payload["blockers"]:
        for item in payload["blockers"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None detected by the automated audit.")
    lines.extend(
        [
            "",
            "## Target Classification",
            "",
            "| Target | n | Strength | Caveat |",
            "|---|---:|---|---|",
        ]
    )
    for row in payload["target_status"]["targets"]:
        lines.append(f"| {row.get('target')} | {row.get('n_examples')} | {row.get('strength')} | {row.get('caveat')} |")
    lines.extend(
        [
            "",
            "## Required Next Action",
            "",
        ]
    )
    if payload["verdict"] == "CONDITIONAL_GO_AUDIT_PAPER_NEEDS_SECOND_EXTERNAL":
        lines.append("Obtain or construct one larger second external/candidate target. Preferred: Cambridge COVID-19 Sounds / ComParE CCS. Backup: CODA TB with reframed generic cough deployment reliability, or DiCOVA as overlap-controlled auxiliary only.")
    elif payload["verdict"].startswith("NO_GO"):
        lines.append("Regenerate missing artifacts and rerun this audit.")
    else:
        lines.append("Proceed to final writing, ICASSP template migration, and reviewer-risk polishing.")
    (OUT / "SUBMISSION_READINESS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    file_status = {name: _exists(path) for name, path in REQUIRED_FILES.items()}
    target_status = _multitarget_status()
    overlap_status = _overlap_status()
    paper_status = _paper_status()
    metric_status = _metric_status()
    verdict, blockers = _readiness_verdict(file_status, target_status, paper_status, metric_status)
    payload = {
        "verdict": verdict,
        "blockers": blockers,
        "file_status": file_status,
        "target_status": target_status,
        "overlap_status": overlap_status,
        "paper_status": paper_status,
        "metric_status": metric_status,
    }
    _write_report(payload)
    print(json.dumps({"verdict": verdict, "blockers": blockers}, indent=2))
    print(OUT / "SUBMISSION_READINESS.md")


if __name__ == "__main__":
    main()
