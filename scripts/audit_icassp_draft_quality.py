"""Audit the ICASSP-style short draft for submission-risk regressions."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper_icassp2027"
OUT = ROOT / "runs" / "icassp_draft_quality"


REQUIRED_FILES = {
    "main_tex": PAPER_DIR / "main.tex",
    "main_pdf": PAPER_DIR / "main.pdf",
    "main_log": PAPER_DIR / "main.log",
    "main_bbl": PAPER_DIR / "main.bbl",
    "readme": PAPER_DIR / "README.md",
    "temporary_style": PAPER_DIR / "spconf.sty",
    "temporary_bibstyle": PAPER_DIR / "IEEEbib.bst",
    "page_rule_audit": ROOT / "runs" / "icassp_page_rule" / "icassp_page_rule.json",
}


REQUIRED_TEX_ANCHORS = {
    "shift_audit_title": "CoughKD-ShiftAudit",
    "bounded_audit_protocol": "bounded audit protocol",
    "ultra_light_student": "ultra-light",
    "two_external_targets": "COUGHVID plus Tos COVID-19",
    "not_sota_or_first_kd": "not a new SOTA cough classifier or a first KD method",
    "audit_template_boundary": "audit template",
    "diagnostic_boundary": "not a diagnostic claim",
    "deployment_lessons": "three deployment lessons",
    "external_validation_claim": "insufficient evidence for deployable KD robustness without external validation",
    "loss_guard_formula": r"\label{eq:guard}",
}


REQUIRED_BIB_ANCHORS = {
    "ShiftKD": "shiftkd2025",
    "Coughprint": "coughprint2025",
    "OPERA": "opera2024",
    "Respiratory KD": "toikkanen2025respiratorykd",
    "DNDT/DNDF": "saravanan2026dndt",
    "CoughViT": "luong2025coughvit",
    "Zimmer 2026 external cohort": "zimmer2026coughacoustic",
    "TB cough 2026": "kafentzis2026tb",
    "BTS metadata-aided respiratory audio-text": "kim2024bts",
    "AcuLa semantic-teacher medical audio": "wang2026acula",
    "RespiraMFM respiratory audio-language": "siam2026respiramfm",
}


README_ANCHORS = {
    "icasSP_style_draft": "ICASSP-style draft workspace",
    "not_final_verified": "not as a final format-verified ICASSP 2027 submission package",
    "replace_author_kit": "replace `spconf.sty` / `IEEEbib.bst`",
}


FORBIDDEN_PATTERNS = {
    "clinical_diagnosis_claim": r"\b(clinically reliable|clinical utility|diagnose covid|diagnosis system|detects covid-19 from cough)\b",
    "method_superiority_claim": r"\b(state[- ]of[- ]the[- ]art|SOTA|outperforms all|best-performing|robustly outperforms)\b",
    "first_kd_claim": r"\b(first|novel)\s+(KD|knowledge distillation|cough representation|respiratory audio|COVID cough|cough KD|respiratory KD)\b",
    "biomarker_claim": r"\b(biomarker|acoustic signature proves|causal acoustic)\b",
}


NEGATION_MARKERS = (
    "not a new",
    "not as a",
    "not support",
    "not sufficient",
    "not a diagnostic",
    "not about",
    "not interpret",
    "do not claim",
    "cannot claim",
    "rather than claiming",
    "without external",
)


def _read(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _exists(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def _log_status(log_text: str) -> dict[str, Any]:
    page_match = re.findall(r"Output written on .*?\((\d+) pages?,", log_text)
    pages = int(page_match[-1]) if page_match else None
    return {
        "pages": pages,
        "has_output_written": bool(page_match),
        "has_undefined": bool(re.search(r"\b(undefined|Undefined)\b", log_text)),
        "has_latex_error": "LaTeX Error" in log_text,
        "has_overfull": "Overfull" in log_text,
    }


def _forbidden_hits(tex: str) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    for label, pattern in FORBIDDEN_PATTERNS.items():
        for match in re.finditer(pattern, tex, flags=re.IGNORECASE):
            start = max(0, match.start() - 160)
            end = min(len(tex), match.end() + 160)
            context = re.sub(r"\s+", " ", tex[start:end]).strip()
            if any(marker in context.lower() for marker in NEGATION_MARKERS):
                continue
            hits.append({"label": label, "match": match.group(0), "context": context})
    return hits


def _write_report(payload: dict[str, Any]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icassp_draft_quality_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# ICASSP Draft Quality Audit",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- PDF pages: `{payload['log_status']['pages']}`",
        f"- Missing files: `{len(payload['missing_files'])}`",
        f"- Missing TeX anchors: `{len(payload['missing_tex_anchors'])}`",
        f"- Missing bibliography anchors: `{len(payload['missing_bib_anchors'])}`",
        f"- Missing README anchors: `{len(payload['missing_readme_anchors'])}`",
        f"- Forbidden claim hits: `{len(payload['forbidden_hits'])}`",
        f"- Page rule verdict: `{payload['page_rule_verdict']}`",
        "",
        "## Log Checks",
        "",
        "| Check | Result |",
        "|---|---:|",
    ]
    for key in ["has_output_written", "has_undefined", "has_latex_error", "has_overfull"]:
        lines.append(f"| {key} | {payload['log_status'][key]} |")

    lines.extend(["", "## Missing Anchors", ""])
    for label, missing in [
        ("Files", payload["missing_files"]),
        ("TeX", payload["missing_tex_anchors"]),
        ("Bibliography", payload["missing_bib_anchors"]),
        ("README", payload["missing_readme_anchors"]),
    ]:
        lines.append(f"### {label}")
        if missing:
            lines.extend(f"- {item}" for item in missing)
        else:
            lines.append("- None.")
        lines.append("")

    lines.extend(["## Forbidden Claim Hits", ""])
    if payload["forbidden_hits"]:
        for hit in payload["forbidden_hits"]:
            lines.append(f"- `{hit['label']}`: {hit['context']}")
    else:
        lines.append("- None.")

    lines.extend(["", "## Page Rule", ""])
    lines.append(
        f"- ICASSP page-rule audit verdict: `{payload['page_rule_verdict']}`"
    )
    lines.append(
        f"- Extracted page count: `{payload['page_rule'].get('page_count', 'missing')}`"
    )

    lines.extend(
        [
            "",
            "## Verdict Interpretation",
            "",
            "A `PASS` means the ICASSP-style draft still satisfies the current automated quality gate. It does not mean the paper is accepted or final-format verified; the official ICASSP 2027 author kit still needs to be checked before submission.",
        ]
    )
    (OUT / "ICASSP_DRAFT_QUALITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    tex = _read(PAPER_DIR / "main.tex")
    log_text = _read(PAPER_DIR / "main.log")
    bbl = _read(PAPER_DIR / "main.bbl")
    readme = _read(PAPER_DIR / "README.md")
    page_rule = _read_json(REQUIRED_FILES["page_rule_audit"])

    file_status = {name: _exists(path) for name, path in REQUIRED_FILES.items()}
    missing_files = [name for name, ok in file_status.items() if not ok]
    tex_status = {name: _contains(tex, needle) for name, needle in REQUIRED_TEX_ANCHORS.items()}
    bib_text = "\n".join([tex, bbl, _read(ROOT / "paper" / "references.bib")])
    bib_status = {name: _contains(bib_text, needle) for name, needle in REQUIRED_BIB_ANCHORS.items()}
    readme_status = {name: _contains(readme, needle) for name, needle in README_ANCHORS.items()}
    log_status = _log_status(log_text)
    forbidden_hits = _forbidden_hits(tex)

    missing_tex_anchors = [name for name, ok in tex_status.items() if not ok]
    missing_bib_anchors = [name for name, ok in bib_status.items() if not ok]
    missing_readme_anchors = [name for name, ok in readme_status.items() if not ok]

    blockers: list[str] = []
    if missing_files:
        blockers.append("missing_files")
    if missing_tex_anchors:
        blockers.append("missing_tex_anchors")
    if missing_bib_anchors:
        blockers.append("missing_bib_anchors")
    if missing_readme_anchors:
        blockers.append("missing_readme_anchors")
    if log_status["pages"] not in {4, 5}:
        blockers.append("page_count_not_4_or_5")
    if log_status["has_undefined"]:
        blockers.append("undefined_reference_or_citation")
    if log_status["has_latex_error"]:
        blockers.append("latex_error")
    if log_status["has_overfull"]:
        blockers.append("overfull_hbox")
    if forbidden_hits:
        blockers.append("forbidden_claim_hits")
    if page_rule.get("verdict") != "PAGE_RULE_PASS":
        blockers.append("icassp_page_rule_not_pass")

    verdict = "PASS" if not blockers else "REVIEW_NEEDED"
    payload = {
        "verdict": verdict,
        "blockers": blockers,
        "file_status": file_status,
        "missing_files": missing_files,
        "tex_status": tex_status,
        "missing_tex_anchors": missing_tex_anchors,
        "bib_status": bib_status,
        "missing_bib_anchors": missing_bib_anchors,
        "readme_status": readme_status,
        "missing_readme_anchors": missing_readme_anchors,
        "log_status": log_status,
        "forbidden_hits": forbidden_hits,
        "page_rule": page_rule,
        "page_rule_verdict": page_rule.get("verdict", "missing"),
    }
    _write_report(payload)
    print(json.dumps({"verdict": verdict, "blockers": blockers}, indent=2))
    print(OUT / "ICASSP_DRAFT_QUALITY.md")


if __name__ == "__main__":
    main()
