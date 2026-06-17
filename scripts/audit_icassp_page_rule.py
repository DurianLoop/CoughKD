"""Audit the local ICASSP-style PDF against the current page-rule record.

The repository records the current ICASSP 2027 rule as 4 technical pages plus
an optional fifth page limited to references, funding acknowledgements, and
ethics statements. This script checks the already-extracted PDF text and does
not run LaTeX, inference, training, or downloads.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper_icassp2027"
OUT = ROOT / "runs" / "icassp_page_rule"


TEXT = PAPER / "main_pdftotext.txt"
PDF = PAPER / "main.pdf"
OFFICIAL_STATUS = ROOT / "runs" / "semantic_router_submission_readiness" / "ICASSP_2027_OFFICIAL_STATUS.md"


FORBIDDEN_PAGE5_PATTERNS = {
    "abstract": r"\babstract\b",
    "introduction": r"\bintroduction\b",
    "method_section": r"\b(coughkd-shiftaudit|semantic-constrained transfer routing|experimental setup|results|discussion|conclusion)\b",
    "table_or_figure": r"\b(table|figure|fig\.)\s+\d+\b",
    "equation_label": r"\b(eq\.|equation)\s*\(?\d+\)?",
    "main_claim_phrase": r"\btarget metadata semantics can act as a safety prior\b",
}


def _read(path: Path) -> str:
    if not path.is_file():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def _page_texts(text: str) -> list[str]:
    pages = text.split("\f")
    if pages and not pages[-1].strip():
        pages = pages[:-1]
    return pages


def _hits(text: str) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for label, pattern in FORBIDDEN_PAGE5_PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            context = re.sub(r"\s+", " ", text[start:end]).strip()
            out.append({"label": label, "match": match.group(0), "context": context})
    return out


def _official_rule_ok(text: str) -> bool:
    required = [
        "4 pages are allowed for technical content",
        "an optional fifth page is allowed",
        "the fifth page may contain only references",
    ]
    lower = text.lower()
    return all(item.lower() in lower for item in required)


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
    text = _read(TEXT)
    official_status = _read(OFFICIAL_STATUS)
    pages = _page_texts(text)
    page_count = len(pages)
    page5 = pages[4] if page_count >= 5 else ""
    page5_hits = _hits(page5)
    official_ok = _official_rule_ok(official_status)

    if not PDF.is_file() or not TEXT.is_file():
        verdict = "PAGE_RULE_MISSING_ARTIFACTS"
    elif page_count <= 4 and official_ok:
        verdict = "PAGE_RULE_PASS"
    elif page_count == 5 and official_ok and not page5_hits and bool(page5.strip()):
        verdict = "PAGE_RULE_PASS"
    elif page_count > 5:
        verdict = "PAGE_RULE_FAIL_TOO_MANY_PAGES"
    else:
        verdict = "PAGE_RULE_REVIEW_NEEDED"

    payload = {
        "verdict": verdict,
        "pdf": str(PDF.relative_to(ROOT)),
        "text": str(TEXT.relative_to(ROOT)),
        "page_count": page_count,
        "official_rule_record_present": official_ok,
        "page5_nonempty": bool(page5.strip()),
        "page5_forbidden_hits": page5_hits,
        "page5_preview": re.sub(r"\s+", " ", page5.strip())[:800],
        "rule": "4 technical pages plus optional fifth page limited to references/funding/ethics",
        "evidence_paths": {
            "official_status": str(OFFICIAL_STATUS.relative_to(ROOT)),
            "pdf_text": str(TEXT.relative_to(ROOT)),
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icassp_page_rule.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# ICASSP Page Rule Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- PDF pages from text extraction: `{page_count}`",
        f"- Official page-rule record present: `{official_ok}`",
        f"- Page 5 nonempty: `{payload['page5_nonempty']}`",
        f"- Page 5 forbidden hits: `{len(page5_hits)}`",
        f"- Rule: {payload['rule']}",
        "",
        "## Page 5 Forbidden Hits",
        "",
        _table(page5_hits),
        "",
        "## Page 5 Preview",
        "",
        payload["page5_preview"] or "_No fifth page._",
        "",
        "## Decision Boundary",
        "",
        "- This validates page-shape only; it does not replace final ICASSP 2027 author-kit verification.",
        "- It reads existing PDF text only and does not compile, train, infer, or download data.",
        "",
    ]
    (OUT / "ICASSP_PAGE_RULE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "page_count": page_count}, indent=2))
    print(OUT / "ICASSP_PAGE_RULE.md")


if __name__ == "__main__":
    main()
