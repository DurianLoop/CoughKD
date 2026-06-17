"""Audit paper text for over-claiming and missing novelty-boundary anchors."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "claim_boundary"

TEXT_FILES = [
    ROOT / "paper" / "main.tex",
    ROOT / "paper" / "sections" / "01_introduction.tex",
    ROOT / "paper" / "sections" / "02_related_work.tex",
    ROOT / "paper" / "sections" / "06_discussion.tex",
    ROOT / "docs" / "paper_draft_shift_audit.md",
]

FORBIDDEN_PATTERNS = {
    "clinical_diagnosis_claim": r"\b(clinically reliable|clinical utility|diagnose covid|diagnosis system|detects covid-19 from cough)\b",
    "sota_claim": r"\b(state[- ]of[- ]the[- ]art|SOTA|outperforms all|best-performing)\b",
    "first_claim": r"\b(first|novel)\b.*\b(KD|knowledge distillation|cough representation|respiratory audio|COVID cough)\b",
    "biomarker_claim": r"\b(biomarker|acoustic signature proves|causal)\b",
}

REQUIRED_ANCHORS = {
    "ShiftKD": "ShiftKD",
    "Coughprint": "Coughprint",
    "OPERA": "OPERA",
    "Respiratory KD": "Respiratory sound KD",
    "DNDT/DNDF": "DNDT",
    "CoughViT": "CoughViT",
    "UK COVID dataset": "UK COVID-19 Vocal Audio Dataset",
    "symptom checker caution": "symptom checker",
    "non-clinical boundary": "does not aim to prove",
    "second external blocker": "second external",
}

NEGATION_MARKERS = (
    "does not aim",
    "do not aim",
    "does not support",
    "not support",
    "not claim",
    "do not claim",
    "cannot claim",
    "avoid wording",
    "not presented as",
    "without strong external",
    "weak evidence",
)


def _read_all() -> dict[str, str]:
    texts = {}
    for path in TEXT_FILES:
        if path.is_file():
            texts[str(path.relative_to(ROOT))] = path.read_text(encoding="utf-8", errors="ignore")
    return texts


def _context(text: str, match: re.Match[str], width: int = 180) -> str:
    start = max(0, match.start() - width)
    end = min(len(text), match.end() + width)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def main() -> None:
    texts = _read_all()
    joined = "\n".join(texts.values())
    lower_joined = joined.lower()

    forbidden_hits = []
    for label, pattern in FORBIDDEN_PATTERNS.items():
        regex = re.compile(pattern, flags=re.IGNORECASE)
        for rel_path, text in texts.items():
            for match in regex.finditer(text):
                context = _context(text, match)
                if any(marker in context.lower() for marker in NEGATION_MARKERS):
                    continue
                forbidden_hits.append(
                    {
                        "label": label,
                        "file": rel_path,
                        "match": match.group(0),
                        "context": context,
                    }
                )

    required_status = {
        label: (needle.lower() in lower_joined)
        for label, needle in REQUIRED_ANCHORS.items()
    }
    missing_required = [label for label, ok in required_status.items() if not ok]
    verdict = "PASS" if not forbidden_hits and not missing_required else "REVIEW_NEEDED"

    payload = {
        "verdict": verdict,
        "num_forbidden_hits": len(forbidden_hits),
        "forbidden_hits": forbidden_hits,
        "required_status": required_status,
        "missing_required": missing_required,
        "files_checked": sorted(texts),
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "claim_boundary_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Claim Boundary Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Files checked: `{len(texts)}`",
        f"- Forbidden/over-claim hits: `{len(forbidden_hits)}`",
        f"- Missing required anchors: `{len(missing_required)}`",
        "",
        "## Required Anchors",
        "",
        "| Anchor | Present |",
        "|---|---:|",
    ]
    for label, ok in required_status.items():
        lines.append(f"| {label} | {ok} |")
    lines.extend(["", "## Forbidden Hits", ""])
    if forbidden_hits:
        for hit in forbidden_hits:
            lines.append(f"- `{hit['label']}` in `{hit['file']}`: {hit['context']}")
    else:
        lines.append("- None.")
    (OUT / "CLAIM_BOUNDARY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"verdict": verdict, "missing_required": missing_required, "num_forbidden_hits": len(forbidden_hits)}, indent=2))
    print(OUT / "CLAIM_BOUNDARY_AUDIT.md")


if __name__ == "__main__":
    main()
