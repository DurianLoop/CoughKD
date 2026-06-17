from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_claim_consistency"


PATHS = {
    "main_tex": ROOT / "paper_icassp2027" / "main.tex",
    "main_text": ROOT / "paper_icassp2027" / "main_pdftotext.txt",
    "supplement_tex": RUNS / "semantic_router_supplement" / "main.tex",
    "supplement_text": RUNS / "semantic_router_supplement" / "main_pdftotext.txt",
    "novelty_matrix_tex": RUNS / "semantic_router_novelty_matrix" / "semantic_router_novelty_matrix.tex",
    "novelty_source_ledger": RUNS / "semantic_router_novelty_matrix" / "SEMANTIC_ROUTER_SOURCE_LEDGER.md",
    "claim_dossier": RUNS / "semantic_router_claim_dossier" / "SEMANTIC_ROUTER_CLAIM_DOSSIER.md",
    "readiness": RUNS / "semantic_router_submission_readiness" / "SEMANTIC_ROUTER_SUBMISSION_READINESS.md",
}


REQUIRED_ANCHORS = {
    "main_pdf_subject_grouped_boundary": {
        "path": "main_text",
        "needles": ["row-level mean", "+3.75", "+3.99", "subject-grouped", "+2.94", "conservative"],
    },
    "main_pdf_nonclinical_boundary": {
        "path": "main_text",
        "needles": [
            "not a diagnostic claim",
            "not about disease biology or clinical utility",
            "metadata-to-text or audio-language respiratory modeling",
        ],
    },
    "supplement_pdf_subject_grouped_boundary": {
        "path": "supplement_text",
        "needles": ["Subject-Grouped Resampling", "+3.99", "+2.94", "multi-record targets"],
    },
    "supplement_pdf_nonclaims": {
        "path": "supplement_text",
        "needles": [
            "does not claim first metadata use",
            "first safe transfer",
            "first respiratory metadata or multimodal modeling",
            "first metadata-guided transfer or adaptation",
            "clinical diagnostic utility",
            "not first metadata use",
        ],
    },
    "dossier_current_status": {
        "path": "claim_dossier",
        "needles": ["CREDIBLE_CANDIDATE_NOT_FINAL_READY", "Subject-disjoint resampling", "Third-target empirical scope"],
    },
    "novelty_source_ledger_boundary": {
        "path": "novelty_source_ledger",
        "needles": [
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
    },
    "novelty_matrix_tex_boundary": {
        "path": "novelty_matrix_tex",
        "needles": [
            "Clinical-metadata cough baselines",
            "TB cough audio + clinical metadata baseline",
            "Not first cough-audio clinical-metadata fusion",
            "metadata field semantics as a safety prior",
        ],
    },
    "readiness_current_status": {
        "path": "readiness",
        "needles": ["NOT_READY_THIRD_TARGET_OR_ARTIFACTS_MISSING", "third_target_ukcovid", "NOT_READY_AUDIO_AND_PREDICTIONS_MISSING"],
    },
}


OVERCLAIM_PHRASES = [
    "we are the first",
    "first to use metadata",
    "first metadata use",
    "first symptom-assisted cough model",
    "first safe-transfer method",
    "first metadata-gated audio adaptation",
    "first metadata-guided transfer",
    "clinical diagnostic utility",
    "general kd method superiority",
    "label-free method",
    "general claim that metadata helps",
    "guarantee against negative transfer",
    "ready for submission",
    "ready for final submission",
]


NEGATION_CUES = ["not ", "does not ", "do not ", "is not ", "not a ", "not first ", "not intended "]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower())


def _anchor_result(name: str, spec: dict[str, Any], texts: dict[str, str]) -> dict[str, Any]:
    text = _norm(texts.get(spec["path"], ""))
    found = {needle: needle.lower() in text for needle in spec["needles"]}
    return {
        "check": name,
        "pass": all(found.values()),
        "path": spec["path"],
        "found": found,
    }


def _is_negated(context: str, phrase: str) -> bool:
    idx = context.find(phrase)
    if idx < 0:
        return False
    prefix = context[max(0, idx - 40) : idx]
    full_prefix = context[:idx]
    return any(cue in prefix for cue in NEGATION_CUES) or "does not claim" in full_prefix


def _overclaim_hits(source_name: str, text: str) -> list[dict[str, str]]:
    hits = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        normalized = _norm(line)
        for phrase in OVERCLAIM_PHRASES:
            if phrase not in normalized:
                continue
            if _is_negated(normalized, phrase):
                continue
            hits.append(
                {
                    "source": source_name,
                    "line": str(line_no),
                    "phrase": phrase,
                    "context": line.strip(),
                }
            )
    return hits


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows._"
    cols = list(rows[0].keys())
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("\n", " ") for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    texts = {name: _read(path) for name, path in PATHS.items()}
    anchors = [_anchor_result(name, spec, texts) for name, spec in REQUIRED_ANCHORS.items()]
    overclaims = []
    for name in ["main_tex", "supplement_tex", "novelty_matrix_tex", "novelty_source_ledger"]:
        overclaims.extend(_overclaim_hits(name, texts[name]))

    missing_files = [name for name, path in PATHS.items() if not path.is_file()]
    passed = not missing_files and all(item["pass"] for item in anchors) and not overclaims
    verdict = "PASS" if passed else "BLOCK"

    payload = {
        "verdict": verdict,
        "missing_files": missing_files,
        "anchors": anchors,
        "overclaims": overclaims,
        "paths": {name: str(path.relative_to(ROOT)) for name, path in PATHS.items()},
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_claim_consistency.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Semantic-Router Claim Consistency Audit",
        "",
        f"- Verdict: `{verdict}`",
        f"- Missing files: `{len(missing_files)}`",
        f"- Overclaim hits: `{len(overclaims)}`",
        "",
        "## Anchor Checks",
        "",
        _markdown_table(
            [
                {
                    "check": item["check"],
                    "status": "PASS" if item["pass"] else "BLOCK",
                    "path": item["path"],
                    "missing": ", ".join(k for k, ok in item["found"].items() if not ok),
                }
                for item in anchors
            ]
        ),
        "",
        "## Overclaim Hits",
        "",
        _markdown_table(overclaims),
        "",
        "## Interpretation",
        "",
    ]
    if passed:
        lines.append("The main paper, supplement, readiness report, and claim dossier are internally consistent with the current conservative semantic-router claim boundary.")
    else:
        lines.append("The claim boundary is not internally consistent. Resolve missing anchors or overclaim hits before treating the draft as stable.")
    lines.append("")
    (OUT / "SEMANTIC_ROUTER_CLAIM_CONSISTENCY.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps({"verdict": verdict, "missing_files": missing_files, "overclaims": len(overclaims)}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_CLAIM_CONSISTENCY.md")


if __name__ == "__main__":
    main()
