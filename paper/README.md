# CoughKD-ShiftAudit Paper Draft

This directory contains a top-conference-style manuscript scaffold for:

> CoughKD-ShiftAudit: Failure Cartography for Ultra-Light Cough Audio Distillation under Dataset Shift

The current draft is intentionally conservative. It is no longer written as a method-superiority paper. The supported story is a deployment reliability audit: KD gains are weak or unstable under cough dataset shift, and aggregate AUROC, calibration, probes, target slices, and clip-vs-subject evaluation can disagree.

## Files

- `main.tex`: compilable LaTeX manuscript scaffold.
- `sections/`: paper sections.
- `../runs/kd_failure_analysis/paper_ready_tables/paper_ready_tables.tex`: current paper-ready result tables.
- `../runs/kd_failure_analysis/figures/`: current paper-ready audit figures.
- `references.bib`: core bibliography for the first draft.
- `../docs/paper_draft_shift_audit.md`: prose draft and claim ledger.
- `../docs/progress/5.47_2026_literature_recheck_and_claim_boundary.md`: novelty boundary.

## Compile

From this directory:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or from the repository root, run the full validation flow:

```bash
scripts/validate_project.sh
```

For a real ICASSP submission, replace the current standard article wrapper in `main.tex` with the official ICASSP 2027 author kit after it is released, then re-check page limits and formatting.

## Writing Rule

Do not convert any target number into a result unless it is backed by a saved experiment log, exact split definition, and reproducible evaluation script.

Measured results now include multi-seed COUGHVID external audit runs, Virufy tiny stress targets, subject-level Virufy segmented aggregation, calibration/efficiency summaries, and overlap-control reports. These support an audit-protocol claim, not a clinical diagnosis claim or a robust new-KD-method claim.
