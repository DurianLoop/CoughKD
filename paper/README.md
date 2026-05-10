# CoughKD Paper Draft

This directory contains a top-conference-style manuscript scaffold for:

> CoughKD: Distilling Audio Foundation Models into Compact Cough Screening Networks

The current draft is intentionally conservative: target results are marked as targets, and empirical claims are tracked in `notes/claim_ledger.md`.

## Files

- `main.tex`: compilable LaTeX manuscript scaffold.
- `sections/`: paper sections.
- `tables/`: planned result and ablation tables.
- `tables/coswara_split_summary.tex`: measured filtered Coswara split summary.
- `tables/coswara_engineering_results.tex`: measured compact teacher/student engineering baseline.
- `references.bib`: core bibliography for the first draft.
- `notes/experimental_protocol.md`: implementation-facing experiment protocol.
- `notes/review_log.md`: research-direction review notes.
- `notes/claim_ledger.md`: claim safety ledger.
- `../docs/long_training_plan.md`: long-duration GPU training readiness gate.
- `../docs/clickup_long_training.md`: ClickUp-style task list for pre-long-training work.

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

For a real AAAI submission, replace the current standard article wrapper in `main.tex` with the official AAAI template after downloading the current author kit, then re-check page limits and formatting.

## Writing Rule

Do not convert any target number into a result unless it is backed by a saved experiment log, exact split definition, and reproducible evaluation script.

Measured results now include the compact Coswara engineering baseline from `runs/coswara_torch_closed_loop_full/` and the first PANNs CNN14 16 kHz foundation-teacher long run from `runs/long_coswara_panns_cnn14_16k_seed7_e30/`. These are valid evidence that the pipeline runs end to end with both compact and one pretrained-teacher path, but they are still single-seed in-domain results rather than final clinical or cross-dataset claims.
