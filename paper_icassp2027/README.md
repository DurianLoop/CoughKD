# ICASSP 2027 Draft Workspace

This directory contains the compact ICASSP-style draft for the current CoughKD-ShiftAudit paper.

## Files

- `main.tex`: self-contained ICASSP-style draft.
- `spconf.sty`: temporary style file from the ICASSP 2026 author kit.
- `IEEEbib.bst`: bibliography style from the ICASSP 2026 author kit.
- `compile_windows.cmd`: one-command Windows compilation script.

The official ICASSP 2027 author kit should replace the temporary 2026 style files once it is released. As of 2026-06-09, the official ICASSP 2027 CFP PDF confirms the 4-page technical-content rule plus an optional fifth page for references/funding/ethics only, but a 2027-specific final author kit/template was not located. This directory should still be treated as an ICASSP-style draft workspace, not as a final format-verified ICASSP 2027 submission package.

## Compile

```cmd
cd /d D:\CoughKD\AAAI\paper_icassp2027
compile_windows.cmd
```

Expected output:

- `main.pdf`
- 5 pages, with page 5 reserved for references under the current ICASSP 2027 CFP page rule
- no undefined citations/references
- no LaTeX errors
- no overfull hbox in the final log

## Current Claim Boundary

The draft supports a bounded audit-plus-routing claim:

> CoughKD-ShiftAudit shows that common ultra-light cough KD variants are target-sensitive under public-dataset shift, and that target metadata semantics can act as a safety prior for moderate-label transfer routing.

The draft should not claim:

- clinical COVID-19 cough diagnosis utility;
- causal acoustic COVID biomarkers;
- SOTA cough classifier performance;
- a new KD method with robust cross-dataset superiority;
- first metadata use, first symptom-assisted cough modeling, first safe transfer, or first metadata-gated audio adaptation.

Supporting novelty matrix:

- `../runs/semantic_router_novelty_matrix/SEMANTIC_ROUTER_NOVELTY_MATRIX.md`
- `../runs/semantic_router_novelty_matrix/semantic_router_novelty_matrix.tex`

Current readiness audit:

- claim-boundary: `CREDIBLE_CANDIDATE`
- submission readiness: `PENDING_FINAL_2027_AUTHOR_KIT_AND_THIRD_TARGET_OPTION`
- official status note: `../runs/semantic_router_submission_readiness/ICASSP_2027_OFFICIAL_STATUS.md`

Before submission:

- re-check the official ICASSP 2027 author kit/template;
- replace `spconf.sty` / `IEEEbib.bst` if the official kit changes;
- recompile and rerun visual QA plus claim/readiness audits.
