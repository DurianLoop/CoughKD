"""Build paper-ready Markdown and LaTeX tables for CoughKD-ShiftAudit."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "kd_failure_analysis"
PAPER_OUT = OUT / "paper_ready_tables"


MAIN_CSV = OUT / "shift_audit_main_table.csv"
MULTI_CSV = OUT / "shift_audit_multitarget_table.csv"
SUBJECT_CSV = RUNS / "subject_aggregate_virufyseg" / "method_summary.csv"


def _read_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                else:
                    try:
                        parsed[key] = float(value)
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def _fmt(value: Any, digits: int = 4, signed: bool = False) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if signed:
            return f"{value:+.{digits}f}"
        return f"{value:.{digits}f}"
    return str(value)


def _escape_tex(text: Any) -> str:
    s = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in s)


def _latex_table(
    headers: list[str],
    rows: list[list[Any]],
    caption: str,
    label: str,
    aligns: str | None = None,
    span_columns: bool = False,
) -> str:
    aligns = aligns or ("l" + "r" * (len(headers) - 1))
    env = "table*" if span_columns else "table"
    width = r"\textwidth" if span_columns else r"\linewidth"
    lines = [
        rf"\begin{{{env}}}[t]",
        r"\centering",
        r"\scriptsize",
        rf"\caption{{{_escape_tex(caption)}}}",
        rf"\label{{{label}}}",
        rf"\resizebox{{{width}}}{{!}}{{%",
        rf"\begin{{tabular}}{{{aligns}}}",
        r"\toprule",
        " & ".join(str(h) for h in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(str(cell) for cell in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"}", rf"\end{{{env}}}", ""])
    return "\n".join(lines)


def _markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _build_main_table(main_rows: list[dict[str, Any]]) -> tuple[str, str]:
    keep = ["source_only", "kd", "tcd_conf035", "candidate_a", "candidate_b", "candidate_c"]
    by_method = {row["method"]: row for row in main_rows}
    rows: list[list[str]] = []
    for method in keep:
        row = by_method[method]
        rows.append(
            [
                str(row["display"]),
                f"{_fmt(row['external_macro_auroc'])} $\\pm$ {_fmt(row['external_macro_std'])}",
                _fmt(row["delta_vs_source"], 4, signed=True),
                _fmt(row["external_covid_auroc"]),
                _fmt(row["external_macro_auprc"]),
                _fmt(row["ece"]),
                _fmt(row["domain_probe_auc"]),
                _fmt(row["task_probe_auc"]),
                _fmt(row["checkpoint_mb"], 3),
            ]
        )
    headers = ["Method", "Macro AUROC", "$\\Delta$", "COVID AUROC", "AUPRC", "ECE", "Domain AUC", "Task AUC", "MB"]
    caption = "COUGHVID external audit. Deltas are against source-only continuation. Lower ECE is better; lower domain AUC alone is not sufficient evidence of robust task transfer."
    return (
        _markdown_table(headers, rows),
        _latex_table(headers, rows, caption, "tab:coughvid_external_audit", aligns="lrrrrrrrr", span_columns=True),
    )


def _build_target_table(multi_rows: list[dict[str, Any]], subject_rows: list[dict[str, Any]]) -> tuple[str, str]:
    subject_by_method = {row["method"]: row for row in subject_rows}
    subject_source = subject_by_method["source_only"]
    subject_best = max(subject_rows, key=lambda row: float(row["macro_ovr_auroc_mean"]))
    md_rows: list[list[str]] = []
    tex_rows: list[list[str]] = []
    for row in multi_rows:
        base = [
            str(row["target"]),
            str(int(row["n_examples"])) if row["n_examples"] is not None else "-",
            "clip",
            _fmt(row["source_macro"]),
            f"{row['best_method']} ({_fmt(row['best_macro'])})",
            _fmt(row["best_delta"], 4, signed=True),
            str(row["guard_selected"]),
            _fmt(row["guard_delta"], 4, signed=True),
            str(row["caveat"]),
        ]
        md_rows.append(base)
        tex_rows.append([_escape_tex(cell) if idx in {0, 4, 6, 8} else cell for idx, cell in enumerate(base)])
    subject_base = [
        "virufyseg",
        str(int(subject_source["num_subjects"])),
        "subject",
        _fmt(subject_source["macro_ovr_auroc_mean"]),
        f"{subject_best['method']} ({_fmt(subject_best['macro_ovr_auroc_mean'])})",
        _fmt(subject_best["macro_delta_vs_source"], 4, signed=True),
        "tcd_conf035",
        _fmt(subject_by_method["tcd_conf035"]["macro_delta_vs_source"], 4, signed=True),
        "subject aggregate of 121 clips",
    ]
    md_rows.append(subject_base)
    tex_rows.append([_escape_tex(cell) if idx in {0, 4, 6, 8} else cell for idx, cell in enumerate(subject_base)])
    headers = ["Target", "n", "Unit", "Source", "Post-hoc best", "Best $\\Delta$", "Guard", "Guard $\\Delta$", "Caveat"]
    caption = "External target and evaluation-unit stress test. Virufy variants are stress targets only; the effective independent size for subject-level Virufy segmented is 16."
    return (
        _markdown_table(headers, md_rows),
        _latex_table(headers, tex_rows, caption, "tab:target_unit_stress", aligns="lrlrlrlrl", span_columns=True),
    )


def _build_claim_box() -> str:
    return "\n".join(
        [
            "# Paper-Ready Claim Box",
            "",
            "## Claim We Can Currently Support",
            "",
            "CoughKD-ShiftAudit is a deployment-oriented audit protocol showing that ultra-light cough audio distillation under dataset shift is method-sensitive, target-composition-sensitive, and evaluation-unit-sensitive. Across two independent public external targets, COUGHVID and Tos COVID-19 2021 official test, common KD/TCD variants provide weak, unstable, or negative gains: the best COUGHVID delta is only `+0.0007`, while on Tos the best post-hoc method is CE rather than KD and the guard-selected method is negative. Virufy stress targets further show that post-hoc best methods and guard selections can change across target size, preprocessing granularity, and clip-vs-subject evaluation.",
            "",
            "## Claim We Cannot Yet Support",
            "",
            "We cannot claim a new KD method with robust cross-dataset superiority, a SOTA COVID cough classifier, or clinical COVID cough diagnosis utility. Larger targets such as UK COVID-19 Vocal Audio or Cambridge COVID-19 Sounds / ComParE CCS would strengthen the empirical scope, but the current evidence is sufficient for a bounded ShiftAudit analysis claim rather than a method-superiority claim.",
            "",
            "## Best Current Title",
            "",
            "**CoughKD-ShiftAudit: Failure Cartography for Ultra-Light Cough Audio Distillation under Dataset Shift**",
            "",
        ]
    )


def main() -> None:
    main_rows = _read_csv(MAIN_CSV)
    multi_rows = _read_csv(MULTI_CSV)
    subject_rows = _read_csv(SUBJECT_CSV)

    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    main_md, main_tex = _build_main_table(main_rows)
    target_md, target_tex = _build_target_table(multi_rows, subject_rows)

    md = "\n\n".join(
        [
            "# CoughKD-ShiftAudit Paper-Ready Tables",
            "## Table 1: COUGHVID External Audit",
            main_md,
            "## Table 2: Target and Evaluation-Unit Stress Test",
            target_md,
            _build_claim_box(),
        ]
    )
    tex = "\n".join([main_tex, target_tex])
    (PAPER_OUT / "paper_ready_tables.md").write_text(md + "\n", encoding="utf-8")
    (PAPER_OUT / "paper_ready_tables.tex").write_text(tex, encoding="utf-8")
    (PAPER_OUT / "claim_box.md").write_text(_build_claim_box(), encoding="utf-8")
    print(PAPER_OUT / "paper_ready_tables.md")
    print(PAPER_OUT / "paper_ready_tables.tex")


if __name__ == "__main__":
    main()
