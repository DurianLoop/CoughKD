"""Paper table generation with run-id traceability."""

from __future__ import annotations

import csv
import json
from pathlib import Path


def load_run_metrics(runs_dir: Path) -> dict[str, dict[str, object]]:
    runs = {}
    for metrics_path in sorted(runs_dir.glob("*/metrics.json")):
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        runs[payload["run_id"]] = payload
    return runs


def generate_paper_tables(runs_dir: Path, out_dir: Path, required_run_ids: list[str]) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_run_metrics(runs_dir)
    missing = [run_id for run_id in required_run_ids if run_id not in runs]
    if missing:
        raise ValueError(f"missing required run IDs: {missing}")

    rows = []
    for run_id in required_run_ids:
        payload = runs[run_id]
        row = {
            "run_id": run_id,
            "teacher": payload["config"].get("teacher", ""),
            "student": payload["config"].get("student", ""),
            "kd": payload["config"].get("kd", ""),
            "deployment": payload["config"].get("deployment", ""),
            "auroc": payload["metrics"].get("auroc", ""),
            "auprc": payload["metrics"].get("auprc", ""),
            "macro_f1": payload["metrics"].get("macro_f1", ""),
            "ece": payload["metrics"].get("ece", ""),
        }
        rows.append(row)

    csv_path = out_dir / "model_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    md_lines = ["# Model Comparison", ""]
    md_lines.append("| " + " | ".join(rows[0].keys()) + " |")
    md_lines.append("| " + " | ".join(["---"] * len(rows[0])) + " |")
    for row in rows:
        md_lines.append("| " + " | ".join(str(value) for value in row.values()) + " |")
    (out_dir / "model_comparison.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    latex_lines = ["\\begin{tabular}{llllrrrr}", "\\toprule"]
    latex_lines.append("Run & Teacher & Student & KD & AUROC & AUPRC & Macro-F1 & ECE \\\\")
    latex_lines.append("\\midrule")
    for row in rows:
        latex_lines.append(
            f"{row['run_id']} & {row['teacher']} & {row['student']} & {row['kd']} & "
            f"{row['auroc']} & {row['auprc']} & {row['macro_f1']} & {row['ece']} \\\\"
        )
    latex_lines.extend(["\\bottomrule", "\\end{tabular}"])
    (out_dir / "model_comparison.tex").write_text("\n".join(latex_lines) + "\n", encoding="utf-8")

    audit = {
        "required_run_ids": required_run_ids,
        "num_rows": len(rows),
        "source_runs_dir": str(runs_dir),
        "tables": {
            "csv": str(csv_path),
            "markdown": str(out_dir / "model_comparison.md"),
            "latex": str(out_dir / "model_comparison.tex"),
        },
    }
    (out_dir / "RESULTS_AUDIT.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    audit_md = ["# Results Audit", ""]
    audit_md.append(f"- Source runs dir: `{runs_dir}`")
    audit_md.append(f"- Required run IDs: `{required_run_ids}`")
    audit_md.append(f"- Rows: {len(rows)}")
    audit_md.append("- No target placeholders are allowed in measured-result tables.")
    (out_dir / "RESULTS_AUDIT.md").write_text("\n".join(audit_md) + "\n", encoding="utf-8")
    return audit
