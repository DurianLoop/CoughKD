"""Build a paper-facing CoughKD-FailBench main table."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "kd_failure_analysis"
METHOD_ORDER = [
    "ce",
    "kd",
    "source_only",
    "tcd_very_strong",
    "tcd_conf035",
    "candidate_a",
    "candidate_b",
    "candidate_c",
]
DISPLAY = {
    "ce": "CE-only",
    "kd": "Vanilla KD",
    "source_only": "Source-only continuation",
    "tcd_very_strong": "Target-consistency KD",
    "tcd_conf035": "Confidence-gated TCD",
    "candidate_a": "Shortcut-suppressed KD",
    "candidate_b": "Disagreement-gated KD",
    "candidate_c": "Probe-adversarial KD",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _collect_rows() -> list[dict[str, Any]]:
    loop = _load_json(RUNS / "innovation_loop_summary.json")
    cal = _load_json(RUNS / "calibration_efficiency" / "summary.json")
    boot = _load_json(RUNS / "coughvid_bootstrap_stability" / "summary.json")
    loop_by_method = {item["method"]: item for item in loop["summaries"]}
    cal_by_method = {item["method"]: item for item in cal["calibration"]}
    boot_by_method = {item["method"]: item for item in boot["methods"]}
    source_macro = loop_by_method["source_only"]["external_macro_auroc_mean"]
    rows: list[dict[str, Any]] = []
    for method in METHOD_ORDER:
        item = loop_by_method[method]
        cal_item = cal_by_method.get(method, {})
        boot_item = boot_by_method.get(method, {})
        macro = item.get("external_macro_auroc_mean")
        row = {
            "method": method,
            "display": DISPLAY[method],
            "external_macro_auroc": macro,
            "external_macro_std": item.get("external_macro_auroc_std"),
            "delta_vs_source": None if macro is None else float(macro) - float(source_macro),
            "external_covid_auroc": item.get("external_covid_auroc_mean"),
            "external_macro_auprc": item.get("external_macro_auprc_mean"),
            "ece": cal_item.get("ece"),
            "brier": cal_item.get("brier"),
            "domain_probe_auc": item.get("probe_domain_auc_mean"),
            "task_probe_auc": item.get("probe_task_auc_mean"),
            "sens95_covid": cal_item.get("sens_at_95_spec_covid"),
            "bootstrap_positive_rate": boot_item.get("positive_rate"),
            "bootstrap_best_rate": boot_item.get("best_posthoc_rate"),
            "checkpoint_mb": cal_item.get("checkpoint_size_mb"),
        }
        rows.append(row)
    return rows


def main() -> None:
    rows = _collect_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "failbench_main_table.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# CoughKD-FailBench Main Table",
        "",
        "| Method | Ext macro AUROC | Delta vs source | COVID AUROC | Macro AUPRC | ECE (lower) | Brier (lower) | Domain probe AUC | Task probe AUC | Sens@95Spec | Bootstrap +rate | Best-posthoc rate | MB |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {display} | {macro}+/-{std} | {delta} | {covid} | {auprc} | {ece} | {brier} | {domain} | {task} | {sens} | {positive} | {best} | {mb} |".format(
                display=row["display"],
                macro=_fmt(row["external_macro_auroc"]),
                std=_fmt(row["external_macro_std"]),
                delta=_fmt(row["delta_vs_source"], 6),
                covid=_fmt(row["external_covid_auroc"]),
                auprc=_fmt(row["external_macro_auprc"]),
                ece=_fmt(row["ece"]),
                brier=_fmt(row["brier"]),
                domain=_fmt(row["domain_probe_auc"]),
                task=_fmt(row["task_probe_auc"]),
                sens=_fmt(row["sens95_covid"]),
                positive=_fmt(row["bootstrap_positive_rate"]),
                best=_fmt(row["bootstrap_best_rate"]),
                mb=_fmt(row["checkpoint_mb"], 3),
            )
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `Delta vs source` uses source-only continuation as the reference.",
            "- Bootstrap rates are computed over 200 random COUGHVID target subsets of size 800.",
            "- Domain probe AUC measures dataset-readability of student representations; lower is not automatically better if task evidence drops.",
        ]
    )
    md_path = OUT / "FAILBENCH_MAIN_TABLE.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
