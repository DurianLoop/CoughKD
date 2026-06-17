from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    if value is None:
        return "n/a"
    return str(value)


def _row(values: list[object]) -> str:
    return "| " + " | ".join(_fmt(value) for value in values) + " |"


def _table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [_row(headers), "| " + " | ".join(["---"] * len(headers)) + " |"]
    lines.extend(_row(row) for row in rows)
    return "\n".join(lines)


def main() -> None:
    out_dir = RUNS / "stage2_summary"
    out_dir.mkdir(parents=True, exist_ok=True)

    in_domain_runs = [
        ("vanilla_kd", "stage1_panns_response_seed7"),
        ("sakd_quality", "stage2_sakd_quality_seed7"),
        ("sakd_domain", "stage2_sakd_domain_seed7"),
        ("sakd_full", "stage2_sakd_full_seed7"),
    ]
    external_runs = [
        ("ce_baseline", "external_coughvid_test_ce_baseline"),
        ("vanilla_kd", "external_coughvid_test_kd_baseline"),
        ("sakd_quality", "external_coughvid_test_stage2_sakd_quality_seed7"),
        ("sakd_domain", "external_coughvid_test_stage2_sakd_domain_seed7"),
        ("sakd_full", "external_coughvid_test_stage2_sakd_full_seed7"),
    ]
    weight_runs = [
        ("quality_only", "quality_only_seed7"),
        ("domain_only", "domain_only_seed7"),
        ("full", "full_seed7"),
    ]

    in_rows = []
    for name, run in in_domain_runs:
        metrics_path = RUNS / run / "metrics.json"
        if not metrics_path.is_file():
            continue
        student = _load(metrics_path)["student_test"]
        in_rows.append(
            [
                name,
                student.get("accuracy"),
                student.get("macro_f1"),
                student.get("macro_ovr_auroc"),
                student.get("macro_ovr_auprc"),
                student.get("ece"),
                student.get("brier"),
            ]
        )

    ext_rows = []
    for name, run in external_runs:
        metrics_path = RUNS / run / "metrics.json"
        if not metrics_path.is_file():
            continue
        metrics = _load(metrics_path)
        ext_rows.append(
            [
                name,
                metrics.get("accuracy"),
                metrics.get("macro_ovr_auroc"),
                metrics.get("macro_ovr_auprc"),
                metrics.get("covid_positive"),
                metrics.get("healthy"),
                metrics.get("respiratory_illness"),
                metrics.get("covid_positive_ovr_auprc"),
            ]
        )

    weight_rows = []
    for name, run in weight_runs:
        path = RUNS / "shortcut_weights" / run / "shortcut_weight_summary.json"
        if not path.is_file():
            continue
        payload = _load(path)
        weight_rows.append(
            [
                name,
                payload["quality"].get("mean"),
                payload["domain"].get("mean"),
                payload["stability"].get("mean"),
                payload["kd_weight"].get("mean"),
                payload["kd_weight"].get("p10"),
                payload["kd_weight"].get("p90"),
                payload["domain_summary"].get("domain_probe_auc"),
            ]
        )

    probe_path = RUNS / "domain_probe_stage2_sakd_seed7" / "domain_probe_results.csv"
    probe_text = ""
    if probe_path.is_file():
        probe_text = probe_path.read_text(encoding="utf-8")

    lines = [
        "# Stage 2 SA-KD Result Summary",
        "",
        "## In-Domain Coswara Test",
        "",
        _table(["method", "acc", "macro_f1", "auroc", "auprc", "ece", "brier"], in_rows),
        "",
        "## External COUGHVID-Test",
        "",
        _table(["method", "acc", "macro_auroc", "macro_auprc", "covid_auc", "healthy_auc", "resp_auc", "covid_auprc"], ext_rows),
        "",
        "## Shortcut Weight Distribution",
        "",
        _table(["method", "quality_mean", "domain_mean", "stability_mean", "kd_mean", "kd_p10", "kd_p90", "domain_probe_auc"], weight_rows),
        "",
        "## Domain Probe CSV",
        "",
        "```csv",
        probe_text.strip(),
        "```",
        "",
        "## Immediate Interpretation",
        "",
        "- SA-KD improves in-domain Coswara AUROC over vanilla KD in this seed.",
        "- SA-KD does not improve COUGHVID-test AUROC in this first setting.",
        "- Domain/full weights suppress KD too aggressively: domain-only mean weight is about 0.17 and full mean weight about 0.13.",
        "- Student embedding domain probe AUC does not decrease for SA-KD; it increases relative to vanilla KD, so the first version does not support a shortcut-removal claim.",
        "- Next experiment should temper weights with a higher floor and smaller domain/stability powers.",
        "",
    ]
    report_path = out_dir / "STAGE2_SUMMARY.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(report_path)


if __name__ == "__main__":
    main()
