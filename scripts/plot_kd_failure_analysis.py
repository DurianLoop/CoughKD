"""Create paper-facing figures for the CoughKD failure analysis route."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "kd_failure_analysis" / "figures"


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
COLORS = {
    "ce": "#4C78A8",
    "kd": "#F58518",
    "source_only": "#54A24B",
    "tcd_very_strong": "#B279A2",
    "tcd_conf035": "#E45756",
    "candidate_a": "#72B7B2",
    "candidate_b": "#EECA3B",
    "candidate_c": "#9D755D",
}


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value == "":
                    parsed[key] = None
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
    return rows


def _method_summary() -> list[dict[str, Any]]:
    loop = _load_json(RUNS / "innovation_loop_summary.json")
    cal = _load_json(RUNS / "calibration_efficiency" / "summary.json")
    cal_by_method = {item["method"]: item for item in cal["calibration"]}
    rows = []
    for item in loop["summaries"]:
        method = item["method"]
        if method not in METHOD_ORDER:
            continue
        merged = {
            "method": method,
            "macro": item["external_macro_auroc_mean"],
            "macro_std": item["external_macro_auroc_std"],
            "covid": item["external_covid_auroc_mean"],
            "domain_probe": item["probe_domain_auc_mean"],
            "task_probe": item["probe_task_auc_mean"],
            **cal_by_method.get(method, {}),
        }
        rows.append(merged)
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method"]))


def _annotate(ax: Any, rows: list[dict[str, Any]], x_key: str, y_key: str) -> None:
    for row in rows:
        ax.annotate(
            row["method"],
            (row[x_key], row[y_key]),
            xytext=(5, 3),
            textcoords="offset points",
            fontsize=8,
        )


def _save(fig: Any, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=220)
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)


def plot_external_vs_calibration(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for row in rows:
        ax.scatter(row["ece"], row["macro"], s=64, color=COLORS[row["method"]], edgecolor="black", linewidth=0.5)
    _annotate(ax, rows, "ece", "macro")
    ax.set_xlabel("External ECE on COUGHVID (lower is better)")
    ax.set_ylabel("External macro AUROC on COUGHVID")
    ax.set_title("Calibration Does Not Predict External Ranking")
    ax.grid(alpha=0.25)
    _save(fig, "external_macro_vs_ece")


def plot_probe_vs_external(rows: list[dict[str, Any]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for row in rows:
        ax.scatter(row["domain_probe"], row["macro"], s=64, color=COLORS[row["method"]], edgecolor="black", linewidth=0.5)
    _annotate(ax, rows, "domain_probe", "macro")
    ax.set_xlabel("Domain probe AUC (higher means more dataset-readable)")
    ax.set_ylabel("External macro AUROC on COUGHVID")
    ax.set_title("Probe Shortcut Reduction Is Not Sufficient")
    ax.grid(alpha=0.25)
    _save(fig, "external_macro_vs_domain_probe")


def plot_slice_gap() -> None:
    rows = _load_csv(RUNS / "coughvid_slice_guard" / "slice_guard_decisions.csv")
    names = [str(row["slice"]) for row in rows]
    selected = [float(row["selected_macro_delta"]) for row in rows]
    best = [float(row["best_macro_delta"]) for row in rows]
    kd = [float(row["always_kd_macro_delta"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    width = 0.25
    ax.bar([i - width for i in x], kd, width=width, label="Vanilla KD", color="#F58518")
    ax.bar(x, selected, width=width, label="Guard selected", color="#E45756")
    ax.bar([i + width for i in x], best, width=width, label="Best post-hoc", color="#54A24B")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro AUROC delta vs source-only")
    ax.set_title("Slice-Level KD Gains Are Small and Method-Dependent")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "slice_delta_gap")


def plot_efficiency(rows: list[dict[str, Any]]) -> None:
    eff = _load_json(RUNS / "calibration_efficiency" / "summary.json")["efficiency"]
    student_params = eff["student_params"]
    teacher_params = eff["teacher_params"]
    source = next(row for row in rows if row["method"] == "source_only")
    teacher_macro_proxy = 0.6134
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter([student_params], [source["macro"]], s=78, color="#E45756", edgecolor="black", linewidth=0.6, label="Ultra-light student")
    ax.scatter([teacher_params], [teacher_macro_proxy], s=78, color="#4C78A8", edgecolor="black", linewidth=0.6, label="PANNs teacher, Coswara proxy")
    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("AUROC")
    ax.set_title("Extreme Compression, Weak External Transfer")
    ax.annotate(f"{eff['param_compression']:.0f}x fewer params", (student_params, source["macro"]), xytext=(12, -10), textcoords="offset points", fontsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, "efficiency_compression_proxy")


def plot_bootstrap_stability() -> None:
    path = RUNS / "coughvid_bootstrap_stability" / "bootstrap_method_rows.csv"
    if not path.is_file():
        return
    grouped: dict[str, list[float]] = {method: [] for method in METHOD_ORDER if method != "source_only"}
    with path.open("r", encoding="utf-8", newline="") as handle:
        import csv

        for row in csv.DictReader(handle):
            method = row["method"]
            if method in grouped:
                grouped[method].append(float(row["macro_delta"]))
    methods = [method for method in METHOD_ORDER if method in grouped and grouped[method]]
    values = [grouped[method] for method in methods]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    box = ax.boxplot(values, tick_labels=methods, patch_artist=True, showfliers=False)
    for patch, method in zip(box["boxes"], methods):
        patch.set_facecolor(COLORS[method])
        patch.set_alpha(0.72)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticklabels(methods, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Macro AUROC delta vs source-only")
    ax.set_title("Bootstrap Target Subsets Reveal Method-Ranking Instability")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, "bootstrap_delta_distribution")


def main() -> None:
    rows = _method_summary()
    plot_external_vs_calibration(rows)
    plot_probe_vs_external(rows)
    plot_slice_gap()
    plot_efficiency(rows)
    plot_bootstrap_stability()
    readme = [
        "# KD Failure Analysis Figures",
        "",
        "| Figure | Purpose |",
        "|---|---|",
        "| `external_macro_vs_ece` | Shows calibration improvements do not imply better external AUROC. |",
        "| `external_macro_vs_domain_probe` | Shows lower dataset-readability/probe behavior does not automatically yield external gains. |",
        "| `slice_delta_gap` | Shows slice-level method selection remains unstable and post-hoc best is not captured by the guard. |",
        "| `efficiency_compression_proxy` | Shows the deployment asset: extreme student compression, with external-transfer caveat. |",
        "| `bootstrap_delta_distribution` | Shows method-ranking instability across random target subsets. |",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    main()
