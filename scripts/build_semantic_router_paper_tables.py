from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


RUNS = {
    "Semantic": "semantic_constrained_transfer_router_threshold1_1000",
    "Inverted": "semantic_constrained_transfer_router_inverted_age_only_1000",
    "All-slice": "semantic_constrained_transfer_router_all_target_slices_1000",
    "No-slice": "semantic_constrained_transfer_router_no_target_slices_1000",
}

BUDGET_RUNS = {
    "50%": "semantic_constrained_transfer_router_threshold1_1000",
    "30%": "semantic_constrained_transfer_router_train0p3_1000",
    "20%": "semantic_constrained_transfer_router_train0p2_1000",
    "10%": "semantic_constrained_transfer_router_train0p1_1000",
}


def _mean_ci(values: pd.Series) -> tuple[float, float]:
    arr = values.to_numpy(dtype=float)
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    mean = float(np.mean(arr))
    return mean - 1.96 * se, mean + 1.96 * se


def _load_run(run_name: str) -> pd.DataFrame:
    path = ROOT / "runs" / run_name / "semantic_constrained_transfer_router.csv"
    df = pd.read_csv(path)
    rows = []
    for target, group in df.groupby("target"):
        low, high = _mean_ci(group["delta_vs_source"])
        rows.append(
            {
                "target": target,
                "mean": float(group["delta_vs_source"].mean()),
                "mean_ci_low": low,
                "mean_ci_high": high,
                "split_q025": float(group["delta_vs_source"].quantile(0.025)),
                "p_neg": float((group["delta_vs_source"] < 0.0).mean()),
                "p_lt_1pt": float((group["delta_vs_source"] < 0.01).mean()),
            }
        )
    return pd.DataFrame(rows)


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}"


def _ci(low: float, high: float) -> str:
    return f"[{100.0 * low:.2f}, {100.0 * high:.2f}]"


def _mk_delta_table() -> pd.DataFrame:
    rows = []
    for label, run_name in RUNS.items():
        df = _load_run(run_name)
        by_target = {row["target"]: row for _, row in df.iterrows()}
        rows.append(
            {
                "Rule": label,
                "COUGHVID mean": _pct(by_target["COUGHVID"]["mean"]),
                "COUGHVID mean CI": _ci(
                    by_target["COUGHVID"]["mean_ci_low"], by_target["COUGHVID"]["mean_ci_high"]
                ),
                "Tos mean": _pct(by_target["TosCOVID"]["mean"]),
                "Tos mean CI": _ci(
                    by_target["TosCOVID"]["mean_ci_low"], by_target["TosCOVID"]["mean_ci_high"]
                ),
                "Failure readout": {
                    "Semantic": "intended metadata semantics",
                    "Inverted": "both targets routed incorrectly",
                    "All-slice": "Tos age slice is unsafe",
                    "No-slice": "COUGHVID symptom slice is disabled",
                }[label],
            }
        )
    return pd.DataFrame(rows)


def _mk_budget_table() -> pd.DataFrame:
    rows = []
    for label, run_name in BUDGET_RUNS.items():
        df = _load_run(run_name)
        for _, row in df.sort_values("target").iterrows():
            rows.append(
                {
                    "Calibration": label,
                    "Target": row["target"],
                    "Mean delta": _pct(row["mean"]),
                    "Mean CI": _ci(row["mean_ci_low"], row["mean_ci_high"]),
                    "Split q2.5": _pct(row["split_q025"]),
                    "P(delta<0)": f"{row['p_neg']:.3f}",
                    "P(delta<1pt)": f"{row['p_lt_1pt']:.3f}",
                }
            )
    return pd.DataFrame(rows)


def _to_latex_tabular(df: pd.DataFrame, column_spec: str) -> str:
    def esc(value: object) -> str:
        text = str(value)
        return text.replace("%", r"\%").replace("<", r"$<$").replace(">", r"$>$")

    lines = [f"\\begin{{tabular}}{{{column_spec}}}", "\\toprule"]
    lines.append(" & ".join(esc(col) for col in df.columns) + r" \\")
    lines.append("\\midrule")
    for _, row in df.iterrows():
        lines.append(" & ".join(esc(row[col]) for col in df.columns) + r" \\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(lines)


def _to_markdown_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/semantic_router_paper_tables")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    delta_table = _mk_delta_table()
    budget_table = _mk_budget_table()

    delta_table.to_csv(args.out / "semantic_router_negative_controls.csv", index=False)
    budget_table.to_csv(args.out / "semantic_router_label_budget.csv", index=False)

    md_lines = [
        "# Semantic Router Paper Tables",
        "",
        "## Negative Controls",
        "",
        _to_markdown_table(delta_table),
        "",
        "## Calibration Budget",
        "",
        _to_markdown_table(budget_table),
        "",
    ]
    (args.out / "SEMANTIC_ROUTER_TABLES.md").write_text("\n".join(md_lines), encoding="utf-8")

    latex_lines = [
        "% Auto-generated by scripts/build_semantic_router_paper_tables.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Semantic-constrained transfer routing and negative controls. Mean deltas are AUROC points relative to source-only; CIs are 95\\% confidence intervals for the mean over 1000 target resamples.}",
        "\\label{tab:semantic_router_controls}",
        "\\resizebox{\\linewidth}{!}{%",
        _to_latex_tabular(delta_table, "lccccl"),
        "}",
        "\\end{table}",
        "",
        "\\begin{table}[t]",
        "\\centering",
        "\\scriptsize",
        "\\caption{Target calibration label budget for semantic-constrained routing. Split q2.5 reports the 2.5\\% quantile of per-split deltas, not the mean confidence interval.}",
        "\\label{tab:semantic_router_budget}",
        "\\resizebox{\\linewidth}{!}{%",
        _to_latex_tabular(budget_table, "llccccc"),
        "}",
        "\\end{table}",
        "",
    ]
    (args.out / "semantic_router_tables.tex").write_text("\n".join(latex_lines), encoding="utf-8")
    print(args.out / "SEMANTIC_ROUTER_TABLES.md")


if __name__ == "__main__":
    main()
