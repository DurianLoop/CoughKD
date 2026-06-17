from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-dir", type=Path, default=ROOT / "runs/crossfit_metadata_slice_grid")
    args = parser.parse_args()
    rows = []
    for path in sorted(args.grid_dir.glob("crossfit_*_summary.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        data["summary_path"] = str(path)
        detail_path = path.with_name(path.name.removesuffix("_summary.json") + ".csv")
        if detail_path.is_file():
            detail = pd.read_csv(detail_path)
            if "actual_train_n" in detail.columns and not detail.empty:
                data["mean_actual_train_n"] = float(detail["actual_train_n"].mean())
            elif "train_n" in detail.columns and not detail.empty:
                data["mean_actual_train_n"] = float(detail["train_n"].mean())
        data.setdefault("train_per_slice", None)
        data.setdefault("switch_margin", 0.0)
        data.setdefault("mean_actual_train_n", None)
        rows.append(data)
    if not rows:
        raise SystemExit(f"No summaries found in {args.grid_dir}")
    df = pd.DataFrame(rows).sort_values(["target", "mean_delta_vs_base"], ascending=[True, False])
    out_csv = args.grid_dir / "crossfit_metadata_slice_grid_summary.csv"
    out_json = args.grid_dir / "crossfit_metadata_slice_grid_summary.json"
    df.to_csv(out_csv, index=False)
    payload = {
        "n_rows": int(len(df)),
        "clears_mean_3pt": bool((df["mean_delta_vs_base"] >= 0.03).any()),
        "clears_median_3pt": bool((df["median_delta_vs_base"] >= 0.03).any()),
        "best_rows": df.head(20).to_dict(orient="records"),
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = [
        "# Crossfit Metadata Slice Grid",
        "",
        _to_md(df),
        "",
        f"Clears mean 3-point gate: `{payload['clears_mean_3pt']}`",
        f"Clears median 3-point gate: `{payload['clears_median_3pt']}`",
        "",
    ]
    report = args.grid_dir / "CROSSFIT_METADATA_SLICE_GRID.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(report)


def _to_md(df: pd.DataFrame) -> str:
    cols = [
        "target",
        "slice_column",
        "train_per_slice",
        "switch_margin",
        "n_repeats_valid",
        "mean_actual_train_n",
        "mean_delta_vs_base",
        "median_delta_vs_base",
        "ci95_low",
        "ci95_high",
        "p_delta_le_0",
        "p_delta_lt_3pt",
    ]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in cols:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
