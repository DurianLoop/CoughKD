"""Summarize bootstrap stability across subset sizes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "coughvid_bootstrap_sensitivity"
INPUTS = [
    (400, RUNS / "coughvid_bootstrap_stability_400" / "summary.json"),
    (800, RUNS / "coughvid_bootstrap_stability" / "summary.json"),
    (1200, RUNS / "coughvid_bootstrap_stability_1200" / "summary.json"),
]


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    rows = []
    for size, path in INPUTS:
        data = json.loads(path.read_text(encoding="utf-8"))
        for item in data["methods"]:
            if item["method"] == "source_only":
                continue
            rows.append({"subset_size": size, **item})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Bootstrap Subset-Size Sensitivity",
        "",
        "| Subset size | Method | Mean delta | Std | Positive rate | Negative rate | Best-posthoc rate |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['subset_size']} | {row['method']} | {_fmt(row['mean_delta'])} | {_fmt(row['std_delta'])} | {_fmt(row['positive_rate'])} | {_fmt(row['negative_rate'])} | {_fmt(row['best_posthoc_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- Smaller target subsets create larger variance, making post-hoc method ranking less stable.",
            "- `tcd_conf035` remains the most consistently positive method across subset sizes, but its mean gain remains very small.",
        ]
    )
    (OUT / "BOOTSTRAP_SENSITIVITY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "BOOTSTRAP_SENSITIVITY.md")


if __name__ == "__main__":
    main()
