"""Build a compact multi-target ShiftAudit table from completed external runs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "kd_failure_analysis"
MULTI = RUNS / "coughkd_guard_multitarget" / "multitarget_signals.csv"
DECISIONS = RUNS / "coughkd_guard_multitarget" / "guard_decisions.csv"


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


def _target_n(target: str) -> int | None:
    if target == "coughvid":
        metrics_path = RUNS / "external_coughvid_test_stage3c_source_only_seed11" / "metrics.json"
    else:
        metrics_path = RUNS / f"external_{target}_source_only_seed11" / "metrics.json"
    if not metrics_path.is_file():
        return None
    data = json.loads(metrics_path.read_text(encoding="utf-8"))
    return int(data["num_examples"]) if "num_examples" in data else None


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def main() -> None:
    rows = _read_csv(MULTI)
    decisions = _read_csv(DECISIONS)
    decision_by_target = {row["target"]: row for row in decisions}
    targets = sorted({str(row["target"]) for row in rows})
    compact_rows = []
    for target in targets:
        target_rows = [row for row in rows if row["target"] == target]
        n = _target_n(target)
        best = max(target_rows, key=lambda row: float(row["external_macro_auroc"]))
        source = next(row for row in target_rows if row["method"] == "source_only")
        kd = next((row for row in target_rows if row["method"] == "kd"), None)
        selected = decision_by_target.get(target)
        if target == "virufy":
            caveat = "tiny original stress target"
        elif target == "virufyseg":
            caveat = "segmented stress target, 16 source subjects"
        else:
            caveat = "external target"
        compact_rows.append(
            {
                "target": target,
                "n_examples": n,
                "source_macro": source["external_macro_auroc"],
                "best_method": best["method"],
                "best_macro": best["external_macro_auroc"],
                "best_delta": best.get("macro_delta"),
                "kd_delta": kd.get("macro_delta") if kd else None,
                "guard_selected": selected.get("selected") if selected else None,
                "guard_delta": selected.get("selected_macro_delta") if selected else None,
                "guard_negative": selected.get("selected_negative_transfer") if selected else None,
                "caveat": caveat,
            }
        )

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "shift_audit_multitarget_table.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(compact_rows[0]))
        writer.writeheader()
        writer.writerows(compact_rows)

    lines = [
        "# CoughKD-ShiftAudit Multi-Target Table",
        "",
        "| Target | n | Source-only macro | Best method | Best delta | Vanilla KD delta | Guard selected | Guard delta | Guard negative? | Caveat |",
        "|---|---:|---:|---|---:|---:|---|---:|---|---|",
    ]
    for row in compact_rows:
        lines.append(
            f"| {row['target']} | {row['n_examples']} | {_fmt(row['source_macro'])} | {row['best_method']} ({_fmt(row['best_macro'])}) | {_fmt(row['best_delta'], 6)} | {_fmt(row['kd_delta'], 6)} | {row['guard_selected']} | {_fmt(row['guard_delta'], 6)} | {row['guard_negative']} | {row['caveat']} |"
        )
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- COUGHVID remains the main external target because it has thousands of samples.",
            "- Virufy original is included only as a tiny stress target. Its 16 examples are useful for sanity-checking failure modes but cannot support an ICASSP-strength external validation claim by themselves.",
            "- Virufy segmented provides more clips but still only 16 source subjects, so it is a segment-level stress target rather than a new independent cohort.",
            "- Guard failures or weak selections on Virufy variants strengthen the argument that target-unlabeled selection is not reliable yet.",
        ]
    )
    (OUT / "SHIFT_AUDIT_MULTITARGET_TABLE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "SHIFT_AUDIT_MULTITARGET_TABLE.md")


if __name__ == "__main__":
    main()
