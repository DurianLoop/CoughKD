from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "semantic_router_third_target_success"


TARGET = "UKCOVID"
MIN_REPEATS = 1000
MIN_MAIN_GAIN = 0.01
MIN_CONTROL_GAP = 0.005
BOOTSTRAP_SAMPLES = 20000
BOOTSTRAP_SEED = 20270609


RUN_DIRS = {
    "main_semantic": RUNS / "semantic_constrained_transfer_router_with_ukcovid_1000",
    "demographic_only_control": RUNS / "semantic_router_ukcovid_inverted_demographic_1000",
    "all_slice_control": RUNS / "semantic_router_ukcovid_all_slice_1000",
    "no_slice_control": RUNS / "semantic_router_ukcovid_no_slice_1000",
}


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "semantic_constrained_transfer_router_summary.csv"
    json_path = run_dir / "semantic_constrained_transfer_router_summary.json"
    if not path.is_file():
        return {"present": False, "path": str(path.relative_to(ROOT))}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    target_rows = [row for row in rows if row.get("target") == TARGET]
    if not target_rows:
        return {"present": False, "path": str(path.relative_to(ROOT)), "reason": f"missing {TARGET} row"}
    row = target_rows[0]
    parsed: dict[str, Any] = {"present": True, "path": str(path.relative_to(ROOT)), "raw": row}
    if json_path.is_file():
        config = json.loads(json_path.read_text(encoding="utf-8"))
        parsed["group_split_column"] = config.get("group_split_column", "")
        parsed["summary_json_path"] = str(json_path.relative_to(ROOT))
    else:
        parsed["group_split_column"] = ""
        parsed["summary_json_path"] = str(json_path.relative_to(ROOT))
    for key in [
        "n_repeats",
        "mean_delta_vs_source",
        "median_delta_vs_source",
        "ci95_low",
        "ci95_high",
        "p_delta_lt_0",
        "p_delta_lt_1pt",
        "p_delta_lt_2pt",
        "p_delta_lt_3pt",
    ]:
        value = row.get(key)
        if value is None or value == "":
            parsed[key] = None
            continue
        parsed[key] = int(float(value)) if key == "n_repeats" else float(value)
    return parsed


def _read_deltas(run_dir: Path) -> list[float]:
    path = run_dir / "semantic_constrained_transfer_router.csv"
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    deltas = []
    for row in rows:
        if row.get("target") != TARGET:
            continue
        value = row.get("delta_vs_source")
        if value is None or value == "":
            continue
        deltas.append(float(value))
    return deltas


def _bootstrap_mean_ci(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"present": False, "n": 0, "mean_ci_low": None, "mean_ci_high": None}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    arr = np.asarray(values, dtype=float)
    idx = rng.integers(0, len(arr), size=(BOOTSTRAP_SAMPLES, len(arr)))
    means = arr[idx].mean(axis=1)
    return {
        "present": True,
        "n": int(len(arr)),
        "mean_ci_low": float(np.quantile(means, 0.025)),
        "mean_ci_high": float(np.quantile(means, 0.975)),
    }


def _fmt_pt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{100.0 * float(value):+.2f}pt"


def _criterion(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"criterion": name, "pass": passed, "detail": detail}


def _summary_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| run | present | n | mean | split q2.5 | split q97.5 | p(delta<0) |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    item["name"],
                    str(item["present"]),
                    str(item.get("n_repeats", "NA")),
                    _fmt_pt(item.get("mean_delta_vs_source")),
                    _fmt_pt(item.get("ci95_low")),
                    _fmt_pt(item.get("ci95_high")),
                    "NA" if item.get("p_delta_lt_0") is None else f"{float(item['p_delta_lt_0']):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main() -> None:
    summaries = {name: _read_summary(path) for name, path in RUN_DIRS.items()}
    main_mean_ci = _bootstrap_mean_ci(_read_deltas(RUN_DIRS["main_semantic"]))
    named_rows = [{"name": name, **summary} for name, summary in summaries.items()]
    main = summaries["main_semantic"]
    demographic = summaries["demographic_only_control"]
    all_slice = summaries["all_slice_control"]
    no_slice = summaries["no_slice_control"]

    artifacts_present = all(summary.get("present") for summary in summaries.values())
    main_mean = main.get("mean_delta_vs_source")
    demographic_mean = demographic.get("mean_delta_vs_source")
    all_slice_mean = all_slice.get("mean_delta_vs_source")
    no_slice_mean = no_slice.get("mean_delta_vs_source")
    all_slice_gap = None
    if main_mean is not None and all_slice_mean is not None:
        all_slice_gap = float(main_mean) - float(all_slice_mean)

    criteria = [
        _criterion(
            "all_pre_registered_artifacts_present",
            artifacts_present,
            "main, demographic-only, all-slice, and no-slice UKCOVID summaries must exist",
        ),
        _criterion(
            "all_runs_subject_grouped",
            artifacts_present and all(summary.get("group_split_column") == "subject_id" for summary in summaries.values()),
            "all UKCOVID semantic-router and control runs must use --group-split-column subject_id",
        ),
        _criterion(
            "main_has_1000_repeats",
            bool(main.get("present")) and int(main.get("n_repeats") or 0) >= MIN_REPEATS,
            f"required n >= {MIN_REPEATS}; observed {main.get('n_repeats', 'missing')}",
        ),
        _criterion(
            "main_gain_at_least_1pt",
            main_mean is not None and float(main_mean) >= MIN_MAIN_GAIN,
            f"required mean delta >= {_fmt_pt(MIN_MAIN_GAIN)}; observed {_fmt_pt(main_mean)}",
        ),
        _criterion(
            "split_tail_risk_reported",
            bool(main.get("present"))
            and all(main.get(key) is not None for key in ["p_delta_lt_0", "p_delta_lt_1pt", "ci95_low", "ci95_high"]),
            "p(delta<0), p(delta<1pt), and split quantiles must be present",
        ),
        _criterion(
            "main_mean_ci_excludes_zero",
            bool(main_mean_ci.get("present")) and float(main_mean_ci.get("mean_ci_low") or 0.0) > 0.0,
            f"bootstrap mean CI must exclude zero; observed [{_fmt_pt(main_mean_ci.get('mean_ci_low'))}, {_fmt_pt(main_mean_ci.get('mean_ci_high'))}] from n={main_mean_ci.get('n')}",
        ),
        _criterion(
            "demographic_control_does_not_match",
            main_mean is not None
            and demographic_mean is not None
            and float(demographic_mean) <= float(main_mean) - MIN_CONTROL_GAP,
            f"demographic-only mean must be at least {_fmt_pt(MIN_CONTROL_GAP)} below main; main={_fmt_pt(main_mean)}, control={_fmt_pt(demographic_mean)}",
        ),
        _criterion(
            "no_slice_control_does_not_match",
            main_mean is not None
            and no_slice_mean is not None
            and float(no_slice_mean) <= float(main_mean) - MIN_CONTROL_GAP,
            f"no-slice mean must be at least {_fmt_pt(MIN_CONTROL_GAP)} below main; main={_fmt_pt(main_mean)}, control={_fmt_pt(no_slice_mean)}",
        ),
    ]

    if not artifacts_present:
        verdict = "NOT_READY_MISSING_ARTIFACTS"
    elif all(item["pass"] for item in criteria):
        verdict = "THIRD_TARGET_SUPPORTS_CLAIM"
    else:
        verdict = "THIRD_TARGET_DOES_NOT_SUPPORT_CLAIM"

    payload = {
        "target": TARGET,
        "verdict": verdict,
        "thresholds": {
            "min_repeats": MIN_REPEATS,
            "min_main_gain": MIN_MAIN_GAIN,
            "min_control_gap": MIN_CONTROL_GAP,
        },
        "criteria": criteria,
        "summaries": summaries,
        "main_mean_bootstrap": main_mean_ci,
        "diagnostics": {
            "all_slice_gap_vs_main": all_slice_gap,
            "all_slice_interpretation": (
                "diagnostic_only: the current router uses each target's registered slice_column; "
                "for UKCOVID, all-slice includes symptom_cough_any and may match the main semantic rule"
            ),
        },
        "run_dirs": {name: str(path.relative_to(ROOT)) for name, path in RUN_DIRS.items()},
    }

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "semantic_router_third_target_success.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# Semantic Router Third-Target Success Gate",
        "",
        f"- Target: `{TARGET}`",
        f"- Verdict: `{verdict}`",
        "",
        "## Pre-Registered Criteria",
        "",
        f"- Main semantic UKCOVID run must have at least `{MIN_REPEATS}` repeats.",
        "- Main and control runs must use `--group-split-column subject_id`.",
        f"- Main mean AUROC delta versus source-only must be at least `{_fmt_pt(MIN_MAIN_GAIN)}`.",
        "- Split-tail risk fields must be reported, not hidden.",
        f"- Main bootstrap mean CI must exclude zero, using `{BOOTSTRAP_SAMPLES}` samples and seed `{BOOTSTRAP_SEED}`.",
        f"- Demographic-only and no-slice controls must each be at least `{_fmt_pt(MIN_CONTROL_GAP)}` below the main semantic rule.",
        "- The all-slice control is required as a diagnostic artifact, but equality with the main rule is not a failure because the current router uses the target's registered `slice_column` rather than searching all columns.",
        "",
        "## UKCOVID Summary Inputs",
        "",
        _summary_table(named_rows),
        "",
        "## Criterion Results",
        "",
        "| criterion | pass | detail |",
        "| --- | --- | --- |",
    ]
    for item in criteria:
        lines.append(f"| {item['criterion']} | {item['pass']} | {item['detail']} |")
    lines.extend(
        [
            "",
            "## Diagnostic Notes",
            "",
            f"- All-slice gap versus main: `{_fmt_pt(all_slice_gap)}`.",
            "- All-slice is diagnostic-only for UKCOVID under the current implementation: the router uses the registered `symptom_cough_any` slice when that field is allowed, so all-slice may match the main semantic rule.",
            "",
            "## Interpretation",
            "",
        ]
    )
    if verdict == "THIRD_TARGET_SUPPORTS_CLAIM":
        lines.append("UKCOVID supports upgrading the current semantic-router candidate from two-target credible to third-target supported, subject to paper integration and final format checks.")
    elif verdict == "THIRD_TARGET_DOES_NOT_SUPPORT_CLAIM":
        lines.append("UKCOVID does not support the current semantic-router claim under the pre-registered criteria. Do not upgrade the claim without a new, separately justified innovation loop.")
    else:
        lines.append("UKCOVID success cannot be judged yet because the required semantic-router and control summaries are missing.")
    lines.append("")
    (OUT / "SEMANTIC_ROUTER_THIRD_TARGET_SUCCESS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"target": TARGET, "verdict": verdict, "criteria_passed": sum(1 for item in criteria if item["pass"]), "criteria_total": len(criteria)}, indent=2))
    print(OUT / "SEMANTIC_ROUTER_THIRD_TARGET_SUCCESS.md")


if __name__ == "__main__":
    main()
