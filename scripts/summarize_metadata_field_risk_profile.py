"""Summarize metadata-field risk profiles for the semantic-router claim.

This is a manifest-only audit plus a read of existing oracle summaries. It does
not train models, run inference, or download data. The goal is to distinguish
"metadata has signal" from "metadata is safe as a slice-gating variable".
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from audit_metadata_slice_oracle import _prepare_manifest


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "metadata_field_risk_profile"

FIELDS = [
    {
        "target": "COUGHVID",
        "manifest": ROOT / "manifests" / "coughvid_external.csv",
        "field": "symptom_resp",
        "field_type": "symptom",
        "router_role": "safe_slice_gate_candidate",
    },
    {
        "target": "COUGHVID",
        "manifest": ROOT / "manifests" / "coughvid_external.csv",
        "field": "age_bin",
        "field_type": "demographic",
        "router_role": "unsafe_as_primary_slice_gate",
    },
    {
        "target": "COUGHVID",
        "manifest": ROOT / "manifests" / "coughvid_external.csv",
        "field": "sex",
        "field_type": "demographic",
        "router_role": "unsafe_as_primary_slice_gate",
    },
    {
        "target": "COUGHVID",
        "manifest": ROOT / "manifests" / "coughvid_external.csv",
        "field": "cough_detected_bin",
        "field_type": "acquisition_quality",
        "router_role": "safe_slice_gate_candidate_but_not_current_primary",
    },
    {
        "target": "TosCOVID",
        "manifest": ROOT / "manifests" / "toscovid2021_test_external.csv",
        "field": "age",
        "field_type": "demographic",
        "router_role": "route_to_calibration_stacking",
    },
    {
        "target": "TosCOVID",
        "manifest": ROOT / "manifests" / "toscovid2021_test_external.csv",
        "field": "sex",
        "field_type": "demographic",
        "router_role": "unsafe_as_primary_slice_gate",
    },
    {
        "target": "TosCOVID",
        "manifest": ROOT / "manifests" / "toscovid2021_test_external.csv",
        "field": "device",
        "field_type": "acquisition",
        "router_role": "safe_in_principle_but_degenerate_locally",
    },
]


def _label_series(df: pd.DataFrame) -> pd.Series:
    labels = df["label"].fillna("missing").astype(str)
    return labels.map(lambda x: "positive" if x == "covid_positive" else "non_positive")


def _cramers_v(labels: pd.Series, groups: pd.Series) -> float:
    table = pd.crosstab(labels, groups)
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    total = observed.sum()
    if total <= 0:
        return 0.0
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / total
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = np.nansum((observed - expected) ** 2 / expected)
    r, k = observed.shape
    denom = total * max(1, min(k - 1, r - 1))
    return float(np.sqrt(chi2 / denom)) if denom > 0 else 0.0


def _profile_field(spec: dict[str, Any], oracle: pd.DataFrame) -> dict[str, Any]:
    df = _prepare_manifest(spec["manifest"])
    field = spec["field"]
    labels = _label_series(df)
    if field not in df.columns:
        return {
            **{k: str(v) for k, v in spec.items() if k != "manifest"},
            "status": "missing_field",
        }
    groups = df[field].fillna("missing").astype(str)
    counts = groups.value_counts(dropna=False)
    valid_groups = counts[counts >= 30].index
    valid = groups.isin(valid_groups)
    rates = (
        pd.DataFrame({"label": labels[valid], "group": groups[valid]})
        .assign(is_positive=lambda x: (x["label"] == "positive").astype(float))
        .groupby("group")["is_positive"]
        .mean()
        .sort_values(ascending=False)
    )
    if rates.empty:
        max_rate_gap = 0.0
    else:
        max_rate_gap = float(rates.max() - rates.min())
    oracle_rows = oracle[
        (oracle["target"].astype(str) == spec["target"])
        & (oracle["slice_column"].astype(str) == field)
    ]
    oracle_delta = None
    oracle_used = None
    if not oracle_rows.empty:
        best = oracle_rows.sort_values("delta_vs_source", ascending=False).iloc[0]
        oracle_delta = float(best["delta_vs_source"])
        oracle_used = int(best["n_slices_used"])

    signal_readout = "strong_label_association" if max_rate_gap >= 0.10 else "weak_or_moderate_label_association"
    if spec["field_type"] == "demographic" and oracle_delta is not None and oracle_delta > 0:
        safety_readout = "signal_exists_but_demographic_gate_is_crowded_or_confound_prone"
    elif spec["field_type"] in {"symptom", "acquisition", "acquisition_quality"}:
        safety_readout = "semantically_plausible_slice_field"
    else:
        safety_readout = "not_a_primary_slice_gate"

    return {
        "target": spec["target"],
        "field": field,
        "field_type": spec["field_type"],
        "router_role": spec["router_role"],
        "rows": int(len(df)),
        "subjects": int(df["subject_id"].nunique()) if "subject_id" in df.columns else None,
        "levels": int(groups.nunique()),
        "valid_levels_n_ge_30": int(len(valid_groups)),
        "min_level_n": int(counts.min()) if not counts.empty else 0,
        "max_level_n": int(counts.max()) if not counts.empty else 0,
        "positive_rate_gap_ge_30": max_rate_gap,
        "cramers_v": _cramers_v(labels, groups),
        "oracle_delta_vs_source": oracle_delta,
        "oracle_slices_used": oracle_used,
        "signal_readout": signal_readout,
        "safety_readout": safety_readout,
        "status": "profiled",
    }


def _to_md(rows: list[dict[str, Any]]) -> str:
    cols = [
        "target",
        "field",
        "field_type",
        "router_role",
        "levels",
        "valid_levels_n_ge_30",
        "positive_rate_gap_ge_30",
        "cramers_v",
        "oracle_delta_vs_source",
        "safety_readout",
    ]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for row in rows:
        values = []
        for col in cols:
            value = row.get(col)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    oracle_path = RUNS / "metadata_slice_oracle" / "metadata_slice_oracle_summary.csv"
    oracle = pd.read_csv(oracle_path) if oracle_path.is_file() else pd.DataFrame()
    rows = [_profile_field(spec, oracle) for spec in FIELDS]
    demographic_positive = [
        row
        for row in rows
        if row.get("field_type") == "demographic"
        and (row.get("oracle_delta_vs_source") or 0.0) > 0
    ]
    symptom_safe = [
        row
        for row in rows
        if row.get("field_type") == "symptom"
        and row.get("router_role") == "safe_slice_gate_candidate"
        and row.get("status") == "profiled"
    ]
    payload = {
        "verdict": "METADATA_FIELD_RISK_PROFILE_ACTIVE",
        "profile_type": "manifest_only_plus_existing_oracle_summary",
        "runs_inference": False,
        "runs_training": False,
        "uses_existing_datasets_only": True,
        "demographic_fields_have_signal": bool(demographic_positive),
        "symptom_safe_field_profiled": bool(symptom_safe),
        "interpretation": (
            "Metadata signal is not sufficient for safe slice gating. "
            "Demographic fields may show label or oracle signal, but remain "
            "crowded/confound-prone and should route to calibration stacking "
            "or non-claim controls under the current semantic-router boundary."
        ),
        "rows": rows,
        "evidence_paths": {
            "oracle_summary": str(oracle_path.relative_to(ROOT)) if oracle_path.is_file() else "",
            "coughvid_manifest": "manifests/coughvid_external.csv",
            "toscovid_manifest": "manifests/toscovid2021_test_external.csv",
        },
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metadata_field_risk_profile.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Metadata Field Risk Profile",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Runs inference: `{payload['runs_inference']}`",
        f"- Runs training: `{payload['runs_training']}`",
        f"- Uses existing datasets only: `{payload['uses_existing_datasets_only']}`",
        f"- Demographic fields have signal: `{payload['demographic_fields_have_signal']}`",
        "",
        "## Field Profiles",
        "",
        _to_md(rows),
        "",
        "## Interpretation",
        "",
        payload["interpretation"],
        "",
        "This supports the current claim boundary. It does not add a new main effect and does not prove clinical utility.",
        "",
    ]
    (OUT / "METADATA_FIELD_RISK_PROFILE.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["verdict", "demographic_fields_have_signal", "symptom_safe_field_profiled"]}, indent=2))
    print(OUT / "METADATA_FIELD_RISK_PROFILE.md")


if __name__ == "__main__":
    main()
