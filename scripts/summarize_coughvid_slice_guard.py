"""CoughKD-Guard audit on unlabeled COUGHVID target slices.

This script reuses existing full-COUGHVID prediction files and evaluates target
slices defined by metadata available before labels are used for auditing:

- cough detector confidence high/low
- sex male/female
- age younger/older
- self-reported respiratory condition true/false

It is not a substitute for an independent external dataset, but it tests whether
the guard score remains useful across multiple target shifts without retraining.
"""

from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
MANIFEST = ROOT / "manifests" / "coughvid_adapt_test.csv"
OUT = RUNS / "coughvid_slice_guard"
REFERENCE = "source_only"
METHOD_RUNS = {
    "ce": {
        7: "external_coughvid_test_ce_baseline",
        11: "external_coughvid_test_ce_seed11",
        23: "external_coughvid_test_ce_seed23",
    },
    "kd": {
        7: "external_coughvid_test_kd_baseline",
        11: "external_coughvid_test_kd_seed11",
        23: "external_coughvid_test_kd_seed23",
    },
    "source_only": {
        7: "external_coughvid_test_stage3b_source_only_seed7",
        11: "external_coughvid_test_stage3c_source_only_seed11",
        23: "external_coughvid_test_stage3c_source_only_seed23",
    },
    "tcd_very_strong": {
        7: "external_coughvid_test_stage3b_tcd_very_strong_seed7",
        11: "external_coughvid_test_stage3c_tcd_very_strong_seed11",
        23: "external_coughvid_test_stage3c_tcd_very_strong_seed23",
    },
    "tcd_conf035": {
        7: "external_coughvid_test_stage3b_tcd_conf035_seed7",
        11: "external_coughvid_test_stage3c_tcd_conf035_seed11",
        23: "external_coughvid_test_stage3c_tcd_conf035_seed23",
    },
    "candidate_a": {
        7: "external_coughvid_test_candidate_a_seed7",
        11: "external_coughvid_test_candidate_a_seed11",
        23: "external_coughvid_test_candidate_a_seed23",
    },
    "candidate_b": {
        7: "external_coughvid_test_candidate_b_seed7",
        11: "external_coughvid_test_candidate_b_seed11",
        23: "external_coughvid_test_candidate_b_seed23",
    },
    "candidate_c": {
        7: "external_coughvid_test_candidate_c_seed7",
        11: "external_coughvid_test_candidate_c_seed11",
        23: "external_coughvid_test_candidate_c_seed23",
    },
}


def _read_predictions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path).fillna("")
    return df


def _prediction_shift_stats(df: pd.DataFrame, reference: pd.DataFrame | None) -> dict[str, float]:
    if reference is None or reference.empty:
        return {
            "agree_source_only": float("nan"),
            "l1_to_source_only": float("nan"),
            "confidence_gap_source_only": float("nan"),
            "entropy_gap_source_only": float("nan"),
        }
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    merged = df.merge(reference[["recording_id", "pred_label", *prob_cols]], on="recording_id", suffixes=("", "_ref"))
    if merged.empty:
        return {
            "agree_source_only": float("nan"),
            "l1_to_source_only": float("nan"),
            "confidence_gap_source_only": float("nan"),
            "entropy_gap_source_only": float("nan"),
        }
    agreement = (merged["pred_label"].astype(str) == merged["pred_label_ref"].astype(str)).mean()
    l1_values = []
    conf_gaps = []
    entropy_gaps = []
    class_count = max(1, len(prob_cols))
    for _, row in merged.iterrows():
        probs = [float(row[col]) for col in prob_cols]
        ref_probs = [float(row[f"{col}_ref"]) for col in prob_cols]
        l1_values.append(sum(abs(a - b) for a, b in zip(probs, ref_probs)))
        conf_gaps.append(max(probs) - max(ref_probs))
        entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(class_count)
        ref_entropy = -sum(p * math.log(max(p, 1e-12)) for p in ref_probs) / math.log(class_count)
        entropy_gaps.append(entropy - ref_entropy)
    return {
        "agree_source_only": float(mean(agreement if isinstance(agreement, list) else [agreement])),
        "l1_to_source_only": float(mean(l1_values)),
        "confidence_gap_source_only": float(mean(conf_gaps)),
        "entropy_gap_source_only": float(mean(entropy_gaps)),
    }


def _slice_definitions(manifest: pd.DataFrame) -> dict[str, set[str]]:
    df = manifest[manifest["split"].eq("test")].copy()
    df["cough_detected_num"] = pd.to_numeric(df["cough_detected"], errors="coerce")
    df["age_num"] = pd.to_numeric(df["age"], errors="coerce")
    cough_median = float(df["cough_detected_num"].median())
    age_median = float(df["age_num"].median())
    respiratory_true = df["symptoms"].astype(str).str.contains("respiratory_condition=True", regex=False)
    respiratory_false = df["symptoms"].astype(str).str.contains("respiratory_condition=False", regex=False)
    slices = {
        "all_test": set(df["recording_id"]),
        "cough_detected_low": set(df[df["cough_detected_num"].le(cough_median)]["recording_id"]),
        "cough_detected_high": set(df[df["cough_detected_num"].gt(cough_median)]["recording_id"]),
        "sex_male": set(df[df["sex"].eq("male")]["recording_id"]),
        "sex_female": set(df[df["sex"].eq("female")]["recording_id"]),
        "age_young": set(df[df["age_num"].le(age_median)]["recording_id"]),
        "age_old": set(df[df["age_num"].gt(age_median)]["recording_id"]),
        "respiratory_condition_true": set(df[respiratory_true]["recording_id"]),
        "respiratory_condition_false": set(df[respiratory_false]["recording_id"]),
    }
    for col, prefix in [("cough_detected_num", "cough_detected_q"), ("age_num", "age_q")]:
        valid = df[df[col].notna()].copy()
        if len(valid) >= 400:
            try:
                valid["_quartile"] = pd.qcut(valid[col], q=4, labels=False, duplicates="drop")
            except ValueError:
                valid["_quartile"] = None
            if valid["_quartile"].notna().any():
                for q in sorted(valid["_quartile"].dropna().unique()):
                    q_int = int(q) + 1
                    slices[f"{prefix}{q_int}"] = set(valid[valid["_quartile"].eq(q)]["recording_id"])

    cross_masks = {
        "male_resp_true": df["sex"].eq("male") & respiratory_true,
        "male_resp_false": df["sex"].eq("male") & respiratory_false,
        "female_resp_true": df["sex"].eq("female") & respiratory_true,
        "female_resp_false": df["sex"].eq("female") & respiratory_false,
        "young_resp_true": df["age_num"].le(age_median) & respiratory_true,
        "young_resp_false": df["age_num"].le(age_median) & respiratory_false,
        "old_resp_true": df["age_num"].gt(age_median) & respiratory_true,
        "old_resp_false": df["age_num"].gt(age_median) & respiratory_false,
    }
    for name, mask in cross_masks.items():
        slices[name] = set(df[mask]["recording_id"])
    return {name: ids for name, ids in slices.items() if len(ids) >= 100}


def _metric_rows(df: pd.DataFrame) -> dict[str, float]:
    prob_cols = [col for col in df.columns if col.startswith("prob_")]
    classes = [col.removeprefix("prob_") for col in prob_cols]
    y_true = df["true_label"].astype(str).tolist()
    y_score = df[prob_cols].astype(float).to_numpy()
    y_bin = [[1 if label == cls else 0 for cls in classes] for label in y_true]
    result: dict[str, float] = {"n": float(len(df))}
    aucs: list[float] = []
    auprcs: list[float] = []
    for idx, cls in enumerate(classes):
        binary = [row[idx] for row in y_bin]
        if len(set(binary)) == 2:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                aucs.append(float(roc_auc_score(binary, y_score[:, idx])))
                auprcs.append(float(average_precision_score(binary, y_score[:, idx])))
        if cls == "covid_positive":
            if len(set(binary)) == 2:
                result["covid_positive"] = float(roc_auc_score(binary, y_score[:, idx]))
            else:
                result["covid_positive"] = float("nan")
    result["macro_ovr_auroc"] = float(mean(aucs)) if aucs else float("nan")
    result["macro_ovr_auprc"] = float(mean(auprcs)) if auprcs else float("nan")
    max_probs = df[prob_cols].astype(float).max(axis=1).tolist()
    class_count = max(1, len(classes))
    entropies = []
    for probs in df[prob_cols].astype(float).to_numpy():
        entropies.append(float(-sum(p * math.log(max(p, 1e-12)) for p in probs) / math.log(class_count)))
    pred_counts = df["pred_label"].value_counts().to_dict()
    result.update(
        {
            "target_confidence": float(mean(max_probs)),
            "target_entropy": float(mean(entropies)),
            "target_covid_prob": float(df["prob_covid_positive"].astype(float).mean()) if "prob_covid_positive" in df else float("nan"),
            "pred_healthy_rate": float(pred_counts.get("healthy", 0) / max(1, len(df))),
            "pred_covid_rate": float(pred_counts.get("covid_positive", 0) / max(1, len(df))),
        }
    )
    return result


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return None, None
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _aggregate(seed_rows: list[dict[str, Any]]) -> dict[str, Any]:
    out = {"n_seeds": len(seed_rows)}
    for key in [
        "macro_ovr_auroc",
        "covid_positive",
        "macro_ovr_auprc",
        "target_confidence",
        "target_entropy",
        "target_covid_prob",
        "pred_healthy_rate",
        "pred_covid_rate",
        "agree_source_only",
        "l1_to_source_only",
        "confidence_gap_source_only",
        "entropy_gap_source_only",
    ]:
        mu, sd = _mean_std([float(row[key]) for row in seed_rows if row.get(key) is not None])
        out[key] = mu
        out[f"{key}_std"] = sd
    out["n"] = mean([float(row["n"]) for row in seed_rows]) if seed_rows else 0.0
    return out


def _zscore(rows: list[dict[str, Any]], key: str, value: float) -> float:
    vals = [float(row[key]) for row in rows if row.get(key) is not None and not math.isnan(float(row[key]))]
    mu = mean(vals)
    sd = math.sqrt(max(mean([(item - mu) ** 2 for item in vals]), 1e-12))
    return (value - mu) / sd


def _guard_score(rows: list[dict[str, Any]], row: dict[str, Any]) -> float:
    return (
        0.40 * _zscore(rows, "target_covid_prob", float(row["target_covid_prob"]))
        + 0.25 * _zscore(rows, "pred_healthy_rate", float(row["pred_healthy_rate"]))
        + 0.20 * _zscore(rows, "target_confidence", float(row["target_confidence"]))
        - 0.15 * _zscore(rows, "target_entropy", float(row["target_entropy"]))
    )


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def _rank(values: list[float]) -> list[float]:
    ordered = sorted((value, idx) for idx, value in enumerate(values))
    ranks = [0.0 for _ in values]
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1][0] == ordered[i][0]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        for _, idx in ordered[i : j + 1]:
            ranks[idx] = rank
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float | None:
    return _pearson(_rank(xs), _rank(ys))


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.6f}"
    return str(value)


def main() -> None:
    manifest = pd.read_csv(MANIFEST).fillna("")
    slices = _slice_definitions(manifest)
    rows: list[dict[str, Any]] = []
    for slice_name, ids in slices.items():
        source_refs: dict[int, pd.DataFrame] = {}
        for seed, run_name in METHOD_RUNS[REFERENCE].items():
            pred_path = RUNS / run_name / "predictions.csv"
            if pred_path.is_file():
                ref = _read_predictions(pred_path)
                source_refs[seed] = ref[ref["recording_id"].isin(ids)].copy()
        for method, seed_runs in METHOD_RUNS.items():
            seed_rows: list[dict[str, Any]] = []
            for seed, run_name in seed_runs.items():
                pred_path = RUNS / run_name / "predictions.csv"
                if not pred_path.is_file():
                    continue
                pred = _read_predictions(pred_path)
                sub = pred[pred["recording_id"].isin(ids)].copy()
                if len(sub) < 50:
                    continue
                reference = source_refs.get(seed) if method != REFERENCE else sub
                seed_rows.append({"seed": seed, **_metric_rows(sub), **_prediction_shift_stats(sub, reference)})
            if not seed_rows:
                continue
            rows.append({"slice": slice_name, "method": method, **_aggregate(seed_rows)})
    for slice_name in slices:
        slice_rows = [row for row in rows if row["slice"] == slice_name]
        ref = next((row for row in slice_rows if row["method"] == REFERENCE), None)
        if ref is None:
            continue
        for row in slice_rows:
            if row["macro_ovr_auroc"] is not None and ref["macro_ovr_auroc"] is not None:
                row["macro_delta"] = float(row["macro_ovr_auroc"]) - float(ref["macro_ovr_auroc"])
            if row["covid_positive"] is not None and ref["covid_positive"] is not None:
                row["covid_delta"] = float(row["covid_positive"]) - float(ref["covid_positive"])
            row["guard_score"] = _guard_score(slice_rows, row)
    decisions = []
    for slice_name in slices:
        slice_rows = [row for row in rows if row["slice"] == slice_name]
        candidates = [row for row in slice_rows if row["method"] != REFERENCE]
        if not candidates:
            continue
        selected = max(candidates, key=lambda row: float(row["guard_score"]))
        best_macro = max(slice_rows, key=lambda row: float(row["macro_ovr_auroc"]))
        always_kd = next((row for row in slice_rows if row["method"] == "kd"), None)
        decisions.append(
            {
                "slice": slice_name,
                "selected": selected["method"],
                "selected_macro_delta": selected["macro_delta"],
                "best_macro": best_macro["method"],
                "best_macro_delta": best_macro.get("macro_delta"),
                "always_kd_macro_delta": always_kd.get("macro_delta") if always_kd else None,
                "selected_negative_transfer": bool(float(selected["macro_delta"]) < 0),
                "n": selected["n"],
            }
        )
    points = [row for row in rows if row["method"] != REFERENCE and row.get("macro_delta") is not None]
    correlations = {}
    for signal in [
        "guard_score",
        "target_covid_prob",
        "pred_healthy_rate",
        "target_confidence",
        "target_entropy",
        "pred_covid_rate",
        "agree_source_only",
        "l1_to_source_only",
        "confidence_gap_source_only",
        "entropy_gap_source_only",
    ]:
        xs = [float(row[signal]) for row in points]
        ys = [float(row["macro_delta"]) for row in points]
        correlations[signal] = {"pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys), "n": len(xs)}
    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with (OUT / "slice_signals.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (OUT / "slice_guard_decisions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(decisions[0]))
        writer.writeheader()
        writer.writerows(decisions)
    (OUT / "slice_correlations.json").write_text(json.dumps(correlations, indent=2), encoding="utf-8")
    lines = [
        "# COUGHVID Slice Guard Audit",
        "",
        "This is a proxy multi-target audit using metadata-defined COUGHVID target slices. It does not replace independent external validation.",
        "",
        "## Decisions",
        "",
        "| Slice | n | Selected | Selected macro delta | Best macro | Always KD macro delta | Negative transfer? |",
        "|---|---:|---|---:|---|---:|---|",
    ]
    for item in decisions:
        lines.append(
            f"| {item['slice']} | {_fmt(item['n'])} | {item['selected']} | {_fmt(item['selected_macro_delta'])} | {item['best_macro']} ({_fmt(item['best_macro_delta'])}) | {_fmt(item['always_kd_macro_delta'])} | {item['selected_negative_transfer']} |"
        )
    lines.extend(["", "## Correlations", "", "| Signal | Pearson | Spearman | n |", "|---|---:|---:|---:|"])
    for signal, item in correlations.items():
        lines.append(f"| {signal} | {_fmt(item['pearson'])} | {_fmt(item['spearman'])} | {item['n']} |")
    negative = sum(1 for item in decisions if item["selected_negative_transfer"])
    lines.extend(
        [
            "",
            "## Readout",
            "",
            f"- Slices: `{len(decisions)}`",
            f"- Guard-selected negative-transfer slices: `{negative}/{len(decisions)}`",
            "- This is useful as a stress test. A publishable claim still requires at least one independent external dataset.",
        ]
    )
    (OUT / "COUGHVID_SLICE_GUARD_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "COUGHVID_SLICE_GUARD_AUDIT.md")


if __name__ == "__main__":
    main()
