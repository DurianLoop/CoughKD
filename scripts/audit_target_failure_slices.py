from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.metrics import average_precision, multiclass_ovr_auroc


METHODS_DEFAULT = {
    "source_only": {
        "coughvid": ROOT / "runs/external_coughvid_test_stage3c_source_only_seed7/predictions.csv",
        "tos": ROOT / "runs/external_toscovid2021_test_source_only_seed7/predictions.csv",
    },
    "candidate_e_tga": {
        "coughvid": ROOT / "runs/external_coughvid_test_candidate_e_tga_seed7/predictions.csv",
        "tos": ROOT / "runs/external_toscovid2021_test_candidate_e_tga_seed7/predictions.csv",
    },
    "candidate_f_artifact_irm": {
        "coughvid": ROOT / "runs/external_coughvid_test_candidate_f_artifact_env_irm_ramp_seed7/predictions.csv",
        "tos": ROOT / "runs/external_toscovid2021_test_candidate_f_artifact_env_irm_ramp_seed7/predictions.csv",
    },
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _prob_classes(rows: list[dict[str, str]]) -> list[str]:
    return [key.removeprefix("prob_") for key in rows[0] if key.startswith("prob_")]


def _safe_float(value: str, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def _metrics(rows: list[dict[str, Any]], classes: list[str]) -> dict[str, Any]:
    labels = [str(row["true_label"]) for row in rows]
    probs = [[float(row[f"prob_{name}"]) for name in classes] for row in rows]
    pred = [classes[max(range(len(classes)), key=lambda idx: prob[idx])] for prob in probs]
    out: dict[str, Any] = {
        "n": len(rows),
        "accuracy": sum(int(left == right) for left, right in zip(labels, pred)) / max(1, len(rows)),
        "pred_distribution": dict(Counter(pred)),
        "label_distribution": dict(Counter(labels)),
    }
    try:
        out.update(multiclass_ovr_auroc(labels, probs, classes))
    except Exception as exc:
        out["auroc_error"] = str(exc)
    auprc = {}
    for idx, class_name in enumerate(classes):
        binary = [1 if label == class_name else 0 for label in labels]
        if sum(binary) == 0:
            continue
        auprc[class_name] = average_precision(binary, [row[idx] for row in probs])
    if auprc:
        out["macro_ovr_auprc"] = sum(auprc.values()) / len(auprc)
        out["ovr_auprc"] = auprc
    return out


def _merge_manifest(pred_rows: list[dict[str, str]], manifest_rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    manifest_by_id = {row["recording_id"]: row for row in manifest_rows}
    merged = []
    for row in pred_rows:
        manifest = manifest_by_id.get(row["recording_id"], {})
        merged.append({**manifest, **row})
    return merged


def _subject_rows(rows: list[dict[str, Any]], classes: list[str]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        subject_id = str(row.get("subject_id", "") or row["recording_id"])
        groups[subject_id].append(row)
    out = []
    for subject_id, items in groups.items():
        labels = Counter(str(item["true_label"]) for item in items)
        label = labels.most_common(1)[0][0]
        rec: dict[str, Any] = {
            "recording_id": subject_id,
            "subject_id": subject_id,
            "true_label": label,
            "n_clips": len(items),
        }
        for class_name in classes:
            rec[f"prob_{class_name}"] = sum(float(item[f"prob_{class_name}"]) for item in items) / len(items)
        out.append(rec)
    return out


def _mean_probs_by_label(rows: list[dict[str, Any]], classes: list[str]) -> dict[str, dict[str, float]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[str(row["true_label"])].append(row)
    out = {}
    for label, items in sorted(by_label.items()):
        out[label] = {
            f"mean_prob_{class_name}": sum(float(item[f"prob_{class_name}"]) for item in items) / len(items)
            for class_name in classes
        }
    return out


def _binary_margin(rows: list[dict[str, Any]]) -> dict[str, float]:
    margins = []
    for row in rows:
        if "prob_covid_positive" in row and "prob_healthy" in row:
            margins.append(float(row["prob_covid_positive"]) - float(row["prob_healthy"]))
    if not margins:
        return {}
    values = sorted(margins)
    return {
        "mean_covid_minus_healthy": sum(values) / len(values),
        "p10": values[int(0.10 * (len(values) - 1))],
        "p50": values[int(0.50 * (len(values) - 1))],
        "p90": values[int(0.90 * (len(values) - 1))],
    }


def _artifact_envs(source_features: list[dict[str, str]], target_features: list[dict[str, str]], seed: int) -> dict[str, int]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    feature_names = [
        "duration_sec",
        "log_rms",
        "peak",
        "clipping_fraction",
        "silence_ratio",
        "zcr",
        "spectral_centroid",
        "spectral_bandwidth",
        "low_band_ratio",
        "mid_band_ratio",
        "high_band_ratio",
        "active_ratio",
    ]
    source_x = np.asarray([[_safe_float(row[name], 0.0) for name in feature_names] for row in source_features], dtype=np.float32)
    target_x = np.asarray([[_safe_float(row[name], 0.0) for name in feature_names] for row in target_features], dtype=np.float32)
    scaler = StandardScaler()
    source_scaled = scaler.fit_transform(source_x)
    kmeans = KMeans(n_clusters=2, random_state=seed, n_init=20)
    kmeans.fit(source_scaled)
    target_envs = kmeans.predict(scaler.transform(target_x))
    return {row["recording_id"]: int(env) for row, env in zip(target_features, target_envs)}


def _slice_metrics(rows: list[dict[str, Any]], classes: list[str], field: str) -> dict[str, Any]:
    out = {}
    values = sorted({str(row.get(field, "")) for row in rows})
    for value in values:
        items = [row for row in rows if str(row.get(field, "")) == value]
        if len(items) < 20:
            continue
        out[value] = _metrics(items, classes)
    return out


def _quantile_slices(rows: list[dict[str, Any]], classes: list[str], field: str, q: int = 4) -> dict[str, Any]:
    values = sorted(_safe_float(str(row.get(field, ""))) for row in rows)
    values = [value for value in values if not math.isnan(value)]
    if len(values) < q * 20:
        return {}
    cuts = [values[int(frac * (len(values) - 1) / q)] for frac in range(1, q)]
    out = {}
    for row in rows:
        value = _safe_float(str(row.get(field, "")))
        if math.isnan(value):
            bucket = "missing"
        else:
            bucket_idx = sum(1 for cut in cuts if value > cut)
            bucket = f"q{bucket_idx + 1}"
        row[f"{field}_quantile"] = bucket
    return _slice_metrics(rows, classes, f"{field}_quantile")


def _compare_method_pair(
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    classes: list[str],
) -> dict[str, Any]:
    base = {row["recording_id"]: row for row in baseline_rows}
    deltas: dict[str, list[float]] = {class_name: [] for class_name in classes}
    by_label: dict[str, dict[str, list[float]]] = defaultdict(lambda: {class_name: [] for class_name in classes})
    for row in candidate_rows:
        prior = base.get(row["recording_id"])
        if prior is None:
            continue
        label = str(row["true_label"])
        for class_name in classes:
            delta = float(row[f"prob_{class_name}"]) - float(prior[f"prob_{class_name}"])
            deltas[class_name].append(delta)
            by_label[label][class_name].append(delta)
    return {
        "mean_delta": {class_name: (sum(values) / len(values) if values else 0.0) for class_name, values in deltas.items()},
        "mean_delta_by_true_label": {
            label: {class_name: (sum(values) / len(values) if values else 0.0) for class_name, values in class_map.items()}
            for label, class_map in by_label.items()
        },
    }


def _audit_target(
    target_name: str,
    manifest: Path,
    artifact_features: Path,
    source_features: Path,
    predictions: dict[str, Path],
    seed: int,
) -> dict[str, Any]:
    manifest_rows = _read_csv(manifest)
    target_features_rows = _read_csv(artifact_features)
    source_features_rows = _read_csv(source_features)
    target_envs = _artifact_envs(source_features_rows, target_features_rows, seed)
    by_method = {}
    merged_by_method = {}
    classes: list[str] | None = None
    for method, path in predictions.items():
        pred_rows = _read_csv(path)
        if classes is None:
            classes = _prob_classes(pred_rows)
        merged = _merge_manifest(pred_rows, manifest_rows)
        features_by_id = {row["recording_id"]: row for row in target_features_rows}
        for row in merged:
            row["artifact_env"] = str(target_envs.get(row["recording_id"], -1))
            feats = features_by_id.get(row["recording_id"], {})
            for key, value in feats.items():
                if key not in row:
                    row[key] = value
        merged_by_method[method] = merged
        subject = _subject_rows(merged, classes)
        by_method[method] = {
            "clip": _metrics(merged, classes),
            "subject": _metrics(subject, classes),
            "mean_probs_by_label": _mean_probs_by_label(merged, classes),
            "binary_margin": _binary_margin(merged),
            "artifact_env_slices": _slice_metrics(merged, classes, "artifact_env"),
            "source_status_slices": _slice_metrics(merged, classes, "source_status"),
            "age_slices": _slice_metrics(merged, classes, "age"),
            "high_band_quantiles": _quantile_slices(list(merged), classes, "high_band_ratio"),
            "silence_quantiles": _quantile_slices(list(merged), classes, "silence_ratio"),
        }
    comparisons = {}
    if "source_only" in merged_by_method:
        for method, rows in merged_by_method.items():
            if method == "source_only":
                continue
            comparisons[f"{method}_minus_source_only"] = _compare_method_pair(merged_by_method["source_only"], rows, classes)
    if "candidate_e_tga" in merged_by_method and "candidate_f_artifact_irm" in merged_by_method:
        comparisons["candidate_f_minus_candidate_e"] = _compare_method_pair(
            merged_by_method["candidate_e_tga"],
            merged_by_method["candidate_f_artifact_irm"],
            classes,
        )
    return {
        "target": target_name,
        "classes": classes,
        "artifact_env_distribution": dict(Counter(str(env) for env in target_envs.values())),
        "methods": by_method,
        "comparisons": comparisons,
    }


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = ["# Target Failure Slice Audit", ""]
    for target, audit in summary["targets"].items():
        lines.extend([f"## {target}", "", "### Clip Metrics", "", "| Method | Macro AUROC | COVID AUROC | Healthy AUROC | Macro AUPRC | Accuracy |", "|---|---:|---:|---:|---:|---:|"])
        for method, item in audit["methods"].items():
            clip = item["clip"]
            lines.append(
                "| {method} | {macro} | {covid} | {healthy} | {auprc} | {acc} |".format(
                    method=method,
                    macro=_fmt(clip.get("macro_ovr_auroc", "")),
                    covid=_fmt(clip.get("covid_positive", "")),
                    healthy=_fmt(clip.get("healthy", "")),
                    auprc=_fmt(clip.get("macro_ovr_auprc", "")),
                    acc=_fmt(clip.get("accuracy", "")),
                )
            )
        lines.extend(["", "### Subject Metrics", "", "| Method | Macro AUROC | COVID AUROC | Healthy AUROC | Macro AUPRC | Accuracy |", "|---|---:|---:|---:|---:|---:|"])
        for method, item in audit["methods"].items():
            subject = item["subject"]
            lines.append(
                "| {method} | {macro} | {covid} | {healthy} | {auprc} | {acc} |".format(
                    method=method,
                    macro=_fmt(subject.get("macro_ovr_auroc", "")),
                    covid=_fmt(subject.get("covid_positive", "")),
                    healthy=_fmt(subject.get("healthy", "")),
                    auprc=_fmt(subject.get("macro_ovr_auprc", "")),
                    acc=_fmt(subject.get("accuracy", "")),
                )
            )
        lines.extend(["", "### Candidate F Minus Source-Only Mean Probability Delta", "", "```json"])
        lines.append(json.dumps(audit["comparisons"].get("candidate_f_artifact_irm_minus_source_only", {}), indent=2))
        lines.extend(["```", "", "### Candidate F Artifact Env Slices", "", "```json"])
        lines.append(json.dumps(audit["methods"].get("candidate_f_artifact_irm", {}).get("artifact_env_slices", {}), indent=2))
        lines.extend(["```", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/target_failure_slice_audit_seed7")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    targets = {
        "coughvid": _audit_target(
            "coughvid",
            ROOT / "manifests/coughvid_adapt_test.csv",
            ROOT / "runs/target_failure_slice_audit_seed7/coughvid_test_artifact_features.csv",
            ROOT / "runs/artifact_environment_audit_seed7/source_artifact_features.csv",
            {name: paths["coughvid"] for name, paths in METHODS_DEFAULT.items()},
            args.seed,
        ),
        "tos": _audit_target(
            "tos",
            ROOT / "manifests/toscovid2021_test_external.csv",
            ROOT / "runs/artifact_environment_audit_seed7/tos_artifact_features.csv",
            ROOT / "runs/artifact_environment_audit_seed7/source_artifact_features.csv",
            {name: paths["tos"] for name, paths in METHODS_DEFAULT.items()},
            args.seed,
        ),
    }
    summary = {"targets": targets}
    (args.out / "target_failure_slice_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "TARGET_FAILURE_SLICE_AUDIT.md", summary)
    print(str(args.out / "TARGET_FAILURE_SLICE_AUDIT.md"), flush=True)


if __name__ == "__main__":
    main()
