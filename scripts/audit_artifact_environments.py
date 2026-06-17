from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.audio import audio_stats, read_wav_mono, resample_linear
from coughkd.config import RunConfig
from coughkd.manifest import read_manifest


FEATURE_NAMES = [
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


def _require_numpy_sklearn() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any]:
    import numpy as np
    from sklearn.cluster import KMeans
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import normalized_mutual_info_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return np, KMeans, LogisticRegression, normalized_mutual_info_score, roc_auc_score, train_test_split, make_pipeline, StandardScaler


def _safe_float(value: str, default: float = math.nan) -> float:
    try:
        if value == "":
            return default
        return float(value)
    except ValueError:
        return default


def _zero_crossing_rate(samples: list[float]) -> float:
    if len(samples) <= 1:
        return 0.0
    crossings = 0
    prev = samples[0]
    for value in samples[1:]:
        if (prev < 0 <= value) or (prev >= 0 > value):
            crossings += 1
        prev = value
    return crossings / (len(samples) - 1)


def _spectral_features(samples: list[float], sample_rate: int) -> dict[str, float]:
    np, *_ = _require_numpy_sklearn()
    if not samples:
        return {
            "spectral_centroid": 0.0,
            "spectral_bandwidth": 0.0,
            "low_band_ratio": 0.0,
            "mid_band_ratio": 0.0,
            "high_band_ratio": 0.0,
        }
    x = np.asarray(samples, dtype=np.float32)
    if x.shape[0] > sample_rate:
        hop = max(1, x.shape[0] // sample_rate)
        x = x[::hop][:sample_rate]
    x = x - float(np.mean(x))
    if x.shape[0] < 256:
        x = np.pad(x, (0, 256 - x.shape[0]))
    window = np.hanning(x.shape[0]).astype(np.float32)
    spec = np.abs(np.fft.rfft(x * window)) ** 2
    freqs = np.fft.rfftfreq(x.shape[0], d=1.0 / sample_rate)
    total = float(spec.sum()) + 1e-12
    centroid = float((freqs * spec).sum() / total)
    bandwidth = float(np.sqrt((((freqs - centroid) ** 2) * spec).sum() / total))

    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs < hi)
        return float(spec[mask].sum() / total)

    return {
        "spectral_centroid": centroid,
        "spectral_bandwidth": bandwidth,
        "low_band_ratio": band(0.0, 1000.0),
        "mid_band_ratio": band(1000.0, 4000.0),
        "high_band_ratio": band(4000.0, sample_rate / 2.0 + 1.0),
    }


def _active_ratio(samples: list[float], frame: int = 400, hop: int = 160) -> float:
    if not samples:
        return 0.0
    energies: list[float] = []
    for start in range(0, max(1, len(samples) - frame + 1), hop):
        chunk = samples[start : start + frame]
        if not chunk:
            continue
        energies.append(math.sqrt(sum(value * value for value in chunk) / len(chunk)))
    if not energies:
        return 0.0
    energies_sorted = sorted(energies)
    lo = energies_sorted[int(0.10 * (len(energies_sorted) - 1))]
    hi = energies_sorted[int(0.90 * (len(energies_sorted) - 1))]
    return hi / max(1e-8, lo)


def _audio_features(row: dict[str, str], root: Path, config: RunConfig, max_duration_sec: float) -> dict[str, Any]:
    samples, sample_rate = read_wav_mono(root / row["path"])
    samples = resample_linear(samples, sample_rate, config.sample_rate)
    if max_duration_sec > 0:
        samples = samples[: int(max_duration_sec * config.sample_rate)]
    stats = audio_stats(samples, config.sample_rate, config)
    peak = max((abs(value) for value in samples), default=0.0)
    silence_ratio = sum(1 for value in samples if abs(value) < 0.005) / max(1, len(samples))
    spec = _spectral_features(samples, config.sample_rate)
    return {
        "recording_id": row.get("recording_id", ""),
        "subject_id": row.get("subject_id", ""),
        "dataset": row.get("dataset", ""),
        "split": row.get("split", ""),
        "label": row.get("label", ""),
        "path": row.get("path", ""),
        "duration_sec": stats.duration_sec,
        "log_rms": math.log10(max(stats.rms, 1e-8)),
        "peak": peak,
        "clipping_fraction": stats.clipping_fraction,
        "silence_ratio": silence_ratio,
        "zcr": _zero_crossing_rate(samples),
        "active_ratio": _active_ratio(samples),
        **spec,
    }


def _load_or_build_cache(
    rows: list[dict[str, str]],
    root: Path,
    config: RunConfig,
    max_duration_sec: float,
    cache_path: Path,
    label: str,
) -> list[dict[str, Any]]:
    if cache_path.is_file():
        with cache_path.open("r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        try:
            records.append(_audio_features(row, root, config, max_duration_sec))
        except Exception as exc:
            errors.append({"recording_id": row.get("recording_id", ""), "path": row.get("path", ""), "error": str(exc)})
            if len(errors) <= 5:
                print(f"[warn] {label} decode failed {row.get('recording_id', '')}: {exc}", flush=True)
        if idx == 1 or idx % 500 == 0:
            print(f"[{label}] {idx}/{len(rows)}", flush=True)
    fieldnames = ["recording_id", "subject_id", "dataset", "split", "label", "path", *FEATURE_NAMES]
    with cache_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record.get(key, "") for key in fieldnames})
    if errors:
        cache_path.with_suffix(".errors.json").write_text(json.dumps(errors[:50], indent=2), encoding="utf-8")
    return records


def _matrix(rows: list[dict[str, Any]]) -> Any:
    np, *_ = _require_numpy_sklearn()
    return np.asarray([[_safe_float(str(row[name]), 0.0) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)


def _domain_probe(source_rows: list[dict[str, Any]], target_rows: list[dict[str, Any]], seed: int) -> dict[str, Any]:
    np, _, LogisticRegression, _, roc_auc_score, train_test_split, make_pipeline, StandardScaler = _require_numpy_sklearn()
    n = min(len(source_rows), len(target_rows))
    if n < 100:
        return {"enabled": False, "reason": f"too few rows: source={len(source_rows)} target={len(target_rows)}"}
    source_rows = source_rows[:n]
    target_rows = target_rows[:n]
    x = np.concatenate([_matrix(source_rows), _matrix(target_rows)], axis=0)
    y = np.asarray([0] * n + [1] * n, dtype=np.int64)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(x_train, y_train)
    probs = clf.predict_proba(x_test)[:, 1]
    return {"enabled": True, "n_per_domain": n, "auc": float(roc_auc_score(y_test, probs))}


def _label_table(rows: list[dict[str, Any]], envs: list[int]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = sorted({str(row.get("label", "")) for row in rows})
    global_counts = Counter(str(row.get("label", "")) for row in rows)
    table: list[dict[str, Any]] = []
    for env in sorted(set(envs)):
        env_rows = [row for row, row_env in zip(rows, envs) if row_env == env]
        counts = Counter(str(row.get("label", "")) for row in env_rows)
        record: dict[str, Any] = {"env": env, "n": len(env_rows)}
        for label in labels:
            record[f"p_{label}"] = counts[label] / max(1, len(env_rows))
        table.append(record)
    max_shift = 0.0
    for record in table:
        for label in labels:
            shift = abs(float(record[f"p_{label}"]) - global_counts[label] / max(1, len(rows)))
            max_shift = max(max_shift, shift)
    return table, {"labels": labels, "max_abs_label_prop_shift": max_shift}


def _cluster_audit(source_rows: list[dict[str, Any]], targets: dict[str, list[dict[str, Any]]], seed: int, k: int) -> dict[str, Any]:
    np, KMeans, _, normalized_mutual_info_score, *_ = _require_numpy_sklearn()
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    source_x = scaler.fit_transform(_matrix(source_rows))
    kmeans = KMeans(n_clusters=k, random_state=seed, n_init=20)
    envs_np = kmeans.fit_predict(source_x)
    envs = [int(value) for value in envs_np.tolist()]
    labels = [str(row.get("label", "")) for row in source_rows]
    label_table, label_summary = _label_table(source_rows, envs)
    env_counts = Counter(envs)
    target_tables = {}
    for name, rows in targets.items():
        if not rows:
            continue
        target_envs = [int(value) for value in kmeans.predict(scaler.transform(_matrix(rows))).tolist()]
        target_counts = Counter(target_envs)
        target_tables[name] = {
            "n": len(rows),
            "env_distribution": {str(env): target_counts[env] / max(1, len(rows)) for env in range(k)},
        }
    return {
        "k": k,
        "source_assignments": [
            {"recording_id": str(row.get("recording_id", "")), "artifact_env": int(env)}
            for row, env in zip(source_rows, envs)
        ],
        "source_env_distribution": {str(env): env_counts[env] / max(1, len(source_rows)) for env in range(k)},
        "source_min_env_fraction": min(env_counts[env] / max(1, len(source_rows)) for env in range(k)),
        "label_nmi": float(normalized_mutual_info_score(labels, envs)),
        "label_table": label_table,
        "label_summary": label_summary,
        "target_envs": target_tables,
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Artifact Environment Audit",
        "",
        "## Inputs",
        "",
        f"- Source records: `{summary['counts']['source']}`",
        f"- COUGHVID records: `{summary['counts']['coughvid']}`",
        f"- Tos records: `{summary['counts']['tos']}`",
        "",
        "## Domain Probe",
        "",
        "| Target | AUC | n/domain |",
        "|---|---:|---:|",
    ]
    for name in ("coughvid", "tos"):
        item = summary["domain_probe"].get(name, {})
        if item.get("enabled"):
            lines.append(f"| {name} | {item['auc']:.6f} | {item['n_per_domain']} |")
        else:
            lines.append(f"| {name} | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Source Pseudo-Environments",
            "",
            f"- k: `{summary['cluster']['k']}`",
            f"- source min env fraction: `{summary['cluster']['source_min_env_fraction']:.6f}`",
            f"- label-env NMI: `{summary['cluster']['label_nmi']:.6f}`",
            f"- max label proportion shift: `{summary['cluster']['label_summary']['max_abs_label_prop_shift']:.6f}`",
            "",
            "Source env distribution:",
            "",
            "```json",
            json.dumps(summary["cluster"]["source_env_distribution"], indent=2),
            "```",
            "",
            "Target env distributions:",
            "",
            "```json",
            json.dumps(summary["cluster"]["target_envs"], indent=2),
            "```",
            "",
            "## Decision Hint",
            "",
            summary["decision_hint"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--coughvid-manifest", type=Path, default=ROOT / "manifests/coughvid_adapt_test.csv")
    parser.add_argument("--tos-manifest", type=Path, default=ROOT / "manifests/toscovid2021_test_external.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/artifact_environment_audit_seed7")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--coughvid-split", default="adapt")
    parser.add_argument("--tos-split", default="test")
    parser.add_argument("--max-source-records", type=int)
    parser.add_argument("--max-coughvid-records", type=int, default=1500)
    parser.add_argument("--max-tos-records", type=int)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--clusters", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    config = RunConfig(experiment_name="artifact_environment_audit", seed=args.seed)
    source_rows = [row for row in read_manifest(args.source_manifest) if row.get("split") == args.source_split]
    coughvid_rows = [row for row in read_manifest(args.coughvid_manifest) if row.get("split") == args.coughvid_split]
    tos_rows = [row for row in read_manifest(args.tos_manifest) if row.get("split") == args.tos_split]
    if args.max_source_records:
        source_rows = source_rows[: args.max_source_records]
    if args.max_coughvid_records:
        coughvid_rows = coughvid_rows[: args.max_coughvid_records]
    if args.max_tos_records:
        tos_rows = tos_rows[: args.max_tos_records]

    source_features = _load_or_build_cache(source_rows, args.root, config, args.max_duration_sec, args.out / "source_artifact_features.csv", "source")
    coughvid_features = _load_or_build_cache(coughvid_rows, args.root, config, args.max_duration_sec, args.out / "coughvid_artifact_features.csv", "coughvid")
    tos_features = _load_or_build_cache(tos_rows, args.root, config, args.max_duration_sec, args.out / "tos_artifact_features.csv", "tos")

    domain_probe = {
        "coughvid": _domain_probe(source_features, coughvid_features, args.seed),
        "tos": _domain_probe(source_features, tos_features, args.seed),
    }
    cluster = _cluster_audit(
        source_features,
        {"coughvid": coughvid_features, "tos": tos_features},
        args.seed,
        args.clusters,
    )
    assignment_path = args.out / f"source_artifact_envs_k{args.clusters}.csv"
    source_assignments = list(cluster.pop("source_assignments"))
    with assignment_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recording_id", "artifact_env"])
        writer.writeheader()
        writer.writerows(source_assignments)
    strong_shift = any(item.get("enabled") and float(item["auc"]) >= 0.75 for item in domain_probe.values())
    usable_envs = float(cluster["source_min_env_fraction"]) >= 0.05
    not_degenerate = float(cluster["label_nmi"]) <= 0.20
    if strong_shift and usable_envs and not_degenerate:
        decision_hint = "Proceed: artifact pseudo-environments are detectable, covered, and not fully label-degenerate."
    else:
        decision_hint = "Do not train yet: artifact pseudo-environments are weak, too imbalanced, or label-degenerate."
    summary = {
        "counts": {
            "source": len(source_features),
            "coughvid": len(coughvid_features),
            "tos": len(tos_features),
        },
        "feature_names": FEATURE_NAMES,
        "domain_probe": domain_probe,
        "cluster": cluster,
        "source_assignment_path": str(assignment_path),
        "decision_hint": decision_hint,
    }
    (args.out / "artifact_environment_audit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "ARTIFACT_ENVIRONMENT_AUDIT.md", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
