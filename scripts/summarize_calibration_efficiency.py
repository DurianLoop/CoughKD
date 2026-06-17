"""Summarize calibration and deployment evidence for completed student runs."""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.torch_models import DepthwiseStudent, count_parameters  # noqa: E402


RUNS = ROOT / "runs"
OUT = RUNS / "calibration_efficiency"
SEEDS = [7, 11, 23]
METHODS = {
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
CHECKPOINTS = {
    "ce": {
        7: "stage1_panns_response_seed7/checkpoints/ce_student_best.pt",
        11: "stage1_panns_response_seed11/checkpoints/ce_student_best.pt",
        23: "stage1_panns_response_seed23/checkpoints/ce_student_best.pt",
    },
    "kd": {
        7: "stage1_panns_response_seed7/checkpoints/student_best.pt",
        11: "stage1_panns_response_seed11/checkpoints/student_best.pt",
        23: "stage1_panns_response_seed23/checkpoints/student_best.pt",
    },
    "source_only": {
        7: "stage3b_source_only_seed7/checkpoints/student_best.pt",
        11: "stage3c_source_only_seed11/checkpoints/student_best.pt",
        23: "stage3c_source_only_seed23/checkpoints/student_best.pt",
    },
    "tcd_very_strong": {
        7: "stage3b_tcd_very_strong_seed7/checkpoints/student_best.pt",
        11: "stage3c_tcd_very_strong_seed11/checkpoints/student_best.pt",
        23: "stage3c_tcd_very_strong_seed23/checkpoints/student_best.pt",
    },
    "tcd_conf035": {
        7: "stage3b_tcd_conf035_seed7/checkpoints/student_best.pt",
        11: "stage3c_tcd_conf035_seed11/checkpoints/student_best.pt",
        23: "stage3c_tcd_conf035_seed23/checkpoints/student_best.pt",
    },
    "candidate_a": {
        7: "candidate_a_shortcut_suppressed_seed7/checkpoints/student_best.pt",
        11: "candidate_a_shortcut_suppressed_seed11/checkpoints/student_best.pt",
        23: "candidate_a_shortcut_suppressed_seed23/checkpoints/student_best.pt",
    },
    "candidate_b": {
        7: "candidate_b_disagreement_gated_seed7/checkpoints/student_best.pt",
        11: "candidate_b_disagreement_gated_seed11/checkpoints/student_best.pt",
        23: "candidate_b_disagreement_gated_seed23/checkpoints/student_best.pt",
    },
    "candidate_c": {
        7: "candidate_c_probe_adv_seed7/student_domain_adv.pt",
        11: "candidate_c_probe_adv_seed11/student_domain_adv.pt",
        23: "candidate_c_probe_adv_seed23/student_domain_adv.pt",
    },
}


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.6f}"
    return str(value)


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    values = [value for value in values if not math.isnan(value)]
    if not values:
        return None, None
    return mean(values), pstdev(values) if len(values) > 1 else 0.0


def _load_predictions(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)
    return rows


def _calibration_metrics(rows: list[dict[str, Any]], bins: int = 10) -> dict[str, float]:
    prob_cols = [key for key in rows[0] if key.startswith("prob_")]
    classes = [col.removeprefix("prob_") for col in prob_cols]
    confidences: list[float] = []
    correct: list[float] = []
    briers: list[float] = []
    nlls: list[float] = []
    covid_scores: list[float] = []
    covid_true: list[int] = []
    for row in rows:
        probs = [float(row[col]) for col in prob_cols]
        pred_idx = max(range(len(probs)), key=lambda idx: probs[idx])
        true = str(row["true_label"])
        confidences.append(probs[pred_idx])
        correct.append(1.0 if classes[pred_idx] == true else 0.0)
        target = [1.0 if cls == true else 0.0 for cls in classes]
        briers.append(sum((prob - tgt) ** 2 for prob, tgt in zip(probs, target)))
        if true in classes:
            nlls.append(-math.log(max(probs[classes.index(true)], 1e-12)))
        if "covid_positive" in classes:
            covid_scores.append(probs[classes.index("covid_positive")])
            covid_true.append(1 if true == "covid_positive" else 0)

    ece = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        include_hi = bin_idx == bins - 1
        indices = [
            idx
            for idx, conf in enumerate(confidences)
            if (lo <= conf < hi) or (include_hi and lo <= conf <= hi)
        ]
        if not indices:
            continue
        acc = mean(correct[idx] for idx in indices)
        conf = mean(confidences[idx] for idx in indices)
        ece += (len(indices) / len(rows)) * abs(acc - conf)

    sens95 = float("nan")
    if covid_scores and len(set(covid_true)) == 2:
        thresholds = sorted(set(covid_scores), reverse=True)
        best = 0.0
        for threshold in thresholds + [min(covid_scores) - 1e-12]:
            tp = sum(1 for y, score in zip(covid_true, covid_scores) if y == 1 and score >= threshold)
            fn = sum(1 for y, score in zip(covid_true, covid_scores) if y == 1 and score < threshold)
            tn = sum(1 for y, score in zip(covid_true, covid_scores) if y == 0 and score < threshold)
            fp = sum(1 for y, score in zip(covid_true, covid_scores) if y == 0 and score >= threshold)
            specificity = tn / max(1, tn + fp)
            sensitivity = tp / max(1, tp + fn)
            if specificity >= 0.95:
                best = max(best, sensitivity)
        sens95 = best

    return {
        "ece": ece,
        "brier": mean(briers),
        "nll": mean(nlls) if nlls else float("nan"),
        "confidence": mean(confidences),
        "sens_at_95_spec_covid": sens95,
    }


def _checkpoint_size_mb(method: str, seed: int) -> float | None:
    rel = CHECKPOINTS[method][seed]
    path = RUNS / rel
    if not path.is_file():
        return None
    return path.stat().st_size / (1024 * 1024)


def _student_latency_ms(device_name: str, repeats: int = 300, warmup: int = 50) -> float | None:
    import torch

    device = torch.device(device_name)
    model = DepthwiseStudent(num_classes=5).to(device).eval()
    x = torch.randn(1, 1, 398, 32, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repeats):
            _ = model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
        end = time.perf_counter()
    return (end - start) * 1000.0 / repeats


def main() -> None:
    import torch

    rows: list[dict[str, Any]] = []
    for method, seed_runs in METHODS.items():
        seed_metrics: list[dict[str, float]] = []
        sizes: list[float] = []
        for seed, run_name in seed_runs.items():
            pred_path = RUNS / run_name / "predictions.csv"
            if not pred_path.is_file():
                continue
            metrics = _calibration_metrics(_load_predictions(pred_path))
            seed_metrics.append(metrics)
            size = _checkpoint_size_mb(method, seed)
            if size is not None:
                sizes.append(size)
        if not seed_metrics:
            continue
        item: dict[str, Any] = {"method": method, "n": len(seed_metrics)}
        for key in ["ece", "brier", "nll", "confidence", "sens_at_95_spec_covid"]:
            mu, sd = _mean_std([float(seed[key]) for seed in seed_metrics])
            item[key] = mu
            item[f"{key}_std"] = sd
        item["checkpoint_size_mb"] = mean(sizes) if sizes else None
        rows.append(item)

    student = DepthwiseStudent(num_classes=5)
    student_params = count_parameters(student)
    teacher_params = None
    teacher_size_mb = None
    metric_path = RUNS / "stage1_panns_response_seed7" / "metrics.json"
    if metric_path.is_file():
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        teacher_params = metric.get("config", {}).get("teacher_params")
    teacher_ckpt = RUNS / "stage1_panns_response_seed7" / "checkpoints" / "teacher_best.pt"
    if teacher_ckpt.is_file():
        teacher_size_mb = teacher_ckpt.stat().st_size / (1024 * 1024)

    latency = {"student_cpu_ms": _student_latency_ms("cpu", repeats=200, warmup=20)}
    if torch.cuda.is_available():
        latency["student_cuda_ms"] = _student_latency_ms("cuda", repeats=500, warmup=50)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(
        json.dumps(
            {
                "calibration": rows,
                "efficiency": {
                    "student_params": student_params,
                    "teacher_params": teacher_params,
                    "param_compression": (float(teacher_params) / student_params) if teacher_params else None,
                    "teacher_checkpoint_mb": teacher_size_mb,
                    "student_checkpoint_mb_mean": mean([float(row["checkpoint_size_mb"]) for row in rows if row.get("checkpoint_size_mb") is not None]),
                    **latency,
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Calibration and Efficiency Summary",
        "",
        "External calibration is computed from existing COUGHVID prediction CSV files. ECE uses 10 confidence bins; Brier is multiclass one-vs-probability squared error over the model label space.",
        "",
        "## Calibration",
        "",
        "| Method | ECE | Brier | NLL | Confidence | COVID sensitivity @ 95% specificity | Checkpoint MB | n |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {_fmt(row['ece'])} | {_fmt(row['brier'])} | {_fmt(row['nll'])} | {_fmt(row['confidence'])} | {_fmt(row['sens_at_95_spec_covid'])} | {_fmt(row['checkpoint_size_mb'])} | {row['n']} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency",
            "",
            f"- Student params: `{student_params}`",
            f"- Teacher params: `{teacher_params}`",
            f"- Parameter compression: `{_fmt((float(teacher_params) / student_params) if teacher_params else None)}x`",
            f"- Teacher checkpoint: `{_fmt(teacher_size_mb)} MB`",
            f"- Mean student checkpoint: `{_fmt(mean([float(row['checkpoint_size_mb']) for row in rows if row.get('checkpoint_size_mb') is not None]))} MB`",
            f"- Student CPU latency, batch=1 synthetic 4s feature: `{_fmt(latency.get('student_cpu_ms'))} ms`",
            f"- Student CUDA latency, batch=1 synthetic 4s feature: `{_fmt(latency.get('student_cuda_ms'))} ms`",
            "",
            "## Readout",
            "",
            "- These deployment numbers remain a strong project asset: the student is roughly four thousand times smaller in parameters than the PANNs CNN14 teacher.",
            "- The calibration table should be used as supporting evidence for the analysis paper, not as a standalone method claim.",
        ]
    )
    (OUT / "CALIBRATION_EFFICIENCY_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(OUT / "CALIBRATION_EFFICIENCY_REPORT.md")


if __name__ == "__main__":
    main()
