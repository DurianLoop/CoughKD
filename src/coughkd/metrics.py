"""Small deterministic metric implementations without third-party dependencies."""

from __future__ import annotations

import json
import random
from pathlib import Path


def _check_binary(labels: list[int], scores: list[float]) -> None:
    if len(labels) != len(scores):
        raise ValueError("labels and scores must have the same length")
    if not labels:
        raise ValueError("labels cannot be empty")
    if not set(labels).issubset({0, 1}):
        raise ValueError("binary labels must be 0/1")


def roc_auc(labels: list[int], scores: list[float]) -> float:
    _check_binary(labels, scores)
    positives = [score for label, score in zip(labels, scores) if label == 1]
    negatives = [score for label, score in zip(labels, scores) if label == 0]
    if not positives or not negatives:
        raise ValueError("AUROC requires both positive and negative labels")
    wins = 0.0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positives) * len(negatives))


def average_precision(labels: list[int], scores: list[float]) -> float:
    _check_binary(labels, scores)
    total_pos = sum(labels)
    if total_pos == 0:
        raise ValueError("AUPRC requires at least one positive label")
    pairs = sorted(zip(scores, labels), reverse=True)
    precisions = []
    tp = 0
    for idx, (_, label) in enumerate(pairs, start=1):
        if label == 1:
            tp += 1
            precisions.append(tp / idx)
    return sum(precisions) / total_pos


def threshold_metrics(labels: list[int], scores: list[float], threshold: float = 0.5) -> dict[str, float]:
    _check_binary(labels, scores)
    preds = [1 if score >= threshold else 0 for score in scores]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    precision_pos = tp / (tp + fp) if tp + fp else 0.0
    recall_pos = tp / (tp + fn) if tp + fn else 0.0
    f1_pos = 2 * precision_pos * recall_pos / (precision_pos + recall_pos) if precision_pos + recall_pos else 0.0
    precision_neg = tn / (tn + fn) if tn + fn else 0.0
    recall_neg = tn / (tn + fp) if tn + fp else 0.0
    f1_neg = 2 * precision_neg * recall_neg / (precision_neg + recall_neg) if precision_neg + recall_neg else 0.0
    return {
        "macro_f1": (f1_pos + f1_neg) / 2.0,
        "sensitivity": recall_pos,
        "specificity": recall_neg,
        "tp": float(tp),
        "tn": float(tn),
        "fp": float(fp),
        "fn": float(fn),
    }


def expected_calibration_error(labels: list[int], scores: list[float], bins: int = 10) -> float:
    _check_binary(labels, scores)
    total = len(labels)
    ece = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        members = [
            (label, score)
            for label, score in zip(labels, scores)
            if (lo <= score < hi) or (bin_idx == bins - 1 and score == 1.0)
        ]
        if not members:
            continue
        accuracy = sum(1 for label, score in members if (score >= 0.5) == bool(label)) / len(members)
        confidence = sum(score if score >= 0.5 else 1.0 - score for _, score in members) / len(members)
        ece += (len(members) / total) * abs(accuracy - confidence)
    return ece


def binary_classification_report(labels: list[int], scores: list[float]) -> dict[str, float]:
    report = {
        "auroc": roc_auc(labels, scores),
        "auprc": average_precision(labels, scores),
        "ece": expected_calibration_error(labels, scores),
    }
    report.update(threshold_metrics(labels, scores))
    return report


def bootstrap_auc_ci(
    labels: list[int],
    scores: list[float],
    n_bootstrap: int = 200,
    confidence: float = 0.95,
    seed: int = 7,
) -> dict[str, float]:
    _check_binary(labels, scores)
    if not 0 < confidence < 1:
        raise ValueError("confidence must be in (0, 1)")
    rng = random.Random(seed)
    aucs: list[float] = []
    indices = list(range(len(labels)))
    for _ in range(n_bootstrap):
        sample_indices = [rng.choice(indices) for _ in indices]
        sample_labels = [labels[idx] for idx in sample_indices]
        sample_scores = [scores[idx] for idx in sample_indices]
        if len(set(sample_labels)) < 2:
            continue
        aucs.append(roc_auc(sample_labels, sample_scores))
    if not aucs:
        raise ValueError("bootstrap failed to sample both classes")
    aucs.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * (len(aucs) - 1))
    hi_idx = int((1.0 - alpha) * (len(aucs) - 1))
    return {"mean": sum(aucs) / len(aucs), "low": aucs[lo_idx], "high": aucs[hi_idx]}


def multiclass_ovr_auroc(labels: list[str], score_rows: list[list[float]], classes: list[str]) -> dict[str, float]:
    if len(labels) != len(score_rows):
        raise ValueError("labels and score_rows must have the same length")
    if not classes:
        raise ValueError("classes cannot be empty")
    per_class: dict[str, float] = {}
    for class_idx, class_name in enumerate(classes):
        binary_labels = [1 if label == class_name else 0 for label in labels]
        binary_scores = [row[class_idx] for row in score_rows]
        if len(set(binary_labels)) < 2:
            continue
        per_class[class_name] = roc_auc(binary_labels, binary_scores)
    if not per_class:
        raise ValueError("no class had both positive and negative examples")
    per_class["macro_ovr_auroc"] = sum(per_class.values()) / len(per_class)
    return per_class


def external_drop_report(in_domain_auroc: float, external_auroc: float) -> dict[str, float]:
    return {
        "in_domain_auroc": in_domain_auroc,
        "external_auroc": external_auroc,
        "external_auroc_drop": in_domain_auroc - external_auroc,
    }


def write_report(report: dict[str, float], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = ["# Metrics Report", ""]
    for key, value in report.items():
        lines.append(f"- {key}: {value:.6f}")
    (out_dir / "metrics.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
