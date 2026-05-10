"""Dependency-free sanity baselines for smoke experiments."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from .metrics import binary_classification_report


def summarize_feature_matrix(features: list[list[float]]) -> list[float]:
    flat = [value for row in features for value in row]
    if not flat:
        return [0.0, 0.0]
    return [sum(flat) / len(flat), max(flat)]


class NearestCentroidBaseline:
    """Simple baseline standing in for classical MFCC/SVM smoke checks."""

    def __init__(self, name: str = "nearest_centroid") -> None:
        self.name = name
        self.centroids: dict[int, list[float]] = {}

    def fit(self, features: list[list[float]], labels: list[int]) -> None:
        buckets: dict[int, list[list[float]]] = defaultdict(list)
        for row, label in zip(features, labels):
            buckets[label].append(row)
        self.centroids = {}
        for label, rows in buckets.items():
            width = len(rows[0])
            self.centroids[label] = [sum(row[idx] for row in rows) / len(rows) for idx in range(width)]

    def score_positive(self, feature: list[float]) -> float:
        if set(self.centroids) != {0, 1}:
            raise ValueError("binary centroids for labels 0 and 1 are required")
        d0 = _squared_distance(feature, self.centroids[0])
        d1 = _squared_distance(feature, self.centroids[1])
        return d0 / (d0 + d1) if (d0 + d1) else 0.5

    def predict_scores(self, features: list[list[float]]) -> list[float]:
        return [self.score_positive(feature) for feature in features]


def _squared_distance(left: list[float], right: list[float]) -> float:
    return sum((a - b) ** 2 for a, b in zip(left, right))


def baseline_smoke_report(out_dir: Path) -> dict[str, object]:
    train_features = [[0.1, 0.2], [0.2, 0.2], [0.8, 0.9], [0.9, 0.8]]
    train_labels = [0, 0, 1, 1]
    test_features = [[0.15, 0.2], [0.85, 0.85], [0.25, 0.2], [0.75, 0.8]]
    test_labels = [0, 1, 0, 1]
    baselines = [
        NearestCentroidBaseline("mfcc_svm_smoke"),
        NearestCentroidBaseline("mfcc_random_forest_smoke"),
        NearestCentroidBaseline("logmel_bilstm_smoke"),
        NearestCentroidBaseline("resnet18_spectrogram_smoke"),
    ]
    rows = []
    for model in baselines:
        model.fit(train_features, train_labels)
        scores = model.predict_scores(test_features)
        rows.append({"model": model.name, "metrics": binary_classification_report(test_labels, scores)})
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {"baselines": rows}
    (out_dir / "baseline_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
