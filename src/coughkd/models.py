"""Minimal model interfaces and smoke models.

These classes are not SOTA models. They provide a dependency-free interface
contract for later PyTorch BEATs/AST/PANNs teachers and compact students.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelOutput:
    logits: list[float]
    features: list[float]
    embedding: list[float]
    attention: list[list[float]]


class BaseAudioModel:
    name = "base"

    def forward(self, features: list[list[float]]) -> ModelOutput:
        raise NotImplementedError

    def extract_features(self, features: list[list[float]]) -> list[float]:
        return self.forward(features).features

    def extract_attention(self, features: list[list[float]]) -> list[list[float]]:
        return self.forward(features).attention

    def parameter_count(self) -> int:
        raise NotImplementedError


def _summary_features(features: list[list[float]]) -> list[float]:
    flat = [value for row in features for value in row]
    if not flat:
        return [0.0, 0.0, 0.0, 0.0]
    mean = sum(flat) / len(flat)
    maximum = max(flat)
    minimum = min(flat)
    energy = sum(value * value for value in flat) / len(flat)
    return [mean, maximum, minimum, energy]


class SmokeTeacher(BaseAudioModel):
    name = "smoke_teacher"

    def __init__(self) -> None:
        self.weights = [
            [1.2, 0.4, -0.5, 0.8],
            [-0.7, 0.2, 0.6, -0.3],
        ]
        self.bias = [0.1, -0.1]

    def forward(self, features: list[list[float]]) -> ModelOutput:
        summary = _summary_features(features)
        logits = [sum(w * x for w, x in zip(row, summary)) + b for row, b in zip(self.weights, self.bias)]
        attention = _attention_proxy(features)
        embedding = [math.tanh(value) for value in summary]
        return ModelOutput(logits=logits, features=summary, embedding=embedding, attention=attention)

    def parameter_count(self) -> int:
        return sum(len(row) for row in self.weights) + len(self.bias)


class SmokeStudent(BaseAudioModel):
    name = "smoke_student"

    def __init__(self) -> None:
        self.weights = [
            [0.8, 0.3, -0.2, 0.4],
            [-0.4, 0.1, 0.4, -0.2],
        ]
        self.bias = [0.0, 0.0]

    def forward(self, features: list[list[float]]) -> ModelOutput:
        summary = _summary_features(features)
        logits = [sum(w * x for w, x in zip(row, summary)) + b for row, b in zip(self.weights, self.bias)]
        attention = _attention_proxy(features)
        embedding = [math.tanh(value * 0.8) for value in summary]
        return ModelOutput(logits=logits, features=summary, embedding=embedding, attention=attention)

    def parameter_count(self) -> int:
        return sum(len(row) for row in self.weights) + len(self.bias)


def _attention_proxy(features: list[list[float]]) -> list[list[float]]:
    if not features:
        return []
    row_sums = [sum(abs(value) for value in row) for row in features]
    denom = sum(row_sums) or 1.0
    weights = [value / denom for value in row_sums]
    return [[weight for _ in row] for weight, row in zip(weights, features)]
