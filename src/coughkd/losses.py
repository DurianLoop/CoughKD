"""Dependency-free loss functions for KD smoke tests."""

from __future__ import annotations

import math


def softmax(logits: list[float], temperature: float = 1.0) -> list[float]:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = [logit / temperature for logit in logits]
    max_logit = max(scaled)
    exps = [math.exp(value - max_logit) for value in scaled]
    denom = sum(exps)
    return [value / denom for value in exps]


def cross_entropy(label: int, logits: list[float]) -> float:
    probs = softmax(logits)
    if label < 0 or label >= len(probs):
        raise ValueError("label out of range")
    return -math.log(max(probs[label], 1e-12))


def kl_divergence(teacher_logits: list[float], student_logits: list[float], temperature: float = 1.0) -> float:
    teacher_probs = softmax(teacher_logits, temperature)
    student_probs = softmax(student_logits, temperature)
    return sum(
        t_prob * math.log(max(t_prob, 1e-12) / max(s_prob, 1e-12))
        for t_prob, s_prob in zip(teacher_probs, student_probs)
    ) * temperature * temperature


def mse(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return sum((a - b) ** 2 for a, b in zip(left, right)) / max(1, len(left))


def flatten(matrix: list[list[float]]) -> list[float]:
    return [value for row in matrix for value in row]


def relation_loss(teacher_embeddings: list[list[float]], student_embeddings: list[list[float]]) -> float:
    if len(teacher_embeddings) != len(student_embeddings):
        raise ValueError("teacher and student batch sizes must match")
    teacher_sim = _similarity_matrix(teacher_embeddings)
    student_sim = _similarity_matrix(student_embeddings)
    return mse(flatten(teacher_sim), flatten(student_sim))


def _similarity_matrix(vectors: list[list[float]]) -> list[list[float]]:
    return [[_cosine(a, b) for b in vectors] for a in vectors]


def _cosine(left: list[float], right: list[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def coughkd_loss(
    label: int,
    teacher_logits: list[float],
    student_logits: list[float],
    teacher_features: list[float],
    student_features: list[float],
    teacher_attention: list[list[float]],
    student_attention: list[list[float]],
    teacher_batch_embeddings: list[list[float]],
    student_batch_embeddings: list[list[float]],
    alpha: float = 1.0,
    beta: float = 1.0,
    delta: float = 1.0,
    eta: float = 1.0,
    temperature: float = 2.0,
) -> dict[str, float]:
    ce = cross_entropy(label, student_logits)
    response = kl_divergence(teacher_logits, student_logits, temperature)
    feature = mse(teacher_features, student_features)
    attention = mse(flatten(teacher_attention), flatten(student_attention))
    relation = relation_loss(teacher_batch_embeddings, student_batch_embeddings)
    total = ce + alpha * response + beta * feature + delta * attention + eta * relation
    return {
        "total": total,
        "ce": ce,
        "response_kd": response,
        "feature_kd": feature,
        "attention_kd": attention,
        "relation_kd": relation,
    }
