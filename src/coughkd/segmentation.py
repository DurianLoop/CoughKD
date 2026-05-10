"""Fallback segmentation and recording-level aggregation utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    start_sec: float
    end_sec: float
    quality: float = 1.0


def sliding_windows(duration_sec: float, window_sec: float = 1.0, hop_sec: float = 0.5) -> list[Segment]:
    if duration_sec <= 0:
        return []
    if window_sec <= 0 or hop_sec <= 0:
        raise ValueError("window_sec and hop_sec must be positive")
    segments: list[Segment] = []
    start = 0.0
    while start < duration_sec:
        end = min(duration_sec, start + window_sec)
        segments.append(Segment(round(start, 6), round(end, 6), 1.0))
        if end >= duration_sec:
            break
        start += hop_sec
    return segments


def merge_intervals(intervals: list[tuple[float, float]], gap_sec: float = 0.1) -> list[tuple[float, float]]:
    if gap_sec < 0:
        raise ValueError("gap_sec cannot be negative")
    valid = sorted((start, end) for start, end in intervals if end > start)
    if not valid:
        return []
    merged = [valid[0]]
    for start, end in valid[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + gap_sec:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def rms_quality(samples: list[float], floor: float = 1e-6) -> float:
    if not samples:
        return 0.0
    rms = (sum(sample * sample for sample in samples) / len(samples)) ** 0.5
    return max(0.0, min(1.0, rms / (rms + floor)))


def aggregate_scores(
    scores: list[float],
    qualities: list[float] | None = None,
    method: str = "mean",
    top_k: int = 3,
) -> float:
    if not scores:
        raise ValueError("scores cannot be empty")
    if qualities is not None and len(qualities) != len(scores):
        raise ValueError("qualities must match scores")

    if method == "mean":
        return sum(scores) / len(scores)
    if method == "max":
        return max(scores)
    if method == "topk":
        chosen = sorted(scores, reverse=True)[: max(1, top_k)]
        return sum(chosen) / len(chosen)
    if method == "quality_topk":
        quality_values = qualities if qualities is not None else scores
        ranked = sorted(zip(scores, quality_values), key=lambda item: item[1], reverse=True)[: max(1, top_k)]
        weight_sum = sum(max(0.0, quality) for _, quality in ranked)
        if weight_sum == 0:
            return sum(score for score, _ in ranked) / len(ranked)
        return sum(score * max(0.0, quality) for score, quality in ranked) / weight_sum
    raise ValueError(f"unknown aggregation method: {method}")
