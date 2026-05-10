"""Deterministic smoke augmentations for waveform and feature matrices."""

from __future__ import annotations

import random


def add_uniform_noise(samples: list[float], amplitude: float, seed: int) -> list[float]:
    if amplitude < 0:
        raise ValueError("amplitude cannot be negative")
    rng = random.Random(seed)
    return [max(-1.0, min(1.0, sample + rng.uniform(-amplitude, amplitude))) for sample in samples]


def time_shift(samples: list[float], shift: int) -> list[float]:
    if not samples:
        return []
    shift = shift % len(samples)
    return samples[-shift:] + samples[:-shift] if shift else list(samples)


def specaugment(
    features: list[list[float]],
    time_mask: int = 0,
    freq_mask: int = 0,
    seed: int = 7,
) -> list[list[float]]:
    if not features:
        return []
    width = len(features[0])
    augmented = [list(row) for row in features]
    rng = random.Random(seed)
    if time_mask > 0:
        start = rng.randrange(0, max(1, len(augmented)))
        for row in augmented[start : start + time_mask]:
            for idx in range(width):
                row[idx] = 0.0
    if freq_mask > 0 and width > 0:
        start = rng.randrange(0, width)
        for row in augmented:
            for idx in range(start, min(width, start + freq_mask)):
                row[idx] = 0.0
    return augmented
