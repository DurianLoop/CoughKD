"""Synthetic fixtures for smoke verification."""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from .audio import write_wav


def make_smoke_data(out_dir: Path, seed: int = 7) -> Path:
    rng = random.Random(seed)
    audio_dir = out_dir / "audio"
    rows = []
    sample_rate = 16000
    labels = ["healthy", "cough_positive"]
    for subject_idx in range(6):
        subject_id = f"subject_{subject_idx:02d}"
        label = labels[subject_idx % 2]
        for rec_idx in range(2):
            recording_id = f"{subject_id}_rec_{rec_idx}"
            freq = 220 + 30 * subject_idx + 5 * rec_idx
            samples = []
            for n in range(int(sample_rate * 0.6)):
                t = n / sample_rate
                cough_burst = math.exp(-((t - 0.25) ** 2) / 0.002)
                tone = math.sin(2 * math.pi * freq * t)
                noise = rng.uniform(-0.01, 0.01)
                samples.append(0.25 * cough_burst * tone + noise)
            wav_path = audio_dir / f"{recording_id}.wav"
            write_wav(wav_path, samples, sample_rate)
            rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "dataset": "synthetic_smoke",
                    "path": str(wav_path.resolve()),
                    "label": label,
                    "split": "",
                    "age": "",
                    "sex": "",
                    "country": "",
                    "device": "synthetic",
                    "symptoms": "",
                    "quality_score": "1.0",
                }
            )

    manifest = out_dir / "manifest.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return manifest
