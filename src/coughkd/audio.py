"""Standard-library audio preprocessing for smoke tests."""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
import sys
import wave
from dataclasses import asdict, dataclass
from pathlib import Path

from .config import RunConfig
from .cache import save_feature_cache
from .manifest import read_manifest


@dataclass(frozen=True)
class AudioStats:
    sample_rate: int
    num_samples: int
    duration_sec: float
    rms: float
    clipping_fraction: float
    valid: bool
    reason: str = ""


def write_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = bytearray()
    for sample in samples:
        clipped = max(-1.0, min(1.0, sample))
        frames.extend(struct.pack("<h", int(clipped * 32767)))
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(bytes(frames))


def read_wav_mono(path: Path) -> tuple[list[float], int]:
    try:
        with wave.open(str(path), "rb") as handle:
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            sample_rate = handle.getframerate()
            raw = handle.readframes(handle.getnframes())
    except (wave.Error, EOFError):
        return _read_with_ffmpeg_mono(path, sample_rate=16000)
    if sample_width != 2:
        raise ValueError(f"unsupported WAV sample width {sample_width}; expected 16-bit PCM")
    ints = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    if channels == 1:
        return [value / 32768.0 for value in ints], sample_rate
    mono: list[float] = []
    for idx in range(0, len(ints), channels):
        mono.append(sum(ints[idx : idx + channels]) / (32768.0 * channels))
    return mono, sample_rate


def _read_with_ffmpeg_mono(path: Path, sample_rate: int = 16000) -> tuple[list[float], int]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        env_prefix = Path(sys.executable).resolve().parent if "sys" in globals() else None
        candidate = env_prefix / "Library" / "bin" / "ffmpeg.exe" if env_prefix is not None else None
        if candidate is not None and candidate.is_file():
            ffmpeg = str(candidate)
    if ffmpeg is None:
        raise RuntimeError(
            f"Cannot decode non-WAV audio without ffmpeg on PATH: {path}. "
            "Install ffmpeg in the CoughKD conda environment first."
        )
    cmd = [
        ffmpeg,
        "-v",
        "error",
        "-i",
        str(path),
        "-f",
        "s16le",
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-",
    ]
    raw = subprocess.check_output(cmd)
    if not raw:
        return [], sample_rate
    ints = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    return [value / 32768.0 for value in ints], sample_rate


def resample_linear(samples: list[float], src_rate: int, dst_rate: int) -> list[float]:
    if src_rate == dst_rate or not samples:
        return samples
    duration = len(samples) / src_rate
    dst_len = max(1, int(round(duration * dst_rate)))
    scale = src_rate / dst_rate
    out: list[float] = []
    for i in range(dst_len):
        pos = i * scale
        left = int(math.floor(pos))
        right = min(left + 1, len(samples) - 1)
        frac = pos - left
        out.append(samples[left] * (1.0 - frac) + samples[right] * frac)
    return out


def audio_stats(samples: list[float], sample_rate: int, config: RunConfig) -> AudioStats:
    num_samples = len(samples)
    duration = num_samples / sample_rate if sample_rate else 0.0
    rms = math.sqrt(sum(sample * sample for sample in samples) / max(1, num_samples))
    clipping = sum(1 for sample in samples if abs(sample) >= 0.999) / max(1, num_samples)
    valid = True
    reason = ""
    if duration < config.min_duration_sec:
        valid = False
        reason = "too_short"
    elif rms < config.min_rms:
        valid = False
        reason = "low_energy"
    elif clipping > config.max_clip_fraction:
        valid = False
        reason = "clipped"
    return AudioStats(sample_rate, num_samples, duration, rms, clipping, valid, reason)


def log_mel_like(samples: list[float], config: RunConfig) -> list[list[float]]:
    """Return deterministic lightweight log-energy bands for smoke tests.

    This is not a production mel frontend. It gives shape-stable features in
    minimal environments until the real torchaudio/librosa frontend is added.
    """

    features: list[list[float]] = []
    frame = config.frame_size
    hop = config.hop_size
    if len(samples) < frame:
        samples = samples + [0.0] * (frame - len(samples))
    for start in range(0, max(1, len(samples) - frame + 1), hop):
        chunk = samples[start : start + frame]
        if len(chunk) < frame:
            chunk = chunk + [0.0] * (frame - len(chunk))
        bands: list[float] = []
        band_size = max(1, frame // config.n_mels)
        for band in range(config.n_mels):
            sub = chunk[band * band_size : (band + 1) * band_size]
            energy = sum(value * value for value in sub) / max(1, len(sub))
            bands.append(math.log1p(energy * 1000.0))
        features.append(bands)
    return features


def preprocess_manifest_smoke(manifest: Path, root: Path, out_dir: Path, config: RunConfig) -> dict[str, object]:
    rows = read_manifest(manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    reports = []
    for row in rows:
        path = root / row["path"]
        samples, sample_rate = read_wav_mono(path)
        samples = resample_linear(samples, sample_rate, config.sample_rate)
        stats = audio_stats(samples, config.sample_rate, config)
        features = log_mel_like(samples, config)
        cache_path = save_feature_cache(out_dir, row["recording_id"], config, features)
        reports.append(
            {
                "recording_id": row["recording_id"],
                "stats": asdict(stats),
                "feature_shape": [len(features), len(features[0]) if features else 0],
                "feature_cache": str(cache_path),
            }
        )
    summary = {
        "num_recordings": len(reports),
        "num_valid": sum(1 for item in reports if item["stats"]["valid"]),
        "records": reports,
    }
    (out_dir / "preprocess_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
