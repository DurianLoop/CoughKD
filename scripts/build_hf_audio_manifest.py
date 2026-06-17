"""Build a CoughKD-style manifest from a Hugging Face audio dataset.

This script is intentionally optional-dependency based. Install `datasets` only
when a selected HF dataset becomes part of the experiment plan.
"""

from __future__ import annotations

import argparse
import csv
import math
import wave
from pathlib import Path
from typing import Any


def _require_datasets() -> Any:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - optional dependency
        raise SystemExit("Missing optional dependency `datasets`. Install with: pip install datasets soundfile") from exc
    return load_dataset


def _parse_map(items: list[str] | None) -> dict[str, str]:
    if not items:
        return {}
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected SOURCE=TARGET mapping, got: {item}")
        src, dst = item.split("=", 1)
        out[src] = dst
    return out


def _safe(value: Any) -> str:
    text = str(value).strip()
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in text)[:120] or "unknown"


def _label_value(row: dict[str, Any], label_col: str, label_names: list[str] | None) -> str:
    value = row[label_col]
    if isinstance(value, int) and label_names and 0 <= value < len(label_names):
        return str(label_names[value])
    return str(value)


def _audio_array(audio: Any) -> tuple[list[float], int] | None:
    if isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        array = audio["array"]
        rate = int(audio["sampling_rate"])
        return [float(item) for item in array], rate
    return None


def _write_wav(path: Path, samples: list[float], sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    peak = max((abs(x) for x in samples), default=1.0)
    scale = 32767.0 / peak if peak > 1.0 else 32767.0
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        frames = bytearray()
        for sample in samples:
            value = int(max(-32768, min(32767, round(sample * scale))))
            frames.extend(value.to_bytes(2, byteorder="little", signed=True))
        handle.writeframes(bytes(frames))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HF dataset name, e.g. vtsouval/flusense")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--audio-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--id-col", default="")
    parser.add_argument("--subject-col", default="")
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--audio-out-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--label-map", nargs="*", default=[])
    parser.add_argument("--keep-labels", nargs="*", default=[])
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--streaming", action="store_true")
    args = parser.parse_args()

    load_dataset = _require_datasets()
    label_map = _parse_map(args.label_map)
    keep = set(args.keep_labels)

    dataset = load_dataset(args.dataset, args.config, split=args.split, streaming=args.streaming)
    label_names = None
    if not args.streaming:
        feature = getattr(dataset, "features", {}).get(args.label_col)
        label_names = getattr(feature, "names", None)

    rows: list[dict[str, str]] = []
    skipped = {"label": 0, "audio": 0}
    for idx, item in enumerate(dataset):
        raw_label = _label_value(item, args.label_col, label_names)
        label = label_map.get(raw_label, raw_label)
        if keep and label not in keep:
            skipped["label"] += 1
            continue
        audio_pair = _audio_array(item[args.audio_col])
        if audio_pair is None:
            skipped["audio"] += 1
            continue
        samples, sample_rate = audio_pair
        if not samples or all(math.isclose(sample, 0.0) for sample in samples[: min(len(samples), 256)]):
            pass
        raw_id = item.get(args.id_col, idx) if args.id_col else idx
        subject = item.get(args.subject_col, raw_id) if args.subject_col else raw_id
        recording_id = f"{args.dataset_name}_{_safe(raw_id)}"
        audio_path = args.audio_out_dir / f"{recording_id}.wav"
        _write_wav(audio_path, samples, sample_rate)
        rows.append(
            {
                "recording_id": recording_id,
                "subject_id": str(subject),
                "dataset": args.dataset_name,
                "path": audio_path.relative_to(args.root).as_posix() if audio_path.is_relative_to(args.root) else audio_path.as_posix(),
                "label": label,
                "split": "test",
                "age": str(item.get("age", "")),
                "sex": str(item.get("sex", item.get("gender", ""))),
                "country": str(item.get("country", "")),
                "device": str(item.get("device", "")),
                "symptoms": "",
                "quality_score": "",
                "source_status": raw_label,
            }
        )
        if args.max_records and len(rows) >= args.max_records:
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "recording_id",
        "subject_id",
        "dataset",
        "path",
        "label",
        "split",
        "age",
        "sex",
        "country",
        "device",
        "symptoms",
        "quality_score",
        "source_status",
    ]
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print({"written": len(rows), "skipped": skipped, "out": str(args.out)})


if __name__ == "__main__":
    main()
