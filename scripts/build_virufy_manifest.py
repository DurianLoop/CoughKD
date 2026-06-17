"""Build an external manifest from a local clone of virufy/virufy-data.

Expected usage:
  git clone https://github.com/virufy/virufy-data.git D:\CoughKD\datasets\virufy-data
  python scripts\build_virufy_manifest.py --dataset-dir D:\CoughKD\datasets\virufy-data --out manifests\virufy_external.csv

The script is intentionally defensive because the repository is small and may
change. It searches for labels.csv and audio files recursively, then infers
common column names for audio filename and PCR/COVID status.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


AUDIO_COL_CANDIDATES = [
    "cough_filename",
    "filename",
    "file_name",
    "audio_filename",
    "audio_file",
    "audio",
    "path",
]
LABEL_COL_CANDIDATES = [
    "corona_test",
    "covid_status",
    "COVID_status",
    "status",
    "pcr_test_status",
    "pcr",
    "label",
]
ID_COL_CANDIDATES = ["patient_id", "subject_id", "id", "uuid", "participant_id"]


def _pick_column(columns: list[str], candidates: list[str], role: str) -> str:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    raise SystemExit(f"Could not infer {role} column. Available columns: {columns}")


def _normalize_label(value: object) -> str | None:
    raw = str(value).strip().lower()
    positive = {"positive", "pos", "covid", "covid-19", "covid_positive", "1", "true"}
    negative = {"negative", "neg", "non-covid", "noncovid", "healthy", "0", "false"}
    if raw in positive:
        return "covid_positive"
    if raw in negative:
        return "healthy"
    if "positive" in raw or "covid" in raw and "non" not in raw and "negative" not in raw:
        return "covid_positive"
    if "negative" in raw or "healthy" in raw or "non" in raw:
        return "healthy"
    return None


def _audio_index(dataset_dir: Path, audio_version: str) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in dataset_dir.rglob("*"):
        if path.suffix.lower() not in {".wav", ".flac", ".mp3", ".ogg", ".webm", ".m4a"}:
            continue
        parts = {part.lower() for part in path.parts}
        if audio_version == "original" and "segmented" in parts:
            continue
        if audio_version == "segmented" and "segmented" not in parts:
            continue
        index[path.name.lower()] = path
        index[path.stem.lower()] = path
    return index


def _find_audio(index: dict[str, Path], value: object) -> Path | None:
    raw = Path(str(value).strip())
    keys = [raw.name.lower(), raw.stem.lower()]
    for key in keys:
        if key in index:
            return index[key]
    return None


def _find_segmented_audio(index: dict[str, Path], value: object) -> list[Path]:
    raw = Path(str(value).strip())
    stem = raw.stem.lower()
    matches = [
        path
        for key, path in index.items()
        if key.startswith(f"{stem}-") and key == path.stem.lower()
    ]
    return sorted(set(matches), key=lambda path: path.name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--out", type=Path, default=Path("manifests/virufy_external.csv"))
    parser.add_argument("--split", default="test")
    parser.add_argument("--audio-version", choices=["original", "segmented"], default="original")
    args = parser.parse_args()

    labels_candidates = list(args.dataset_dir.rglob("labels.csv"))
    if not labels_candidates:
        raise SystemExit(f"Could not find labels.csv under {args.dataset_dir}")
    labels_path = labels_candidates[0]
    df = pd.read_csv(labels_path).fillna("")
    columns = list(df.columns)
    audio_col = _pick_column(columns, AUDIO_COL_CANDIDATES, "audio")
    label_col = _pick_column(columns, LABEL_COL_CANDIDATES, "label")
    id_col = None
    for candidate in ID_COL_CANDIDATES:
        if candidate.lower() in {col.lower() for col in columns}:
            id_col = next(col for col in columns if col.lower() == candidate.lower())
            break

    audio_files = _audio_index(args.dataset_dir, args.audio_version)
    rows: list[dict[str, str]] = []
    skipped = {"unknown_label": 0, "missing_audio": 0}
    for idx, item in enumerate(df.to_dict(orient="records")):
        label = _normalize_label(item[label_col])
        if label is None:
            skipped["unknown_label"] += 1
            continue
        if args.audio_version == "segmented":
            audios = _find_segmented_audio(audio_files, item[audio_col])
        else:
            audio = _find_audio(audio_files, item[audio_col])
            audios = [audio] if audio is not None else []
        if not audios:
            skipped["missing_audio"] += 1
            continue
        raw_id = str(item[id_col]) if id_col else Path(str(item[audio_col])).stem
        subject_id = f"virufy_{raw_id or idx}"
        for audio in audios:
            recording_id = f"virufy_{audio.stem}"
            rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject_id,
                    "dataset": "virufy",
                    "path": audio.relative_to(args.root).as_posix() if audio.is_relative_to(args.root) else audio.as_posix(),
                    "label": label,
                    "split": args.split,
                    "age": str(item.get("age", "")),
                    "sex": str(item.get("gender", item.get("sex", ""))),
                    "country": str(item.get("country", "")),
                    "device": str(item.get("device", "")),
                    "symptoms": str(item.get("medical_history", "")),
                    "quality_score": "",
                    "source_status": str(item[label_col]),
                }
            )

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
    counts = pd.Series([row["label"] for row in rows]).value_counts().to_dict() if rows else {}
    print(
        {
            "labels_csv": str(labels_path),
            "audio_col": audio_col,
            "label_col": label_col,
            "id_col": id_col,
            "audio_version": args.audio_version,
            "written": len(rows),
            "labels": counts,
            "skipped": skipped,
            "out": str(args.out),
        }
    )


if __name__ == "__main__":
    main()
