"""Build a CoughKD external manifest from a generic metadata CSV.

Use this for DiCOVA or other cough datasets when the metadata schema differs
from COUGHVID. The script is deliberately column-configurable so it can adapt to
official challenge CSV files without editing code.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


def _find_audio(dataset_dir: Path, value: str) -> Path | None:
    raw = Path(value)
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
    else:
        candidates.append(dataset_dir / raw)
        candidates.append(dataset_dir / raw.name)
    if raw.suffix:
        stem = raw.with_suffix("").name
    else:
        stem = raw.name
    for ext in (".wav", ".flac", ".ogg", ".webm", ".mp3"):
        candidates.append(dataset_dir / f"{stem}{ext}")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def _parse_label_map(items: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"label map item must be SOURCE=TARGET: {item}")
        src, dst = item.split("=", 1)
        mapping[src] = dst
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--audio-col", required=True)
    parser.add_argument("--label-col", required=True)
    parser.add_argument("--subject-col", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--label-map",
        nargs="+",
        required=True,
        help="Mappings like COVID-19=covid_positive nonCOVID=healthy.",
    )
    parser.add_argument("--keep-labels", nargs="*", default=["covid_positive", "healthy", "respiratory_illness"])
    args = parser.parse_args()

    label_map = _parse_label_map(args.label_map)
    df = pd.read_csv(args.metadata).fillna("")
    required = [args.id_col, args.audio_col, args.label_col]
    if args.subject_col:
        required.append(args.subject_col)
    missing_cols = [col for col in required if col not in df.columns]
    if missing_cols:
        raise SystemExit(f"missing columns in {args.metadata}: {missing_cols}; available={list(df.columns)}")

    rows: list[dict[str, str]] = []
    skipped = {"unknown_label": 0, "missing_audio": 0}
    keep = set(args.keep_labels)
    for item in df.to_dict(orient="records"):
        raw_label = str(item[args.label_col])
        label = label_map.get(raw_label)
        if label is None or label not in keep:
            skipped["unknown_label"] += 1
            continue
        audio = _find_audio(args.dataset_dir, str(item[args.audio_col]))
        if audio is None:
            skipped["missing_audio"] += 1
            continue
        recording_id = f"{args.dataset_name}_{item[args.id_col]}"
        subject_id = str(item[args.subject_col]) if args.subject_col else str(item[args.id_col])
        rows.append(
            {
                "recording_id": recording_id,
                "subject_id": subject_id,
                "dataset": args.dataset_name,
                "path": audio.relative_to(args.root).as_posix() if audio.is_relative_to(args.root) else audio.as_posix(),
                "label": label,
                "split": args.split,
                "age": str(item.get("age", "")),
                "sex": str(item.get("sex", item.get("gender", ""))),
                "country": str(item.get("country", "")),
                "device": str(item.get("device", "")),
                "symptoms": "",
                "quality_score": "",
                "source_status": raw_label,
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
    print({"written": len(rows), "labels": counts, "skipped": skipped, "out": str(args.out)})


if __name__ == "__main__":
    main()
