from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pandas as pd


LABEL_MAP = {
    "healthy": "healthy",
    "symptomatic": "respiratory_illness",
    "COVID-19": "covid_positive",
}


def _find_audio(dataset_dir: Path, uuid: str) -> Path | None:
    for ext in (".webm", ".ogg", ".wav"):
        path = dataset_dir / f"{uuid}{ext}"
        if path.is_file():
            return path
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, default=Path(r"D:\CoughKD\datasets\public_dataset\metadata_compiled.csv"))
    parser.add_argument("--dataset-dir", type=Path, default=Path(r"D:\CoughKD\datasets\public_dataset"))
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--out", type=Path, default=Path(r"D:\CoughKD\AAAI\manifests\coughvid_external.csv"))
    parser.add_argument("--min-cough-detected", type=float, default=0.8)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    df = pd.read_csv(args.metadata)
    df["cough_detected_num"] = pd.to_numeric(df["cough_detected"], errors="coerce")
    rows = []
    skipped = {"missing_status": 0, "unknown_status": 0, "low_cough_detected": 0, "missing_audio": 0}
    for item in df.itertuples(index=False):
        uuid = str(item.uuid)
        status = getattr(item, "status")
        if not isinstance(status, str) or not status:
            skipped["missing_status"] += 1
            continue
        if status not in LABEL_MAP:
            skipped["unknown_status"] += 1
            continue
        cough_detected = getattr(item, "cough_detected_num")
        if pd.isna(cough_detected) or float(cough_detected) < args.min_cough_detected:
            skipped["low_cough_detected"] += 1
            continue
        audio_path = _find_audio(args.dataset_dir, uuid)
        if audio_path is None:
            skipped["missing_audio"] += 1
            continue
        rel_path = audio_path.relative_to(args.root).as_posix()
        rows.append(
            {
                "recording_id": f"coughvid_{uuid}",
                "subject_id": uuid,
                "dataset": "coughvid",
                "path": rel_path,
                "label": LABEL_MAP[status],
                "split": args.split,
                "age": getattr(item, "age", ""),
                "sex": getattr(item, "gender", ""),
                "country": "",
                "device": "",
                "symptoms": f"respiratory_condition={getattr(item, 'respiratory_condition', '')};fever_muscle_pain={getattr(item, 'fever_muscle_pain', '')}",
                "quality_score": "",
                "cough_detected": f"{float(cough_detected):.4f}",
                "source_status": status,
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
        "cough_detected",
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
