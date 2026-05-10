"""Dataset adapters for converting public cough corpora into project manifests."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .manifest import read_manifest, validate_manifest, write_manifest
from .smoke import make_smoke_data


AUDIO_EXTENSIONS = {".wav", ".wave", ".flac", ".mp3", ".m4a", ".ogg", ".webm"}
PATH_COLUMNS = ["path", "audio_path", "file", "filename", "file_path", "wav_path", "cough_path"]
SUBJECT_COLUMNS = ["subject_id", "participant_id", "user_id", "patient_id", "id", "uuid"]
LABEL_COLUMNS = ["label", "covid_status", "status", "diagnosis", "class", "result"]
SPLIT_COLUMNS = ["split", "fold", "partition", "subset"]
COSWARA_DEFAULT_RECORDINGS = ["cough-heavy", "cough-shallow"]
COSWARA_SYMPTOM_COLUMNS = [
    "asthma",
    "cough",
    "smoker",
    "ht",
    "cold",
    "diabetes",
    "diarrhoea",
    "ihd",
    "bd",
    "st",
    "fever",
    "ftg",
    "mp",
    "loss_of_smell",
    "cld",
    "pneumonia",
    "others_resp",
    "others_preexist",
]


@dataclass(frozen=True)
class DatasetBuildReport:
    dataset: str
    num_records: int
    num_subjects: int
    labels: dict[str, int]
    manifest: str
    validation_errors: int


def build_manifest_from_metadata(
    root: Path,
    metadata_csv: Path,
    out_manifest: Path,
    dataset: str,
    path_column: str | None = None,
    subject_column: str | None = None,
    label_column: str | None = None,
    split_column: str | None = None,
    default_split: str = "unassigned",
) -> DatasetBuildReport:
    """Build a project manifest from a dataset metadata CSV.

    The function accepts explicit column names, but also includes conservative
    aliases for common public cough datasets. Audio paths remain relative to
    `root`, which keeps manifests portable across local and remote workspaces.
    """

    metadata_path = metadata_csv if metadata_csv.is_absolute() else root / metadata_csv
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("metadata CSV is empty")
        fieldnames = list(reader.fieldnames)
        path_col = _choose_column(fieldnames, path_column, PATH_COLUMNS, required=True, purpose="audio path")
        subject_col = _choose_column(fieldnames, subject_column, SUBJECT_COLUMNS, required=False, purpose="subject")
        label_col = _choose_column(fieldnames, label_column, LABEL_COLUMNS, required=True, purpose="label")
        split_col = _choose_column(fieldnames, split_column, SPLIT_COLUMNS, required=False, purpose="split")
        rows = []
        for idx, raw in enumerate(reader, start=1):
            audio_rel = _normalize_relative_path(raw.get(path_col, ""), root)
            if not audio_rel:
                continue
            subject = _clean(raw.get(subject_col, "")) if subject_col else f"unknown_subject_{idx:06d}"
            label = normalize_label(raw.get(label_col, ""))
            split = normalize_split(raw.get(split_col, "")) if split_col else default_split
            recording_id = _recording_id(dataset, subject, audio_rel, idx)
            rows.append(
                {
                    "recording_id": recording_id,
                    "subject_id": subject or f"unknown_subject_{idx:06d}",
                    "dataset": dataset,
                    "path": audio_rel,
                    "label": label,
                    "split": split,
                }
            )
    return _write_and_report(root, rows, out_manifest, dataset)


def build_manifest_from_directory(
    root: Path,
    out_manifest: Path,
    dataset: str,
    label_from_parent: bool = True,
    default_label: str = "unknown",
    default_split: str = "unassigned",
) -> DatasetBuildReport:
    """Build a manifest by recursively discovering audio files under `root`."""

    rows = []
    for idx, audio_path in enumerate(sorted(_iter_audio_files(root)), start=1):
        rel_path = audio_path.relative_to(root).as_posix()
        label = normalize_label(audio_path.parent.name if label_from_parent else default_label)
        subject = infer_subject_id(audio_path)
        rows.append(
            {
                "recording_id": _recording_id(dataset, subject, rel_path, idx),
                "subject_id": subject,
                "dataset": dataset,
                "path": rel_path,
                "label": label,
                "split": default_split,
            }
        )
    return _write_and_report(root, rows, out_manifest, dataset)


def build_coswara_manifest(
    root: Path,
    metadata_csv: Path,
    out_manifest: Path,
    extracted_dir: str = "Extracted_data",
    recording_types: list[str] | None = None,
    default_split: str = "unassigned",
) -> DatasetBuildReport:
    """Build a cough-only Coswara manifest from extracted audio and metadata.

    Coswara metadata identifies participants by `id`; audio files live under
    `Extracted_data/<collection_date>/<id>/<recording_type>.wav`. This adapter
    scans the extracted tree first, then joins rows by participant ID.
    """

    metadata_path = metadata_csv if metadata_csv.is_absolute() else root / metadata_csv
    recordings = set(recording_types or COSWARA_DEFAULT_RECORDINGS)
    audio_index = _index_coswara_audio(root / extracted_dir, root, recordings)
    rows: list[dict[str, str]] = []
    with metadata_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("metadata CSV is empty")
        for raw in reader:
            subject = _clean(raw.get("id", ""))
            if not subject:
                continue
            for rel_path in audio_index.get(subject, []):
                rows.append(
                    {
                        "recording_id": _recording_id("coswara", subject, rel_path, len(rows) + 1),
                        "subject_id": subject,
                        "dataset": "coswara",
                        "path": rel_path,
                        "label": normalize_label(raw.get("covid_status", "")),
                        "split": default_split,
                        "age": _clean(raw.get("a", "")),
                        "sex": normalize_sex(raw.get("g", "")),
                        "country": _clean(raw.get("l_c", "")),
                        "symptoms": _coswara_symptoms(raw),
                        "device": "",
                        "quality_score": "",
                    }
                )
    rows.sort(key=lambda row: (row["subject_id"], row["path"]))
    return _write_and_report(root, rows, out_manifest, "coswara")


def dataset_smoke_report(out_dir: Path) -> dict[str, object]:
    """Exercise metadata import against the synthetic WAV fixture."""

    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_manifest = make_smoke_data(out_dir / "source")
    smoke_rows = read_manifest(smoke_manifest)
    metadata = out_dir / "metadata.csv"
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["audio_path", "participant_id", "diagnosis", "split"])
        writer.writeheader()
        for row in smoke_rows:
            writer.writerow(
                {
                    "audio_path": row["path"],
                    "participant_id": row["subject_id"],
                    "diagnosis": row["label"],
                    "split": row["split"],
                }
            )
    manifest = out_dir / "manifest.csv"
    report = build_manifest_from_metadata(
        root=out_dir,
        metadata_csv=metadata,
        out_manifest=manifest,
        dataset="synthetic_import",
        path_column="audio_path",
        subject_column="participant_id",
        label_column="diagnosis",
        split_column="split",
    )
    issues, summary = validate_manifest(manifest, out_dir)
    payload = {"build": asdict(report), "validation": summary, "errors": [issue.__dict__ for issue in issues]}
    (out_dir / "dataset_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def normalize_label(value: str | None) -> str:
    label = _clean(value).lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "positive": "covid_positive",
        "positive_mild": "covid_positive",
        "positive_moderate": "covid_positive",
        "positive_asymp": "covid_positive",
        "pos": "covid_positive",
        "covid": "covid_positive",
        "covid19": "covid_positive",
        "covid_19": "covid_positive",
        "negative": "healthy",
        "neg": "healthy",
        "normal": "healthy",
        "control": "healthy",
        "healthy": "healthy",
        "recovered_full": "covid_recovered",
        "recovered_resp": "covid_recovered",
        "no_resp_illness_exposed": "exposed",
        "resp_illness_not_identified": "respiratory_illness",
    }
    return aliases.get(label, label or "unknown")


def normalize_sex(value: str | None) -> str:
    sex = _clean(value).lower()
    aliases = {"m": "male", "f": "female", "man": "male", "woman": "female"}
    return aliases.get(sex, sex)


def normalize_split(value: str | None) -> str:
    split = _clean(value).lower().replace(" ", "_").replace("-", "_")
    aliases = {
        "dev": "val",
        "valid": "val",
        "validation": "val",
        "eval": "test",
        "evaluation": "test",
    }
    return aliases.get(split, split or "unassigned")


def infer_subject_id(path: Path) -> str:
    parts = [part for part in path.parts[:-1] if part not in {"audio", "wav", "wavs", "clips", "cough"}]
    if parts:
        return _clean(parts[-1]) or "unknown_subject"
    stem = path.stem
    for sep in ["_", "-"]:
        if sep in stem:
            return _clean(stem.split(sep)[0]) or "unknown_subject"
    return _clean(stem) or "unknown_subject"


def _index_coswara_audio(audio_root: Path, dataset_root: Path, recording_types: set[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for audio_path in sorted(_iter_audio_files(audio_root)):
        if audio_path.stem not in recording_types:
            continue
        try:
            rel_path = audio_path.relative_to(dataset_root).as_posix()
        except ValueError:
            rel_path = audio_path.as_posix()
        parts = Path(rel_path).parts
        if len(parts) < 4:
            continue
        subject = _clean(parts[-2])
        if not subject:
            continue
        index.setdefault(subject, []).append(rel_path)
    return index


def _coswara_symptoms(row: dict[str, str]) -> str:
    symptoms = [column for column in COSWARA_SYMPTOM_COLUMNS if _is_truthy(row.get(column, ""))]
    return ";".join(symptoms)


def _is_truthy(value: str | None) -> bool:
    return _clean(value).lower() in {"true", "t", "yes", "y", "1", "p", "positive"}


def _write_and_report(root: Path, rows: list[dict[str, str]], out_manifest: Path, dataset: str) -> DatasetBuildReport:
    write_manifest(rows, out_manifest)
    issues, _ = validate_manifest(out_manifest, root)
    labels: dict[str, int] = {}
    for row in rows:
        labels[row["label"]] = labels.get(row["label"], 0) + 1
    return DatasetBuildReport(
        dataset=dataset,
        num_records=len(rows),
        num_subjects=len({row["subject_id"] for row in rows}),
        labels=labels,
        manifest=str(out_manifest),
        validation_errors=sum(1 for issue in issues if issue.severity == "error"),
    )


def _iter_audio_files(root: Path) -> Iterable[Path]:
    return (path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS)


def _choose_column(
    fieldnames: list[str],
    explicit: str | None,
    candidates: list[str],
    required: bool,
    purpose: str,
) -> str | None:
    if explicit:
        if explicit not in fieldnames:
            raise ValueError(f"{purpose} column not found: {explicit}")
        return explicit
    by_lower = {field.lower(): field for field in fieldnames}
    for candidate in candidates:
        if candidate.lower() in by_lower:
            return by_lower[candidate.lower()]
    if required:
        raise ValueError(f"could not infer {purpose} column from: {fieldnames}")
    return None


def _normalize_relative_path(value: str, root: Path) -> str:
    cleaned = _clean(value).replace("\\", "/")
    if not cleaned:
        return ""
    path = Path(cleaned)
    if path.is_absolute():
        try:
            return path.relative_to(root).as_posix()
        except ValueError:
            return path.as_posix()
    return path.as_posix()


def _recording_id(dataset: str, subject: str, rel_path: str, idx: int) -> str:
    stem = Path(rel_path).stem
    safe = "_".join(part for part in [dataset, subject, stem, f"{idx:06d}"] if part)
    return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in safe)


def _clean(value: str | None) -> str:
    return (value or "").strip()
