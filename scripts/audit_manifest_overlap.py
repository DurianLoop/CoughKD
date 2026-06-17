"""Audit possible overlap between a source manifest and an external target.

This is intended for Coswara-derived auxiliary targets such as DiCOVA, where
an external-looking protocol may still share participants or recordings with
the source data. The default checks are string/stem based and fast. Optional
hashing can be enabled when audio files are locally available.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


TOKEN_RE = re.compile(r"[a-z0-9]+")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [{key: (value or "") for key, value in row.items()} for row in csv.DictReader(handle)]


def _norm(value: str) -> str:
    return value.strip().lower().replace("\\", "/")


def _path_stem(value: str) -> str:
    return Path(_norm(value)).stem


def _path_tail(value: str, depth: int) -> str:
    parts = Path(_norm(value)).parts
    return "/".join(parts[-depth:]) if parts else ""


def _tokens(*values: str) -> set[str]:
    out: set[str] = set()
    for value in values:
        for token in TOKEN_RE.findall(_norm(value)):
            if len(token) >= 6:
                out.add(token)
    return out


def _resolve_audio(root: Path, path_value: str) -> Path | None:
    path = Path(path_value)
    if path.is_absolute():
        return path if path.is_file() else None
    candidate = root / path
    return candidate if candidate.is_file() else None


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _index(rows: list[dict[str, str]], root: Path, enable_hash: bool) -> dict[str, dict[str, list[dict[str, str]]]]:
    index: dict[str, dict[str, list[dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        rid = row.get("recording_id", "")
        sid = row.get("subject_id", "")
        path = row.get("path", "")
        keys = {
            "recording_id": _norm(rid),
            "subject_id": _norm(sid),
            "path": _norm(path),
            "path_stem": _path_stem(path),
            "path_tail2": _path_tail(path, 2),
            "path_tail3": _path_tail(path, 3),
        }
        for token in _tokens(rid, sid, path):
            index["token"][token].append(row)
        for kind, key in keys.items():
            if key:
                index[kind][key].append(row)
        if enable_hash:
            audio = _resolve_audio(root, path)
            if audio is not None:
                index["sha256"][_sha256(audio)].append(row)
    return index


def _row_id(row: dict[str, str]) -> str:
    return row.get("recording_id") or row.get("path") or row.get("subject_id") or "<unknown>"


def _find_matches(
    source_index: dict[str, dict[str, list[dict[str, str]]]],
    target_rows: list[dict[str, str]],
    target_root: Path,
    enable_hash: bool,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for target in target_rows:
        rid = target.get("recording_id", "")
        sid = target.get("subject_id", "")
        path = target.get("path", "")
        checks = {
            "recording_id": _norm(rid),
            "subject_id": _norm(sid),
            "path": _norm(path),
            "path_stem": _path_stem(path),
            "path_tail2": _path_tail(path, 2),
            "path_tail3": _path_tail(path, 3),
        }
        for kind, key in checks.items():
            if not key:
                continue
            for source in source_index.get(kind, {}).get(key, []):
                matches.append(
                    {
                        "match_type": kind,
                        "match_key": key,
                        "source_recording_id": _row_id(source),
                        "source_subject_id": source.get("subject_id", ""),
                        "source_path": source.get("path", ""),
                        "target_recording_id": _row_id(target),
                        "target_subject_id": target.get("subject_id", ""),
                        "target_path": target.get("path", ""),
                    }
                )
        for token in _tokens(rid, sid, path):
            for source in source_index.get("token", {}).get(token, []):
                matches.append(
                    {
                        "match_type": "token",
                        "match_key": token,
                        "source_recording_id": _row_id(source),
                        "source_subject_id": source.get("subject_id", ""),
                        "source_path": source.get("path", ""),
                        "target_recording_id": _row_id(target),
                        "target_subject_id": target.get("subject_id", ""),
                        "target_path": target.get("path", ""),
                    }
                )
        if enable_hash:
            audio = _resolve_audio(target_root, path)
            if audio is not None:
                digest = _sha256(audio)
                for source in source_index.get("sha256", {}).get(digest, []):
                    matches.append(
                        {
                            "match_type": "sha256",
                            "match_key": digest,
                            "source_recording_id": _row_id(source),
                            "source_subject_id": source.get("subject_id", ""),
                            "source_path": source.get("path", ""),
                            "target_recording_id": _row_id(target),
                            "target_subject_id": target.get("subject_id", ""),
                            "target_path": target.get("path", ""),
                        }
                    )
    deduped = []
    seen = set()
    for match in matches:
        key = tuple(match.items())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(match)
    return deduped


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "match_type",
        "match_key",
        "source_recording_id",
        "source_subject_id",
        "source_path",
        "target_recording_id",
        "target_subject_id",
        "target_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(matches: list[dict[str, Any]], source_rows: list[dict[str, str]], target_rows: list[dict[str, str]], enable_hash: bool) -> dict[str, Any]:
    return {
        "source_rows": len(source_rows),
        "target_rows": len(target_rows),
        "num_matches": len(matches),
        "match_types": dict(sorted({kind: sum(1 for row in matches if row["match_type"] == kind) for kind in {row["match_type"] for row in matches}}.items())),
        "target_rows_with_any_match": len({row["target_recording_id"] for row in matches}),
        "source_rows_with_any_match": len({row["source_recording_id"] for row in matches}),
        "hash_check_enabled": enable_hash,
        "verdict": "overlap_detected" if matches else "no_overlap_detected_by_configured_checks",
    }


def _write_report(path: Path, summary: dict[str, Any], matches: list[dict[str, Any]], source_manifest: Path, target_manifest: Path) -> None:
    lines = [
        "# Manifest Overlap Audit",
        "",
        f"- Source manifest: `{source_manifest}`",
        f"- Target manifest: `{target_manifest}`",
        f"- Verdict: `{summary['verdict']}`",
        f"- Source rows: `{summary['source_rows']}`",
        f"- Target rows: `{summary['target_rows']}`",
        f"- Matches: `{summary['num_matches']}`",
        f"- Match types: `{summary['match_types']}`",
        f"- Hash check enabled: `{summary['hash_check_enabled']}`",
        "",
        "## Interpretation",
        "",
    ]
    if matches:
        lines.extend(
            [
                "Potential overlap was detected. Do not describe this target as an independent external dataset until the matches are manually resolved or excluded.",
                "",
                "## First Matches",
                "",
                "| Type | Key | Source | Target |",
                "|---|---|---|---|",
            ]
        )
        for row in matches[:30]:
            lines.append(
                f"| {row['match_type']} | `{row['match_key']}` | `{row['source_recording_id']}` | `{row['target_recording_id']}` |"
            )
    else:
        lines.append(
            "No overlap was detected by the configured string/stem/token/hash checks. This is not proof of full independence, but it is the required first-pass audit before external evaluation."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--target-manifest", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--target-root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--enable-hash", action="store_true", help="Compute SHA-256 for existing audio files; slower but stronger.")
    args = parser.parse_args()

    source_rows = _read_csv(args.source_manifest)
    target_rows = _read_csv(args.target_manifest)
    source_index = _index(source_rows, args.source_root, args.enable_hash)
    matches = _find_matches(source_index, target_rows, args.target_root, args.enable_hash)
    summary = _summary(matches, source_rows, target_rows, args.enable_hash)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "overlap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_csv(args.out / "overlap_matches.csv", matches)
    _write_report(args.out / "OVERLAP_AUDIT.md", summary, matches, args.source_manifest, args.target_manifest)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
