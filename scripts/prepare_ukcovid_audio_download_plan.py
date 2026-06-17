"""Prepare a resumable download plan for the UK COVID-19 audio archive."""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = Path(r"D:\CoughKD\external\ukcovid_open")
ZENODO_JSON = BASE / "zenodo_record_10043978.json"
OUT = ROOT / "runs" / "ukcovid_open_metadata_audit"


def _archive_sort_key(name: str) -> tuple[int, int]:
    match = re.fullmatch(r"covid_data\.z(\d+)", name)
    if match:
        return (0, int(match.group(1)))
    if name == "covid_data.zip":
        return (1, 999)
    return (2, 999)


def main() -> None:
    data = json.loads(ZENODO_JSON.read_text(encoding="utf-8"))
    files = data.get("files", [])
    archive = []
    metadata = []
    for item in files:
        name = item.get("key", "")
        url = item.get("links", {}).get("self") or item.get("links", {}).get("download")
        row = {
            "name": name,
            "size_bytes": int(item.get("size") or 0),
            "size_gb": f"{int(item.get('size') or 0) / (1024**3):.3f}",
            "checksum": item.get("checksum", ""),
            "url": url,
        }
        if name.startswith("covid_data."):
            archive.append(row)
        else:
            metadata.append(row)
    archive.sort(key=lambda row: _archive_sort_key(row["name"]))

    OUT.mkdir(parents=True, exist_ok=True)
    csv_path = OUT / "ukcovid_audio_archive_files.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["name", "size_bytes", "size_gb", "checksum", "url"])
        writer.writeheader()
        writer.writerows(archive)

    cmd_path = OUT / "download_ukcovid_audio_archive_windows.cmd"
    lines = [
        "@echo off",
        "setlocal",
        "set OUT_DIR=%~1",
        "if \"%OUT_DIR%\"==\"\" set OUT_DIR=D:\\CoughKD\\external\\ukcovid_open",
        "if not exist \"%OUT_DIR%\" mkdir \"%OUT_DIR%\"",
        "set CURL=curl.exe",
        "set RETRY=--retry 10 --retry-all-errors --retry-delay 10 --connect-timeout 30 --max-time 0",
        "set TLS=--ssl-no-revoke --http1.1",
        "echo Downloading UK COVID-19 audio split archive to %OUT_DIR%",
        "echo This is about 53.6GB. Use Ctrl+C to stop; rerun to resume missing files.",
        "",
    ]
    for row in archive:
        name = row["name"]
        url = row["url"]
        size = row["size_bytes"]
        lines.extend(
            [
                f"if exist \"%OUT_DIR%\\{name}\" (",
                f"  for %%A in (\"%OUT_DIR%\\{name}\") do if %%~zA GEQ {size} (echo [skip] {name} & goto done_{name.replace('.', '_')})",
                ")",
                f"%CURL% -L --fail %TLS% %RETRY% --continue-at - -o \"%OUT_DIR%\\{name}\" \"{url}\"",
                "if errorlevel 1 goto fail",
                f":done_{name.replace('.', '_')}",
                "",
            ]
        )
    lines.extend(
        [
            "echo.",
            "echo Download complete. Extract with 7-Zip or another split-zip capable tool, then rebuild manifests without --allow-missing-audio.",
            "exit /b 0",
            "",
            ":fail",
            "echo Download failed or interrupted. Rerun this script to continue missing/incomplete files.",
            "exit /b 1",
        ]
    )
    cmd_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    total = sum(row["size_bytes"] for row in archive)
    report = [
        "# UK COVID-19 Audio Archive Download Plan",
        "",
        f"- Archive files: `{len(archive)}`",
        f"- Total size: `{total / (1024**3):.2f} GB`",
        f"- File list CSV: `{csv_path.relative_to(ROOT)}`",
        f"- Download script: `{cmd_path.relative_to(ROOT)}`",
        "",
        "## Usage",
        "",
        "```cmd",
        "cd /d D:\\CoughKD\\AAAI",
        "set HTTPS_PROXY=http://127.0.0.1:7897",
        "set HTTP_PROXY=http://127.0.0.1:7897",
        "runs\\ukcovid_open_metadata_audit\\download_ukcovid_audio_archive_windows.cmd D:\\CoughKD\\external\\ukcovid_open",
        "```",
        "",
        "After extraction, rebuild `manifests\\ukcovid_open_test_external.csv` without `--allow-missing-audio` and run `scripts\\onboard_external_target.py --target-tag ukcovid_open`.",
        "",
        "## Archive Files",
        "",
        "| File | Size GB | Checksum |",
        "|---|---:|---|",
    ]
    for row in archive:
        report.append(f"| {row['name']} | {row['size_gb']} | {row['checksum']} |")
    (OUT / "UKCOVID_AUDIO_DOWNLOAD_PLAN.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print({"archive_files": len(archive), "total_gb": round(total / (1024**3), 2), "cmd": str(cmd_path)})


if __name__ == "__main__":
    main()
