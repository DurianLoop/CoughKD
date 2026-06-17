"""Record the current external approval state for UKCOVID audio download."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "ukcovid_open_metadata_audit"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--status",
        choices=["not_requested", "requested_timeout", "approved", "denied"],
        default="requested_timeout",
    )
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--checked", default="2026-06-09")
    parser.add_argument(
        "--note",
        default=(
            "Approval-time attempts to run scripts\\run_ukcovid_audio_download_verify_windows.cmd "
            "timed out before approval. No UKCOVID audio download was started."
        ),
    )
    args = parser.parse_args()

    payload = {
        "verdict": "DOWNLOAD_APPROVAL_NOT_GRANTED" if args.status != "approved" else "DOWNLOAD_APPROVED",
        "status": args.status,
        "attempts": args.attempts,
        "checked": args.checked,
        "large_external_data": True,
        "download_driver": "scripts\\run_ukcovid_audio_download_verify_windows.cmd",
        "note": args.note,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ukcovid_download_approval_status.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    lines = [
        "# UKCOVID Download Approval Status",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Status: `{payload['status']}`",
        f"- Attempts: `{payload['attempts']}`",
        f"- Checked: `{payload['checked']}`",
        f"- Large external data: `{payload['large_external_data']}`",
        f"- Download driver: `{payload['download_driver']}`",
        "",
        "## Note",
        "",
        payload["note"],
        "",
    ]
    (OUT / "UKCOVID_DOWNLOAD_APPROVAL_STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(OUT / "UKCOVID_DOWNLOAD_APPROVAL_STATUS.md")


if __name__ == "__main__":
    main()
