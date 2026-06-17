from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"missing": str(path)}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Next External-State Actions",
        "",
        "This summarizes actions that can materially move the ICASSP 2027 positive-claim search.",
        "",
        "## Current Blockers",
        "",
        "| Route | Current state | Why it matters | Approval/action needed |",
        "|---|---|---|---|",
    ]
    for item in summary["actions"]:
        lines.append(f"| {item['route']} | {item['state']} | {item['why']} | {item['action']} |")
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            summary["recommendation"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/next_external_state_actions")
    args = parser.parse_args()
    uk = _load_json(ROOT / "runs/ukcovid_open_metadata_audit/ukcovid_audio_ready_summary.json")
    hear = _load_json(ROOT / "runs/hear_pytorch_embedding_upper_bound_seed7/hear_pytorch_embedding_upper_bound_audit.json")
    opera = _load_json(ROOT / "runs/opera_embedding_upper_bound_seed7/opera_embedding_upper_bound_audit.json")
    gain = _load_json(ROOT / "runs/existing_external_gain_matrix/bootstrap_virufyseg_clip_kd_vs_source_seed11.json")
    actions = [
        {
            "route": "UKCOVID large external evaluation",
            "state": (
                f"{uk.get('verdict', 'unknown')}; "
                f"{uk.get('manifest_audio_status', {}).get('unique_audio_found', 0)}/"
                f"{uk.get('manifest_audio_status', {}).get('unique_audio_filenames', 0)} audio files found"
            ),
            "why": "Largest available external target: 22,242 test rows and 11,121 subjects; can validate or reject 3-5 point gains robustly.",
            "action": "Approve/download and extract the 49.95 GiB UKCOVID audio archive, then run scripts/run_ukcovid_after_audio_windows.cmd.",
        },
        {
            "route": "HeAR PyTorch frozen embedding gate",
            "state": (
                "environment_ready="
                + str(hear.get("preflight", {}).get("environment_ready"))
                + "; asset_present="
                + str(hear.get("preflight", {}).get("asset_present"))
            ),
            "why": "Best health-acoustic foundation route for Tos env1; gate can decide whether a positive KD method is worth designing.",
            "action": "Accept google/hear-pytorch Hugging Face terms, download weights, then run the HeAR gate.",
        },
        {
            "route": "OPERA respiratory foundation gate",
            "state": (
                "environment_ready="
                + str(opera.get("preflight", {}).get("environment_ready"))
                + "; asset_ready="
                + str(opera.get("preflight", {}).get("asset_ready"))
            ),
            "why": "Respiratory-specific fallback if HeAR is inaccessible.",
            "action": "Clone OPERA, install timm/omegaconf/hydra-core, then approve --run-gate because extractor may download checkpoints.",
        },
        {
            "route": "Existing Virufyseg positive signal",
            "state": (
                "seed11 clip KD delta="
                + str(round(float(gain.get("point_delta", 0.0)), 6))
                + "; CI="
                + str(gain.get("bootstrap", {}).get("ci95_low"))
                + ".."
                + str(gain.get("bootstrap", {}).get("ci95_high"))
            ),
            "why": "Shows possible 3-5 point gains, but only on 121 clips / 16 subjects and unstable across seeds.",
            "action": "Use as auxiliary evidence only; do not make it the main method claim.",
        },
    ]
    summary = {
        "actions": actions,
        "recommendation": (
            "For a positive 3-5 point ICASSP claim, the highest-value next external-state action is UKCOVID audio evaluation. "
            "For a new method route, the highest-value action is HeAR PyTorch gated-weight access. Without one of these, current evidence supports "
            "a guarded-transfer/failure-slice paper rather than a strong positive method claim."
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "next_external_state_actions.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "NEXT_EXTERNAL_STATE_ACTIONS.md", summary)
    print(str(args.out / "NEXT_EXTERNAL_STATE_ACTIONS.md"), flush=True)


if __name__ == "__main__":
    main()
