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


def _metric(path: Path, key: str = "macro_ovr_auroc") -> float | None:
    data = _load_json(path)
    value = data.get(key)
    return float(value) if value is not None else None


def _fmt(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value:.6f}"


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Post-5.102 Decision Gate",
        "",
        "This no-download gate decides whether local positive-method hunting should continue after the symptom-risk KD hard gate.",
        "",
        "## Latest Hard-Gate Evidence",
        "",
        "| target | source-only | symptom-risk KD | delta | hard-gate result |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for row in summary["latest_external_gate"]:
        lines.append(
            f"| {row['target']} | {_fmt(row['source_only'])} | {_fmt(row['symptom_risk_kd'])} | "
            f"{_fmt(row['delta_vs_source'])} | {row['result']} |"
        )
    lines.extend(
        [
            "",
            "## External-State Options",
            "",
            "| priority | route | state | why | required approval | next hard gate |",
            "| ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for option in summary["options"]:
        lines.append(
            f"| {option['priority']} | {option['route']} | {option['state']} | {option['why']} | "
            f"{option['approval']} | {option['next_gate']} |"
        )
    lines.extend(
        [
            "",
            "## Recommendation",
            "",
            summary["recommendation"],
            "",
            "## Stop Rule",
            "",
            summary["stop_rule"],
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/post_5102_decision_gate")
    args = parser.parse_args()

    cough_source = _metric(ROOT / "runs/external_coughvid_test_stage3c_source_only_seed7/metrics.json")
    cough_symptom = _metric(ROOT / "runs/external_coughvid_test_symptom_risk_equalized_kd_seed7/metrics.json")
    tos_source = _metric(ROOT / "runs/external_toscovid2021_test_source_only_seed7/metrics.json")
    tos_symptom = _metric(ROOT / "runs/external_toscovid2021_test_symptom_risk_equalized_kd_seed7/metrics.json")

    latest = [
        {
            "target": "COUGHVID",
            "source_only": cough_source,
            "symptom_risk_kd": cough_symptom,
            "delta_vs_source": _delta(cough_symptom, cough_source),
            "result": "positive but below +2pt continuation gate",
        },
        {
            "target": "TosCOVID",
            "source_only": tos_source,
            "symptom_risk_kd": tos_symptom,
            "delta_vs_source": _delta(tos_symptom, tos_source),
            "result": "fail: below source-only",
        },
    ]

    uk = _load_json(ROOT / "runs/ukcovid_open_metadata_audit/ukcovid_audio_ready_summary.json")
    foundation = _load_json(ROOT / "runs/foundation_asset_gate_plan/foundation_asset_gate_plan.json")
    hear = next((item for item in foundation.get("assets", []) if item.get("name") == "hear_pytorch"), {})
    opera = next((item for item in foundation.get("assets", []) if item.get("name") == "opera"), {})
    ast = next((item for item in foundation.get("assets", []) if item.get("name") == "ast"), {})

    options = [
        {
            "priority": 1,
            "route": "UKCOVID audio external validation",
            "state": (
                f"{uk.get('verdict', 'unknown')}; "
                f"{uk.get('manifest_audio_status', {}).get('unique_audio_found', 0)}/"
                f"{uk.get('manifest_audio_status', {}).get('unique_audio_filenames', 0)} files local"
            ),
            "why": "Best way to validate whether the COUGHVID symptom-slice signal generalizes to a large symptom-rich external target.",
            "approval": "Approve the ~53.6 GB UKCOVID audio archive download/extraction.",
            "next_gate": "Run source-only, Candidate B/F, symptom-risk KD, and metadata/random-slice controls on UKCOVID.",
        },
        {
            "priority": 2,
            "route": "HeAR PyTorch frozen embedding gate",
            "state": (
                f"asset_present={hear.get('asset_present')}; "
                f"environment_ready={hear.get('environment_ready')}"
            ),
            "why": "Best method-route test for TosCOVID: health-acoustic embeddings may contain the missing target signal.",
            "approval": "Accept google/hear-pytorch terms and download/place weights under pretrained/teachers/hear_pytorch.",
            "next_gate": "Frozen HeAR Tos env1 AUROC >= 0.60 and all-Tos AUROC > 0.564254 before any KD training.",
        },
        {
            "priority": 3,
            "route": "OPERA respiratory foundation gate",
            "state": (
                f"asset_present={opera.get('asset_present')}; "
                f"environment_ready={opera.get('environment_ready')}"
            ),
            "why": "Respiratory-specific fallback if HeAR is inaccessible, but heavier integration.",
            "approval": "Clone OPERA and approve dependency/checkpoint downloads.",
            "next_gate": "Frozen OPERA Tos env1 AUROC >= 0.60 before any KD training.",
        },
        {
            "priority": 4,
            "route": "AST AudioSet public control",
            "state": (
                f"asset_present={ast.get('asset_present')}; "
                f"environment_ready={ast.get('environment_ready')}"
            ),
            "why": "Public general-audio control; less health-specific than HeAR/OPERA.",
            "approval": "Download AST AudioSet model files.",
            "next_gate": "Use only as a control after HeAR/OPERA or if gated assets are impossible.",
        },
    ]

    recommendation = (
        "Do not launch another local scalar-loss KD variant. The next approval should be either UKCOVID audio "
        "for external validation, or HeAR PyTorch for a foundation-embedding gate. If the goal is a 3-5 point "
        "positive ICASSP claim, UKCOVID is the stronger evidence route; if the goal is a new method mechanism, "
        "HeAR is the stronger method route."
    )
    stop_rule = (
        "If UKCOVID or HeAR/OPERA cannot be approved, stop positive-method hunting locally and pivot the paper "
        "to a guarded-transfer/failure-slice audit, because COUGHVID-positive/TosCOVID-negative behavior has now "
        "repeated across post-hoc gates, artifact-risk KD, and symptom-risk KD."
    )
    summary = {
        "latest_external_gate": latest,
        "options": options,
        "recommendation": recommendation,
        "stop_rule": stop_rule,
        "claim_status": "not_claimable",
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "post_5102_decision_gate.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = args.out / "POST_5102_DECISION_GATE.md"
    _write_report(report, summary)
    print(report)


if __name__ == "__main__":
    main()
