from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _exists(path: Path) -> bool:
    return path.exists() and (path.is_dir() or path.stat().st_size > 0)


def _asset_rows() -> list[dict[str, Any]]:
    return [
        {
            "name": "hear_pytorch",
            "display": "Google HeAR PyTorch",
            "source": "https://huggingface.co/google/hear-pytorch",
            "expected_path": ROOT / "pretrained/teachers/hear_pytorch",
            "required_modules": ["huggingface_hub", "torch", "soundfile"],
            "approval": "Requires accepting Google Health AI Developer Foundations terms on Hugging Face and downloading gated weights.",
            "integration": "Preferred next gate: PyTorch-native health acoustic embeddings should be easiest to plug into the current PyTorch audit.",
            "priority": 1,
        },
        {
            "name": "hear_tf",
            "display": "Google HeAR TensorFlow SavedModel",
            "source": "https://huggingface.co/google/hear",
            "expected_path": ROOT / "pretrained/teachers/hear",
            "required_modules": ["huggingface_hub", "tensorflow", "soundfile"],
            "approval": "Requires accepting Google Health AI Developer Foundations terms on Hugging Face and downloading gated weights.",
            "integration": "Lower priority than hear_pytorch because TensorFlow is not installed in the current CoughKD environment.",
            "priority": 4,
        },
        {
            "name": "opera",
            "display": "OPERA respiratory acoustic foundation models",
            "source": "https://github.com/evelyn0414/OPERA",
            "expected_path": ROOT / "external/teacher_repos/OPERA",
            "required_modules": ["torch", "torchaudio", "librosa", "numpy", "timm", "omegaconf", "hydra", "soundfile"],
            "approval": "Requires cloning the OPERA repo and allowing model downloads from the upstream release locations.",
            "integration": "Scientifically aligned with respiratory audio, but likely heavier because OPERA has its own repo, feature pipeline, and checkpoint conventions.",
            "priority": 2,
        },
        {
            "name": "beats",
            "display": "BEATs general audio SSL",
            "source": "https://arxiv.org/abs/2212.09058",
            "expected_path": ROOT / "pretrained/teachers/beats/BEATs_iter3_plus_AS2M.pt",
            "required_modules": ["torch", "soundfile"],
            "approval": "Requires downloading BEATs checkpoint and sparse-cloning the BEATs source under microsoft/unilm.",
            "integration": "Strong general audio baseline, but less health-specific than HeAR/OPERA.",
            "priority": 3,
        },
        {
            "name": "ast",
            "display": "AST AudioSet",
            "source": "https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593",
            "expected_path": ROOT / "pretrained/teachers/ast/ast-finetuned-audioset-10-10-0.4593/model.safetensors",
            "required_modules": ["transformers", "torch", "soundfile"],
            "approval": "Requires downloading the public AST AudioSet model files.",
            "integration": "Useful general audio transformer control, but previous docs rank health/respiratory-specific embeddings higher for Tos env1.",
            "priority": 5,
        },
    ]


def _summarize_asset(row: dict[str, Any]) -> dict[str, Any]:
    expected_path = Path(row["expected_path"])
    modules = {name: _module_available(name) for name in row["required_modules"]}
    return {
        **{key: value for key, value in row.items() if key != "expected_path"},
        "expected_path": str(expected_path),
        "asset_present": _exists(expected_path),
        "required_modules": modules,
        "environment_ready": all(modules.values()),
        "ready_without_download": _exists(expected_path) and all(modules.values()),
    }


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    rows = sorted(summary["assets"], key=lambda item: int(item["priority"]))
    lines = [
        "# Foundation Asset Gate Plan",
        "",
        "This is a no-download readiness plan for the next Tos env-1 frozen-embedding gate.",
        "",
        "## Priority",
        "",
        "| Priority | Asset | Present | Env ready | Approval needed | Expected path |",
        "|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['priority']} | {row['display']} | {row['asset_present']} | {row['environment_ready']} | "
            f"{row['approval']} | `{row['expected_path']}` |"
        )
    lines.extend(
        [
            "",
            "## Recommended Next Gate",
            "",
            summary["recommendation"],
            "",
            "## Sources",
            "",
        ]
    )
    for row in rows:
        lines.append(f"- {row['display']}: `{row['source']}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=ROOT / "runs/foundation_asset_gate_plan")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    assets = [_summarize_asset(row) for row in _asset_rows()]
    recommendation = (
        "Request user approval for `hear_pytorch` first. It is health-acoustic, PyTorch-native, and the current environment "
        "already has `huggingface_hub`, `torch`, and `soundfile`. If HeAR cannot be accessed because of gated terms, move to OPERA. "
        "Do not run another KD training experiment until one of these frozen embeddings clears the Tos env-1 AUROC >= 0.60 gate."
    )
    summary = {
        "assets": assets,
        "gate": {
            "target": "Tos env 1 COVID-vs-healthy oracle AUROC",
            "minimum_env1_auroc": 0.60,
            "minimum_all_tos_auroc_gt": 0.564254,
        },
        "recommendation": recommendation,
    }
    (args.out / "foundation_asset_gate_plan.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "FOUNDATION_ASSET_GATE_PLAN.md", summary)
    print(str(args.out / "FOUNDATION_ASSET_GATE_PLAN.md"), flush=True)


if __name__ == "__main__":
    main()
