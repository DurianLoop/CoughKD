"""Check local datasets and teacher-model assets before COUGHKD experiments.

This script is intentionally dependency-light so it can run before the ML
environment is fully ready.
"""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_FOR_STAGE1 = [
    ("coswara_root", "directory", "Coswara extracted dataset root"),
    ("coswara_metadata", "file", "Coswara combined_data.csv metadata"),
]

REQUIRED_FOR_PANNS = [
    ("panns_checkpoint", "file", "PANNs CNN14 16 kHz checkpoint"),
    ("panns_repo", "directory", "PANNs audioset_tagging_cnn repository"),
]

OPTIONAL_TEACHERS = [
    ("beats_checkpoint", "file", "BEATs checkpoint, for future BEATs wrapper"),
    ("ast_checkpoint", "file", "AST AudioSet model.safetensors, for future AST wrapper"),
    ("passt_root", "directory", "PaSST checkpoint/package root, optional teacher"),
]


def _exists(path: Path, kind: str) -> bool:
    if kind == "file":
        return path.is_file()
    if kind == "directory":
        return path.is_dir()
    raise ValueError(f"unsupported kind: {kind}")


def _print_group(title: str, specs: list[tuple[str, str, str]], values: dict[str, Path]) -> bool:
    print(f"\n[{title}]")
    ok = True
    for key, kind, description in specs:
        path = values[key]
        present = _exists(path, kind)
        mark = "OK" if present else "MISSING"
        print(f"{mark:7} {key:18} {path}  # {description}")
        if not present:
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--coswara-root", default=r"D:\CoughKD\datasets\coswara_zip")
    parser.add_argument("--coswara-metadata", default=r"D:\CoughKD\datasets\coswara_zip\combined_data.csv")
    parser.add_argument("--panns-checkpoint", default=r"D:\CoughKD\AAAI\pretrained\teachers\panns\Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--panns-repo", default=r"D:\CoughKD\AAAI\external\teacher_repos\audioset_tagging_cnn_zip")
    parser.add_argument("--beats-checkpoint", default=r"D:\CoughKD\AAAI\pretrained\teachers\beats\BEATs_iter3_plus_AS2M.pt")
    parser.add_argument("--ast-checkpoint", default=r"D:\CoughKD\AAAI\pretrained\teachers\ast\ast-finetuned-audioset-10-10-0.4593\model.safetensors")
    parser.add_argument("--passt-root", default=r"D:\CoughKD\AAAI\pretrained\teachers\passt")
    args = parser.parse_args()

    values = {
        "coswara_root": Path(args.coswara_root),
        "coswara_metadata": Path(args.coswara_metadata),
        "panns_checkpoint": Path(args.panns_checkpoint),
        "panns_repo": Path(args.panns_repo),
        "beats_checkpoint": Path(args.beats_checkpoint),
        "ast_checkpoint": Path(args.ast_checkpoint),
        "passt_root": Path(args.passt_root),
    }

    stage1_ok = _print_group("Required for Stage 1 Coswara reproduction", REQUIRED_FOR_STAGE1, values)
    panns_ok = _print_group("Required for PANNs teacher baseline", REQUIRED_FOR_PANNS, values)
    _print_group("Optional future teacher assets", OPTIONAL_TEACHERS, values)

    print("\nNext steps:")
    if not stage1_ok:
        print("- Download/extract Coswara before running Stage 1.")
    else:
        print("- Stage 1 compact-teacher and DepthwiseStudent baselines can start.")
    if not panns_ok:
        print("- Download PANNs checkpoint and source repo before running panns_cnn14_16k.")
    else:
        print("- PANNs CNN14 16 kHz teacher baseline can start.")
    print("- AST/BEATs/PaSST assets are optional until their wrappers are implemented.")
    return 0 if stage1_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
