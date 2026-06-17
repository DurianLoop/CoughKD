from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.config import RunConfig
from coughkd.torch_models import (
    CoughManifestDataset,
    DepthwiseStudent,
    _build_teacher,
    _forward_model,
    pad_collate,
)


CLASSES = ["covid_positive", "covid_recovered", "exposed", "healthy", "respiratory_illness"]


def _require_torch() -> Any:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    return torch, DataLoader, np


def _seed_everything(seed: int) -> None:
    torch, _, np = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _entropy(probs: list[float]) -> float:
    return -sum(prob * math.log(max(prob, 1e-12)) for prob in probs)


def _margin(probs: list[float]) -> float:
    ordered = sorted(probs, reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def _kl(p: list[float], q: list[float]) -> float:
    return sum(pi * math.log(max(pi, 1e-12) / max(qi, 1e-12)) for pi, qi in zip(p, q))


def _load_student(checkpoint: Path, device: Any) -> tuple[Any, dict[str, int], dict[int, str]]:
    torch, _, _ = _require_torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(key): int(value) for key, value in payload["label_to_index"].items()}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    model = DepthwiseStudent(num_classes=len(label_to_index))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, label_to_index, index_to_label


def _load_teacher(args: argparse.Namespace, num_classes: int, checkpoint: Path, device: Any) -> Any:
    torch, _, _ = _require_torch()
    teacher = _build_teacher(
        teacher_kind=args.teacher_kind,
        num_classes=num_classes,
        checkpoint=Path("__skip_pretrained__") if args.teacher_kind == "panns_cnn14_16k" else args.teacher_checkpoint,
        repo=args.teacher_repo,
    ).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "runs/stage1_panns_response_seed7")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/source_transfer_gap_seed7")
    parser.add_argument("--split", default="train")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--teacher-kind", default="panns_cnn14_16k", choices=["compact", "panns_cnn14_16k"])
    parser.add_argument("--teacher-checkpoint", type=Path, default=ROOT / "pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--teacher-repo", type=Path, default=ROOT / "external/teacher_repos/audioset_tagging_cnn_zip")
    parser.add_argument("--teacher-finetuned-checkpoint", type=Path)
    args = parser.parse_args()

    torch, DataLoader, _ = _require_torch()
    _seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)

    kd_ckpt = args.run_dir / "checkpoints/student_best.pt"
    ce_ckpt = args.run_dir / "checkpoints/ce_student_best.pt"
    teacher_ckpt = args.teacher_finetuned_checkpoint or (args.run_dir / "checkpoints/teacher_best.pt")
    kd_student, label_to_index, index_to_label = _load_student(kd_ckpt, device)
    ce_student, _, _ = _load_student(ce_ckpt, device)
    teacher = _load_teacher(args, len(label_to_index), teacher_ckpt, device)

    dataset = CoughManifestDataset(
        args.manifest,
        args.root,
        RunConfig(experiment_name="source_transfer_gap", seed=args.seed),
        label_to_index=label_to_index,
        splits={args.split},
        max_records=args.max_records,
        max_duration_sec=args.max_duration_sec,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate)

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            labels = batch["label"].to(device)
            teacher_probs = torch.softmax(_forward_model(teacher, batch, device)["logits"], dim=1).cpu().tolist()
            kd_probs = torch.softmax(kd_student(batch["features"].to(device))["logits"], dim=1).cpu().tolist()
            ce_probs = torch.softmax(ce_student(batch["features"].to(device))["logits"], dim=1).cpu().tolist()
            for rid, true_idx, tp, kp, cp in zip(batch["recording_id"], labels.cpu().tolist(), teacher_probs, kd_probs, ce_probs):
                teacher_pred = int(max(range(len(tp)), key=lambda idx: tp[idx]))
                kd_pred = int(max(range(len(kp)), key=lambda idx: kp[idx]))
                ce_pred = int(max(range(len(cp)), key=lambda idx: cp[idx]))
                row = {
                    "recording_id": rid,
                    "true_label": index_to_label[int(true_idx)],
                    "teacher_pred": index_to_label[teacher_pred],
                    "kd_pred": index_to_label[kd_pred],
                    "ce_pred": index_to_label[ce_pred],
                    "teacher_entropy": f"{_entropy(tp):.8f}",
                    "teacher_margin": f"{_margin(tp):.8f}",
                    "kd_kl_from_teacher": f"{_kl(tp, kp):.8f}",
                    "ce_kl_from_teacher": f"{_kl(tp, cp):.8f}",
                    "kd_agree_teacher": str(int(kd_pred == teacher_pred)),
                    "ce_agree_teacher": str(int(ce_pred == teacher_pred)),
                    "teacher_correct": str(int(teacher_pred == int(true_idx))),
                }
                row.update({f"teacher_prob_{label}": f"{tp[idx]:.8f}" for idx, label in index_to_label.items()})
                row.update({f"kd_prob_{label}": f"{kp[idx]:.8f}" for idx, label in index_to_label.items()})
                row.update({f"ce_prob_{label}": f"{cp[idx]:.8f}" for idx, label in index_to_label.items()})
                rows.append(row)
            if batch_idx == 1 or batch_idx % 40 == 0:
                print(f"[cache] batch {batch_idx}/{len(loader)}", flush=True)

    fieldnames = list(rows[0]) if rows else []
    csv_path = args.out / "source_transfer_gap.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    def _mean(key: str) -> float:
        values = [float(row[key]) for row in rows]
        return sum(values) / len(values) if values else 0.0

    summary = {
        "num_records": len(rows),
        "split": args.split,
        "run_dir": str(args.run_dir),
        "teacher_entropy": _mean("teacher_entropy"),
        "teacher_margin": _mean("teacher_margin"),
        "kd_kl_from_teacher": _mean("kd_kl_from_teacher"),
        "ce_kl_from_teacher": _mean("ce_kl_from_teacher"),
        "kd_agree_teacher": _mean("kd_agree_teacher"),
        "ce_agree_teacher": _mean("ce_agree_teacher"),
        "teacher_correct": _mean("teacher_correct"),
        "csv": str(csv_path),
    }
    (args.out / "source_transfer_gap_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
