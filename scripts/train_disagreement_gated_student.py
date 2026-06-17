from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
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
    KdLossConfig,
    _build_teacher,
    _evaluate_model,
    _forward_model,
    _save_checkpoint,
    _selection_metric,
    count_parameters,
    pad_collate,
    torch_kd_loss,
)


def _require_torch() -> Any:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    return torch, F, DataLoader, np


def _seed_everything(seed: int) -> None:
    torch, _, _, np = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read_manifest_labels(path: Path) -> dict[str, int]:
    import csv

    labels = set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label", "")
            if label:
                labels.add(label)
    return {label: idx for idx, label in enumerate(sorted(labels))}


def _class_weights(dataset: CoughManifestDataset, num_classes: int, device: Any) -> Any:
    torch, _, _, _ = _require_torch()
    counts = [0 for _ in range(num_classes)]
    for row in dataset.rows:
        counts[dataset.label_to_index[row["label"]]] += 1
    total = sum(counts)
    weights = [total / max(1, num_classes * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _load_teacher(args: argparse.Namespace, num_classes: int, device: Any) -> Any:
    torch, _, _, _ = _require_torch()
    print("[teacher] build architecture", flush=True)
    teacher = _build_teacher(
        teacher_kind=args.teacher_kind,
        num_classes=num_classes,
        checkpoint=Path("__skip_pretrained__") if args.teacher_kind == "panns_cnn14_16k" else args.teacher_checkpoint,
        repo=args.teacher_repo,
    ).to(device)
    print("[teacher] load fine-tuned checkpoint", flush=True)
    payload = torch.load(args.init_teacher_checkpoint, map_location=device, weights_only=False)
    print("[teacher] apply state dict", flush=True)
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    print("[teacher] ready", flush=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _disagreement_weights(labels: Any, teacher_logits: Any, student_logits: Any, args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    torch, F, _, _ = _require_torch()
    teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
    student_probs = F.softmax(student_logits.detach(), dim=1)
    teacher_conf, teacher_pred = teacher_probs.max(dim=1)
    student_pred = student_probs.argmax(dim=1)
    label_agree = teacher_pred.eq(labels)
    student_agree = teacher_pred.eq(student_pred)
    entropy = -(teacher_probs * torch.log(teacher_probs.clamp(min=1e-12))).sum(dim=1)
    entropy_norm = entropy / max(1e-8, math.log(float(teacher_probs.shape[1])))
    reliability = (1.0 - entropy_norm).clamp(0.0, 1.0) * teacher_conf.clamp(0.0, 1.0)
    weights = args.min_kd_weight + (args.max_kd_weight - args.min_kd_weight) * reliability
    high_conf_mismatch = teacher_conf.ge(args.high_confidence_threshold) & (~label_agree)
    weights = torch.where(high_conf_mismatch, weights * args.high_conf_mismatch_scale, weights)
    weights = torch.where(label_agree, weights * args.label_agree_boost, weights)
    weights = torch.where(student_agree, weights * args.student_agree_boost, weights)
    weights = weights.clamp(args.min_kd_weight, args.max_kd_weight)
    return weights.detach(), {
        "mean_kd_weight": float(weights.mean().detach().cpu()),
        "teacher_confidence": float(teacher_conf.mean().detach().cpu()),
        "teacher_label_agreement": float(label_agree.float().mean().detach().cpu()),
        "teacher_student_agreement": float(student_agree.float().mean().detach().cpu()),
        "high_conf_mismatch_ratio": float(high_conf_mismatch.float().mean().detach().cpu()),
    }


def _train_epoch(
    student: Any,
    teacher: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    class_weights: Any,
    kd_config: KdLossConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    student.train()
    teacher.eval()
    totals = {
        "total": 0.0,
        "ce": 0.0,
        "response_kd": 0.0,
        "feature_kd": 0.0,
        "embedding_kd": 0.0,
        "relation_kd": 0.0,
        "mean_kd_weight": 0.0,
        "teacher_confidence": 0.0,
        "teacher_label_agreement": 0.0,
        "teacher_student_agreement": 0.0,
        "high_conf_mismatch_ratio": 0.0,
    }
    total_items = 0
    for step, batch in enumerate(loader, start=1):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with _require_torch()[0].no_grad():
            teacher_out = _forward_model(teacher, batch, device)
        student_out = student(features)
        kd_weights, weight_stats = _disagreement_weights(labels, teacher_out["logits"], student_out["logits"], args)
        losses = torch_kd_loss(
            labels,
            teacher_out,
            student_out,
            config=kd_config,
            class_weights=class_weights,
            kd_weights=kd_weights,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_items += batch_size
        for key in ("total", "ce", "response_kd", "feature_kd", "embedding_kd", "relation_kd"):
            totals[key] += float(losses[key].detach().cpu()) * batch_size
        for key, value in weight_stats.items():
            totals[key] += value * batch_size
        if step == 1 or step % args.log_every == 0:
            print(
                f"[train] step {step}/{len(loader)} total={float(losses['total'].detach().cpu()):.4f} "
                f"w={weight_stats['mean_kd_weight']:.3f} agree={weight_stats['teacher_label_agreement']:.3f}",
                flush=True,
            )
    return {key: value / max(1, total_items) for key, value in totals.items()}


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    test = summary["source_test"]
    best = summary["best_val"]
    config = summary["config"]
    lines = [
        "# Disagreement-Gated KD",
        "",
        "## Config",
        "",
        f"- Source manifest: `{config['source_manifest']}`",
        f"- Teacher checkpoint: `{config['init_teacher_checkpoint']}`",
        f"- Student params: `{config['student_params']}`",
        f"- Teacher params: `{config['teacher_params']}`",
        f"- Seed: `{config['seed']}`",
        "",
        "## Best Validation",
        "",
        "```json",
        json.dumps(best, indent=2),
        "```",
        "",
        "## Source Test",
        "",
        "```json",
        json.dumps(test, indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/candidate_b_disagreement_gated_seed7")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-records-per-split", type=int)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--teacher-kind", default="panns_cnn14_16k", choices=["compact", "panns_cnn14_16k"])
    parser.add_argument("--teacher-checkpoint", type=Path, default=ROOT / "pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--teacher-repo", type=Path, default=ROOT / "external/teacher_repos/audioset_tagging_cnn_zip")
    parser.add_argument("--init-teacher-checkpoint", type=Path, default=ROOT / "runs/stage1_panns_response_seed7/checkpoints/teacher_best.pt")
    parser.add_argument("--kd-temperature", type=float, default=2.0)
    parser.add_argument("--kd-response-weight", type=float, default=0.7)
    parser.add_argument("--kd-feature-weight", type=float, default=0.1)
    parser.add_argument("--kd-embedding-weight", type=float, default=0.0)
    parser.add_argument("--kd-relation-weight", type=float, default=0.0)
    parser.add_argument("--min-kd-weight", type=float, default=0.05)
    parser.add_argument("--max-kd-weight", type=float, default=1.0)
    parser.add_argument("--high-confidence-threshold", type=float, default=0.45)
    parser.add_argument("--high-conf-mismatch-scale", type=float, default=0.25)
    parser.add_argument("--label-agree-boost", type=float, default=1.10)
    parser.add_argument("--student-agree-boost", type=float, default=1.05)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    torch, _, DataLoader, _ = _require_torch()
    _seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(parents=True, exist_ok=True)

    label_to_index = _read_manifest_labels(args.source_manifest)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    run_config = RunConfig(experiment_name="disagreement_gated_kd", seed=args.seed)
    datasets = {
        split: CoughManifestDataset(
            args.source_manifest,
            args.root,
            run_config,
            label_to_index=label_to_index,
            splits={split},
            max_records=args.max_records_per_split,
            max_duration_sec=args.max_duration_sec,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=pad_collate,
            num_workers=0,
            pin_memory=bool(str(device).startswith("cuda")),
        )
        for split, ds in datasets.items()
    }
    num_classes = len(label_to_index)
    teacher = _load_teacher(args, num_classes, device)
    student = DepthwiseStudent(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = _class_weights(datasets["train"], num_classes, device)
    kd_config = KdLossConfig(
        temperature=args.kd_temperature,
        response_weight=args.kd_response_weight,
        feature_weight=args.kd_feature_weight,
        embedding_weight=args.kd_embedding_weight,
        relation_weight=args.kd_relation_weight,
    )
    config = {
        "source_manifest": str(args.source_manifest),
        "root": str(args.root),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "teacher_kind": args.teacher_kind,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_repo": str(args.teacher_repo),
        "init_teacher_checkpoint": str(args.init_teacher_checkpoint),
        "kd": {
            "temperature": args.kd_temperature,
            "response_weight": args.kd_response_weight,
            "feature_weight": args.kd_feature_weight,
            "embedding_weight": args.kd_embedding_weight,
            "relation_weight": args.kd_relation_weight,
        },
        "gating": {
            "min_kd_weight": args.min_kd_weight,
            "max_kd_weight": args.max_kd_weight,
            "high_confidence_threshold": args.high_confidence_threshold,
            "high_conf_mismatch_scale": args.high_conf_mismatch_scale,
            "label_agree_boost": args.label_agree_boost,
            "student_agree_boost": args.student_agree_boost,
        },
        "split_sizes": {split: len(ds) for split, ds in datasets.items()},
        "label_to_index": label_to_index,
        "teacher_params": count_parameters(teacher),
        "student_params": count_parameters(student),
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_metric = -1.0
    best_path = args.out / "checkpoints" / "student_best.pt"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = _train_epoch(student, teacher, loaders["train"], optimizer, device, class_weights, kd_config, args)
        val = _evaluate_model(student, loaders["val"], device, index_to_label)["metrics"]
        entry = {"epoch": epoch, "train": train, "val": val, "seconds": time.time() - started}
        history.append(entry)
        metric = _selection_metric(val)
        if metric > best_metric:
            best_metric = metric
            _save_checkpoint(best_path, student, label_to_index, config, entry)
        print(json.dumps({"stage": "epoch", **entry}), flush=True)

    student.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
    source_test = _evaluate_model(student, loaders["test"], device, index_to_label)["metrics"]
    final_path = args.out / "checkpoints" / "student_final.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "label_to_index": label_to_index,
            "config": config,
            "test_metrics": source_test,
            "history": history,
        },
        final_path,
    )
    summary = {
        "config": config,
        "history": history,
        "best_val": max(history, key=lambda item: _selection_metric(item["val"])),
        "source_test": source_test,
        "artifacts": {
            "student_best": str(best_path),
            "student_final": str(final_path),
        },
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(args.out / "RESULTS.md", summary)
    print(json.dumps({"student_best": str(best_path), "source_test": source_test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
