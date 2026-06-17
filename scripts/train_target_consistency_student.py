from __future__ import annotations

import argparse
import itertools
import json
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
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    return torch, F, DataLoader


def _load_student(checkpoint: Path, device: Any) -> tuple[Any, dict[str, int], dict[str, Any]]:
    torch, _, _ = _require_torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(key): int(value) for key, value in payload["label_to_index"].items()}
    student = DepthwiseStudent(num_classes=len(label_to_index))
    student.load_state_dict(payload["model_state_dict"], strict=True)
    student.to(device)
    return student, label_to_index, payload


def _load_teacher(
    teacher_kind: str,
    num_classes: int,
    teacher_checkpoint: Path | None,
    teacher_repo: Path | None,
    init_teacher_checkpoint: Path,
    device: Any,
) -> Any:
    torch, _, _ = _require_torch()
    teacher = _build_teacher(
        teacher_kind=teacher_kind,
        num_classes=num_classes,
        checkpoint=teacher_checkpoint,
        repo=teacher_repo,
    ).to(device)
    payload = torch.load(init_teacher_checkpoint, map_location=device, weights_only=False)
    teacher.load_state_dict(payload["model_state_dict"], strict=True)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    return teacher


def _weak_augment(x: Any, noise_std: float) -> Any:
    if noise_std <= 0:
        return x
    torch, _, _ = _require_torch()
    return x + torch.randn_like(x) * noise_std


def _strong_augment(x: Any, noise_std: float, time_mask_ratio: float, freq_mask_ratio: float) -> Any:
    torch, _, _ = _require_torch()
    out = x.clone()
    if noise_std > 0:
        out = out + torch.randn_like(out) * noise_std
    batch, _, frames, bins = out.shape
    time_width = max(1, int(frames * time_mask_ratio)) if time_mask_ratio > 0 else 0
    freq_width = max(1, int(bins * freq_mask_ratio)) if freq_mask_ratio > 0 else 0
    for idx in range(batch):
        if time_width > 0 and frames > time_width:
            start = random.randint(0, frames - time_width)
            out[idx, :, start : start + time_width, :] = 0.0
        if freq_width > 0 and bins > freq_width:
            start = random.randint(0, bins - freq_width)
            out[idx, :, :, start : start + freq_width] = 0.0
    return out


def _target_consistency_loss(student: Any, target_x: Any, args: argparse.Namespace) -> tuple[Any, dict[str, float]]:
    torch, F, _ = _require_torch()
    weak_x = _weak_augment(target_x, args.weak_noise_std)
    strong_x = _strong_augment(target_x, args.strong_noise_std, args.time_mask_ratio, args.freq_mask_ratio)
    weak_logits = student(weak_x)["logits"]
    strong_logits = student(strong_x)["logits"]
    weak_probs = F.softmax(weak_logits.detach() / args.consistency_temperature, dim=1)
    confidence, _ = weak_probs.max(dim=1)
    per_sample = F.kl_div(
        F.log_softmax(strong_logits / args.consistency_temperature, dim=1),
        weak_probs,
        reduction="none",
    ).sum(dim=1) * (args.consistency_temperature * args.consistency_temperature)
    mask = (confidence >= args.confidence_threshold).float()
    if float(mask.sum().detach().cpu()) > 0:
        consistency = (per_sample * mask).sum() / mask.sum().clamp(min=1.0)
    else:
        consistency = per_sample.mean() * 0.0
    entropy = -(weak_probs * torch.log(weak_probs.clamp(min=1e-12))).sum(dim=1).mean()
    total = consistency + args.entropy_weight * entropy
    return total, {
        "consistency": float(consistency.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
        "target_confidence": float(confidence.mean().detach().cpu()),
        "target_keep_ratio": float(mask.mean().detach().cpu()),
    }


def _train_epoch(
    student: Any,
    teacher: Any,
    source_loader: Any,
    target_loader: Any,
    optimizer: Any,
    device: Any,
    kd_config: KdLossConfig,
    args: argparse.Namespace,
) -> dict[str, float]:
    torch, F, _ = _require_torch()
    student.train()
    teacher.eval()
    totals = {
        "total": 0.0,
        "ce": 0.0,
        "kd": 0.0,
        "target": 0.0,
        "target_entropy": 0.0,
        "target_confidence": 0.0,
        "target_keep_ratio": 0.0,
        "items": 0.0,
        "steps": 0.0,
    }
    source_iter = itertools.cycle(source_loader)
    target_iter = itertools.cycle(target_loader)
    steps = max(len(source_loader), len(target_loader))
    for step in range(1, steps + 1):
        source_batch = next(source_iter)
        target_batch = next(target_iter)
        source_x = source_batch["features"].to(device, non_blocking=True)
        source_y = source_batch["label"].to(device, non_blocking=True)
        target_x = target_batch["features"].to(device, non_blocking=True)

        with torch.no_grad():
            teacher_out = _forward_model(teacher, source_batch, device)
        student_out = student(source_x)
        kd_losses = torch_kd_loss(source_y, teacher_out, student_out, config=kd_config)
        target_loss, target_stats = _target_consistency_loss(student, target_x, args)
        total = kd_losses["ce"] + args.kd_weight * (
            args.response_weight * kd_losses["response_kd"]
            + args.feature_weight * kd_losses["feature_kd"]
            + args.embedding_weight * kd_losses["embedding_kd"]
            + args.relation_weight * kd_losses["relation_kd"]
        ) + args.target_weight * target_loss

        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        batch_items = int(source_y.shape[0])
        totals["total"] += float(total.detach().cpu()) * batch_items
        totals["ce"] += float(kd_losses["ce"].detach().cpu()) * batch_items
        kd_value = (
            args.response_weight * kd_losses["response_kd"]
            + args.feature_weight * kd_losses["feature_kd"]
            + args.embedding_weight * kd_losses["embedding_kd"]
            + args.relation_weight * kd_losses["relation_kd"]
        )
        totals["kd"] += float(kd_value.detach().cpu()) * batch_items
        totals["target"] += float(target_loss.detach().cpu()) * batch_items
        totals["target_entropy"] += target_stats["entropy"] * batch_items
        totals["target_confidence"] += target_stats["target_confidence"] * batch_items
        totals["target_keep_ratio"] += target_stats["target_keep_ratio"] * batch_items
        totals["items"] += batch_items
        totals["steps"] += 1
        if step == 1 or step % args.log_every == 0:
            print(
                f"[train] step {step}/{steps} total={float(total.detach().cpu()):.4f} "
                f"ce={float(kd_losses['ce'].detach().cpu()):.4f} kd={float(kd_value.detach().cpu()):.4f} "
                f"target={float(target_loss.detach().cpu()):.4f} keep={target_stats['target_keep_ratio']:.3f}",
                flush=True,
            )
    denom = max(1.0, totals["items"])
    return {key: value / denom for key, value in totals.items() if key not in {"items", "steps"}} | {"steps": totals["steps"]}


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    config = summary["config"]
    best = summary["best_val"]
    test = summary["source_test"]
    lines = [
        "# Target-Consistent Distillation",
        "",
        "## Config",
        "",
        f"- Source manifest: `{config['source_manifest']}`",
        f"- Target manifest: `{config['target_manifest']}`",
        f"- Init student: `{config['init_student_checkpoint']}`",
        f"- Init teacher: `{config['init_teacher_checkpoint']}`",
        f"- Target weight: `{config['target_weight']}`",
        f"- Confidence threshold: `{config['confidence_threshold']}`",
        f"- Source train records: `{config['source_train_records']}`",
        f"- Target adapt records: `{config['target_adapt_records']}`",
        f"- Student params: `{config['student_params']}`",
        "",
        "## Source Validation Selection",
        "",
        "```json",
        json.dumps(best, indent=2),
        "```",
        "",
        "## Source Test Metrics",
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
    parser.add_argument("--target-manifest", type=Path, default=ROOT / "manifests/coughvid_adapt_test.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--init-student-checkpoint", type=Path, default=ROOT / "runs/stage1_panns_response_seed7/checkpoints/student_best.pt")
    parser.add_argument("--init-teacher-checkpoint", type=Path, default=ROOT / "runs/stage1_panns_response_seed7/checkpoints/teacher_best.pt")
    parser.add_argument("--teacher-kind", default="panns_cnn14_16k", choices=["compact", "panns_cnn14_16k"])
    parser.add_argument("--teacher-checkpoint", type=Path, default=ROOT / "pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth")
    parser.add_argument("--teacher-repo", type=Path, default=ROOT / "external/teacher_repos/audioset_tagging_cnn_zip")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/stage3_target_consistency_seed7")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-source-records", type=int)
    parser.add_argument("--max-target-records", type=int, default=3000)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--kd-weight", type=float, default=1.0)
    parser.add_argument("--response-weight", type=float, default=0.7)
    parser.add_argument("--feature-weight", type=float, default=0.0)
    parser.add_argument("--embedding-weight", type=float, default=0.0)
    parser.add_argument("--relation-weight", type=float, default=0.0)
    parser.add_argument("--target-weight", type=float, default=0.3)
    parser.add_argument("--entropy-weight", type=float, default=0.0)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--consistency-temperature", type=float, default=1.0)
    parser.add_argument("--weak-noise-std", type=float, default=0.005)
    parser.add_argument("--strong-noise-std", type=float, default=0.02)
    parser.add_argument("--time-mask-ratio", type=float, default=0.08)
    parser.add_argument("--freq-mask-ratio", type=float, default=0.08)
    parser.add_argument("--log-every", type=int, default=20)
    args = parser.parse_args()

    torch, _, DataLoader = _require_torch()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoints = args.out / "checkpoints"
    predictions = args.out / "predictions"
    checkpoints.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)

    student, label_to_index, init_payload = _load_student(args.init_student_checkpoint, device)
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    teacher = _load_teacher(
        args.teacher_kind,
        len(label_to_index),
        args.teacher_checkpoint,
        args.teacher_repo,
        args.init_teacher_checkpoint,
        device,
    )

    config = RunConfig(experiment_name="target_consistency", seed=args.seed)
    source_train = CoughManifestDataset(
        args.source_manifest,
        args.root,
        config,
        label_to_index=label_to_index,
        splits={"train"},
        max_records=args.max_source_records,
        max_duration_sec=args.max_duration_sec,
    )
    source_val = CoughManifestDataset(
        args.source_manifest,
        args.root,
        config,
        label_to_index=label_to_index,
        splits={"val"},
        max_records=args.max_source_records,
        max_duration_sec=args.max_duration_sec,
    )
    source_test = CoughManifestDataset(
        args.source_manifest,
        args.root,
        config,
        label_to_index=label_to_index,
        splits={"test"},
        max_records=args.max_source_records,
        max_duration_sec=args.max_duration_sec,
    )
    target_adapt = CoughManifestDataset(
        args.target_manifest,
        args.root,
        config,
        label_to_index=label_to_index,
        splits={"adapt"},
        max_records=args.max_target_records,
        max_duration_sec=args.max_duration_sec,
    )
    if not source_train or not source_val or not source_test:
        raise RuntimeError("source train/val/test splits must be non-empty")
    if not target_adapt:
        raise RuntimeError("target adapt split is empty")

    source_loader = DataLoader(source_train, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0)
    target_loader = DataLoader(target_adapt, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0)
    val_loader = DataLoader(source_val, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
    test_loader = DataLoader(source_test, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    kd_config = KdLossConfig(temperature=2.0, response_weight=args.response_weight, feature_weight=args.feature_weight)
    run_config = {
        "source_manifest": str(args.source_manifest),
        "target_manifest": str(args.target_manifest),
        "root": str(args.root),
        "init_student_checkpoint": str(args.init_student_checkpoint),
        "init_teacher_checkpoint": str(args.init_teacher_checkpoint),
        "teacher_kind": args.teacher_kind,
        "target_weight": args.target_weight,
        "entropy_weight": args.entropy_weight,
        "confidence_threshold": args.confidence_threshold,
        "weak_noise_std": args.weak_noise_std,
        "strong_noise_std": args.strong_noise_std,
        "time_mask_ratio": args.time_mask_ratio,
        "freq_mask_ratio": args.freq_mask_ratio,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "source_train_records": len(source_train),
        "source_val_records": len(source_val),
        "source_test_records": len(source_test),
        "target_adapt_records": len(target_adapt),
        "student_params": count_parameters(student),
    }
    (args.out / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    events = args.out / "events.jsonl"
    events.write_text("", encoding="utf-8")

    best_metric = -1.0
    best_record: dict[str, Any] = {}
    best_path = checkpoints / "student_best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = _train_epoch(student, teacher, source_loader, target_loader, optimizer, device, kd_config, args)
        val = _evaluate_model(student, val_loader, device, index_to_label)["metrics"]
        metric = _selection_metric(val)
        record = {"epoch": epoch, "train": train, "val": val, "seconds": time.time() - started}
        history.append(record)
        if metric > best_metric:
            best_metric = metric
            best_record = record
            _save_checkpoint(best_path, student, label_to_index, run_config, record)
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        print(json.dumps(record, ensure_ascii=True), flush=True)

    payload = torch.load(best_path, map_location=device, weights_only=False)
    student.load_state_dict(payload["model_state_dict"], strict=True)
    test_result = _evaluate_model(student, test_loader, device, index_to_label)
    final_path = checkpoints / "student_final.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "label_to_index": label_to_index,
            "config": run_config,
            "selection_record": best_record,
            "test_metrics": test_result["metrics"],
        },
        final_path,
    )
    summary = {
        "config": run_config,
        "history": history,
        "best_val": best_record,
        "source_test": test_result["metrics"],
        "artifacts": {"student_best": str(best_path), "student_final": str(final_path)},
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(args.out / "RESULTS.md", summary)
    print(json.dumps({"status": "ok", "source_test": test_result["metrics"], "student_best": str(best_path)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
