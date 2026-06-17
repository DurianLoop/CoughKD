from __future__ import annotations

import argparse
import csv
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


def _label_to_index(manifest: Path) -> dict[str, int]:
    labels = set()
    with manifest.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = row.get("label", "")
            if label:
                labels.add(label)
    return {label: idx for idx, label in enumerate(sorted(labels))}


def _load_envs(path: Path) -> dict[str, int]:
    envs: dict[str, int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rid = row.get("recording_id", "")
            if not rid:
                continue
            envs[rid] = int(row.get("artifact_env", "0"))
    return envs


def _class_weights(dataset: CoughManifestDataset, num_classes: int, device: Any) -> Any:
    torch, _, _, _ = _require_torch()
    counts = [0 for _ in range(num_classes)]
    for row in dataset.rows:
        counts[dataset.label_to_index[row["label"]]] += 1
    total = sum(counts)
    return torch.tensor([total / max(1, num_classes * count) for count in counts], dtype=torch.float32, device=device)


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
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    print("[teacher] ready", flush=True)
    return teacher


def _per_sample_loss(
    labels: Any,
    teacher_out: dict[str, Any],
    student_out: dict[str, Any],
    class_weights: Any,
    kd_config: KdLossConfig,
) -> tuple[Any, Any, Any]:
    torch, F, _, _ = _require_torch()
    ce = F.cross_entropy(student_out["logits"], labels, weight=class_weights, reduction="none")
    temperature = float(kd_config.temperature)
    teacher_probs = torch.softmax(teacher_out["logits"] / temperature, dim=1)
    student_log_probs = torch.log_softmax(student_out["logits"] / temperature, dim=1)
    response_kd = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=1) * (temperature**2)
    total = ce + float(kd_config.response_weight) * response_kd
    return total, ce, response_kd


def _risk_matching_penalty(per_sample_loss: Any, env_ids: Any) -> Any:
    torch, _, _, _ = _require_torch()
    env_values = torch.unique(env_ids)
    if int(env_values.numel()) <= 1:
        return torch.zeros((), dtype=per_sample_loss.dtype, device=per_sample_loss.device)
    means = []
    for env in env_values:
        mask = env_ids == env
        if int(mask.sum().item()) > 0:
            means.append(per_sample_loss[mask].mean())
    if len(means) <= 1:
        return torch.zeros((), dtype=per_sample_loss.dtype, device=per_sample_loss.device)
    stacked = torch.stack(means)
    return ((stacked - stacked.mean()) ** 2).mean()


def _train_epoch(
    student: Any,
    teacher: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    class_weights: Any,
    kd_config: KdLossConfig,
    envs: dict[str, int],
    env_risk_weight: float,
    epoch: int,
    epochs: int,
    env_risk_sigmoid_k: float,
) -> dict[str, float]:
    torch, _, _, _ = _require_torch()
    student.train()
    teacher.eval()
    totals = {
        "total": 0.0,
        "ce": 0.0,
        "response_kd": 0.0,
        "env_penalty": 0.0,
        "effective_env_weight": 0.0,
        "num_env_batches": 0.0,
    }
    total_items = 0
    for batch_idx, batch in enumerate(loader, start=1):
        progress = ((epoch - 1) + batch_idx / max(1, len(loader))) / max(1, epochs)
        if env_risk_sigmoid_k > 0:
            effective_env_weight = float(env_risk_weight) / (1.0 + math.exp(-float(env_risk_sigmoid_k) * (progress - 0.5)))
        else:
            effective_env_weight = float(env_risk_weight)
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        env_ids = torch.tensor([envs.get(rid, 0) for rid in batch["recording_id"]], dtype=torch.long, device=device)
        with torch.no_grad():
            teacher_out = _forward_model(teacher, batch, device)
        student_out = student(features)
        per_sample, ce, response_kd = _per_sample_loss(labels, teacher_out, student_out, class_weights, kd_config)
        base = per_sample.mean()
        env_penalty = _risk_matching_penalty(per_sample, env_ids)
        total = base + effective_env_weight * env_penalty
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()

        batch_size = int(labels.shape[0])
        total_items += batch_size
        totals["total"] += float(total.detach().cpu()) * batch_size
        totals["ce"] += float(ce.mean().detach().cpu()) * batch_size
        totals["response_kd"] += float(response_kd.mean().detach().cpu()) * batch_size
        totals["env_penalty"] += float(env_penalty.detach().cpu()) * batch_size
        totals["effective_env_weight"] += effective_env_weight * batch_size
        if int(torch.unique(env_ids).numel()) > 1:
            totals["num_env_batches"] += batch_size
        if batch_idx == 1 or batch_idx % 40 == 0:
            print(
                f"[train] batch {batch_idx}/{len(loader)} total={float(total.detach().cpu()):.4f} "
                f"env={float(env_penalty.detach().cpu()):.5f} env_w={effective_env_weight:.4f}",
                flush=True,
            )
    return {key: value / max(1, total_items) for key, value in totals.items()}


def _write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Artifact-Env IRM KD",
        "",
        "## Config",
        "",
        f"- Source manifest: `{summary['config']['source_manifest']}`",
        f"- Artifact envs: `{summary['config']['artifact_envs']}`",
        f"- Seed: `{summary['config']['seed']}`",
        f"- Env risk weight: `{summary['config']['env_risk_weight']}`",
        f"- Teacher params: `{summary['config']['teacher_params']}`",
        f"- Student params: `{summary['config']['student_params']}`",
        "",
        "## Source Test",
        "",
        "```json",
        json.dumps(summary["source_test"], indent=2),
        "```",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--artifact-envs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/candidate_f_artifact_env_irm_seed7")
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
    parser.add_argument("--env-risk-weight", type=float, default=1.0)
    parser.add_argument("--env-risk-sigmoid-k", type=float, default=0.0)
    args = parser.parse_args()

    print("[init] import torch", flush=True)
    torch, _, DataLoader, _ = _require_torch()
    _seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[init] device={device}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(parents=True, exist_ok=True)

    print("[init] labels and artifact envs", flush=True)
    labels = _label_to_index(args.source_manifest)
    index_to_label = {idx: label for label, idx in labels.items()}
    envs = _load_envs(args.artifact_envs)
    config = RunConfig(experiment_name="artifact_env_irm_kd", seed=args.seed)
    print("[init] datasets", flush=True)
    datasets = {
        split: CoughManifestDataset(
            args.source_manifest,
            args.root,
            config,
            label_to_index=labels,
            splits={split},
            max_records=args.max_records_per_split,
            max_duration_sec=args.max_duration_sec,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=(split == "train"),
            collate_fn=pad_collate,
            num_workers=0,
            pin_memory=bool(str(device).startswith("cuda")),
        )
        for split, dataset in datasets.items()
    }

    print("[init] teacher", flush=True)
    teacher = _load_teacher(args, len(labels), device)
    print("[init] student", flush=True)
    student = DepthwiseStudent(num_classes=len(labels)).to(device)
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = _class_weights(datasets["train"], len(labels), device)
    kd_config = KdLossConfig(
        temperature=args.kd_temperature,
        response_weight=args.kd_response_weight,
        feature_weight=0.0,
        embedding_weight=0.0,
        relation_weight=0.0,
    )
    run_config = {
        "source_manifest": str(args.source_manifest),
        "root": str(args.root),
        "artifact_envs": str(args.artifact_envs),
        "device": str(device),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "teacher_kind": args.teacher_kind,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_repo": str(args.teacher_repo),
        "init_teacher_checkpoint": str(args.init_teacher_checkpoint),
        "student_params": count_parameters(student),
        "teacher_params": count_parameters(teacher),
        "label_to_index": labels,
        "kd": {"temperature": args.kd_temperature, "response_weight": args.kd_response_weight},
        "env_risk_weight": args.env_risk_weight,
        "env_risk_sigmoid_k": args.env_risk_sigmoid_k,
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
    }
    (args.out / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    history = []
    best_metric = -1.0
    best_path = args.out / "checkpoints/student_best.pt"
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = _train_epoch(
            student,
            teacher,
            loaders["train"],
            optimizer,
            device,
            class_weights,
            kd_config,
            envs,
            args.env_risk_weight,
            epoch,
            args.epochs,
            args.env_risk_sigmoid_k,
        )
        val = _evaluate_model(student, loaders["val"], device, index_to_label)["metrics"]
        entry = {"epoch": epoch, "train": train, "val": val, "seconds": time.time() - started}
        history.append(entry)
        metric = _selection_metric(val)
        if metric > best_metric:
            best_metric = metric
            _save_checkpoint(best_path, student, labels, run_config, entry)
        print(json.dumps({"epoch": epoch, "train": train, "val": val}, indent=2), flush=True)

    student.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model_state_dict"])
    source_test = _evaluate_model(student, loaders["test"], device, index_to_label)["metrics"]
    final_path = args.out / "checkpoints/student_final.pt"
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "label_to_index": labels,
            "config": run_config,
            "history": history,
            "test_metrics": source_test,
        },
        final_path,
    )
    summary = {
        "config": run_config,
        "history": history,
        "best_val": max(history, key=lambda item: _selection_metric(item["val"])),
        "source_test": source_test,
        "artifacts": {"student_best": str(best_path), "student_final": str(final_path)},
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_report(args.out / "RESULTS.md", summary)
    print(json.dumps({"student_best": str(best_path), "source_test": source_test}, indent=2), flush=True)


if __name__ == "__main__":
    main()
