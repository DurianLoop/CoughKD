"""Optional PyTorch datasets, models, and KD training path."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
import platform
import random
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .audio import log_mel_like, read_wav_mono, resample_linear
from .config import RunConfig, git_hash
from .manifest import read_manifest, subject_disjoint_split, write_manifest
from .metrics import average_precision, multiclass_ovr_auroc
from .smoke import make_smoke_data


try:  # pragma: no cover - exercised on the remote torch environment.
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset
except Exception:  # pragma: no cover - local foundation env intentionally has no torch.
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    DataLoader = None  # type: ignore[assignment]
    Dataset = object  # type: ignore[assignment,misc]

_DatasetBase = Dataset
_ModuleBase = nn.Module if nn is not None else object


@dataclass(frozen=True)
class TorchSmokeReport:
    torch_version: str
    device: str
    cuda_available: bool
    num_classes: int
    batch_shape: list[int]
    teacher_params: int
    student_params: int
    initial_loss: float
    final_loss: float
    manifest: str


@dataclass(frozen=True)
class KdLossConfig:
    temperature: float = 2.0
    response_weight: float = 0.7
    feature_weight: float = 0.1
    embedding_weight: float = 0.0
    relation_weight: float = 0.0
    label_smoothing: float = 0.0


def require_torch() -> Any:
    if torch is None:
        raise RuntimeError("PyTorch is not installed. Install torch in the conda environment before training.")
    return torch


class CoughManifestDataset(_DatasetBase):  # type: ignore[misc,valid-type]
    """Load project manifest rows as log-mel-like tensors for PyTorch training."""

    def __init__(
        self,
        manifest: Path,
        root: Path,
        config: RunConfig | None = None,
        label_to_index: dict[str, int] | None = None,
        splits: set[str] | None = None,
        max_records: int | None = None,
        max_duration_sec: float | None = 4.0,
    ):
        require_torch()
        self.root = root
        self.config = config or RunConfig()
        rows = read_manifest(manifest)
        if splits is not None:
            rows = [row for row in rows if row.get("split", "") in splits]
        else:
            rows = [row for row in rows if row.get("split", "") in {"train", "val", "test", "unassigned"}]
        if max_records is not None:
            rows = rows[:max_records]
        self.rows = rows
        self.max_duration_sec = max_duration_sec
        labels = sorted({row["label"] for row in self.rows})
        self.label_to_index = label_to_index or {label: idx for idx, label in enumerate(labels)}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        samples, sample_rate = read_wav_mono(self.root / row["path"])
        samples = resample_linear(samples, sample_rate, self.config.sample_rate)
        if self.max_duration_sec is not None:
            max_samples = max(1, int(self.max_duration_sec * self.config.sample_rate))
            samples = samples[:max_samples]
        features = log_mel_like(samples, self.config)
        waveform = torch.tensor(samples, dtype=torch.float32)
        tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(self.label_to_index[row["label"]], dtype=torch.long)
        return {"features": tensor, "waveform": waveform, "label": label, "recording_id": row["recording_id"]}


def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    require_torch()
    max_frames = max(item["features"].shape[-2] for item in batch)
    n_mels = batch[0]["features"].shape[-1]
    max_samples = max(item["waveform"].shape[-1] for item in batch)
    features = torch.zeros((len(batch), 1, max_frames, n_mels), dtype=torch.float32)
    waveforms = torch.zeros((len(batch), max_samples), dtype=torch.float32)
    labels = torch.stack([item["label"] for item in batch])
    recording_ids = []
    for idx, item in enumerate(batch):
        frames = item["features"].shape[-2]
        samples = item["waveform"].shape[-1]
        features[idx, :, :frames, :] = item["features"]
        waveforms[idx, :samples] = item["waveform"]
        recording_ids.append(item["recording_id"])
    return {"features": features, "waveform": waveforms, "label": labels, "recording_id": recording_ids}


class ConvTeacher(_ModuleBase):  # type: ignore[misc,valid-type]
    """Compact convolutional teacher placeholder for remote training plumbing."""

    def __init__(self, num_classes: int, embedding_dim: int = 128):
        require_torch()
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(128, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: Any) -> dict[str, Any]:
        fmap = self.encoder(x)
        pooled = self.pool(fmap).flatten(1)
        embedding = F.normalize(self.embedding(pooled), dim=1)
        return {"logits": self.classifier(embedding), "features": pooled, "embedding": embedding, "attention": fmap.mean(dim=1)}


class PannsCnn14Teacher(_ModuleBase):  # type: ignore[misc,valid-type]
    """Frozen PANNs CNN14 16 kHz backbone with a trainable cough-label head."""

    def __init__(
        self,
        num_classes: int,
        checkpoint: Path = Path("pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth"),
        repo: Path = Path("external/teacher_repos/audioset_tagging_cnn"),
    ):
        require_torch()
        super().__init__()
        repo = repo.resolve()
        pytorch_dir = repo / "pytorch"
        if not pytorch_dir.is_dir():
            raise RuntimeError(f"PANNs source directory not found: {pytorch_dir}")
        if not checkpoint.is_file():
            raise RuntimeError(f"PANNs checkpoint not found: {checkpoint}")
        sys.path.insert(0, str(pytorch_dir))
        try:
            panns_models = importlib.import_module("models")
        finally:
            try:
                sys.path.remove(str(pytorch_dir))
            except ValueError:
                pass
        self.backbone = panns_models.Cnn14_16k(
            sample_rate=16000,
            window_size=512,
            hop_size=160,
            mel_bins=64,
            fmin=50,
            fmax=8000,
            classes_num=527,
        )
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.backbone.load_state_dict(checkpoint_payload["model"], strict=True)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(2048, num_classes)
        self.checkpoint = str(checkpoint)
        self.repo = str(repo)

    def train(self, mode: bool = True) -> Any:
        super().train(mode)
        self.backbone.eval()
        self.classifier.train(mode)
        return self

    def forward_batch(self, batch: dict[str, Any], device: Any) -> dict[str, Any]:
        waveform = batch["waveform"].to(device, non_blocking=True)
        with torch.no_grad():
            backbone_out = self.backbone(waveform)
            embedding = backbone_out["embedding"]
        logits = self.classifier(embedding)
        return {
            "logits": logits,
            "features": embedding,
            "embedding": F.normalize(embedding, dim=1),
            "attention": torch.empty((waveform.shape[0], 1), device=device),
        }


class DepthwiseStudent(_ModuleBase):  # type: ignore[misc,valid-type]
    """MobileNet-style student for cough spectrogram distillation."""

    def __init__(self, num_classes: int, embedding_dim: int = 64, width: int = 24):
        require_torch()
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(1, width, kernel_size=3, padding=1), nn.BatchNorm2d(width), nn.SiLU())
        self.blocks = nn.Sequential(
            _DepthwiseBlock(width, width * 2, stride=2),
            _DepthwiseBlock(width * 2, width * 3, stride=2),
            _DepthwiseBlock(width * 3, width * 4, stride=2),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding = nn.Linear(width * 4, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: Any) -> dict[str, Any]:
        fmap = self.blocks(self.stem(x))
        pooled = self.pool(fmap).flatten(1)
        embedding = F.normalize(self.embedding(pooled), dim=1)
        return {"logits": self.classifier(embedding), "features": pooled, "embedding": embedding, "attention": fmap.mean(dim=1)}


class _DepthwiseBlock(_ModuleBase):  # type: ignore[misc,valid-type]
    def __init__(self, in_channels: int, out_channels: int, stride: int):
        require_torch()
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels),
            nn.BatchNorm2d(in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(),
        )

    def forward(self, x: Any) -> Any:
        return self.block(x)


def torch_kd_loss(
    labels: Any,
    teacher_out: dict[str, Any],
    student_out: dict[str, Any],
    config: KdLossConfig | None = None,
    class_weights: Any | None = None,
    feature_projector: Any | None = None,
) -> dict[str, Any]:
    config = config or KdLossConfig()
    ce = F.cross_entropy(
        student_out["logits"],
        labels,
        weight=class_weights,
        label_smoothing=config.label_smoothing,
    )
    response = F.kl_div(
        F.log_softmax(student_out["logits"] / config.temperature, dim=1),
        F.softmax(teacher_out["logits"] / config.temperature, dim=1),
        reduction="batchmean",
    ) * (config.temperature * config.temperature)
    student_features = student_out["features"]
    teacher_features = teacher_out["features"]
    if feature_projector is not None:
        student_features = feature_projector(student_features)
    feature = F.mse_loss(_align_features(student_features, teacher_features), teacher_features)
    embedding = _cosine_embedding_kd(student_out.get("embedding"), teacher_out.get("embedding"))
    relation = _relation_kd(student_out.get("embedding"), teacher_out.get("embedding"))
    total = (
        ce
        + config.response_weight * response
        + config.feature_weight * feature
        + config.embedding_weight * embedding
        + config.relation_weight * relation
    )
    return {
        "total": total,
        "ce": ce,
        "response_kd": response,
        "feature_kd": feature,
        "embedding_kd": embedding,
        "relation_kd": relation,
    }


def run_torch_smoke(out_dir: Path, device: str = "auto", batch_size: int = 4) -> dict[str, object]:
    require_torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = make_smoke_data(out_dir / "data")
    rows = subject_disjoint_split(read_manifest(source_manifest), seed=7)
    manifest = out_dir / "manifest_split.csv"
    write_manifest(rows, manifest)

    resolved_device = _resolve_device(device)
    dataset = CoughManifestDataset(manifest, out_dir, RunConfig(experiment_name="torch_smoke"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    batch = next(iter(loader))
    features = batch["features"].to(resolved_device)
    labels = batch["label"].to(resolved_device)
    num_classes = len(dataset.label_to_index)

    teacher = ConvTeacher(num_classes=num_classes).to(resolved_device)
    student = DepthwiseStudent(num_classes=num_classes).to(resolved_device)
    teacher.eval()
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    with torch.no_grad():
        teacher_out = teacher(features)
    student_out = student(features)
    losses = torch_kd_loss(labels, teacher_out, student_out)
    initial_loss = float(losses["total"].detach().cpu())
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    with torch.no_grad():
        final_losses = torch_kd_loss(labels, teacher(features), student(features))
    report = TorchSmokeReport(
        torch_version=torch.__version__,
        device=str(resolved_device),
        cuda_available=bool(torch.cuda.is_available()),
        num_classes=num_classes,
        batch_shape=list(features.shape),
        teacher_params=count_parameters(teacher),
        student_params=count_parameters(student),
        initial_loss=initial_loss,
        final_loss=float(final_losses["total"].detach().cpu()),
        manifest=str(manifest),
    )
    payload = asdict(report)
    (out_dir / "torch_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_torch_manifest_smoke(
    manifest: Path,
    root: Path,
    out_dir: Path,
    device: str = "auto",
    batch_size: int = 4,
) -> dict[str, object]:
    """Run one CUDA/CPU KD optimization step on a real project manifest."""

    require_torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = _resolve_device(device)
    dataset = CoughManifestDataset(manifest, root, RunConfig(experiment_name="torch_manifest_smoke"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_collate)
    batch = next(iter(loader))
    features = batch["features"].to(resolved_device)
    labels = batch["label"].to(resolved_device)
    num_classes = len(dataset.label_to_index)

    teacher = ConvTeacher(num_classes=num_classes).to(resolved_device)
    student = DepthwiseStudent(num_classes=num_classes).to(resolved_device)
    teacher.eval()
    student.train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=1e-3)

    with torch.no_grad():
        teacher_out = teacher(features)
    student_out = student(features)
    losses = torch_kd_loss(labels, teacher_out, student_out)
    initial_loss = float(losses["total"].detach().cpu())
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()

    with torch.no_grad():
        final_losses = torch_kd_loss(labels, teacher(features), student(features))

    payload = {
        "torch_version": str(torch.__version__),
        "device": str(resolved_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "manifest": str(manifest),
        "root": str(root),
        "dataset_records": len(dataset),
        "num_classes": num_classes,
        "label_to_index": dataset.label_to_index,
        "batch_shape": list(features.shape),
        "recording_ids": batch["recording_id"],
        "teacher_params": count_parameters(teacher),
        "student_params": count_parameters(student),
        "initial_loss": initial_loss,
        "final_loss": float(final_losses["total"].detach().cpu()),
    }
    (out_dir / "torch_manifest_smoke.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def run_torch_training(
    manifest: Path,
    root: Path,
    out_dir: Path,
    device: str = "auto",
    teacher_epochs: int = 8,
    student_epochs: int = 8,
    batch_size: int = 64,
    lr: float = 1e-3,
    seed: int = 7,
    num_workers: int = 0,
    max_records_per_split: int | None = None,
    max_duration_sec: float = 4.0,
    teacher_kind: str = "compact",
    teacher_checkpoint: Path | None = None,
    teacher_repo: Path | None = None,
    kd_temperature: float = 2.0,
    kd_response_weight: float = 0.7,
    kd_feature_weight: float = 0.1,
    kd_embedding_weight: float = 0.0,
    kd_relation_weight: float = 0.0,
    label_smoothing: float = 0.0,
) -> dict[str, object]:
    """Train teacher and KD student, then run test-set inference."""

    require_torch()
    _seed_torch(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = out_dir / "checkpoints"
    predictions_dir = out_dir / "predictions"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_manifest(manifest)
    label_to_index = {label: idx for idx, label in enumerate(sorted({row["label"] for row in all_rows}))}
    index_to_label = {idx: label for label, idx in label_to_index.items()}
    config = RunConfig(experiment_name="torch_train", seed=seed)
    resolved_device = _resolve_device(device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    datasets = {
        split: CoughManifestDataset(
            manifest,
            root,
            config,
            label_to_index=label_to_index,
            splits={split},
            max_records=max_records_per_split,
            max_duration_sec=max_duration_sec,
        )
        for split in ("train", "val", "test")
    }
    if not datasets["train"]:
        raise RuntimeError("training split is empty")
    if not datasets["val"]:
        raise RuntimeError("validation split is empty")
    if not datasets["test"]:
        raise RuntimeError("test split is empty")

    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=pad_collate,
            num_workers=num_workers,
            pin_memory=bool(str(resolved_device).startswith("cuda")),
        )
        for split, dataset in datasets.items()
    }

    num_classes = len(label_to_index)
    class_weights = _class_weights(datasets["train"], num_classes, resolved_device)
    teacher = _build_teacher(
        teacher_kind=teacher_kind,
        num_classes=num_classes,
        checkpoint=teacher_checkpoint,
        repo=teacher_repo,
    ).to(resolved_device)
    ce_student = DepthwiseStudent(num_classes=num_classes).to(resolved_device)
    student = DepthwiseStudent(num_classes=num_classes).to(resolved_device)
    kd_config = KdLossConfig(
        temperature=kd_temperature,
        response_weight=kd_response_weight,
        feature_weight=kd_feature_weight,
        embedding_weight=kd_embedding_weight,
        relation_weight=kd_relation_weight,
        label_smoothing=label_smoothing,
    )
    feature_projector = _build_feature_projector(
        teacher,
        student,
        loaders["train"],
        resolved_device,
        enabled=kd_feature_weight > 0.0,
    )
    teacher_optimizer = torch.optim.AdamW(teacher.parameters(), lr=lr, weight_decay=1e-4)
    ce_student_optimizer = torch.optim.AdamW(ce_student.parameters(), lr=lr, weight_decay=1e-4)
    student_parameters = list(student.parameters())
    if feature_projector is not None:
        student_parameters.extend(feature_projector.parameters())
    student_optimizer = torch.optim.AdamW(student_parameters, lr=lr, weight_decay=1e-4)

    run_config = {
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "root": str(root),
        "device": str(resolved_device),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "python_version": platform.python_version(),
        "teacher_epochs": teacher_epochs,
        "student_epochs": student_epochs,
        "batch_size": batch_size,
        "lr": lr,
        "seed": seed,
        "num_workers": num_workers,
        "max_records_per_split": max_records_per_split,
        "max_duration_sec": max_duration_sec,
        "teacher_kind": teacher_kind,
        "teacher_checkpoint": str(teacher_checkpoint) if teacher_checkpoint else "",
        "teacher_repo": str(teacher_repo) if teacher_repo else "",
        "kd": asdict(kd_config),
        "feature_projector": _projector_description(feature_projector),
        "label_to_index": label_to_index,
        "git_hash": git_hash(),
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
        "teacher_params": count_parameters(teacher),
        "student_params": count_parameters(student),
        "ce_student_params": count_parameters(ce_student),
        "environment": _environment_snapshot(),
    }
    (out_dir / "config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    events_path = out_dir / "events.jsonl"
    events_path.write_text("", encoding="utf-8")

    teacher_history: list[dict[str, object]] = []
    best_teacher_metric = -1.0
    best_teacher_path = checkpoints_dir / "teacher_best.pt"
    for epoch in range(1, teacher_epochs + 1):
        started = time.time()
        train_loss = _train_teacher_epoch(
            teacher,
            loaders["train"],
            teacher_optimizer,
            resolved_device,
            class_weights,
            label_smoothing=label_smoothing,
        )
        val_result = _evaluate_model(teacher, loaders["val"], resolved_device, index_to_label)
        metric = _selection_metric(val_result["metrics"])
        entry = {"epoch": epoch, "train_loss": train_loss, "val": val_result["metrics"], "seconds": time.time() - started}
        teacher_history.append(entry)
        if metric > best_teacher_metric:
            best_teacher_metric = metric
            _save_checkpoint(best_teacher_path, teacher, label_to_index, run_config, entry)
        _log_event(events_path, {"stage": "teacher", **entry})

    teacher.load_state_dict(torch.load(best_teacher_path, map_location=resolved_device, weights_only=False)["model_state_dict"])
    teacher_test = _evaluate_model(teacher, loaders["test"], resolved_device, index_to_label)
    _write_predictions(predictions_dir / "teacher_test_predictions.csv", teacher_test["predictions"], index_to_label)
    _log_event(events_path, {"stage": "teacher_test", "metrics": teacher_test["metrics"]})
    torch.save(
        {
            "model_state_dict": teacher.state_dict(),
            "label_to_index": label_to_index,
            "config": run_config,
            "test_metrics": teacher_test["metrics"],
        },
        checkpoints_dir / "teacher_final.pt",
    )

    ce_student_history: list[dict[str, object]] = []
    best_ce_student_metric = -1.0
    best_ce_student_path = checkpoints_dir / "ce_student_best.pt"
    for epoch in range(1, student_epochs + 1):
        started = time.time()
        train_loss = _train_teacher_epoch(
            ce_student,
            loaders["train"],
            ce_student_optimizer,
            resolved_device,
            class_weights,
            label_smoothing=label_smoothing,
        )
        val_result = _evaluate_model(ce_student, loaders["val"], resolved_device, index_to_label)
        metric = _selection_metric(val_result["metrics"])
        entry = {"epoch": epoch, "train_loss": train_loss, "val": val_result["metrics"], "seconds": time.time() - started}
        ce_student_history.append(entry)
        if metric > best_ce_student_metric:
            best_ce_student_metric = metric
            _save_checkpoint(best_ce_student_path, ce_student, label_to_index, run_config, entry)
        _log_event(events_path, {"stage": "ce_student", **entry})

    ce_student.load_state_dict(torch.load(best_ce_student_path, map_location=resolved_device, weights_only=False)["model_state_dict"])
    ce_student_test = _evaluate_model(ce_student, loaders["test"], resolved_device, index_to_label)
    _write_predictions(predictions_dir / "ce_student_test_predictions.csv", ce_student_test["predictions"], index_to_label)
    _log_event(events_path, {"stage": "ce_student_test", "metrics": ce_student_test["metrics"]})
    torch.save(
        {
            "model_state_dict": ce_student.state_dict(),
            "label_to_index": label_to_index,
            "config": run_config,
            "test_metrics": ce_student_test["metrics"],
        },
        checkpoints_dir / "ce_student_final.pt",
    )

    student_history: list[dict[str, object]] = []
    best_student_metric = -1.0
    best_student_path = checkpoints_dir / "student_best.pt"
    teacher.eval()
    for epoch in range(1, student_epochs + 1):
        started = time.time()
        train_report = _train_student_epoch(
            student,
            teacher,
            loaders["train"],
            student_optimizer,
            resolved_device,
            class_weights,
            kd_config,
            feature_projector,
        )
        val_result = _evaluate_model(student, loaders["val"], resolved_device, index_to_label)
        metric = _selection_metric(val_result["metrics"])
        entry = {"epoch": epoch, "train": train_report, "val": val_result["metrics"], "seconds": time.time() - started}
        student_history.append(entry)
        if metric > best_student_metric:
            best_student_metric = metric
            _save_checkpoint(best_student_path, student, label_to_index, run_config, entry)
        _log_event(events_path, {"stage": "student", **entry})

    student.load_state_dict(torch.load(best_student_path, map_location=resolved_device, weights_only=False)["model_state_dict"])
    student_test = _evaluate_model(student, loaders["test"], resolved_device, index_to_label)
    _write_predictions(predictions_dir / "student_test_predictions.csv", student_test["predictions"], index_to_label)
    _log_event(events_path, {"stage": "student_test", "metrics": student_test["metrics"]})
    torch.save(
        {
            "model_state_dict": student.state_dict(),
            "feature_projector_state_dict": feature_projector.state_dict() if feature_projector is not None else None,
            "label_to_index": label_to_index,
            "config": run_config,
            "test_metrics": student_test["metrics"],
        },
        checkpoints_dir / "student_final.pt",
    )

    summary = {
        "config": run_config,
        "teacher_history": teacher_history,
        "ce_student_history": ce_student_history,
        "student_history": student_history,
        "teacher_test": teacher_test["metrics"],
        "ce_student_test": ce_student_test["metrics"],
        "student_test": student_test["metrics"],
        "artifacts": {
            "teacher_best": str(best_teacher_path),
            "ce_student_best": str(best_ce_student_path),
            "student_best": str(best_student_path),
            "teacher_final": str(checkpoints_dir / "teacher_final.pt"),
            "ce_student_final": str(checkpoints_dir / "ce_student_final.pt"),
            "student_final": str(checkpoints_dir / "student_final.pt"),
            "teacher_predictions": str(predictions_dir / "teacher_test_predictions.csv"),
            "ce_student_predictions": str(predictions_dir / "ce_student_test_predictions.csv"),
            "student_predictions": str(predictions_dir / "student_test_predictions.csv"),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_summary_markdown(out_dir / "RESULTS.md", summary)
    return summary


def run_prelong_check(
    manifest: Path,
    root: Path,
    out_dir: Path,
    device: str = "auto",
    expected_python: str | None = None,
) -> dict[str, object]:
    """Validate the minimum gates required before launching a long run."""

    require_torch()
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(manifest)
    issues: list[dict[str, object]] = []
    splits = {row.get("split", "") for row in rows}
    for required in ("train", "val", "test"):
        if required not in splits:
            issues.append({"severity": "error", "message": f"missing required split: {required}"})
    subject_splits: dict[str, set[str]] = {}
    for row in rows:
        subject = row.get("subject_id", "")
        split = row.get("split", "")
        if subject and split:
            subject_splits.setdefault(subject, set()).add(split)
    leaked = {subject: sorted(values) for subject, values in subject_splits.items() if len(values) > 1}
    if leaked:
        examples = dict(list(leaked.items())[:5])
        issues.append({"severity": "error", "message": "subject leakage across splits", "examples": examples})

    missing_examples = []
    for row in rows[:200]:
        if not (root / row["path"]).is_file():
            missing_examples.append(row["path"])
            if len(missing_examples) >= 5:
                break
    if missing_examples:
        issues.append({"severity": "error", "message": "missing audio files", "examples": missing_examples})

    resolved_device = _resolve_device(device)
    python_version = platform.python_version()
    if expected_python and not python_version.startswith(expected_python):
        issues.append(
            {
                "severity": "warning",
                "message": f"python version {python_version} does not match expected prefix {expected_python}",
            }
        )

    dataset = CoughManifestDataset(manifest, root, RunConfig(experiment_name="prelong_check"), splits={"train"}, max_records=8)
    loader = DataLoader(dataset, batch_size=min(8, len(dataset)), shuffle=False, collate_fn=pad_collate)
    started = time.time()
    batch = next(iter(loader))
    load_seconds = time.time() - started
    features = batch["features"].to(resolved_device)
    labels = batch["label"].to(resolved_device)
    model = DepthwiseStudent(num_classes=len(dataset.label_to_index)).to(resolved_device)
    with torch.no_grad():
        out = model(features)
        loss = F.cross_entropy(out["logits"], labels)
    payload = {
        "status": "ok" if not any(issue["severity"] == "error" for issue in issues) else "failed",
        "manifest": str(manifest),
        "manifest_sha256": _sha256_file(manifest),
        "root": str(root),
        "num_records": len(rows),
        "splits": {split: sum(1 for row in rows if row.get("split", "") == split) for split in sorted(splits)},
        "num_subjects": len({row.get("subject_id", "") for row in rows if row.get("subject_id", "")}),
        "label_counts": _label_counts(rows),
        "device": str(resolved_device),
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "torch_version": torch.__version__,
        "python_version": python_version,
        "batch_shape": list(features.shape),
        "batch_load_seconds": load_seconds,
        "forward_loss": float(loss.detach().cpu()),
        "issues": issues,
        "environment": _environment_snapshot(),
    }
    (out_dir / "prelong_check.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_prelong_markdown(out_dir / "PRELONG_CHECK.md", payload)
    return payload


def count_parameters(model: Any) -> int:
    return sum(param.numel() for param in model.parameters())


def _build_teacher(
    teacher_kind: str,
    num_classes: int,
    checkpoint: Path | None = None,
    repo: Path | None = None,
) -> Any:
    if teacher_kind == "compact":
        return ConvTeacher(num_classes=num_classes)
    if teacher_kind == "panns_cnn14_16k":
        return PannsCnn14Teacher(
            num_classes=num_classes,
            checkpoint=checkpoint or Path("pretrained/teachers/panns/Cnn14_16k_mAP=0.438.pth"),
            repo=repo or Path("external/teacher_repos/audioset_tagging_cnn"),
        )
    raise ValueError(f"unsupported teacher_kind: {teacher_kind}")


def _forward_model(model: Any, batch: dict[str, Any], device: Any) -> dict[str, Any]:
    if hasattr(model, "forward_batch"):
        return model.forward_batch(batch, device)
    features = batch["features"].to(device, non_blocking=True)
    return model(features)


def _seed_torch(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _class_weights(dataset: CoughManifestDataset, num_classes: int, device: Any) -> Any:
    counts = [0 for _ in range(num_classes)]
    for row in dataset.rows:
        counts[dataset.label_to_index[row["label"]]] += 1
    total = sum(counts)
    weights = [total / max(1, num_classes * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _train_teacher_epoch(
    model: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    class_weights: Any,
    label_smoothing: float = 0.0,
) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        labels = batch["label"].to(device, non_blocking=True)
        out = _forward_model(model, batch, device)
        loss = F.cross_entropy(out["logits"], labels, weight=class_weights, label_smoothing=label_smoothing)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
    return total_loss / max(1, total_items)


def _train_student_epoch(
    student: Any,
    teacher: Any,
    loader: Any,
    optimizer: Any,
    device: Any,
    class_weights: Any,
    kd_config: KdLossConfig,
    feature_projector: Any | None = None,
) -> dict[str, float]:
    teacher.eval()
    student.train()
    if feature_projector is not None:
        feature_projector.train()
    totals = {"total": 0.0, "ce": 0.0, "response_kd": 0.0, "feature_kd": 0.0, "embedding_kd": 0.0, "relation_kd": 0.0}
    total_items = 0
    for batch in loader:
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        with torch.no_grad():
            teacher_out = _forward_model(teacher, batch, device)
        student_out = student(features)
        losses = torch_kd_loss(
            labels,
            teacher_out,
            student_out,
            config=kd_config,
            class_weights=class_weights,
            feature_projector=feature_projector,
        )
        total = losses["total"]
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        batch_size = labels.shape[0]
        total_items += batch_size
        totals["total"] += float(total.detach().cpu()) * batch_size
        totals["ce"] += float(losses["ce"].detach().cpu()) * batch_size
        totals["response_kd"] += float(losses["response_kd"].detach().cpu()) * batch_size
        totals["feature_kd"] += float(losses["feature_kd"].detach().cpu()) * batch_size
        totals["embedding_kd"] += float(losses["embedding_kd"].detach().cpu()) * batch_size
        totals["relation_kd"] += float(losses["relation_kd"].detach().cpu()) * batch_size
    return {key: value / max(1, total_items) for key, value in totals.items()}


def _evaluate_model(model: Any, loader: Any, device: Any, index_to_label: dict[int, str]) -> dict[str, object]:
    model.eval()
    true_indices: list[int] = []
    pred_indices: list[int] = []
    prob_rows: list[list[float]] = []
    recording_ids: list[str] = []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for batch in loader:
            labels = batch["label"].to(device, non_blocking=True)
            out = _forward_model(model, batch, device)
            loss = F.cross_entropy(out["logits"], labels)
            probs = F.softmax(out["logits"], dim=1)
            preds = probs.argmax(dim=1)
            batch_size = labels.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            true_indices.extend(int(item) for item in labels.detach().cpu().tolist())
            pred_indices.extend(int(item) for item in preds.detach().cpu().tolist())
            prob_rows.extend([[float(value) for value in row] for row in probs.detach().cpu().tolist()])
            recording_ids.extend(batch["recording_id"])
    metrics = _classification_metrics(true_indices, pred_indices, prob_rows, index_to_label)
    metrics["loss"] = total_loss / max(1, total_items)
    predictions = [
        {"recording_id": rid, "true_index": true_idx, "pred_index": pred_idx, "probabilities": probs}
        for rid, true_idx, pred_idx, probs in zip(recording_ids, true_indices, pred_indices, prob_rows)
    ]
    return {"metrics": metrics, "predictions": predictions}


def _classification_metrics(
    true_indices: list[int],
    pred_indices: list[int],
    prob_rows: list[list[float]],
    index_to_label: dict[int, str],
) -> dict[str, float | int | None]:
    num_classes = len(index_to_label)
    total = len(true_indices)
    correct = sum(1 for true, pred in zip(true_indices, pred_indices) if true == pred)
    per_class_f1: list[float] = []
    per_class_metrics: dict[str, float] = {}
    confusion = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    for true, pred in zip(true_indices, pred_indices):
        confusion[true][pred] += 1
    for class_idx in range(num_classes):
        tp = sum(1 for true, pred in zip(true_indices, pred_indices) if true == class_idx and pred == class_idx)
        fp = sum(1 for true, pred in zip(true_indices, pred_indices) if true != class_idx and pred == class_idx)
        fn = sum(1 for true, pred in zip(true_indices, pred_indices) if true == class_idx and pred != class_idx)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class_f1.append(f1)
        class_name = index_to_label[class_idx]
        per_class_metrics[f"{class_name}_precision"] = precision
        per_class_metrics[f"{class_name}_recall"] = recall
        per_class_metrics[f"{class_name}_f1"] = f1
    labels = [index_to_label[idx] for idx in true_indices]
    classes = [index_to_label[idx] for idx in range(num_classes)]
    try:
        auc_report = multiclass_ovr_auroc(labels, prob_rows, classes)
        auc = auc_report.get("macro_ovr_auroc")
        for class_name, value in auc_report.items():
            if class_name != "macro_ovr_auroc":
                per_class_metrics[f"{class_name}_ovr_auroc"] = value
    except Exception:
        auc = None
    auprc = _macro_ovr_auprc(labels, prob_rows, classes)
    result: dict[str, float | int | None | list[list[int]]] = {
        "num_examples": total,
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(per_class_f1) / max(1, len(per_class_f1)),
        "macro_ovr_auroc": auc,
        "macro_ovr_auprc": auprc,
        "ece": _multiclass_ece(true_indices, pred_indices, prob_rows),
        "brier": _multiclass_brier(true_indices, prob_rows, num_classes),
        "confusion_matrix": confusion,
    }
    result.update(per_class_metrics)
    return result


def _macro_ovr_auprc(labels: list[str], prob_rows: list[list[float]], classes: list[str]) -> float | None:
    per_class: dict[str, float] = {}
    for class_idx, class_name in enumerate(classes):
        binary_labels = [1 if label == class_name else 0 for label in labels]
        if sum(binary_labels) == 0:
            continue
        binary_scores = [row[class_idx] for row in prob_rows]
        per_class[class_name] = average_precision(binary_labels, binary_scores)
    if not per_class:
        return None
    return sum(per_class.values()) / len(per_class)


def _multiclass_ece(
    true_indices: list[int],
    pred_indices: list[int],
    prob_rows: list[list[float]],
    bins: int = 10,
) -> float:
    total = len(true_indices)
    if total == 0:
        return 0.0
    confidences = [max(row) if row else 0.0 for row in prob_rows]
    ece = 0.0
    for bin_idx in range(bins):
        lo = bin_idx / bins
        hi = (bin_idx + 1) / bins
        indices = [
            idx
            for idx, confidence in enumerate(confidences)
            if (lo <= confidence < hi) or (bin_idx == bins - 1 and confidence == 1.0)
        ]
        if not indices:
            continue
        accuracy = sum(1 for idx in indices if true_indices[idx] == pred_indices[idx]) / len(indices)
        confidence = sum(confidences[idx] for idx in indices) / len(indices)
        ece += (len(indices) / total) * abs(accuracy - confidence)
    return ece


def _multiclass_brier(true_indices: list[int], prob_rows: list[list[float]], num_classes: int) -> float:
    if not true_indices:
        return 0.0
    total = 0.0
    for true_idx, probs in zip(true_indices, prob_rows):
        for class_idx in range(num_classes):
            target = 1.0 if class_idx == true_idx else 0.0
            total += (probs[class_idx] - target) ** 2
    return total / len(true_indices)


def _selection_metric(metrics: dict[str, Any]) -> float:
    value = metrics.get("macro_ovr_auroc")
    if isinstance(value, (int, float)):
        return float(value)
    return float(metrics.get("accuracy", 0.0))


def _save_checkpoint(path: Path, model: Any, label_to_index: dict[str, int], config: dict[str, object], metrics: dict[str, object]) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "label_to_index": label_to_index,
            "config": config,
            "selection_record": metrics,
        },
        path,
    )


def _write_predictions(path: Path, predictions: list[dict[str, object]], index_to_label: dict[int, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    classes = [index_to_label[idx] for idx in sorted(index_to_label)]
    fieldnames = ["recording_id", "true_label", "pred_label"] + [f"prob_{label}" for label in classes]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in predictions:
            probs = item["probabilities"]
            row = {
                "recording_id": item["recording_id"],
                "true_label": index_to_label[int(item["true_index"])],
                "pred_label": index_to_label[int(item["pred_index"])],
            }
            row.update({f"prob_{label}": f"{float(probs[idx]):.8f}" for idx, label in enumerate(classes)})
            writer.writerow(row)


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    config = summary["config"]
    teacher = summary["teacher_test"]
    ce_student = summary.get("ce_student_test")
    student = summary["student_test"]
    artifacts = summary["artifacts"]
    lines = [
        "# COUGHKD Training Results",
        "",
        "## Run",
        "",
        f"- Manifest: `{config['manifest']}`",
        f"- Root: `{config['root']}`",
        f"- Device: `{config['device']}`",
        f"- CUDA device: `{config['cuda_device']}`",
        f"- Split sizes: `{config['split_sizes']}`",
        f"- Labels: `{config['label_to_index']}`",
        f"- Teacher params: `{config['teacher_params']}`",
        f"- CE-only student params: `{config.get('ce_student_params', config['student_params'])}`",
        f"- Student params: `{config['student_params']}`",
        f"- Manifest SHA256: `{config.get('manifest_sha256', '')}`",
        f"- Python: `{config.get('python_version', '')}`",
        f"- Torch: `{config.get('torch_version', '')}`",
        "",
        "## Test Metrics",
        "",
        "| Model | Accuracy | Macro-F1 | Macro OVR AUROC | Macro OVR AUPRC | ECE | Brier | Loss |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| Teacher {config.get('teacher_kind', 'compact')} | {_fmt(teacher.get('accuracy'))} | {_fmt(teacher.get('macro_f1'))} | {_fmt(teacher.get('macro_ovr_auroc'))} | {_fmt(teacher.get('macro_ovr_auprc'))} | {_fmt(teacher.get('ece'))} | {_fmt(teacher.get('brier'))} | {_fmt(teacher.get('loss'))} |",
        f"| Student Depthwise CE-only | {_fmt(ce_student.get('accuracy') if isinstance(ce_student, dict) else None)} | {_fmt(ce_student.get('macro_f1') if isinstance(ce_student, dict) else None)} | {_fmt(ce_student.get('macro_ovr_auroc') if isinstance(ce_student, dict) else None)} | {_fmt(ce_student.get('macro_ovr_auprc') if isinstance(ce_student, dict) else None)} | {_fmt(ce_student.get('ece') if isinstance(ce_student, dict) else None)} | {_fmt(ce_student.get('brier') if isinstance(ce_student, dict) else None)} | {_fmt(ce_student.get('loss') if isinstance(ce_student, dict) else None)} |",
        f"| Student Depthwise KD | {_fmt(student.get('accuracy'))} | {_fmt(student.get('macro_f1'))} | {_fmt(student.get('macro_ovr_auroc'))} | {_fmt(student.get('macro_ovr_auprc'))} | {_fmt(student.get('ece'))} | {_fmt(student.get('brier'))} | {_fmt(student.get('loss'))} |",
        "",
        "## Artifacts",
        "",
    ]
    for name, value in artifacts.items():
        lines.append(f"- {name}: `{value}`")
    lines.extend(
        [
            "",
            "## Paper Conclusion Draft",
            "",
            f"This run is a real closed-loop Coswara cough experiment using a subject-disjoint split, a `{config.get('teacher_kind', 'compact')}` teacher, a CE-only MobileNet-style depthwise student control, and a KD depthwise student trained with response and feature distillation. The reported numbers are measured on the held-out test split and should replace target-only placeholders only for this exact engineering baseline. The result supports the feasibility of the repository pipeline, checkpointing, inference export, and KD-vs-CE control reporting; it does not establish clinical utility because no external dataset validation, repeated-seed analysis, or prospective validation has been run.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _fmt(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{float(value):.6f}"
    return "n/a"


def _log_event(path: Path, payload: dict[str, object]) -> None:
    line = json.dumps(payload, ensure_ascii=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _environment_snapshot() -> dict[str, object]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__) if torch is not None else "",
        "cuda_available": bool(torch.cuda.is_available()) if torch is not None else False,
        "cuda_device": torch.cuda.get_device_name(0) if torch is not None and torch.cuda.is_available() else "",
        "git_hash": git_hash(),
        "conda_list": _run_capture(["conda", "list"]),
        "pip_freeze": _run_capture(["python", "-m", "pip", "freeze"]),
    }


def _run_capture(command: list[str]) -> str:
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except Exception as exc:
        return f"unavailable: {exc}"
    text = result.stdout.strip() or result.stderr.strip()
    return text[:20000]


def _label_counts(rows: list[dict[str, str]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        label = row.get("label", "")
        counts[label] = counts.get(label, 0) + 1
    return counts


def _write_prelong_markdown(path: Path, payload: dict[str, object]) -> None:
    lines = [
        "# Pre-Long-Training Check",
        "",
        f"- Status: `{payload['status']}`",
        f"- Manifest: `{payload['manifest']}`",
        f"- Root: `{payload['root']}`",
        f"- Records: `{payload['num_records']}`",
        f"- Subjects: `{payload['num_subjects']}`",
        f"- Splits: `{payload['splits']}`",
        f"- Labels: `{payload['label_counts']}`",
        f"- Device: `{payload['device']}`",
        f"- CUDA device: `{payload['cuda_device']}`",
        f"- Python: `{payload['python_version']}`",
        f"- Torch: `{payload['torch_version']}`",
        f"- Batch shape: `{payload['batch_shape']}`",
        f"- Batch load seconds: `{payload['batch_load_seconds']}`",
        f"- Forward loss: `{payload['forward_loss']}`",
        "",
        "## Issues",
        "",
    ]
    issues = payload.get("issues", [])
    if isinstance(issues, list) and issues:
        for issue in issues:
            lines.append(f"- `{issue.get('severity')}`: {issue.get('message')}")
    else:
        lines.append("No blocking issues found.")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _build_feature_projector(teacher: Any, student: Any, loader: Any, device: Any, enabled: bool) -> Any | None:
    if not enabled:
        return None
    try:
        batch = next(iter(loader))
    except StopIteration:
        return None
    teacher_was_training = bool(getattr(teacher, "training", False))
    student_was_training = bool(getattr(student, "training", False))
    teacher.eval()
    student.eval()
    with torch.no_grad():
        features = batch["features"].to(device, non_blocking=True)
        teacher_out = _forward_model(teacher, batch, device)
        student_out = student(features)
    if teacher_was_training:
        teacher.train()
    if student_was_training:
        student.train()
    student_dim = int(student_out["features"].shape[1])
    teacher_dim = int(teacher_out["features"].shape[1])
    if student_dim == teacher_dim:
        return None
    return nn.Linear(student_dim, teacher_dim, bias=False).to(device)


def _projector_description(projector: Any | None) -> dict[str, object]:
    if projector is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "in_features": int(projector.in_features),
        "out_features": int(projector.out_features),
        "parameters": count_parameters(projector),
    }


def _cosine_embedding_kd(student_embedding: Any | None, teacher_embedding: Any | None) -> Any:
    if student_embedding is None or teacher_embedding is None:
        device = student_embedding.device if student_embedding is not None else (teacher_embedding.device if teacher_embedding is not None else torch.device("cpu"))
        return torch.zeros((), dtype=torch.float32, device=device)
    student_aligned = _align_features(student_embedding, teacher_embedding)
    student_norm = F.normalize(student_aligned, dim=1)
    teacher_norm = F.normalize(teacher_embedding, dim=1)
    return 1.0 - F.cosine_similarity(student_norm, teacher_norm, dim=1).mean()


def _relation_kd(student_embedding: Any | None, teacher_embedding: Any | None) -> Any:
    if student_embedding is None or teacher_embedding is None:
        device = student_embedding.device if student_embedding is not None else (teacher_embedding.device if teacher_embedding is not None else torch.device("cpu"))
        return torch.zeros((), dtype=torch.float32, device=device)
    if student_embedding.shape[0] <= 1:
        return torch.zeros((), dtype=torch.float32, device=student_embedding.device)
    student_aligned = _align_features(student_embedding, teacher_embedding)
    student_norm = F.normalize(student_aligned, dim=1)
    teacher_norm = F.normalize(teacher_embedding, dim=1)
    student_relation = student_norm @ student_norm.transpose(0, 1)
    teacher_relation = teacher_norm @ teacher_norm.transpose(0, 1)
    return F.mse_loss(student_relation, teacher_relation)


def _align_features(student_features: Any, teacher_features: Any) -> Any:
    if student_features.shape == teacher_features.shape:
        return student_features
    if student_features.shape[1] < teacher_features.shape[1]:
        pad = teacher_features.shape[1] - student_features.shape[1]
        return F.pad(student_features, (0, pad))
    return student_features[:, : teacher_features.shape[1]]


def _resolve_device(device: str) -> Any:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)
