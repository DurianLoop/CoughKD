from __future__ import annotations

import argparse
import csv
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

from coughkd.audio import log_mel_like, read_wav_mono, resample_linear
from coughkd.config import RunConfig
from coughkd.manifest import read_manifest
from coughkd.metrics import average_precision, binary_classification_report, roc_auc
from coughkd.torch_models import DepthwiseStudent, _save_checkpoint, count_parameters, pad_collate


def _require_torch() -> Any:
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Dataset

    return torch, F, DataLoader, Dataset, np


def _seed_everything(seed: int) -> None:
    torch, _, _, _, np = _require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _binary_label(label: str) -> int:
    return 1 if label == "covid_positive" else 0


class BinaryCoughDataset(_require_torch()[3]):  # type: ignore[misc,valid-type]
    def __init__(
        self,
        manifest: Path,
        root: Path,
        config: RunConfig,
        splits: set[str] | None,
        max_records: int | None = None,
        max_duration_sec: float = 4.0,
    ) -> None:
        rows = read_manifest(manifest)
        if splits is not None:
            rows = [row for row in rows if row.get("split", "") in splits]
        if max_records:
            rows = rows[:max_records]
        self.rows = rows
        self.root = root
        self.config = config
        self.max_duration_sec = max_duration_sec

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        torch, *_ = _require_torch()
        row = self.rows[idx]
        samples, sample_rate = read_wav_mono(self.root / row["path"])
        samples = resample_linear(samples, sample_rate, self.config.sample_rate)
        if self.max_duration_sec > 0:
            samples = samples[: int(self.max_duration_sec * self.config.sample_rate)]
        features = log_mel_like(samples, self.config)
        return {
            "features": torch.tensor(features, dtype=torch.float32).unsqueeze(0),
            "waveform": torch.tensor(samples, dtype=torch.float32),
            "label": torch.tensor(_binary_label(row["label"]), dtype=torch.long),
            "kd_weight": torch.tensor(1.0, dtype=torch.float32),
            "recording_id": row["recording_id"],
        }


def _class_weights(dataset: BinaryCoughDataset, device: Any) -> Any:
    torch, *_ = _require_torch()
    counts = [0, 0]
    for row in dataset.rows:
        counts[_binary_label(row["label"])] += 1
    total = sum(counts)
    return torch.tensor([total / max(1, 2 * count) for count in counts], dtype=torch.float32, device=device)


def _evaluate(model: Any, loader: Any, device: Any, rows_by_id: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    torch, *_ = _require_torch()
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    predictions: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            features = batch["features"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)
            probs = torch.softmax(model(features)["logits"], dim=1)[:, 1]
            for rid, label, score in zip(batch["recording_id"], y.cpu().tolist(), probs.cpu().tolist()):
                labels.append(int(label))
                scores.append(float(score))
                manifest = rows_by_id.get(rid, {}) if rows_by_id else {}
                predictions.append(
                    {
                        "recording_id": rid,
                        "subject_id": manifest.get("subject_id", ""),
                        "true_label": "covid_positive" if int(label) == 1 else "non_covid",
                        "score_covid_positive": f"{float(score):.8f}",
                    }
                )
            if batch_idx == 1 or batch_idx % 40 == 0:
                print(f"[eval] batch {batch_idx}", flush=True)
    metrics = binary_classification_report(labels, scores)
    metrics["num_examples"] = len(labels)
    metrics["positive_rate"] = sum(labels) / max(1, len(labels))
    return {"metrics": metrics, "predictions": predictions}


def _subject_metrics(predictions: list[dict[str, Any]]) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        sid = row.get("subject_id") or row["recording_id"]
        groups.setdefault(str(sid), []).append(row)
    labels = []
    scores = []
    for items in groups.values():
        label = 1 if any(item["true_label"] == "covid_positive" for item in items) else 0
        score = sum(float(item["score_covid_positive"]) for item in items) / len(items)
        labels.append(label)
        scores.append(score)
    out = binary_classification_report(labels, scores)
    out["num_subjects"] = len(labels)
    out["positive_rate"] = sum(labels) / max(1, len(labels))
    return out


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["recording_id", "subject_id", "true_label", "score_covid_positive"])
        writer.writeheader()
        writer.writerows(rows)


def _train_epoch(model: Any, loader: Any, optimizer: Any, device: Any, class_weights: Any) -> dict[str, float]:
    torch, F, *_ = _require_torch()
    model.train()
    total_loss = 0.0
    total_items = 0
    for batch_idx, batch in enumerate(loader, start=1):
        features = batch["features"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        loss = F.cross_entropy(model(features)["logits"], labels, weight=class_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_items += batch_size
        if batch_idx == 1 or batch_idx % 40 == 0:
            print(f"[train] batch {batch_idx}/{len(loader)} loss={float(loss.detach().cpu()):.4f}", flush=True)
    return {"loss": total_loss / max(1, total_items)}


def _manifest_by_id(path: Path) -> dict[str, dict[str, str]]:
    return {row["recording_id"]: row for row in read_manifest(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--coughvid-manifest", type=Path, default=ROOT / "manifests/coughvid_adapt_test.csv")
    parser.add_argument("--tos-manifest", type=Path, default=ROOT / "manifests/toscovid2021_test_external.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/binary_covid_student_seed7")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-records-per-split", type=int)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    args = parser.parse_args()

    torch, _, DataLoader, _, _ = _require_torch()
    _seed_everything(args.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "checkpoints").mkdir(parents=True, exist_ok=True)
    config = RunConfig(experiment_name="binary_covid_student", seed=args.seed)
    datasets = {
        split: BinaryCoughDataset(
            args.source_manifest,
            args.root,
            config,
            splits={split},
            max_records=args.max_records_per_split,
            max_duration_sec=args.max_duration_sec,
        )
        for split in ("train", "val", "test")
    }
    loaders = {
        split: DataLoader(dataset, batch_size=args.batch_size, shuffle=(split == "train"), collate_fn=pad_collate, num_workers=0)
        for split, dataset in datasets.items()
    }
    model = DepthwiseStudent(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    class_weights = _class_weights(datasets["train"], device)
    run_config = {
        "label_to_index": {"non_covid": 0, "covid_positive": 1},
        "source_manifest": str(args.source_manifest),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "student_params": count_parameters(model),
        "split_sizes": {split: len(dataset) for split, dataset in datasets.items()},
    }
    best_metric = -1.0
    best_path = args.out / "checkpoints/student_best.pt"
    history = []
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train = _train_epoch(model, loaders["train"], optimizer, device, class_weights)
        val_eval = _evaluate(model, loaders["val"], device)
        val = val_eval["metrics"]
        entry = {"epoch": epoch, "train": train, "val": val, "seconds": time.time() - started}
        history.append(entry)
        if val["auroc"] > best_metric:
            best_metric = val["auroc"]
            _save_checkpoint(best_path, model, run_config["label_to_index"], run_config, entry)
        print(json.dumps({"epoch": epoch, "train": train, "val": val}, indent=2), flush=True)
    payload = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    source_test = _evaluate(model, loaders["test"], device)["metrics"]

    external = {}
    for name, manifest in (("coughvid", args.coughvid_manifest), ("tos", args.tos_manifest)):
        rows_by_id = _manifest_by_id(manifest)
        ds = BinaryCoughDataset(
            manifest,
            args.root,
            config,
            splits={"test", "val", "train", "external"},
            max_duration_sec=args.max_duration_sec,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=0)
        result = _evaluate(model, loader, device, rows_by_id)
        _write_predictions(args.out / f"{name}_binary_predictions.csv", result["predictions"])
        external[name] = {
            "clip": result["metrics"],
            "subject": _subject_metrics(result["predictions"]),
        }
    summary = {
        "config": run_config,
        "history": history,
        "best_val": max(history, key=lambda item: item["val"]["auroc"]),
        "source_test": source_test,
        "external": external,
        "artifacts": {"student_best": str(best_path)},
    }
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
