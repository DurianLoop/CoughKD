from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.config import RunConfig
from coughkd.metrics import average_precision, multiclass_ovr_auroc
from coughkd.torch_models import CoughManifestDataset, DepthwiseStudent, pad_collate


def _require_torch() -> Any:
    import torch
    from torch.utils.data import DataLoader

    return torch, DataLoader


def _metrics(true_indices: list[int], pred_indices: list[int], prob_rows: list[list[float]], index_to_label: dict[int, str]) -> dict[str, Any]:
    labels = [index_to_label[idx] for idx in true_indices]
    classes = [index_to_label[idx] for idx in sorted(index_to_label)]
    result: dict[str, Any] = {
        "num_examples": len(true_indices),
        "accuracy": sum(int(t == p) for t, p in zip(true_indices, pred_indices)) / max(1, len(true_indices)),
    }
    try:
        result.update(multiclass_ovr_auroc(labels, prob_rows, classes))
    except Exception as exc:
        result["macro_ovr_auroc_error"] = str(exc)
    auprc = {}
    for idx, class_name in enumerate(classes):
        binary = [1 if label == class_name else 0 for label in labels]
        if sum(binary) == 0:
            continue
        auprc[class_name] = average_precision(binary, [row[idx] for row in prob_rows])
    if auprc:
        result["macro_ovr_auprc"] = sum(auprc.values()) / len(auprc)
        for key, value in auprc.items():
            result[f"{key}_ovr_auprc"] = value
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(r"D:\CoughKD"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-duration-sec", type=float, default=4.0)
    parser.add_argument("--max-records", type=int, default=None)
    args = parser.parse_args()

    torch, DataLoader = _require_torch()
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(k): int(v) for k, v in payload["label_to_index"].items()}
    index_to_label = {idx: label for label, idx in label_to_index.items()}

    dataset = CoughManifestDataset(
        manifest=args.manifest,
        root=args.root,
        config=RunConfig(experiment_name="external_eval"),
        label_to_index=label_to_index,
        splits={"test", "val", "train", "external"},
        max_records=args.max_records,
        max_duration_sec=args.max_duration_sec,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_collate, num_workers=args.num_workers)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = DepthwiseStudent(num_classes=len(label_to_index))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    model.eval()

    true_indices: list[int] = []
    pred_indices: list[int] = []
    prob_rows: list[list[float]] = []
    rows: list[dict[str, Any]] = []
    classes = [index_to_label[idx] for idx in sorted(index_to_label)]
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            features = batch["features"].to(device)
            labels = batch["label"].to(device)
            out = model(features)
            probs = torch.softmax(out["logits"], dim=1)
            preds = probs.argmax(dim=1)
            for rid, true_idx, pred_idx, prob in zip(batch["recording_id"], labels.cpu().tolist(), preds.cpu().tolist(), probs.cpu().tolist()):
                true_indices.append(int(true_idx))
                pred_indices.append(int(pred_idx))
                prob_rows.append([float(item) for item in prob])
                row = {
                    "recording_id": rid,
                    "true_label": index_to_label[int(true_idx)],
                    "pred_label": index_to_label[int(pred_idx)],
                }
                row.update({f"prob_{label}": f"{float(prob[idx]):.8f}" for idx, label in enumerate(classes)})
                rows.append(row)
            if batch_idx == 1 or batch_idx % 20 == 0:
                print(f"[eval] batch {batch_idx}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = _metrics(true_indices, pred_indices, prob_rows, index_to_label)
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    with (args.out / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        fieldnames = ["recording_id", "true_label", "pred_label"] + [f"prob_{label}" for label in classes]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# External Evaluation", "", f"- Manifest: `{args.manifest}`", f"- Checkpoint: `{args.checkpoint}`", "", "```json", json.dumps(metrics, indent=2), "```", ""]
    (args.out / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
