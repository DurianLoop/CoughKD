from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from coughkd.config import RunConfig
from coughkd.torch_models import CoughManifestDataset, DepthwiseStudent, pad_collate


def _require_torch() -> Any:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import DataLoader

    return torch, nn, F, DataLoader


class _GradReverse:
    @staticmethod
    def apply(x: Any, weight: float) -> Any:
        torch, _, _, _ = _require_torch()

        class Fn(torch.autograd.Function):
            @staticmethod
            def forward(ctx: Any, inp: Any) -> Any:
                ctx.weight = weight
                return inp.view_as(inp)

            @staticmethod
            def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:
                return -ctx.weight * grad_output, None

        return Fn.apply(x)


def _load_student(checkpoint: Path, device: Any) -> tuple[Any, dict[str, int], dict[str, Any]]:
    torch, _, _, _ = _require_torch()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    label_to_index = {str(k): int(v) for k, v in payload["label_to_index"].items()}
    model = DepthwiseStudent(num_classes=len(label_to_index))
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    return model, label_to_index, payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "runs/coswara_cough_filtered_split/manifest_split.csv")
    parser.add_argument("--target-manifest", type=Path, default=ROOT / "manifests/coughvid_adapt_test.csv")
    parser.add_argument("--root", type=Path, default=ROOT.parent)
    parser.add_argument("--init-checkpoint", type=Path, default=ROOT / "runs/stage1_panns_response_seed7/checkpoints/student_best.pt")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/domain_adv_kd_student")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--domain-weight", type=float, default=0.2)
    parser.add_argument("--grl-weight", type=float, default=1.0)
    parser.add_argument("--max-source-records", type=int, default=None)
    parser.add_argument("--max-target-records", type=int, default=2000)
    args = parser.parse_args()

    torch, nn, F, DataLoader = _require_torch()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.out.mkdir(parents=True, exist_ok=True)
    student, label_to_index, init_payload = _load_student(args.init_checkpoint, device)
    domain_head = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)

    source_ds = CoughManifestDataset(
        args.source_manifest,
        args.root,
        RunConfig(experiment_name="domain_adv_source"),
        label_to_index=label_to_index,
        splits={"train"},
        max_records=args.max_source_records,
        max_duration_sec=4.0,
    )
    target_ds = CoughManifestDataset(
        args.target_manifest,
        args.root,
        RunConfig(experiment_name="domain_adv_target"),
        label_to_index=label_to_index,
        splits={"adapt"},
        max_records=args.max_target_records,
        max_duration_sec=4.0,
    )
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0)
    target_loader = DataLoader(target_ds, batch_size=args.batch_size, shuffle=True, collate_fn=pad_collate, num_workers=0)
    optimizer = torch.optim.AdamW(list(student.parameters()) + list(domain_head.parameters()), lr=args.lr)

    history = []
    for epoch in range(1, args.epochs + 1):
        student.train()
        domain_head.train()
        totals = {"ce": 0.0, "domain": 0.0, "total": 0.0, "items": 0}
        source_iter = itertools.cycle(source_loader)
        target_iter = itertools.cycle(target_loader)
        steps = max(len(source_loader), len(target_loader))
        for step in range(1, steps + 1):
            source_batch = next(source_iter)
            target_batch = next(target_iter)
            source_x = source_batch["features"].to(device)
            source_y = source_batch["label"].to(device)
            target_x = target_batch["features"].to(device)

            source_out = student(source_x)
            target_out = student(target_x)
            ce = F.cross_entropy(source_out["logits"], source_y)
            domain_emb = torch.cat([source_out["embedding"], target_out["embedding"]], dim=0)
            domain_y = torch.cat(
                [
                    torch.zeros(source_out["embedding"].shape[0], dtype=torch.long, device=device),
                    torch.ones(target_out["embedding"].shape[0], dtype=torch.long, device=device),
                ],
                dim=0,
            )
            domain_logits = domain_head(_GradReverse.apply(domain_emb, args.grl_weight))
            domain_loss = F.cross_entropy(domain_logits, domain_y)
            total = ce + args.domain_weight * domain_loss
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            optimizer.step()

            batch_items = int(source_y.shape[0])
            totals["ce"] += float(ce.detach().cpu()) * batch_items
            totals["domain"] += float(domain_loss.detach().cpu()) * batch_items
            totals["total"] += float(total.detach().cpu()) * batch_items
            totals["items"] += batch_items
            if step == 1 or step % 20 == 0:
                print(f"[epoch {epoch}] step {step}/{steps} ce={float(ce.detach().cpu()):.4f} domain={float(domain_loss.detach().cpu()):.4f}", flush=True)
        rec = {
            "epoch": epoch,
            "ce": totals["ce"] / max(1, totals["items"]),
            "domain": totals["domain"] / max(1, totals["items"]),
            "total": totals["total"] / max(1, totals["items"]),
        }
        history.append(rec)
        print(json.dumps(rec), flush=True)

    ckpt = {
        "model_state_dict": student.state_dict(),
        "label_to_index": label_to_index,
        "config": {
            "init_checkpoint": str(args.init_checkpoint),
            "source_manifest": str(args.source_manifest),
            "target_manifest": str(args.target_manifest),
            "domain_weight": args.domain_weight,
            "grl_weight": args.grl_weight,
            "epochs": args.epochs,
            "max_target_records": args.max_target_records,
        },
        "selection_record": {"history": history},
    }
    torch.save(ckpt, args.out / "student_domain_adv.pt")
    torch.save({"model_state_dict": domain_head.state_dict(), "history": history}, args.out / "domain_head.pt")
    (args.out / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print({"checkpoint": str(args.out / "student_domain_adv.pt"), "history": history})


if __name__ == "__main__":
    main()
