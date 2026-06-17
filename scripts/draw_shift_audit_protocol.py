"""Draw a simple CoughKD-ShiftAudit protocol diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "kd_failure_analysis" / "figures"


def box(ax, xy, wh, text, fc, ec="#263238"):
    patch = FancyBboxPatch(
        xy,
        wh[0],
        wh[1],
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1.1,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(patch)
    ax.text(xy[0] + wh[0] / 2, xy[1] + wh[1] / 2, text, ha="center", va="center", fontsize=9)
    return patch


def arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=12,
            linewidth=1.1,
            color="#37474F",
            shrinkA=4,
            shrinkB=4,
        )
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.4, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, (0.04, 0.64), (0.16, 0.16), "Coswara\nsource cough split", "#DDECCF")
    box(ax, (0.04, 0.28), (0.16, 0.16), "COUGHVID\nexternal target", "#F8E3C5")

    box(ax, (0.28, 0.64), (0.18, 0.16), "PANNs CNN14\nteacher", "#D3E5F3")
    box(ax, (0.28, 0.38), (0.18, 0.20), "Ultra-light\nDepthwise student\n20.7K params", "#EADCF1")

    box(ax, (0.54, 0.58), (0.18, 0.26), "Training variants\nCE-only\nVanilla KD\nTCD / gated TCD\nShortcut / disagreement\nProbe-adversarial", "#FFF1B8")

    box(ax, (0.54, 0.25), (0.18, 0.20), "Target stress tests\n25 metadata slices\nBootstrap subsets\nLOSO guard audit", "#FFD6D6")

    box(ax, (0.79, 0.58), (0.17, 0.26), "Evidence outputs\nExternal AUROC/AUPRC\nECE / Brier / NLL\nDomain/task probes\nParams / latency", "#D7F0EE")
    box(ax, (0.79, 0.22), (0.17, 0.22), "ShiftAudit readout\nWeak aggregate gains\nSlice-dependent wins\nLabel-free guard fails\nMetric disagreement", "#ECEFF1")

    arrow(ax, (0.20, 0.72), (0.28, 0.72))
    arrow(ax, (0.20, 0.72), (0.28, 0.48))
    arrow(ax, (0.46, 0.48), (0.54, 0.70))
    arrow(ax, (0.20, 0.36), (0.54, 0.35))
    arrow(ax, (0.72, 0.70), (0.79, 0.70))
    arrow(ax, (0.72, 0.35), (0.79, 0.33))
    arrow(ax, (0.875, 0.58), (0.875, 0.44))

    ax.text(
        0.5,
        0.94,
        "CoughKD-ShiftAudit Protocol",
        ha="center",
        va="center",
        fontsize=15,
        weight="bold",
    )
    ax.text(
        0.5,
        0.90,
        "Deployment-oriented failure cartography for ultra-light cough audio distillation under dataset shift",
        ha="center",
        va="center",
        fontsize=9,
        color="#455A64",
    )
    fig.tight_layout()
    fig.savefig(OUT / "shift_audit_protocol.png", dpi=220)
    fig.savefig(OUT / "shift_audit_protocol.pdf")
    plt.close(fig)
    print(OUT / "shift_audit_protocol.pdf")


if __name__ == "__main__":
    main()
