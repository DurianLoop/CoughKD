from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    rng = np.random.default_rng(23)
    semantic = pd.read_csv(
        ROOT / "runs/semantic_constrained_transfer_router_threshold1/semantic_constrained_transfer_router.csv"
    )[["target", "repeat", "delta_vs_source"]].rename(columns={"delta_vs_source": "semantic"})
    controller = pd.read_csv(
        ROOT
        / "runs/target_calibrated_transfer_controller_threshold1_confirm/target_calibrated_transfer_controller.csv"
    )[["target", "repeat", "delta_vs_source", "selected_strategy"]].rename(
        columns={"delta_vs_source": "controller"}
    )
    merged = semantic.merge(controller, on=["target", "repeat"], validate="one_to_one")
    rows = []
    for target, group in merged.groupby("target"):
        diffs = (group["semantic"] - group["controller"]).to_numpy(dtype=float)
        n = len(diffs)
        boot = np.empty(20000, dtype=float)
        for i in range(len(boot)):
            boot[i] = float(np.mean(diffs[rng.integers(0, n, size=n)]))
        rows.append(
            {
                "target": target,
                "n": n,
                "mean_diff": float(np.mean(diffs)),
                "ci_low": float(np.quantile(boot, 0.025)),
                "ci_high": float(np.quantile(boot, 0.975)),
                "p_boot_mean_le_0": float(np.mean(boot <= 0.0)),
                "paired_win_rate": float(np.mean(diffs > 0.0)),
                "paired_tie_rate": float(np.mean(np.isclose(diffs, 0.0, atol=1e-12))),
            }
        )
    result = pd.DataFrame(rows)
    out = ROOT / "runs/semantic_vs_controller_bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / "semantic_vs_controller_bootstrap.csv", index=False)
    lines = [
        "# Semantic Router vs Nested Controller Paired Bootstrap",
        "",
        "| target | n | mean diff | 95% CI | P(mean<=0) | win | tie |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in result.iterrows():
        lines.append(
            f"| {row['target']} | {int(row['n'])} | {100 * row['mean_diff']:.2f} | "
            f"[{100 * row['ci_low']:.2f}, {100 * row['ci_high']:.2f}] | "
            f"{row['p_boot_mean_le_0']:.4f} | {row['paired_win_rate']:.3f} | "
            f"{row['paired_tie_rate']:.3f} |"
        )
    (out / "SEMANTIC_VS_CONTROLLER_BOOTSTRAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(out / "SEMANTIC_VS_CONTROLLER_BOOTSTRAP.md")


if __name__ == "__main__":
    main()
