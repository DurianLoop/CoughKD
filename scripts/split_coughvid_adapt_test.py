from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path(r"D:\CoughKD\AAAI\manifests\coughvid_external.csv"))
    parser.add_argument("--out", type=Path, default=Path(r"D:\CoughKD\AAAI\manifests\coughvid_adapt_test.csv"))
    parser.add_argument("--test-size", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    df = pd.read_csv(args.manifest).fillna("")
    train_idx, test_idx = train_test_split(
        df.index,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=df["label"],
    )
    out = df.copy()
    out.loc[train_idx, "split"] = "adapt"
    out.loc[test_idx, "split"] = "test"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(
        {
            "out": str(args.out),
            "split_counts": out["split"].value_counts().to_dict(),
            "adapt_labels": out[out["split"].eq("adapt")]["label"].value_counts().to_dict(),
            "test_labels": out[out["split"].eq("test")]["label"].value_counts().to_dict(),
        }
    )


if __name__ == "__main__":
    main()
