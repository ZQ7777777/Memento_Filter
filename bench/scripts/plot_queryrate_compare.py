#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def latest_file(input_dir: Path, pattern: str) -> Path:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matching {pattern} in {input_dir}")
    return files[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="paper_results/results/overflow_sweep")
    parser.add_argument("--csv", default="")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_path = Path(args.csv) if args.csv else latest_file(input_dir, "queryrate_fp13_m10_n5000_q20000_*.csv")
    out_path = Path(args.out) if args.out else input_dir / "fig_queryrate_fp13_m10_n5000_q20000_x1_256.png"

    df = pd.read_csv(csv_path)
    xhot = df[df["mode"] == "x-hot"].sort_values("requested_hot_slots")
    random_df = df[df["mode"] == "random"]

    random_point_pos = random_df["point_pos_qps"].mean() if not random_df.empty else None
    random_point_neg = random_df["point_neg_qps"].mean() if not random_df.empty else None
    random_short_pos = random_df["short_range_pos_qps"].mean() if not random_df.empty else None
    random_short_neg = random_df["short_range_neg_qps"].mean() if not random_df.empty else None
    random_long_pos = random_df["long_range_pos_qps"].mean() if not random_df.empty else None
    random_long_neg = random_df["long_range_neg_qps"].mean() if not random_df.empty else None

    fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    series = [
        ("point_pos_qps", "point_neg_qps", "Point queries (R=1)", random_point_pos, random_point_neg),
        ("short_range_pos_qps", "short_range_neg_qps", "Short-range queries (R=32)", random_short_pos, random_short_neg),
        ("long_range_pos_qps", "long_range_neg_qps", "Long-range queries (R=1024)", random_long_pos, random_long_neg),
    ]

    for ax, (pos_col, neg_col, title, pos_base, neg_base) in zip(axes, series):
        ax.plot(xhot["requested_hot_slots"], xhot[pos_col], lw=1.2, color="tab:blue", label="x-hot positive")
        ax.plot(xhot["requested_hot_slots"], xhot[neg_col], lw=1.2, color="tab:orange", label="x-hot negative")
        if pos_base is not None:
            ax.axhline(pos_base, color="tab:blue", linestyle="--", lw=1.2, label="random positive")
        if neg_base is not None:
            ax.axhline(neg_base, color="tab:orange", linestyle="--", lw=1.2, label="random negative")
        ax.set_ylabel("queries/sec")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()

    axes[-1].set_xlabel("x (number of contiguous hot buckets)")
    plt.suptitle("Query rate vs x-hot width (fp_bits=13, memento_bits=10, n=5000, q=20000/mode)")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")

    print(f"[+] csv: {csv_path}")
    print(f"[+] figure: {out_path}")


if __name__ == "__main__":
    main()
