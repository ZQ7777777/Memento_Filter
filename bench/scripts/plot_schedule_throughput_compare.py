#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--prefix", default="schedule_compare")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = csv_path.parent
    df = pd.read_csv(csv_path)

    xhot = df[df["schedule"].isin(["round-robin", "burst", "zipf"])].copy()
    rb = df[df["schedule"] == "random"].iloc[0]

    # Query throughput comparison (3 panels)
    fig1, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    metrics = [
        ("point_pos_qps", "Point queries (R=1)"),
        ("short_range_pos_qps", "Short-range queries (R=32)"),
        ("long_range_pos_qps", "Long-range queries (R=1024)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        for sch in ["round-robin", "burst", "zipf"]:
            g = xhot[xhot["schedule"] == sch].sort_values("x")
            ax.plot(g["x"], g[col], lw=1.2, label=f"x-hot {sch}")
        ax.axhline(float(rb[col]), color="tab:red", ls="--", lw=1.2, label="random")
        ax.set_title(title)
        ax.set_ylabel("queries/sec")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("x (1..256 step1, then step8)")
    fig1.suptitle("Query throughput comparison: random vs x-hot variants")
    fig1.tight_layout()
    out1 = out_dir / f"fig_{args.prefix}_query_qps_compare.png"
    fig1.savefig(out1, dpi=220, bbox_inches="tight")

    # Insert throughput comparison
    fig2, ax2 = plt.subplots(figsize=(10, 4.5))
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch].sort_values("x")
        ax2.plot(g["x"], g["insert_qps"], lw=1.2, label=f"x-hot {sch}")
    ax2.axhline(float(rb["insert_qps"]), color="tab:red", ls="--", lw=1.2, label="random")
    ax2.set_xlabel("x (1..256 step1, then step8)")
    ax2.set_ylabel("insert/sec")
    ax2.set_title("Insert throughput comparison: random vs x-hot variants")
    ax2.grid(alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    out2 = out_dir / f"fig_{args.prefix}_insert_qps_compare.png"
    fig2.savefig(out2, dpi=220, bbox_inches="tight")

    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
