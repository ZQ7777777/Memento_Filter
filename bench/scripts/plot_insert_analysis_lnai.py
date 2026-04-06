#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_lnai_style() -> None:
    # LNAI single-column friendly defaults
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.1,
            "lines.markersize": 2.5,
            "figure.dpi": 220,
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    args = parser.parse_args()

    csv_path = Path(args.csv)
    out_dir = csv_path.parent

    setup_lnai_style()

    df = pd.read_csv(csv_path)
    xhot = df[df["schedule"].isin(["round-robin", "burst", "zipf"])].copy()
    rb = df[df["schedule"] == "random"].iloc[0]

    # Figure A: insert throughput vs x
    fig_a, ax_a = plt.subplots(figsize=(3.35, 2.3))
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch].sort_values("x")
        ax_a.plot(g["x"], g["insert_qps"], label=sch)
    ax_a.axhline(float(rb["insert_qps"]), color="tab:red", ls="--", label="random")
    ax_a.set_xlabel("x")
    ax_a.set_ylabel("insert/sec")
    ax_a.grid(alpha=0.3)
    ax_a.legend(loc="best", ncol=2)
    fig_a.tight_layout()
    out_a = out_dir / "insert_qps_x.png"
    fig_a.savefig(out_a, bbox_inches="tight")

    # Figure B: insert slowdown vs random
    fig_b, ax_b = plt.subplots(figsize=(3.35, 2.3))
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch].sort_values("x")
        slowdown = float(rb["insert_qps"]) / g["insert_qps"]
        ax_b.plot(g["x"], slowdown, label=sch)
    ax_b.set_xlabel("x")
    ax_b.set_ylabel("random/x-hot")
    ax_b.grid(alpha=0.3)
    ax_b.legend(loc="best", ncol=2)
    fig_b.tight_layout()
    out_b = out_dir / "insert_slowdown_x.png"
    fig_b.savefig(out_b, bbox_inches="tight")

    # Figure C: structural pressure vs x
    fig_c, axes_c = plt.subplots(2, 1, figsize=(3.35, 3.7), sharex=True)
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch].sort_values("x")
        axes_c[0].plot(g["x"], g["final_saturated_ratio"], label=sch)
        axes_c[1].plot(g["x"], g["first_overflow_insert_idx"], label=sch)
    axes_c[0].set_ylabel("sat. ratio")
    axes_c[1].set_ylabel("1st overflow idx")
    axes_c[1].set_xlabel("x")
    axes_c[0].grid(alpha=0.3)
    axes_c[1].grid(alpha=0.3)
    axes_c[0].legend(loc="best", ncol=2)
    fig_c.tight_layout()
    out_c = out_dir / "insert_structure_x.png"
    fig_c.savefig(out_c, bbox_inches="tight")

    # Figure D: insert qps vs saturation scatter
    fig_d, ax_d = plt.subplots(figsize=(3.35, 2.3))
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch]
        ax_d.scatter(g["final_saturated_ratio"], g["insert_qps"], s=8, alpha=0.7, label=sch)
    ax_d.set_xlabel("final_saturated_ratio")
    ax_d.set_ylabel("insert/sec")
    ax_d.grid(alpha=0.3)
    ax_d.legend(loc="best", ncol=2)
    fig_d.tight_layout()
    out_d = out_dir / "insert_qps_saturation.png"
    fig_d.savefig(out_d, bbox_inches="tight")

    print(out_a)
    print(out_b)
    print(out_c)
    print(out_d)


if __name__ == "__main__":
    main()
