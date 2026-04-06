#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.0,
            "lines.markersize": 2.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ratio_label(r: float) -> str:
    return f"{int(round(r * 100))}%"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out-prefix", default="query_ratio")
    args = ap.parse_args()

    setup_style()

    csv_path = Path(args.csv)
    out_dir = csv_path.parent
    df = pd.read_csv(csv_path).sort_values(["attack_ratio", "x"])

    ratios = sorted(df["attack_ratio"].unique())

    fig, axes = plt.subplots(3, 1, figsize=(3.35, 6.0), sharex=True)
    specs = [
        ("point_pos_qps", "random_point_qps", "Point (R=1)"),
        ("short_range_pos_qps", "random_short_qps", "Short Range (R=32)"),
        ("long_range_pos_qps", "random_long_qps", "Long Range (R=1024)"),
    ]

    for ax, (col, rb_col, ttl) in zip(axes, specs):
        for r in ratios:
            g = df[df["attack_ratio"] == r]
            ax.plot(g["x"], g[col], label=ratio_label(r))
        ax.axhline(float(df[rb_col].iloc[0]), color="black", ls="--", label="random")
        ax.set_ylabel("qps")
        ax.set_title(ttl)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(handles, labels, ncol=4, loc="best")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    out1 = out_dir / f"{args.out_prefix}_qps_x.pdf"
    fig.savefig(out1, bbox_inches="tight")

    # Drop ratio plot for short/long
    fig2, axes2 = plt.subplots(2, 1, figsize=(3.35, 4.2), sharex=True)
    specs2 = [
        ("short_range_pos_qps", "random_short_qps", "Short Range Drop"),
        ("long_range_pos_qps", "random_long_qps", "Long Range Drop"),
    ]
    for ax, (col, rb_col, ttl) in zip(axes2, specs2):
        for r in ratios:
            g = df[df["attack_ratio"] == r]
            drop = (g[rb_col] - g[col]) / g[rb_col]
            ax.plot(g["x"], drop, label=ratio_label(r))
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("drop")
        ax.set_title(ttl)
        ax.grid(alpha=0.25)
    axes2[0].legend(ncol=4, loc="best")
    axes2[-1].set_xlabel("x")
    fig2.tight_layout()
    out2 = out_dir / f"{args.out_prefix}_drop_x.pdf"
    fig2.savefig(out2, bbox_inches="tight")

    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
