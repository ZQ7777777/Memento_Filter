#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_lnai_style() -> None:
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = csv_path.parent
    df = pd.read_csv(csv_path).sort_values("x")
    setup_lnai_style()

    # query_qps_x
    fig1, axes = plt.subplots(3, 1, figsize=(3.35, 5.8), sharex=True)
    metrics = [
        ("point", "R=1"),
        ("short", "R=32"),
        ("long", "R=1024"),
    ]
    for ax, (name, ttl) in zip(axes, metrics):
        ax.plot(df["x"], df[f"start_{name}_qps"], label="x-hot start")
        ax.plot(df["x"], df[f"contains_{name}_qps"], label="x-hot contains")
        ax.axhline(float(df[f"random_start_{name}_qps"].iloc[0]), color="tab:red", ls="--", label="random start")
        ax.axhline(float(df[f"random_contains_{name}_qps"].iloc[0]), color="tab:orange", ls=":", label="random contains")
        ax.set_ylabel("qps")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", ncol=2)
    axes[-1].set_xlabel("x")
    fig1.tight_layout()
    out1 = out_dir / "query_qps_x.png"
    fig1.savefig(out1, bbox_inches="tight")

    # query_drop_x protocol compare
    fig2, axes2 = plt.subplots(2, 1, figsize=(3.35, 4.1), sharex=True)
    for ax, (name, ttl) in zip(axes2, [("short", "R=32"), ("long", "R=1024")]):
        drop_start = (df[f"random_start_{name}_qps"] - df[f"start_{name}_qps"]) / df[f"random_start_{name}_qps"]
        drop_contains = (df[f"random_contains_{name}_qps"] - df[f"contains_{name}_qps"]) / df[f"random_contains_{name}_qps"]
        ax.plot(df["x"], drop_start, label="start drop")
        ax.plot(df["x"], drop_contains, label="contains drop")
        ax.axhline(0.0, color="gray", ls="--", lw=0.8)
        ax.set_ylabel("drop")
        ax.set_title(ttl)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes2[-1].set_xlabel("x")
    fig2.tight_layout()
    out2 = out_dir / "query_drop_x.png"
    fig2.savefig(out2, bbox_inches="tight")

    print(out1)
    print(out2)


if __name__ == "__main__":
    main()
