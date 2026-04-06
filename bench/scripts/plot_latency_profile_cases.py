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
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def parse_tag(tag: str) -> tuple[str, int]:
    # tags: xhot_x1, random_x64, zipf_x1024
    p, x = tag.split("_x")
    return p, int(x)


def add_meta(df: pd.DataFrame) -> pd.DataFrame:
    m = df.copy()
    mode_type = []
    xvals = []
    for t in m["tag"]:
        md, xv = parse_tag(str(t))
        mode_type.append(md)
        xvals.append(xv)
    m["mode_type"] = mode_type
    m["x"] = xvals
    return m


def plot_box(samples: pd.DataFrame, segment: str, out: Path, title: str) -> None:
    xs = [1, 2, 3, 64, 1024]
    modes = ["xhot", "random", "zipf"]

    fig, ax = plt.subplots(figsize=(6.8, 2.5))
    data = []
    labels = []

    for x in xs:
        for m in modes:
            g = samples[(samples["segment"] == segment) & (samples["x"] == x) & (samples["mode_type"] == m)]
            data.append(g["latency_ns"].astype(float).values)
            labels.append(f"{x}\n{m}")

    ax.boxplot(data, showfliers=False, widths=0.55)
    ax.set_yscale("log")
    ax.set_ylabel("latency (ns, log)")
    ax.set_title(title)
    ax.set_xticklabels(labels, rotation=0)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-csv", required=True)
    ap.add_argument("--samples-csv", required=True)
    args = ap.parse_args()

    setup_style()

    summary = add_meta(pd.read_csv(args.summary_csv))
    samples = add_meta(pd.read_csv(args.samples_csv))

    out_dir = Path(args.samples_csv).parent

    # Insert segmented means by x
    fig1, axes = plt.subplots(3, 1, figsize=(3.35, 5.6), sharex=True)
    segs = ["insert_prepare", "insert_core", "insert_post"]
    for ax, seg in zip(axes, segs):
        for m, c in [("xhot", "tab:blue"), ("random", "tab:red"), ("zipf", "tab:green")]:
            g = summary[(summary["segment"] == seg) & (summary["mode_type"] == m)].sort_values("x")
            ax.plot(g["x"], g["mean_ns"], label=m, color=c)
        ax.set_ylabel("mean ns")
        ax.set_title(seg)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("x")
    fig1.tight_layout()
    fig1.savefig(out_dir / "latency_insert_segments.pdf", bbox_inches="tight")

    # Query segmented means (point/short/long core)
    fig2, axes2 = plt.subplots(3, 1, figsize=(3.35, 5.6), sharex=True)
    segs2 = ["point_core", "short_core", "long_core"]
    for ax, seg in zip(axes2, segs2):
        for m, c in [("xhot", "tab:blue"), ("random", "tab:red"), ("zipf", "tab:green")]:
            g = summary[(summary["segment"] == seg) & (summary["mode_type"] == m)].sort_values("x")
            ax.plot(g["x"], g["mean_ns"], label=m, color=c)
        ax.set_ylabel("mean ns")
        ax.set_title(seg)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes2[-1].set_xlabel("x")
    fig2.tight_layout()
    fig2.savefig(out_dir / "latency_query_segments.pdf", bbox_inches="tight")

    # Distribution boxplots for core segments
    plot_box(samples, "insert_core", out_dir / "latency_dist_insert_core.pdf", "Insert core latency distribution")
    plot_box(samples, "point_core", out_dir / "latency_dist_point_core.pdf", "Point core latency distribution")
    plot_box(samples, "short_core", out_dir / "latency_dist_short_core.pdf", "Short-range core latency distribution")
    plot_box(samples, "long_core", out_dir / "latency_dist_long_core.pdf", "Long-range core latency distribution")

    print(out_dir / "latency_insert_segments.pdf")
    print(out_dir / "latency_query_segments.pdf")
    print(out_dir / "latency_dist_insert_core.pdf")
    print(out_dir / "latency_dist_point_core.pdf")
    print(out_dir / "latency_dist_short_core.pdf")
    print(out_dir / "latency_dist_long_core.pdf")


if __name__ == "__main__":
    main()
