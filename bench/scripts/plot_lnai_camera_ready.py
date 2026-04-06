#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def setup_lnai() -> None:
    # LNAI/LLNCS single-column friendly settings
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.05,
            "lines.markersize": 2.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_query_figures(query_start_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(query_start_csv)
    xhot = df[df["mode"] == "x-hot"].sort_values("requested_hot_slots")
    rb = df[df["mode"] == "random"].iloc[0]

    # Query throughput vs x (start-at-hit only)
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 5.9), sharex=True)
    metrics = [
        ("point_pos_qps", "Point Query (R=1)"),
        ("short_range_pos_qps", "Short Range Query (R=32)"),
        ("long_range_pos_qps", "Long Range Query (R=1024)"),
    ]
    for ax, (col, title) in zip(axes, metrics):
        ax.plot(xhot["requested_hot_slots"], xhot[col], label="x-hot")
        ax.axhline(float(rb[col]), color="tab:red", ls="--", label="random")
        ax.set_ylabel("qps")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best")
    axes[-1].set_xlabel("x")
    fig.tight_layout()
    fig.savefig(out_dir / "query_qps_x.pdf", bbox_inches="tight")

    # Saturation ratio vs x (query experiment)
    fig2, ax2 = plt.subplots(figsize=(3.35, 2.25))
    ax2.plot(xhot["requested_hot_slots"], xhot["final_saturated_ratio"], color="tab:purple")
    ax2.set_xlabel("x")
    ax2.set_ylabel("saturated ratio")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(out_dir / "query_saturation_x.pdf", bbox_inches="tight")


def save_insert_figures(schedule_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(schedule_csv)
    xhot = df[df["schedule"].isin(["round-robin", "burst", "zipf"])].copy()
    rb = df[df["schedule"] == "random"].iloc[0]

    # Insert throughput vs x
    fig, ax = plt.subplots(figsize=(3.35, 2.25))
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch].sort_values("x")
        ax.plot(g["x"], g["insert_qps"], label=sch)
    ax.axhline(float(rb["insert_qps"]), color="tab:red", ls="--", label="random")
    ax.set_xlabel("x")
    ax.set_ylabel("insert/sec")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", ncol=2)
    fig.tight_layout()
    fig.savefig(out_dir / "insert_qps_x.pdf", bbox_inches="tight")

    # Insert slowdown vs random
    fig2, ax2 = plt.subplots(figsize=(3.35, 2.25))
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch].sort_values("x")
        slowdown = float(rb["insert_qps"]) / g["insert_qps"]
        ax2.plot(g["x"], slowdown, label=sch)
    ax2.set_xlabel("x")
    ax2.set_ylabel("random/x-hot")
    ax2.grid(alpha=0.3)
    ax2.legend(loc="best", ncol=2)
    fig2.tight_layout()
    fig2.savefig(out_dir / "insert_slowdown_x.pdf", bbox_inches="tight")

    # Insert structure pressure vs x
    fig3, axes3 = plt.subplots(2, 1, figsize=(3.35, 3.6), sharex=True)
    for sch in ["round-robin", "burst", "zipf"]:
        g = xhot[xhot["schedule"] == sch].sort_values("x")
        axes3[0].plot(g["x"], g["final_saturated_ratio"], label=sch)
        axes3[1].plot(g["x"], g["first_overflow_insert_idx"], label=sch)
    axes3[0].set_ylabel("sat. ratio")
    axes3[1].set_ylabel("1st overflow idx")
    axes3[1].set_xlabel("x")
    axes3[0].grid(alpha=0.3)
    axes3[1].grid(alpha=0.3)
    axes3[0].legend(loc="best", ncol=2)
    fig3.tight_layout()
    fig3.savefig(out_dir / "insert_structure_x.pdf", bbox_inches="tight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-start-csv", default="paper_results/results/overflow_sweep/query_start.csv")
    parser.add_argument("--schedule-csv", default="paper_results/results/overflow_sweep/schedule.csv")
    parser.add_argument("--out-dir", default="paper_results/results/overflow_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    setup_lnai()
    save_query_figures(Path(args.query_start_csv), out_dir)
    save_insert_figures(Path(args.schedule_csv), out_dir)

    for name in [
        "query_qps_x.pdf",
        "query_saturation_x.pdf",
        "insert_qps_x.pdf",
        "insert_slowdown_x.pdf",
        "insert_structure_x.pdf",
    ]:
        print(out_dir / name)


if __name__ == "__main__":
    main()
