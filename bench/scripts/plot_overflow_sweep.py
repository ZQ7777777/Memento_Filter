#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def latest_file(input_dir: Path, pattern: str) -> Path:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No file matches pattern '{pattern}' in {input_dir}")
    return files[-1]


def load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"CSV is empty: {path}")
    return df


def savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_threshold_first_overflow(th_df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    for mode, g in th_df.groupby("mode"):
        g = g.sort_values("requested_inserts")
        y = g["first_overflow_insert_idx"].replace(-1, np.nan)
        plt.plot(g["requested_inserts"], y, marker="o", linewidth=1.2, label=mode)
    plt.xlabel("requested_inserts")
    plt.ylabel("first_overflow_insert_idx")
    plt.title("First overflow position vs requested inserts")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig(out_dir / "fig_overflow_threshold_first_idx.png")


def plot_threshold_saturated_ratio(th_df: pd.DataFrame, out_dir: Path) -> None:
    plt.figure(figsize=(7, 4))
    for mode, g in th_df.groupby("mode"):
        g = g.sort_values("requested_inserts")
        plt.plot(g["requested_inserts"], g["final_saturated_ratio"], marker="s", linewidth=1.2, label=mode)
    plt.xlabel("requested_inserts")
    plt.ylabel("final_saturated_ratio")
    plt.title("Saturated block ratio vs requested inserts")
    plt.grid(alpha=0.3)
    plt.legend()
    savefig(out_dir / "fig_overflow_threshold_saturated_ratio.png")


def plot_matrix_first_overflow(matrix_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    modes = ["single-prefix", "dual-hot"]
    series_col = "bpk" if "bpk" in matrix_df.columns else "fingerprint_bits"
    series_label = "bpk" if series_col == "bpk" else "fp_bits"

    for i, mode in enumerate(modes):
        ax = axes[i]
        g = matrix_df[matrix_df["mode"] == mode].copy()
        if g.empty:
            ax.set_visible(False)
            continue
        g = g.groupby(["memento_bits", series_col], as_index=False)["first_overflow_insert_idx"].mean()
        for s, gb in g.groupby(series_col):
            gb = gb.sort_values("memento_bits")
            y = gb["first_overflow_insert_idx"].replace(-1, np.nan)
            ax.plot(gb["memento_bits"], y, marker="o", linewidth=1.2, label=f"{series_label}={s}")
        ax.set_title(mode)
        ax.set_xlabel("memento_bits")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("first_overflow_insert_idx")
    axes[1].legend()
    plt.suptitle("First overflow index vs memento bits (matrix sweep)")
    savefig(out_dir / "fig_overflow_matrix_first_idx.png")


def plot_matrix_query_time(matrix_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    modes = ["single-prefix", "dual-hot"]
    series_col = "bpk" if "bpk" in matrix_df.columns else "fingerprint_bits"
    series_label = "bpk" if series_col == "bpk" else "fp_bits"

    for i, mode in enumerate(modes):
        ax = axes[i]
        g = matrix_df[matrix_df["mode"] == mode].copy()
        if g.empty:
            ax.set_visible(False)
            continue
        g = g.groupby(["memento_bits", series_col], as_index=False)["point_query_ms"].mean()
        for s, gb in g.groupby(series_col):
            gb = gb.sort_values("memento_bits")
            ax.plot(gb["memento_bits"], gb["point_query_ms"], marker="^", linewidth=1.2, label=f"{series_label}={s}")
        ax.set_title(mode)
        ax.set_xlabel("memento_bits")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("point_query_ms")
    axes[1].legend()
    plt.suptitle("Point query time vs memento bits (matrix sweep)")
    savefig(out_dir / "fig_overflow_matrix_point_query_ms.png")


def plot_xhot_first_overflow(xhot_df: pd.DataFrame, out_dir: Path) -> None:
    if xhot_df.empty:
        return

    x_col = "requested_hot_slots" if "requested_hot_slots" in xhot_df.columns else None
    if x_col is None:
        return

    g = xhot_df.groupby(x_col, as_index=False)["first_overflow_insert_idx"].mean().sort_values(x_col)
    y = g["first_overflow_insert_idx"].replace(-1, np.nan)

    plt.figure(figsize=(7, 4))
    plt.plot(g[x_col], y, marker="o", linewidth=1.4)
    plt.xlabel("x (contiguous hot slots)")
    plt.ylabel("first_overflow_insert_idx")
    plt.title("First overflow index vs x-hot width")
    plt.grid(alpha=0.3)
    savefig(out_dir / "fig_overflow_xhot_first_idx.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot figures from overflow sweep CSV outputs")
    parser.add_argument("--input-dir", default="paper_results/results/overflow_sweep", help="Directory containing overflow CSV files")
    parser.add_argument("--matrix-csv", default="", help="Path to matrix csv (optional, auto-detect latest)")
    parser.add_argument("--threshold-csv", default="", help="Path to threshold csv (optional, auto-detect latest)")
    parser.add_argument("--xhot-csv", default="", help="Path to x-hot csv (optional, auto-detect latest if available)")
    parser.add_argument("--out-dir", default="", help="Output directory for figures (default: input-dir)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    matrix_csv = Path(args.matrix_csv) if args.matrix_csv else latest_file(input_dir, "overflow_matrix_*.csv")
    threshold_csv = Path(args.threshold_csv) if args.threshold_csv else latest_file(input_dir, "overflow_threshold_m8_*.csv")
    xhot_csv = Path(args.xhot_csv) if args.xhot_csv else None
    if xhot_csv is None:
        xhot_candidates = sorted(input_dir.glob("overflow_xhot_m8_*.csv"))
        if xhot_candidates:
            xhot_csv = xhot_candidates[-1]
    out_dir = Path(args.out_dir) if args.out_dir else input_dir

    matrix_df = load_csv(matrix_csv)
    threshold_df = load_csv(threshold_csv)
    xhot_df = load_csv(xhot_csv) if xhot_csv else pd.DataFrame()

    plot_threshold_first_overflow(threshold_df, out_dir)
    plot_threshold_saturated_ratio(threshold_df, out_dir)
    plot_matrix_first_overflow(matrix_df, out_dir)
    plot_matrix_query_time(matrix_df, out_dir)
    plot_xhot_first_overflow(xhot_df, out_dir)

    print(f"[+] matrix_csv: {matrix_csv}")
    print(f"[+] threshold_csv: {threshold_csv}")
    if xhot_csv:
        print(f"[+] xhot_csv: {xhot_csv}")
    print(f"[+] figures written to: {out_dir}")


if __name__ == "__main__":
    main()
