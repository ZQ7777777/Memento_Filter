#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import pandas as pd


QUERY_LEGEND_FONT_SIZE = 7
QUERY_Saturation_LEGEND_FONT_SIZE = 5.5
INSERT_LEGEND_FONT_SIZE = 5.5
TITLE_FONT_SIZE = 9.5
YLABEL_FONT_SIZE = 9.5
XLABEL_FONT_SIZE = 9.5

# Local line widths for insert exp1 and query_exp12 first-row (x-sweep) only.
QUERY_EXP12_ROW1_LINEWIDTH = 0.2
INSERT_X_SWEEP_LINEWIDTH = 0.65

LINE_STYLES = {
    0.01: {"color": "#1f77b4", "linestyle": "-", "marker": "o", "label": "1%"},
    0.05: {"color": "#ff7f0e", "linestyle": "--", "marker": "s", "label": "5%"},
    0.10: {"color": "#2ca02c", "linestyle": "-.", "marker": "^", "label": "10%"},
    1.00: {"color": "#d62728", "linestyle": ":", "marker": "D", "label": "100%"},
}

METRICS = [
    ("point_pos_qps", "Point Queries"),
    ("short_range_pos_qps", "Short Range Queries"),
    ("long_range_pos_qps", "Long Range Queries"),
]

OUTPUT_FORMATS = ["pdf", "svg"]
USE_LOG_Y_FOR_QUERY_THROUGHPUT = True


def convert_svg_to_emf(svg_path: Path, emf_path: Path) -> bool:
    inkscape = shutil.which("inkscape")
    if inkscape:
        cmd = [inkscape, str(svg_path), "--export-type=emf", f"--export-filename={emf_path}"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0 and emf_path.exists():
            return True
        print(f"[warn] inkscape emf export failed for {svg_path}: {res.stderr.strip()}")
        return False

    print("[warn] emf requested but inkscape is not installed; skipping emf export")
    return False


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.titlesize": TITLE_FONT_SIZE,
            "axes.labelsize": YLABEL_FONT_SIZE,
            "legend.fontsize": QUERY_LEGEND_FONT_SIZE,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "lines.linewidth": 0.9,
            "lines.markersize": 4,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            # Keep text as text in SVG so Word preserves vector quality better.
            "svg.fonttype": "none",
        }
    )


def save_figure_multi(fig: plt.Figure, fig_dir: Path, base_name: str) -> None:
    non_emf_formats = [fmt for fmt in OUTPUT_FORMATS if fmt != "emf"]
    for fmt in non_emf_formats:
        out = fig_dir / f"{base_name}.{fmt}"
        fig.savefig(out, bbox_inches="tight", pad_inches=0.01)
        print(out)

    if "emf" in OUTPUT_FORMATS:
        svg_path = fig_dir / f"{base_name}.svg"
        temp_svg = False
        if not svg_path.exists():
            fig.savefig(svg_path, bbox_inches="tight", pad_inches=0.01)
            temp_svg = True
        emf_path = fig_dir / f"{base_name}.emf"
        if convert_svg_to_emf(svg_path, emf_path):
            print(emf_path)
        if temp_svg:
            svg_path.unlink(missing_ok=True)

    plt.close(fig)


def resolve_uplift_baseline(base_dir: Path, baseline_dir: Path | None, baseline_filename: str, fallback_filename: str) -> Path:
    local_path = base_dir / fallback_filename
    if local_path.exists():
        return local_path
    if baseline_dir is not None:
        candidate = baseline_dir / baseline_filename
        if candidate.exists():
            return candidate
    return local_path


def power_two_markevery(xs: list[int]) -> list[int]:
    idx = []
    for i, x in enumerate(xs):
        if x > 0 and (x & (x - 1)) == 0 and (x == 1 or x > 64):
            idx.append(i)
    return idx


def format_exp_tick(exp: int) -> str:
    return rf"$2^{{{exp}}}$"


def filter_ok(df: pd.DataFrame) -> pd.DataFrame:
    if "status" not in df.columns:
        return df.copy()
    return df[df["status"] == "ok"].copy()


def metric_columns(df: pd.DataFrame, metric: str) -> tuple[str, str | None]:
    mean_col = f"{metric}_mean" if f"{metric}_mean" in df.columns else metric
    ci_col = f"{metric}_ci95" if f"{metric}_ci95" in df.columns else None
    return mean_col, ci_col


def metric_values(
    df: pd.DataFrame,
    metric: str,
    scale: float = 1.0,
    *,
    use_ci: bool = True,
) -> tuple[list[float], list[float] | None]:
    mean_col, ci_col = metric_columns(df, metric)
    y_vals = (df[mean_col].astype(float) / scale).tolist()
    y_err = None
    if use_ci and ci_col is not None and ci_col in df.columns:
        y_err = (df[ci_col].astype(float) / scale).tolist()
        if all(pd.isna(y_err)):
            y_err = None
    return y_vals, y_err


def compute_uplift_frame(def_df: pd.DataFrame, base_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    def_col, _ = metric_columns(def_df, metric)
    base_col, _ = metric_columns(base_df, metric)
    if def_col not in def_df.columns or base_col not in base_df.columns:
        return pd.DataFrame()
    join_cols = [c for c in ["m_exp", "attack_ratio"] if c in def_df.columns and c in base_df.columns]
    if not join_cols:
        return pd.DataFrame()
    merged = def_df[join_cols + [def_col]].merge(
        base_df[join_cols + [base_col]],
        on=join_cols,
        how="inner",
        suffixes=("_def", "_base"),
    )
    if merged.empty:
        return merged
    merged["uplift_pct"] = (merged[f"{def_col}_def"] / merged[f"{base_col}_base"] - 1.0) * 100.0
    return merged


def baseline_values(
    df: pd.DataFrame,
    metric: str,
    scale: float = 1.0,
    *,
    use_ci: bool = True,
) -> tuple[float | None, float | None]:
    if df.empty:
        return None, None
    mean_col, ci_col = metric_columns(df, metric)
    baseline = float(df[mean_col].iloc[0]) / scale
    baseline_ci = None
    if use_ci and ci_col is not None and ci_col in df.columns:
        baseline_ci = float(df[ci_col].iloc[0]) / scale
    return baseline, baseline_ci


def add_ratio_style(
    ax: plt.Axes,
    x_vals: list[int],
    y_vals: list[float],
    y_err: list[float] | None,
    ratio: float,
    markevery: list[int],
    linewidth: float | None = None,
) -> None:
    style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
    if y_err is None:
        if linewidth is None:
            ax.plot(x_vals, y_vals, markevery=markevery, **style)
        else:
            ax.plot(x_vals, y_vals, markevery=markevery, linewidth=linewidth, **style)
    else:
        if linewidth is not None:
            style = {**style, "linewidth": linewidth}
        ax.errorbar(x_vals, y_vals, yerr=y_err, capsize=2.0, markevery=markevery, **style)


def add_sq_style(
    ax: plt.Axes,
    x_vals: list[int],
    y_vals: list[float],
    y_err: list[float] | None,
    ratio: float,
    linewidth: float | None = None,
) -> None:
    base = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
    style = {
        "color": base["color"],
        "linestyle": ":",
        "marker": "v",
        "label": f"sq {base['label']}",
    }
    if linewidth is not None:
        style["linewidth"] = linewidth
    if y_err is None:
        ax.plot(x_vals, y_vals, **style)
    else:
        ax.errorbar(x_vals, y_vals, yerr=y_err, capsize=2.0, **style)


def draw_baseline_band(ax: plt.Axes, xs: list[int], baseline: float, ci: float | None) -> None:
    if ci is None or pd.isna(ci) or ci <= 0 or not xs:
        return
    y_low = baseline - ci
    y_high = baseline + ci
    if ax.get_yscale() == "log" and y_low <= 0:
        y_low = max(baseline * 1e-6, baseline * 0.5)
    ax.fill_between([min(xs), max(xs)], [y_low, y_low], [y_high, y_high], color="black", alpha=0.08, linewidth=0.0)


def tune_query_axis_ylim(ax: plt.Axes, y_series: list[list[float]], baseline: float) -> bool:
    vals = [v for seq in y_series for v in seq if pd.notna(v)]
    if not vals:
        return False
    y_min = min(vals)
    y_max = max(vals)
    if y_min == y_max:
        y_min *= 0.95
        y_max *= 1.05

    # If baseline is much higher, keep it at the top edge so x-hot curves are readable.
    if baseline > y_max * 1.25:
        bottom = max(0.0, y_min * 0.97)
        top = y_max * 1.08
        ax.set_ylim(bottom, top)
        return True
    else:
        low = min(y_min, baseline) * 0.97
        high = max(y_max, baseline) * 1.05
        ax.set_ylim(max(0.0, low), high)
        return False


def draw_baseline_reference(ax: plt.Axes, baseline: float, clipped: bool) -> None:
    if clipped:
        # Keep baseline reference visible when real value is outside the shown y-range.
        y0, y1 = ax.get_ylim()
        span = max(y1 - y0, 1e-9)
        y_top = y1 - 0.03 * span
        ax.axhline(y_top, color="black", linestyle="--", linewidth=0.9)
        ax.text(0.98, 0.98, f"baseline={baseline:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7)

        # Visual y-axis break marker to indicate omitted coordinate range.
        ax.plot([-0.015, 0.005], [0.93, 0.97], transform=ax.transAxes, color="black", linewidth=0.9, clip_on=False)
        ax.plot([0.005, 0.025], [0.93, 0.97], transform=ax.transAxes, color="black", linewidth=0.9, clip_on=False)

        # Keep lower y ticks for x-hot curves and a top tick for baseline coordinate.
        ticks = [float(t) for t in ax.get_yticks()]
        cutoff = y_top - 0.15 * span
        lower_ticks = [t for t in ticks if t <= cutoff]
        if len(lower_ticks) > 4:
            step = max(1, len(lower_ticks) // 4)
            lower_ticks = lower_ticks[::step][:4]
        tick_vals = lower_ticks + [y_top]
        tick_labels = [f"{t:g}" for t in lower_ticks] + [f"{baseline:.2f}"]
        ax.set_yticks(tick_vals)
        ax.set_yticklabels(tick_labels)
    else:
        ax.axhline(baseline, color="black", linestyle="--", linewidth=0.9)


def _set_inner_y_tick_spacing(ax: plt.Axes, is_left_col: bool) -> None:
    # Tighten y-tick label spacing on inner columns to avoid overlap with the previous subplot frame.
    ax.tick_params(axis="y", pad=(2 if is_left_col else 0.5))


def _mark_baseline_at_x(ax: plt.Axes, x_pos: float, y_val: float) -> None:
    # Mark baseline with an in-plot x marker (same visual language as the baseline legend marker).
    ax.plot([x_pos], [y_val], color="black", marker="x", linestyle="None", markersize=4.0)


def plot_query_exp1(base_dir: Path, fig_dir: Path) -> None:
    if not (base_dir / "query_exp1_x_sweep.csv").exists() or not (base_dir / "query_exp1_random_baseline.csv").exists():
        print(f"[skip] missing query_exp1 csvs under {base_dir}")
        return
    df = filter_ok(pd.read_csv(base_dir / "query_exp1_x_sweep.csv"))
    rb = filter_ok(pd.read_csv(base_dir / "query_exp1_random_baseline.csv"))

    ratios = sorted(df["attack_ratio"].unique())
    xs = sorted(df["x"].unique())
    markevery = power_two_markevery(xs)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.3), sharex=True)

    for col_idx, (metric, title) in enumerate(METRICS):
        ax_top = axes[col_idx]

        rb_qps, rb_ci = baseline_values(rb, metric, scale=1e7)
        rb_plot_qps = rb_qps if rb_qps is not None else 0.0
        y_series_col: list[list[float]] = []

        for ratio in ratios:
            g = df[df["attack_ratio"] == ratio].sort_values("x")
            x_vals = g["x"].astype(int).tolist()
            y_vals, y_err = metric_values(g, metric, scale=1e7)
            y_series_col.append(y_vals)

            add_ratio_style(ax_top, x_vals, y_vals, y_err, float(ratio), markevery)

        if USE_LOG_Y_FOR_QUERY_THROUGHPUT:
            ax_top.set_yscale("log")
            if rb_qps is not None:
                ax_top.axhline(rb_plot_qps, color="black", linestyle="--", linewidth=0.9)
                draw_baseline_band(ax_top, xs, rb_plot_qps, rb_ci)
        else:
            baseline_clipped = tune_query_axis_ylim(ax_top, y_series_col, rb_plot_qps)
            draw_baseline_reference(ax_top, rb_plot_qps, baseline_clipped)
        ax_top.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax_top.grid(alpha=0.25)
        if col_idx > 0:
            try:
                ax_top.ticklabel_format(style="plain", axis="y", useOffset=False)
            except AttributeError:
                pass

        if col_idx == 0:
            ax_top.set_ylabel("Query Throughput\n(queries/s)", fontsize=YLABEL_FONT_SIZE)
            ax_top.text(0.0, 1.02, "1e7", transform=ax_top.transAxes, ha="left", va="bottom", fontsize=8)
        _set_inner_y_tick_spacing(ax_top, is_left_col=(col_idx == 0))

        ax_top.set_xlabel("x", fontsize=XLABEL_FONT_SIZE)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(
            mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"x-hot {s['label']}")
        )
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="baseline"))

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.86, 0.5),
        frameon=True,
        facecolor="none",
        edgecolor="#c0c0c0",
        framealpha=1.0,
        fontsize=QUERY_LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(wspace=0.22, right=0.84)

    save_figure_multi(fig, fig_dir, "query_exp1_throughput_and_ratio_vs_x")


def plot_query_exp1_saturation(base_dir: Path, fig_dir: Path) -> None:
    if not (base_dir / "query_exp1_x_sweep.csv").exists() or not (base_dir / "query_exp1_random_baseline.csv").exists():
        print(f"[skip] missing query_exp1 csvs under {base_dir}")
        return
    df = filter_ok(pd.read_csv(base_dir / "query_exp1_x_sweep.csv"))
    rb = filter_ok(pd.read_csv(base_dir / "query_exp1_random_baseline.csv"))

    ratios = sorted(df["attack_ratio"].unique())
    xs = sorted(df["x"].unique())
    markevery = power_two_markevery(xs)

    fig, ax = plt.subplots(figsize=(3.5, 2))
    for ratio in ratios:
        g = df[df["attack_ratio"] == ratio].sort_values("x")
        x_vals = g["x"].astype(int).tolist()
        y_vals, y_err = metric_values(g, "final_saturated_ratio")
        add_ratio_style(ax, x_vals, y_vals, y_err, float(ratio), markevery)

    if not rb.empty:
        rb_val, rb_ci = baseline_values(rb, "final_saturated_ratio")
        if rb_val is not None:
            ax.axhline(rb_val, color="black", linestyle="--", linewidth=0.8)
            draw_baseline_band(ax, xs, rb_val, rb_ci)

    ax.set_xlabel("x", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel("Saturated Block Ratio", fontsize=YLABEL_FONT_SIZE)
    ax.grid(alpha=0.25)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=s["label"]))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="baseline"))
    ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(1.0, 0.4),
        frameon=True,
        facecolor="none",
        edgecolor="#c0c0c0",
        framealpha=1.0,
        fontsize=QUERY_Saturation_LEGEND_FONT_SIZE,
    )

    save_figure_multi(fig, fig_dir, "query_exp1_saturation_vs_x")


def plot_query_exp2(base_dir: Path, fig_dir: Path, *, use_ci: bool = True, baseline_dir: Path | None = None) -> None:
    if not (base_dir / "query_exp2_m_sweep.csv").exists() or not (base_dir / "query_exp2_random_baseline.csv").exists():
        print(f"[skip] missing query_exp2 csvs under {base_dir}")
        return
    df = filter_ok(pd.read_csv(base_dir / "query_exp2_m_sweep.csv"))
    rb = filter_ok(pd.read_csv(base_dir / "query_exp2_random_baseline.csv"))
    nd = pd.DataFrame()
    if baseline_dir is not None:
        nd_path = baseline_dir / "query_exp2_m_sweep.csv"
        if nd_path.exists():
            nd = filter_ok(pd.read_csv(nd_path))

    if df.empty:
        return

    ratios = sorted(df["attack_ratio"].unique())
    m_exps = sorted(df["m_exp"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.3), sharex=True)
    for col_idx, (metric, title) in enumerate(METRICS):
        ax = axes[col_idx]
        y_series_col: list[list[float]] = []
        for ratio in ratios:
            g = df[df["attack_ratio"] == ratio].sort_values("m_exp")
            y_vals, y_err = metric_values(g, metric, scale=1e7, use_ci=use_ci)
            y_series_col.append(y_vals)
            style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
            if y_err is None:
                ax.plot(g["m_exp"].tolist(), y_vals, **style)
            else:
                ax.errorbar(g["m_exp"].tolist(), y_vals, yerr=y_err, capsize=2.0, **style)

        if not rb.empty:
            g_rb = rb.sort_values("m_exp")
            rb_vals, rb_err = metric_values(g_rb, metric, scale=1e7, use_ci=use_ci)
            if rb_err is None:
                ax.plot(g_rb["m_exp"].tolist(), rb_vals, color="black", linestyle="--", marker="x", label="baseline")
            else:
                ax.errorbar(g_rb["m_exp"].tolist(), rb_vals, yerr=rb_err, capsize=2.0, color="black", linestyle="--", marker="x", label="baseline")
            baseline_ref = max(rb_vals) if rb_vals else 0.0
        else:
            baseline_ref = 0.0

        if not nd.empty:
            g_nd = nd.sort_values("m_exp")
            nd_vals, nd_err = metric_values(g_nd, metric, scale=1e7, use_ci=use_ci)
            if nd_err is None:
                ax.plot(g_nd["m_exp"].tolist(), nd_vals, color="#111111", linestyle="-.", marker="d", label="no defense")
            else:
                ax.errorbar(g_nd["m_exp"].tolist(), nd_vals, yerr=nd_err, capsize=2.0, color="#111111", linestyle="-.", marker="d", label="no defense")

        if USE_LOG_Y_FOR_QUERY_THROUGHPUT:
            ax.set_yscale("log")
        else:
            baseline_clipped = tune_query_axis_ylim(ax, y_series_col, baseline_ref)
            if baseline_ref > 0.0:
                draw_baseline_reference(ax, baseline_ref, baseline_clipped)

        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(alpha=0.25)
        if col_idx > 0:
            try:
                ax.ticklabel_format(style="plain", axis="y", useOffset=False)
            except AttributeError:
                pass
        ax.set_xticks(m_exps)
        ax.set_xticklabels([format_exp_tick(int(e)) for e in m_exps])
        ax.set_xlabel("m", fontsize=XLABEL_FONT_SIZE)
        _set_inner_y_tick_spacing(ax, is_left_col=(col_idx == 0))

    axes[0].set_ylabel("Query Throughput\n(1e7 queries/s)", fontsize=YLABEL_FONT_SIZE)
    axes[0].set_ylabel("Query Throughput\n(queries/s)", fontsize=YLABEL_FONT_SIZE)
    axes[0].text(0.0, 1.02, "1e7", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=8)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"x-hot {s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", marker="x", label="baseline"))

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.848, 0.5),
        frameon=True,
        facecolor="none",
        edgecolor="#c0c0c0",
        framealpha=1.0,
        fontsize=QUERY_LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(wspace=0.22, right=0.84)

    save_figure_multi(fig, fig_dir, "query_exp2_throughput_vs_m")


def plot_query_exp2_sq(base_dir: Path, fig_dir: Path, *, use_ci: bool = True, baseline_dir: Path | None = None) -> None:
    xhot_path = base_dir / "query_exp2_m_sweep_xhot.csv"
    sq_path = base_dir / "query_exp2_m_sweep_sq.csv"
    rb_path = base_dir / "query_exp2_random_baseline.csv"
    if not xhot_path.exists() or not sq_path.exists() or not rb_path.exists():
        print(f"[skip] missing query_exp2 sq/xhot csvs under {base_dir}")
        return

    xhot = filter_ok(pd.read_csv(xhot_path))
    sq = filter_ok(pd.read_csv(sq_path))
    rb = filter_ok(pd.read_csv(rb_path))
    nd = pd.DataFrame()
    if baseline_dir is not None:
        nd_path = baseline_dir / "query_exp2_m_sweep.csv"
        if nd_path.exists():
            nd = filter_ok(pd.read_csv(nd_path))
    if xhot.empty and sq.empty:
        return

    ratios = sorted(set(xhot.get("attack_ratio", [])).union(set(sq.get("attack_ratio", []))))
    m_exps = sorted(set(xhot.get("m_exp", [])).union(set(sq.get("m_exp", []))))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.3), sharex=True)
    for col_idx, (metric, title) in enumerate(METRICS):
        ax = axes[col_idx]
        y_series_col: list[list[float]] = []
        for ratio in ratios:
            if not xhot.empty:
                g = xhot[xhot["attack_ratio"] == ratio].sort_values("m_exp")
                if not g.empty:
                    y_vals, y_err = metric_values(g, metric, scale=1e7, use_ci=use_ci)
                    y_series_col.append(y_vals)
                    style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
                    if y_err is None:
                        ax.plot(g["m_exp"].tolist(), y_vals, **style)
                    else:
                        ax.errorbar(g["m_exp"].tolist(), y_vals, yerr=y_err, capsize=2.0, **style)
            if not sq.empty:
                g_sq = sq[sq["attack_ratio"] == ratio].sort_values("m_exp")
                if not g_sq.empty:
                    y_vals, y_err = metric_values(g_sq, metric, scale=1e7, use_ci=use_ci)
                    y_series_col.append(y_vals)
                    add_sq_style(ax, g_sq["m_exp"].tolist(), y_vals, y_err, ratio)

        if not rb.empty:
            g_rb = rb.sort_values("m_exp")
            rb_vals, rb_err = metric_values(g_rb, metric, scale=1e7, use_ci=use_ci)
            if rb_err is None:
                ax.plot(g_rb["m_exp"].tolist(), rb_vals, color="black", linestyle="--", marker="x", label="baseline")
            else:
                ax.errorbar(g_rb["m_exp"].tolist(), rb_vals, yerr=rb_err, capsize=2.0, color="black", linestyle="--", marker="x", label="baseline")
            baseline_ref = max(rb_vals) if rb_vals else 0.0
        else:
            baseline_ref = 0.0

        if not nd.empty:
            g_nd = nd.sort_values("m_exp")
            nd_vals, nd_err = metric_values(g_nd, metric, scale=1e7, use_ci=use_ci)
            if nd_err is None:
                ax.plot(g_nd["m_exp"].tolist(), nd_vals, color="#111111", linestyle="-.", marker="d", label="no defense")
            else:
                ax.errorbar(g_nd["m_exp"].tolist(), nd_vals, yerr=nd_err, capsize=2.0, color="#111111", linestyle="-.", marker="d", label="no defense")

        if USE_LOG_Y_FOR_QUERY_THROUGHPUT:
            ax.set_yscale("log")
        else:
            baseline_clipped = tune_query_axis_ylim(ax, y_series_col, baseline_ref)
            if baseline_ref > 0.0:
                draw_baseline_reference(ax, baseline_ref, baseline_clipped)

        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(alpha=0.25)
        if col_idx > 0:
            try:
                ax.ticklabel_format(style="plain", axis="y", useOffset=False)
            except AttributeError:
                pass
        ax.set_xticks(m_exps)
        ax.set_xticklabels([format_exp_tick(int(e)) for e in m_exps])
        ax.set_xlabel("m", fontsize=XLABEL_FONT_SIZE)
        _set_inner_y_tick_spacing(ax, is_left_col=(col_idx == 0))

    axes[0].set_ylabel("Query Throughput\n(queries/s)", fontsize=YLABEL_FONT_SIZE)
    axes[0].text(0.0, 1.02, "1e7", transform=axes[0].transAxes, ha="left", va="bottom", fontsize=8)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"x-hot {s['label']}"))
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=":", marker="v", label=f"sq {s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", marker="x", label="baseline"))

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.848, 0.5),
        frameon=True,
        facecolor="none",
        edgecolor="#c0c0c0",
        framealpha=1.0,
        fontsize=QUERY_LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(wspace=0.22, right=0.84)

    save_figure_multi(fig, fig_dir, "query_exp2_throughput_vs_m_sq")


def plot_query_exp12_combined(base_dir: Path, fig_dir: Path) -> None:
    q1_path = base_dir / "query_exp1_x_sweep.csv"
    q1_rb_path = base_dir / "query_exp1_random_baseline.csv"
    q2_path = base_dir / "query_exp2_m_sweep.csv"
    q2_rb_path = base_dir / "query_exp2_random_baseline.csv"
    if not q1_path.exists() or not q1_rb_path.exists() or not q2_path.exists() or not q2_rb_path.exists():
        print(f"[skip] missing query_exp1/query_exp2 csvs under {base_dir}")
        return

    q1 = filter_ok(pd.read_csv(q1_path))
    q1_rb = filter_ok(pd.read_csv(q1_rb_path))
    q2 = filter_ok(pd.read_csv(q2_path))
    q2_rb = filter_ok(pd.read_csv(q2_rb_path))
    if q1.empty or q2.empty:
        return

    ratios = sorted(set(q1["attack_ratio"].unique()) | set(q2["attack_ratio"].unique()))
    xs = sorted(q1["x"].unique())
    markevery = power_two_markevery(xs)
    m_exps = sorted(q2["m_exp"].unique())

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 2.6), sharex=False)

    # Row 1: q1 (x sweep)
    for col_idx, (metric, title) in enumerate(METRICS):
        ax = axes[0, col_idx]
        rb_qps, rb_ci = baseline_values(q1_rb, metric, scale=1e7)
        rb_qps = rb_qps if rb_qps is not None else 0.0
        y_series_col: list[list[float]] = []

        for ratio in ratios:
            g = q1[q1["attack_ratio"] == ratio].sort_values("x")
            if g.empty:
                continue
            x_vals = g["x"].astype(int).tolist()
            y_vals, y_err = metric_values(g, metric, scale=1e7)
            y_series_col.append(y_vals)
            add_ratio_style(ax, x_vals, y_vals, y_err, float(ratio), markevery, linewidth=QUERY_EXP12_ROW1_LINEWIDTH)

        if USE_LOG_Y_FOR_QUERY_THROUGHPUT:
            ax.set_yscale("log")
            if rb_qps > 0.0:
                ax.axhline(rb_qps, color="black", linestyle="--", linewidth=0.9)
                draw_baseline_band(ax, xs, rb_qps, rb_ci)
            _mark_baseline_at_x(ax, 500, rb_qps)
        else:
            clipped = tune_query_axis_ylim(ax, y_series_col, rb_qps)
            draw_baseline_reference(ax, rb_qps, clipped)
            _mark_baseline_at_x(ax, 500, rb_qps)

        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(alpha=0.25)
        if col_idx > 0:
            try:
                ax.ticklabel_format(style="plain", axis="y", useOffset=False)
            except AttributeError:
                pass
        if col_idx == 0:
            ax.text(0.0, 1.02, "1e7", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xlabel("x", fontsize=XLABEL_FONT_SIZE, labelpad=1.0)
        _set_inner_y_tick_spacing(ax, is_left_col=(col_idx == 0))

    # Row 2: q2 (m sweep)
    for col_idx, (metric, _) in enumerate(METRICS):
        ax = axes[1, col_idx]
        y_series_col = []
        for ratio in ratios:
            g = q2[q2["attack_ratio"] == ratio].sort_values("m_exp")
            if g.empty:
                continue
            y_vals, y_err = metric_values(g, metric, scale=1e7)
            y_series_col.append(y_vals)
            style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
            if y_err is None:
                ax.plot(g["m_exp"].tolist(), y_vals, **style)
            else:
                ax.errorbar(g["m_exp"].tolist(), y_vals, yerr=y_err, capsize=2.0, **style)

        if not q2_rb.empty:
            g_rb = q2_rb.sort_values("m_exp")
            rb_vals, rb_err = metric_values(g_rb, metric, scale=1e7)
            if rb_err is None:
                ax.plot(g_rb["m_exp"].tolist(), rb_vals, color="black", linestyle="--", marker="x", label="baseline")
            else:
                ax.errorbar(g_rb["m_exp"].tolist(), rb_vals, yerr=rb_err, capsize=2.0, color="black", linestyle="--", marker="x", label="baseline")
            baseline_ref = max(rb_vals) if rb_vals else 0.0
        else:
            baseline_ref = 0.0

        if USE_LOG_Y_FOR_QUERY_THROUGHPUT:
            ax.set_yscale("log")
        else:
            clipped = tune_query_axis_ylim(ax, y_series_col, baseline_ref)
            if baseline_ref > 0.0:
                draw_baseline_reference(ax, baseline_ref, clipped)

        ax.grid(alpha=0.25)
        if col_idx > 0:
            try:
                ax.ticklabel_format(style="plain", axis="y", useOffset=False)
            except AttributeError:
                pass
        if col_idx == 0:
            ax.text(0.0, 1.02, "1e7", transform=ax.transAxes, ha="left", va="bottom", fontsize=8)
        else:
            ax.tick_params(axis="y", labelleft=False)
        ax.set_xticks(m_exps)
        ax.set_xticklabels([format_exp_tick(int(e)) for e in m_exps])
        ax.set_xlabel("m", fontsize=XLABEL_FONT_SIZE, labelpad=1.0)
        _set_inner_y_tick_spacing(ax, is_left_col=(col_idx == 0))

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"x-hot {s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", marker="x", label="baseline"))

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.84, 0.5),
        frameon=True,
        facecolor="none",
        edgecolor="#c0c0c0",
        framealpha=1.0,
        fontsize=QUERY_LEGEND_FONT_SIZE,
    )
    fig.text(0.02, 0.5, "Query Throughput (queries/s)", rotation=90, va="center", ha="left", fontsize=YLABEL_FONT_SIZE)
    fig.subplots_adjust(left=0.09, wspace=0.09, hspace=0.5, right=0.835)

    save_figure_multi(fig, fig_dir, "query_exp12_combined")


def plot_insert_exp1(base_dir: Path, fig_dir: Path) -> None:
    if not (base_dir / "insert_exp1_x_sweep.csv").exists() or not (base_dir / "insert_exp1_random_baseline.csv").exists():
        print(f"[skip] missing insert_exp1 csvs under {base_dir}")
        return
    df = filter_ok(pd.read_csv(base_dir / "insert_exp1_x_sweep.csv"))
    rb = filter_ok(pd.read_csv(base_dir / "insert_exp1_random_baseline.csv"))

    if df.empty:
        return

    ratios = sorted(df["attack_ratio"].unique())
    xs = sorted(df["x"].unique())
    markevery = power_two_markevery(xs)

    fig, ax = plt.subplots(figsize=(3.5, 2))
    for ratio in ratios:
        g = df[df["attack_ratio"] == ratio].sort_values("x")
        x_vals = g["x"].astype(int).tolist()
        y_vals, y_err = metric_values(g, "insert_qps")
        add_ratio_style(ax, x_vals, y_vals, y_err, float(ratio), markevery, linewidth=INSERT_X_SWEEP_LINEWIDTH)

    if not rb.empty:
        rb_val, rb_ci = baseline_values(rb, "insert_qps")
        if rb_val is not None:
            ax.axhline(rb_val, color="black", linestyle="--", linewidth=0.9)
            draw_baseline_band(ax, xs, rb_val, rb_ci)

    ax.set_yscale("log")
    ax.set_xlabel("x", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel("Insert Throughput\n(inserts/s)", fontsize=YLABEL_FONT_SIZE)
    ax.grid(alpha=0.25)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"{s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="baseline"))
    ax.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.80, 0.2),
        frameon=True,
        facecolor="white",
        edgecolor="#c0c0c0",
        framealpha=1.0,
        fontsize=INSERT_LEGEND_FONT_SIZE,
    )

    save_figure_multi(fig, fig_dir, "insert_exp1_throughput_vs_x")


def plot_insert_exp2(base_dir: Path, fig_dir: Path, *, use_ci: bool = True, baseline_dir: Path | None = None) -> None:
    if not (base_dir / "insert_exp2_m_sweep.csv").exists() or not (base_dir / "insert_exp2_random_baseline.csv").exists():
        print(f"[skip] missing insert_exp2 csvs under {base_dir}")
        return
    df = filter_ok(pd.read_csv(base_dir / "insert_exp2_m_sweep.csv"))
    rb = filter_ok(pd.read_csv(base_dir / "insert_exp2_random_baseline.csv"))
    nd = pd.DataFrame()
    if baseline_dir is not None:
        nd_path = baseline_dir / "insert_exp2_m_sweep.csv"
        if nd_path.exists():
            nd = filter_ok(pd.read_csv(nd_path))

    if df.empty:
        return

    ratios = sorted(df["attack_ratio"].unique())
    m_exps = sorted(df["m_exp"].unique())

    fig, ax = plt.subplots(figsize=(3.5, 2))
    for ratio in ratios:
        g = df[df["attack_ratio"] == ratio].sort_values("m_exp")
        y_vals, y_err = metric_values(g, "insert_qps", use_ci=use_ci)
        style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        if y_err is None:
            ax.plot(g["m_exp"].tolist(), y_vals, **style)
        else:
            ax.errorbar(g["m_exp"].tolist(), y_vals, yerr=y_err, capsize=2.0, **style)

    if not rb.empty:
        g_rb = rb.sort_values("m_exp")
        rb_vals, rb_err = metric_values(g_rb, "insert_qps", use_ci=use_ci)
        if rb_err is None:
            ax.plot(g_rb["m_exp"].tolist(), rb_vals, color="black", linestyle="--", marker="x", label="random baseline")
        else:
            ax.errorbar(g_rb["m_exp"].tolist(), rb_vals, yerr=rb_err, capsize=2.0, color="black", linestyle="--", marker="x", label="random baseline")

    if not nd.empty:
        g_nd = nd.sort_values("m_exp")
        nd_vals, nd_err = metric_values(g_nd, "insert_qps", use_ci=use_ci)
        if nd_err is None:
            ax.plot(g_nd["m_exp"].tolist(), nd_vals, color="#111111", linestyle="-.", marker="d", label="no defense")
        else:
            ax.errorbar(g_nd["m_exp"].tolist(), nd_vals, yerr=nd_err, capsize=2.0, color="#111111", linestyle="-.", marker="d", label="no defense")

    ax.set_yscale("log")
    ax.set_xticks(m_exps)
    ax.set_xticklabels([format_exp_tick(int(e)) for e in m_exps])
    ax.set_xlabel("m", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel("Insert Throughput\n(inserts/s)", fontsize=YLABEL_FONT_SIZE)
    ax.grid(alpha=0.25)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"{s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", marker="x", label="baseline"))
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="#c0c0c0", framealpha=1.0, fontsize=INSERT_LEGEND_FONT_SIZE)

    save_figure_multi(fig, fig_dir, "insert_exp2_throughput_vs_m")


def plot_insert_exp2_sq(base_dir: Path, fig_dir: Path, *, use_ci: bool = True, baseline_dir: Path | None = None) -> None:
    xhot_path = base_dir / "insert_exp2_m_sweep_xhot.csv"
    sq_path = base_dir / "insert_exp2_m_sweep_sq.csv"
    rb_path = base_dir / "insert_exp2_random_baseline.csv"
    if not xhot_path.exists() or not sq_path.exists() or not rb_path.exists():
        print(f"[skip] missing insert_exp2 sq/xhot csvs under {base_dir}")
        return
    xhot = filter_ok(pd.read_csv(xhot_path))
    sq = filter_ok(pd.read_csv(sq_path))
    rb = filter_ok(pd.read_csv(rb_path))
    nd = pd.DataFrame()
    if baseline_dir is not None:
        nd_path = baseline_dir / "insert_exp2_m_sweep.csv"
        if nd_path.exists():
            nd = filter_ok(pd.read_csv(nd_path))
    if xhot.empty and sq.empty:
        return

    ratios = sorted(set(xhot.get("attack_ratio", [])).union(set(sq.get("attack_ratio", []))))
    m_exps = sorted(set(xhot.get("m_exp", [])).union(set(sq.get("m_exp", []))))

    fig, ax = plt.subplots(figsize=(3.5, 2))
    for ratio in ratios:
        if not xhot.empty:
            g = xhot[xhot["attack_ratio"] == ratio].sort_values("m_exp")
            if not g.empty:
                y_vals, y_err = metric_values(g, "insert_qps", use_ci=use_ci)
                style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
                if y_err is None:
                    ax.plot(g["m_exp"].tolist(), y_vals, **style)
                else:
                    ax.errorbar(g["m_exp"].tolist(), y_vals, yerr=y_err, capsize=2.0, **style)
        if not sq.empty:
            g_sq = sq[sq["attack_ratio"] == ratio].sort_values("m_exp")
            if not g_sq.empty:
                y_vals, y_err = metric_values(g_sq, "insert_qps", use_ci=use_ci)
                add_sq_style(ax, g_sq["m_exp"].tolist(), y_vals, y_err, ratio)

    if not rb.empty:
        g_rb = rb.sort_values("m_exp")
        rb_vals, rb_err = metric_values(g_rb, "insert_qps", use_ci=use_ci)
        if rb_err is None:
            ax.plot(g_rb["m_exp"].tolist(), rb_vals, color="black", linestyle="--", marker="x", label="random baseline")
        else:
            ax.errorbar(g_rb["m_exp"].tolist(), rb_vals, yerr=rb_err, capsize=2.0, color="black", linestyle="--", marker="x", label="random baseline")

    if not nd.empty:
        g_nd = nd.sort_values("m_exp")
        nd_vals, nd_err = metric_values(g_nd, "insert_qps", use_ci=use_ci)
        if nd_err is None:
            ax.plot(g_nd["m_exp"].tolist(), nd_vals, color="#111111", linestyle="-.", marker="d", label="no defense")
        else:
            ax.errorbar(g_nd["m_exp"].tolist(), nd_vals, yerr=nd_err, capsize=2.0, color="#111111", linestyle="-.", marker="d", label="no defense")

    ax.set_yscale("log")
    ax.set_xticks(m_exps)
    ax.set_xticklabels([format_exp_tick(int(e)) for e in m_exps])
    ax.set_xlabel("m", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel("Insert Throughput\n(inserts/s)", fontsize=YLABEL_FONT_SIZE)
    ax.grid(alpha=0.25)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"x-hot {s['label']}"))
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=":", marker="v", label=f"sq {s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", marker="x", label="baseline"))
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="#c0c0c0", framealpha=1.0, fontsize=INSERT_LEGEND_FONT_SIZE)

    save_figure_multi(fig, fig_dir, "insert_exp2_throughput_vs_m_sq")


def plot_query_exp2_uplift(base_dir: Path, baseline_dir: Path, fig_dir: Path) -> None:
    def_path = base_dir / "query_exp2_m_sweep.csv"
    base_path = resolve_uplift_baseline(base_dir, baseline_dir, "query_exp2_m_sweep.csv", "query_exp2_no_defense.csv")
    if not def_path.exists() or not base_path.exists():
        print(f"[skip] missing query_exp2 uplift csvs under {base_dir} or {baseline_dir}")
        return

    df = filter_ok(pd.read_csv(def_path))
    base = filter_ok(pd.read_csv(base_path))
    if df.empty or base.empty:
        return

    ratios = sorted(df["attack_ratio"].unique())
    m_exps = sorted(df["m_exp"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 1.3), sharex=True)
    for col_idx, (metric, title) in enumerate(METRICS):
        ax = axes[col_idx]
        for ratio in ratios:
            d = df[df["attack_ratio"] == ratio].sort_values("m_exp")
            b = base[base["attack_ratio"] == ratio].sort_values("m_exp")
            merged = compute_uplift_frame(d, b, metric)
            if merged.empty:
                continue
            ax.plot(
                merged["m_exp"].tolist(),
                merged["uplift_pct"].tolist(),
                **LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"}),
            )
        ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(title, fontsize=TITLE_FONT_SIZE)
        ax.grid(alpha=0.25)
        ax.set_xticks(m_exps)
        ax.set_xticklabels([format_exp_tick(int(e)) for e in m_exps])
        ax.set_xlabel("m", fontsize=XLABEL_FONT_SIZE)
        _set_inner_y_tick_spacing(ax, is_left_col=(col_idx == 0))

    axes[0].set_ylabel("Throughput uplift (%)", fontsize=YLABEL_FONT_SIZE)
    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"attack {s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="0% uplift"))
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.848, 0.5),
        frameon=True,
        facecolor="none",
        edgecolor="#c0c0c0",
        framealpha=1.0,
        fontsize=QUERY_LEGEND_FONT_SIZE,
    )
    fig.subplots_adjust(wspace=0.22, right=0.84)
    save_figure_multi(fig, fig_dir, "query_exp2_throughput_uplift_vs_m")


def plot_insert_exp2_uplift(base_dir: Path, baseline_dir: Path, fig_dir: Path) -> None:
    def_path = base_dir / "insert_exp2_m_sweep.csv"
    base_path = resolve_uplift_baseline(base_dir, baseline_dir, "insert_exp2_m_sweep.csv", "insert_exp2_no_defense.csv")
    if not def_path.exists() or not base_path.exists():
        print(f"[skip] missing insert_exp2 uplift csvs under {base_dir} or {baseline_dir}")
        return

    df = filter_ok(pd.read_csv(def_path))
    base = filter_ok(pd.read_csv(base_path))
    if df.empty or base.empty:
        return

    ratios = sorted(df["attack_ratio"].unique())
    m_exps = sorted(df["m_exp"].unique())

    fig, ax = plt.subplots(figsize=(3.5, 2))
    for ratio in ratios:
        d = df[df["attack_ratio"] == ratio].sort_values("m_exp")
        b = base[base["attack_ratio"] == ratio].sort_values("m_exp")
        merged = compute_uplift_frame(d, b, "insert_qps")
        if merged.empty:
            continue
        style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        ax.plot(merged["m_exp"].tolist(), merged["uplift_pct"].tolist(), **style)

    ax.axhline(0.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(m_exps)
    ax.set_xticklabels([format_exp_tick(int(e)) for e in m_exps])
    ax.set_xlabel("m", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel("Throughput uplift (%)", fontsize=YLABEL_FONT_SIZE)
    ax.grid(alpha=0.25)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"attack {s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="0% uplift"))
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="#c0c0c0", framealpha=1.0, fontsize=INSERT_LEGEND_FONT_SIZE)

    save_figure_multi(fig, fig_dir, "insert_exp2_throughput_uplift_vs_m")


def plot_insert_exp3(base_dir: Path, fig_dir: Path) -> None:
    if not (base_dir / "insert_exp3_skew_sweep.csv").exists() or not (base_dir / "insert_exp3_random_baseline.csv").exists():
        print(f"[skip] missing insert_exp3 csvs under {base_dir}")
        return
    df = filter_ok(pd.read_csv(base_dir / "insert_exp3_skew_sweep.csv"))
    rb = filter_ok(pd.read_csv(base_dir / "insert_exp3_random_baseline.csv"))

    if df.empty:
        return

    ratios = sorted(df["attack_ratio"].unique())
    skews = sorted(df["zipf_s"].unique())

    fig, ax = plt.subplots(figsize=(3.5, 2))
    for ratio in ratios:
        g = df[df["attack_ratio"] == ratio].sort_values("zipf_s")
        y_vals, y_err = metric_values(g, "insert_qps")
        style = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        if y_err is None:
            ax.plot(g["zipf_s"].tolist(), y_vals, **style)
        else:
            ax.errorbar(g["zipf_s"].tolist(), y_vals, yerr=y_err, capsize=2.0, **style)

    if not rb.empty:
        rb_val, rb_ci = baseline_values(rb, "insert_qps")
        if rb_val is not None:
            ax.axhline(rb_val, color="black", linestyle="--", linewidth=0.9)
            draw_baseline_band(ax, skews, rb_val, rb_ci)

    ax.set_yscale("log")
    ax.set_xticks(skews)
    ax.set_xlabel("Zipf Skewness", fontsize=XLABEL_FONT_SIZE)
    ax.set_ylabel("Insert Throughput\n(inserts/s)", fontsize=YLABEL_FONT_SIZE)
    ax.grid(alpha=0.25)

    handles = []
    for ratio in ratios:
        s = LINE_STYLES.get(float(ratio), {"color": "black", "linestyle": "-", "marker": "o", "label": f"{ratio:.2f}"})
        handles.append(mlines.Line2D([], [], color=s["color"], linestyle=s["linestyle"], marker=s["marker"], label=f"{s['label']}"))
    handles.append(mlines.Line2D([], [], color="black", linestyle="--", label="baseline"))
    ax.legend(handles=handles, frameon=True, facecolor="white", edgecolor="#c0c0c0", framealpha=1.0, fontsize=INSERT_LEGEND_FONT_SIZE)

    save_figure_multi(fig, fig_dir, "insert_exp3_throughput_vs_skewness")


def main() -> None:
    global OUTPUT_FORMATS

    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="/home/zq/code2/Memento_Filter/paper_results/results/throughput")
    ap.add_argument("--baseline-dir", default="", help="optional no-defense result dir to overlay")
    ap.add_argument("--figure-dir", default="/home/zq/code2/Memento_Filter/paper_results/figures/throughput")
    ap.add_argument(
        "--figures",
        default="all",
        help="comma-separated ids: q1,q1sat,q2,q2u,i1,i2,i2u,i3,q2sq,i2sq or all",
    )
    ap.add_argument(
        "--formats",
        default="pdf,svg",
        help="comma-separated output formats (recommended for Word: pdf,svg)",
    )
    ap.add_argument(
        "--no-ci",
        action="store_true",
        help="disable error bars (use mean values only)",
    )
    args = ap.parse_args()

    base_dir = Path(args.result_dir)
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else None
    fig_dir = Path(args.figure_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    OUTPUT_FORMATS = [x.strip().lower() for x in args.formats.split(",") if x.strip()]
    if not OUTPUT_FORMATS:
        OUTPUT_FORMATS = ["pdf", "svg"]

    setup_style()

    selected = {x.strip() for x in args.figures.split(",") if x.strip()}
    if "all" in selected:
        selected = {"q1", "q1sat", "q2", "q2u", "i1", "i2", "i2u", "i3", "q2sq", "i2sq"}

    # If both q1 and q2 are requested, render a combined multi-panel figure.
    if "q1" in selected and "q2" in selected:
        plot_query_exp12_combined(base_dir, fig_dir)
        selected.discard("q1")
        selected.discard("q2")

    if "q1" in selected:
        plot_query_exp1(base_dir, fig_dir)
    if "q1sat" in selected:
        plot_query_exp1_saturation(base_dir, fig_dir)
    if "q2" in selected:
        plot_query_exp2(base_dir, fig_dir, use_ci=not args.no_ci, baseline_dir=baseline_dir)
    if "q2u" in selected:
        plot_query_exp2_uplift(base_dir, baseline_dir, fig_dir)
    if "q2sq" in selected:
        plot_query_exp2_sq(base_dir, fig_dir, use_ci=not args.no_ci, baseline_dir=baseline_dir)
    if "i1" in selected:
        plot_insert_exp1(base_dir, fig_dir)
    if "i2" in selected:
        plot_insert_exp2(base_dir, fig_dir, use_ci=not args.no_ci, baseline_dir=baseline_dir)
    if "i2u" in selected:
        plot_insert_exp2_uplift(base_dir, baseline_dir, fig_dir)
    if "i2sq" in selected:
        plot_insert_exp2_sq(base_dir, fig_dir, use_ci=not args.no_ci, baseline_dir=baseline_dir)
    if "i3" in selected:
        plot_insert_exp3(base_dir, fig_dir)


if __name__ == "__main__":
    main()
