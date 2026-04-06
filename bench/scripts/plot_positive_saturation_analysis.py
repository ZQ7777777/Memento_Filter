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
    parser.add_argument("--csv-contains", default="")
    parser.add_argument("--prefix", default="positive_n1e5_q1e5")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    csv_path = Path(args.csv) if args.csv else latest_file(input_dir, "queryrate_positive_startathit_n100000_q100000_*.csv")

    df = pd.read_csv(csv_path)
    xhot = df[df["mode"] == "x-hot"].sort_values("requested_hot_slots")
    random_df = df[df["mode"] == "random"]

    xhot_contains = pd.DataFrame()
    random_contains_df = pd.DataFrame()
    if args.csv_contains:
        df_contains = pd.read_csv(Path(args.csv_contains))
        xhot_contains = df_contains[df_contains["mode"] == "x-hot"].sort_values("requested_hot_slots")
        random_contains_df = df_contains[df_contains["mode"] == "random"]

    if xhot.empty or random_df.empty:
        raise ValueError("CSV must contain both x-hot rows and one random baseline row")

    rb = random_df.iloc[0]

    fp_bits = int(xhot["fingerprint_bits"].iloc[0])
    m_bits = int(xhot["memento_bits"].iloc[0])
    n_req = int(xhot["requested_inserts"].iloc[0])
    q_count = int(xhot["point_pos_checks"].iloc[0])
    protocol = str(xhot["positive_range_protocol"].iloc[0])
    load_est = float(n_req / xhot["nslots"].iloc[0]) if xhot["nslots"].iloc[0] else 0.0

    title_suffix = f"fp_bits={fp_bits}, memento_bits={m_bits}, n={n_req}, q={q_count}, protocol={protocol}, n/m≈{load_est:.4f}"

    # Figure 1: QPS vs x
    fig1, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    series = [
        ("point_pos_qps", "Point queries (R=1)", float(rb["point_pos_qps"])),
        ("short_range_pos_qps", "Short-range queries (R=32)", float(rb["short_range_pos_qps"])),
        ("long_range_pos_qps", "Long-range queries (R=1024)", float(rb["long_range_pos_qps"])),
    ]
    for ax, (col, ttl, base) in zip(axes, series):
        ax.plot(xhot["requested_hot_slots"], xhot[col], lw=1.2, label="x-hot start-at-hit")
        if not xhot_contains.empty:
            ax.plot(xhot_contains["requested_hot_slots"], xhot_contains[col], lw=1.2, label="x-hot contains-hit")
        ax.axhline(base, color="tab:red", ls="--", lw=1.2, label="random start-at-hit")
        if not random_contains_df.empty:
            base_contains = float(random_contains_df.iloc[0][col])
            ax.axhline(base_contains, color="tab:orange", ls=":", lw=1.2, label="random contains-hit")
        ax.set_title(ttl)
        ax.set_ylabel("queries/sec")
        ax.grid(alpha=0.3)
        ax.legend()
    axes[-1].set_xlabel("x (number of contiguous hot buckets)")
    fig1.suptitle(f"Positive query throughput vs x ({title_suffix})")
    fig1.tight_layout()
    out1 = input_dir / f"fig_{args.prefix}_qps_vs_x.png"
    fig1.savefig(out1, dpi=220, bbox_inches="tight")

    # Figure 2: Saturation ratio vs x
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(xhot["requested_hot_slots"], xhot["final_saturated_ratio"], lw=1.2, color="tab:purple")
    ax2.set_xlabel("x (number of contiguous hot buckets)")
    ax2.set_ylabel("final_saturated_ratio")
    ax2.set_title(f"Saturation ratio vs x ({title_suffix})")
    ax2.grid(alpha=0.3)
    fig2.tight_layout()
    out2 = input_dir / f"fig_{args.prefix}_saturation_vs_x.png"
    fig2.savefig(out2, dpi=220, bbox_inches="tight")

    # Figure 3: QPS vs saturation ratio + Pearson r
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
    corr_rows = []
    for ax, (col, ttl, _) in zip(axes3, series):
        r = xhot[[col, "final_saturated_ratio"]].corr().iloc[0, 1]
        corr_rows.append({"metric": col, "pearson_r": float(r)})
        ax.scatter(xhot["final_saturated_ratio"], xhot[col], s=10, alpha=0.7)
        ax.set_xlabel("final_saturated_ratio")
        ax.set_ylabel("queries/sec")
        ax.set_title(f"{ttl}\nPearson r={r:.4f}")
        ax.grid(alpha=0.3)
    fig3.suptitle("QPS vs saturation ratio (Pearson correlation)")
    fig3.tight_layout()
    out3 = input_dir / f"fig_{args.prefix}_qps_vs_saturation_corr.png"
    fig3.savefig(out3, dpi=220, bbox_inches="tight")

    # Figure 4: First overflow index vs x (extra diagnostic)
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.plot(xhot["requested_hot_slots"], xhot["first_overflow_insert_idx"], lw=1.2, color="tab:green")
    ax4.set_xlabel("x (number of contiguous hot buckets)")
    ax4.set_ylabel("first_overflow_insert_idx")
    ax4.set_title(f"First overflow insert index vs x ({title_suffix})")
    ax4.grid(alpha=0.3)
    fig4.tight_layout()
    out4 = input_dir / f"fig_{args.prefix}_first_overflow_vs_x.png"
    fig4.savefig(out4, dpi=220, bbox_inches="tight")

    # Figure 5: Throughput drop ratio vs saturation ratio
    # drop_ratio = (qps_random - qps_xhot) / qps_random
    fig5, axes5 = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
    drop_corr_rows = []
    for ax, (col, ttl, base) in zip(axes5, series):
        drop_col = f"{col}_drop_ratio"
        xhot[drop_col] = (base - xhot[col]) / base
        r_drop = xhot[[drop_col, "final_saturated_ratio"]].corr().iloc[0, 1]
        drop_corr_rows.append({"metric": drop_col, "pearson_r": float(r_drop)})
        ax.scatter(xhot["final_saturated_ratio"], xhot[drop_col], s=10, alpha=0.7)
        ax.set_xlabel("final_saturated_ratio")
        ax.set_ylabel("throughput_drop_ratio")
        ax.set_title(f"{ttl}\nPearson r={r_drop:.4f}")
        ax.grid(alpha=0.3)
    fig5.suptitle("Relative throughput drop vs saturation ratio")
    fig5.tight_layout()
    out5 = input_dir / f"fig_{args.prefix}_drop_ratio_vs_saturation_corr.png"
    fig5.savefig(out5, dpi=220, bbox_inches="tight")

    corr_df = pd.DataFrame(corr_rows)
    corr_df["definition"] = "Pearson r = cov(X,Y)/(sigma_X*sigma_Y), X=final_saturated_ratio, Y=query_qps"
    out_corr = input_dir / f"fig_{args.prefix}_correlations.csv"
    corr_df.to_csv(out_corr, index=False)

    drop_corr_df = pd.DataFrame(drop_corr_rows)
    drop_corr_df["definition"] = "Pearson r = cov(X,Y)/(sigma_X*sigma_Y), X=final_saturated_ratio, Y=(qps_random-qps_xhot)/qps_random"
    out_drop_corr = input_dir / f"fig_{args.prefix}_drop_ratio_correlations.csv"
    drop_corr_df.to_csv(out_drop_corr, index=False)

    # Figure 6: Slowdown ratio vs saturation ratio
    # slowdown_ratio = qps_random / qps_xhot
    fig6, axes6 = plt.subplots(1, 3, figsize=(15, 4), sharex=False, sharey=False)
    slowdown_corr_rows = []
    for ax, (col, ttl, base) in zip(axes6, series):
        ratio_col = f"{col}_slowdown_ratio"
        xhot[ratio_col] = base / xhot[col]
        r_ratio = xhot[[ratio_col, "final_saturated_ratio"]].corr().iloc[0, 1]
        slowdown_corr_rows.append({"metric": ratio_col, "pearson_r": float(r_ratio)})
        ax.scatter(xhot["final_saturated_ratio"], xhot[ratio_col], s=10, alpha=0.7)
        ax.set_xlabel("final_saturated_ratio")
        ax.set_ylabel("qps_random / qps_xhot")
        ax.set_title(f"{ttl}\nPearson r={r_ratio:.4f}")
        ax.grid(alpha=0.3)
    fig6.suptitle("Slowdown ratio vs saturation ratio")
    fig6.tight_layout()
    out6 = input_dir / f"fig_{args.prefix}_slowdown_ratio_vs_saturation_corr.png"
    fig6.savefig(out6, dpi=220, bbox_inches="tight")

    slowdown_corr_df = pd.DataFrame(slowdown_corr_rows)
    slowdown_corr_df["definition"] = "Pearson r = cov(X,Y)/(sigma_X*sigma_Y), X=final_saturated_ratio, Y=qps_random/qps_xhot"
    out_slowdown_corr = input_dir / f"fig_{args.prefix}_slowdown_ratio_correlations.csv"
    slowdown_corr_df.to_csv(out_slowdown_corr, index=False)

    # Figure 7: Drop ratio vs x (start-at-hit vs contains-hit)
    out7 = None
    out_drop_vs_x = None
    if not xhot_contains.empty and not random_contains_df.empty:
        rb_contains = random_contains_df.iloc[0]
        fig7, axes7 = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
        drop_vs_x_rows = []
        for ax, (col, ttl, base_start) in zip(axes7, series):
            base_contains = float(rb_contains[col])
            start_drop = (base_start - xhot[col]) / base_start
            contains_drop = (base_contains - xhot_contains[col]) / base_contains

            ax.plot(xhot["requested_hot_slots"], start_drop, lw=1.2, label="start-at-hit drop ratio")
            ax.plot(xhot_contains["requested_hot_slots"], contains_drop, lw=1.2, label="contains-hit drop ratio")
            ax.axhline(0.0, color="gray", ls="--", lw=0.8)
            ax.set_title(ttl)
            ax.set_ylabel("drop ratio")
            ax.grid(alpha=0.3)
            ax.legend()

            x_start = xhot["requested_hot_slots"].astype(float)
            x_contains = xhot_contains["requested_hot_slots"].astype(float)
            r_start_x = pd.DataFrame({"x": x_start, "y": start_drop}).corr().iloc[0, 1]
            r_contains_x = pd.DataFrame({"x": x_contains, "y": contains_drop}).corr().iloc[0, 1]
            drop_vs_x_rows.append({"metric": col, "protocol": "start-at-hit", "pearson_r_x_drop": float(r_start_x)})
            drop_vs_x_rows.append({"metric": col, "protocol": "contains-hit", "pearson_r_x_drop": float(r_contains_x)})

        axes7[-1].set_xlabel("x (number of contiguous hot buckets)")
        fig7.suptitle("Throughput drop ratio vs x: start-at-hit vs contains-hit")
        fig7.tight_layout()
        out7 = input_dir / f"fig_{args.prefix}_drop_ratio_vs_x_protocol_compare.png"
        fig7.savefig(out7, dpi=220, bbox_inches="tight")

        drop_vs_x_df = pd.DataFrame(drop_vs_x_rows)
        drop_vs_x_df["definition"] = "drop_ratio=(qps_random_protocol-qps_xhot_protocol)/qps_random_protocol; pearson_r_x_drop=corr(x,drop_ratio)"
        out_drop_vs_x = input_dir / f"fig_{args.prefix}_drop_ratio_vs_x_protocol_compare.csv"
        drop_vs_x_df.to_csv(out_drop_vs_x, index=False)

    print(f"[+] csv: {csv_path}")
    print(f"[+] figure: {out1}")
    print(f"[+] figure: {out2}")
    print(f"[+] figure: {out3}")
    print(f"[+] figure: {out4}")
    print(f"[+] figure: {out5}")
    print(f"[+] figure: {out6}")
    if out7 is not None:
        print(f"[+] figure: {out7}")
    print(f"[+] correlations: {out_corr}")
    print(f"[+] drop correlations: {out_drop_corr}")
    print(f"[+] slowdown correlations: {out_slowdown_corr}")
    if out_drop_vs_x is not None:
        print(f"[+] drop-vs-x protocol compare: {out_drop_vs_x}")


if __name__ == "__main__":
    main()
