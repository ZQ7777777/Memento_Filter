#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run_case(bin_path: Path, cwd: Path, mode: str, x: int, schedule: str, tag: str, out_dir: Path) -> None:
    summary_csv = out_dir / "latency_summary.csv"
    samples_csv = out_dir / "latency_samples.csv"

    cmd = [
        str(bin_path),
        "--mode", mode,
        "--hot-slots", str(x if mode != "random" else 0),
        "--xhot-schedule", schedule,
        "--attack-insert-ratio", "1.0",
        "--requested-inserts", "100000",
        "--query-count", "100000",
        "--target-load", "0.95",
        "--strict-load-factor",
        "--positive-range-protocol", "start-at-hit",
        "--memento-size", "10",
        "--allow-duplicate-mementos",
        "--latency-profile",
        "--latency-sample-size", "5000",
        "--latency-summary-csv", str(summary_csv),
        "--latency-samples-csv", str(samples_csv),
        "--latency-tag", tag,
        "26.5",
    ]

    subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zq/code2/Memento_Filter")
    ap.add_argument("--out-dir", default="paper_results/results/overflow_sweep/latency_profile")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in ["latency_summary.csv", "latency_samples.csv"]:
        p = out_dir / f
        if p.exists():
            p.unlink()

    bin_path = root / "build/bench/bench_memento_overflow"
    xs = [1, 2, 3, 64, 1024]

    for x in xs:
        run_case(bin_path, root, "x-hot", x, "round-robin", f"xhot_x{x}", out_dir)
        run_case(bin_path, root, "random", x, "round-robin", f"random_x{x}", out_dir)
        run_case(bin_path, root, "x-hot", x, "zipf", f"zipf_x{x}", out_dir)
        print(f"done x={x}")

    print(out_dir / "latency_summary.csv")
    print(out_dir / "latency_samples.csv")


if __name__ == "__main__":
    main()
