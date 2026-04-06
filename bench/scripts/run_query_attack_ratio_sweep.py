#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import subprocess
from pathlib import Path

import pandas as pd


def x_values() -> list[int]:
    return list(range(1, 257)) + list(range(264, 1025, 8))


def run_case(bin_path: Path, cwd: Path, x: int, ratio: float, args: argparse.Namespace) -> dict:
    cmd = [
        str(bin_path),
        "--mode", "x-hot",
        "--hot-slots", str(x),
        "--xhot-schedule", "round-robin",
        "--attack-insert-ratio", str(ratio),
        "--requested-inserts", str(args.requested_inserts),
        "--query-count", str(args.query_count),
        "--target-load", str(args.target_load),
        "--strict-load-factor",
        "--positive-range-protocol", "start-at-hit",
        "--memento-size", str(args.memento_bits),
        "--allow-duplicate-mementos",
        str(args.bpk),
    ]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    kv = {}
    for line in p.stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()

    return {
        "x": x,
        "attack_ratio": ratio,
        "requested_inserts": int(kv.get("requested_inserts", args.requested_inserts)),
        "actual_inserts": int(float(kv.get("actual_inserts", 0))),
        "attack_inserts_actual": int(float(kv.get("attack_inserts_actual", 0))),
        "random_inserts_actual": int(float(kv.get("random_inserts_actual", 0))),
        "final_saturated_ratio": float(kv.get("final_saturated_ratio", 0.0)),
        "first_overflow_insert_idx": float(kv.get("first_overflow_insert_idx", -1.0)),
        "point_pos_qps": float(kv.get("point_pos_qps", 0.0)),
        "short_range_pos_qps": float(kv.get("short_range_pos_qps", 0.0)),
        "long_range_pos_qps": float(kv.get("long_range_pos_qps", 0.0)),
    }


def run_random_baseline(bin_path: Path, cwd: Path, args: argparse.Namespace) -> dict:
    cmd = [
        str(bin_path),
        "--mode", "random",
        "--hot-slots", "0",
        "--requested-inserts", str(args.requested_inserts),
        "--query-count", str(args.query_count),
        "--target-load", str(args.target_load),
        "--strict-load-factor",
        "--positive-range-protocol", "start-at-hit",
        "--memento-size", str(args.memento_bits),
        "--allow-duplicate-mementos",
        str(args.bpk),
    ]
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    kv = {}
    for line in p.stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        kv[k.strip()] = v.strip()
    return {
        "point_pos_qps": float(kv.get("point_pos_qps", 0.0)),
        "short_range_pos_qps": float(kv.get("short_range_pos_qps", 0.0)),
        "long_range_pos_qps": float(kv.get("long_range_pos_qps", 0.0)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zq/code2/Memento_Filter")
    ap.add_argument("--out-csv", default="paper_results/results/overflow_sweep/query_attack_ratio.csv")
    ap.add_argument("--ratios", default="0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0")
    ap.add_argument("--requested-inserts", type=int, default=100000)
    ap.add_argument("--query-count", type=int, default=100000)
    ap.add_argument("--target-load", type=float, default=0.95)
    ap.add_argument("--bpk", type=float, default=26.5)
    ap.add_argument("--memento-bits", type=int, default=10)
    ap.add_argument("--jobs", type=int, default=20)
    args = ap.parse_args()

    root = Path(args.root)
    bin_path = root / "build/bench/bench_memento_overflow"

    ratios = [float(x) for x in args.ratios.split(",") if x.strip()]
    xs = x_values()

    baseline = run_random_baseline(bin_path, root, args)

    tasks = [(x, r) for r in ratios for x in xs]
    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_case, bin_path, root, x, r, args) for (x, r) in tasks]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            rows.append(fut.result())
            if i % 100 == 0:
                print(f"done {i}/{len(tasks)}")

    df = pd.DataFrame(rows).sort_values(["attack_ratio", "x"]).reset_index(drop=True)
    df["random_point_qps"] = baseline["point_pos_qps"]
    df["random_short_qps"] = baseline["short_range_pos_qps"]
    df["random_long_qps"] = baseline["long_range_pos_qps"]

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    meta = {
        "ratios": ratios,
        "x_count": len(xs),
        "rows": len(df),
        "baseline": baseline,
    }
    (out_csv.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(out_csv)
    print(out_csv.with_suffix(".meta.json"))


if __name__ == "__main__":
    main()
