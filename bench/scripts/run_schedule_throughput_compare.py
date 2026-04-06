#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import subprocess
from pathlib import Path
import pandas as pd


def parse_kv(stdout: str) -> dict[str, str]:
    m: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        m[k.strip()] = v.strip()
    return m


def run_one(bin_path: Path, root: Path, schedule: str, x: int, args: argparse.Namespace) -> dict:
    cmd = [
        str(bin_path),
        "--mode", "x-hot",
        "--hot-slots", str(x),
        "--xhot-schedule", schedule,
        "--requested-inserts", str(args.requested_inserts),
        "--query-count", str(args.query_count),
        "--target-load", str(args.target_load),
        "--strict-load-factor",
        "--positive-range-protocol", args.protocol,
        "--memento-size", str(args.memento_bits),
        "--allow-duplicate-mementos",
    ]
    if schedule == "burst":
        cmd += ["--burst-len", str(args.burst_len)]
    if schedule == "zipf":
        cmd += ["--zipf-s", str(args.zipf_s)]
    cmd += [str(args.bpk)]

    p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
    kv = parse_kv(p.stdout)

    actual = float(kv.get("actual_inserts", "0"))
    insert_ms = float(kv.get("insert_ms", "0"))
    insert_qps = (1000.0 * actual / insert_ms) if insert_ms > 0 else 0.0

    return {
        "schedule": schedule,
        "x": x,
        "protocol": args.protocol,
        "requested_inserts": args.requested_inserts,
        "query_count": args.query_count,
        "bpk": args.bpk,
        "memento_bits": args.memento_bits,
        "zipf_s": args.zipf_s if schedule == "zipf" else None,
        "burst_len": args.burst_len if schedule == "burst" else None,
        "actual_inserts": int(actual),
        "insert_ms": insert_ms,
        "insert_qps": insert_qps,
        "point_pos_qps": float(kv.get("point_pos_qps", "0")),
        "short_range_pos_qps": float(kv.get("short_range_pos_qps", "0")),
        "long_range_pos_qps": float(kv.get("long_range_pos_qps", "0")),
        "final_saturated_ratio": float(kv.get("final_saturated_ratio", "0")),
        "first_overflow_insert_idx": float(kv.get("first_overflow_insert_idx", "-1")),
        "final_max_offset": float(kv.get("final_max_offset", "0")),
    }


def run_random(bin_path: Path, root: Path, args: argparse.Namespace) -> dict:
    cmd = [
        str(bin_path),
        "--mode", "random",
        "--hot-slots", "0",
        "--requested-inserts", str(args.requested_inserts),
        "--query-count", str(args.query_count),
        "--target-load", str(args.target_load),
        "--strict-load-factor",
        "--positive-range-protocol", args.protocol,
        "--memento-size", str(args.memento_bits),
        "--allow-duplicate-mementos",
        str(args.bpk),
    ]
    p = subprocess.run(cmd, cwd=root, capture_output=True, text=True, check=True)
    kv = parse_kv(p.stdout)

    actual = float(kv.get("actual_inserts", "0"))
    insert_ms = float(kv.get("insert_ms", "0"))
    insert_qps = (1000.0 * actual / insert_ms) if insert_ms > 0 else 0.0

    return {
        "schedule": "random",
        "x": 0,
        "protocol": args.protocol,
        "requested_inserts": args.requested_inserts,
        "query_count": args.query_count,
        "bpk": args.bpk,
        "memento_bits": args.memento_bits,
        "zipf_s": None,
        "burst_len": None,
        "actual_inserts": int(actual),
        "insert_ms": insert_ms,
        "insert_qps": insert_qps,
        "point_pos_qps": float(kv.get("point_pos_qps", "0")),
        "short_range_pos_qps": float(kv.get("short_range_pos_qps", "0")),
        "long_range_pos_qps": float(kv.get("long_range_pos_qps", "0")),
        "final_saturated_ratio": float(kv.get("final_saturated_ratio", "0")),
        "first_overflow_insert_idx": float(kv.get("first_overflow_insert_idx", "-1")),
        "final_max_offset": float(kv.get("final_max_offset", "0")),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zq/code2/Memento_Filter")
    ap.add_argument("--out-csv", default="paper_results/results/overflow_sweep/schedule_compare_n100000_q100000.csv")
    ap.add_argument("--requested-inserts", type=int, default=100000)
    ap.add_argument("--query-count", type=int, default=100000)
    ap.add_argument("--target-load", type=float, default=0.95)
    ap.add_argument("--bpk", type=float, default=26.5)
    ap.add_argument("--memento-bits", type=int, default=10)
    ap.add_argument("--protocol", default="start-at-hit")
    ap.add_argument("--burst-len", type=int, default=256)
    ap.add_argument("--zipf-s", type=float, default=1.3)
    ap.add_argument("--jobs", type=int, default=12)
    args = ap.parse_args()

    root = Path(args.root)
    bin_path = root / "build/bench/bench_memento_overflow"

    x_values = list(range(1, 257)) + list(range(264, 1025, 8))
    schedules = ["round-robin", "burst", "zipf"]

    rows: list[dict] = []
    rows.append(run_random(bin_path, root, args))

    tasks = [(s, x) for s in schedules for x in x_values]
    with cf.ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = [ex.submit(run_one, bin_path, root, s, x, args) for s, x in tasks]
        for i, fut in enumerate(cf.as_completed(futs), 1):
            row = fut.result()
            rows.append(row)
            if i % 50 == 0:
                print(f"done {i}/{len(tasks)}")

    df = pd.DataFrame(rows)
    df = df.sort_values(["schedule", "x"]).reset_index(drop=True)

    out_csv = root / args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(out_csv)
    print(f"rows={len(df)}")


if __name__ == "__main__":
    main()
