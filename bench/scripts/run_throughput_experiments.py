#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import math
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


# ===========================
# Editable experiment config
# ===========================
BPK = 20.0
MEMENTO_BITS = 10
TARGET_LOAD = 0.95
SEED = 1380
POSITIVE_RANGE_PROTOCOL = "start-at-hit"

QUERY_COUNT = 200000
INSERT_ONLY_QUERY_COUNT = 0

QUERY_EXP1_M_EXP = 18
QUERY_EXP1_X_VALUES = list(range(1, 1025))
QUERY_EXP1_ATTACK_RATIOS = [0.01, 0.05, 0.10]

# Requested by user as 2^16, 2^17, 2^18, 2^19, 2^20.
QUERY_EXP2_M_EXPS = [16, 17, 18, 19, 20]
QUERY_EXP2_X = 512
QUERY_EXP2_ATTACK_RATIOS = [0.01, 0.05, 0.10]

INSERT_EXP1_M_EXP = 18
INSERT_EXP1_X_VALUES = list(range(1, 1025))
INSERT_EXP1_ATTACK_RATIOS = [0.01, 0.05, 0.10, 1.00]
INSERT_EXP1_ZIPF_S = 1.0

INSERT_EXP2_M_EXPS = [16, 17, 18, 19, 20]
INSERT_EXP2_X = 512
INSERT_EXP2_ATTACK_RATIOS = [0.01, 0.05, 0.10, 1.00]
INSERT_EXP2_ZIPF_S = 1.0

INSERT_EXP3_M_EXP = 18
INSERT_EXP3_X = 512
INSERT_EXP3_ATTACK_RATIOS = [0.01, 0.05, 0.10, 1.00]
INSERT_EXP3_ZIPF_S_VALUES = [0.5, 1.0, 1.5, 2.0]

# Safety guard to avoid impossible allocations by accident.
MAX_RUNNABLE_M_EXP = 26


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zq/code2/Memento_Filter")
    ap.add_argument("--out-dir", default="paper_results/results/throughput")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--max-runnable-m-exp", type=int, default=MAX_RUNNABLE_M_EXP)
    ap.add_argument(
        "--experiments",
        default="all",
        help="comma-separated experiment ids: q1,q2,i1,i2,i3,i1rb,i2rb,i3rb or all",
    )
    return ap.parse_args()


def parse_kv(stdout: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def to_float(kv: dict[str, str], key: str, default: float = 0.0) -> float:
    try:
        return float(kv.get(key, default))
    except (TypeError, ValueError):
        return default


def to_int(kv: dict[str, str], key: str, default: int = 0) -> int:
    try:
        return int(float(kv.get(key, default)))
    except (TypeError, ValueError):
        return default


def derive_n(m_slots: int) -> int:
    return int(math.floor(m_slots * TARGET_LOAD))


def run_bench(cmd: list[str], cwd: Path) -> dict[str, str]:
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=True)
    return parse_kv(p.stdout)


def make_common_prefix(bin_path: Path, m_slots: int, n_inserts: int, query_count: int) -> list[str]:
    return [
        str(bin_path),
        "--requested-inserts", str(n_inserts),
        "--query-count", str(query_count),
        "--target-load", str(TARGET_LOAD),
        "--strict-load-factor",
        "--fixed-nslots", str(m_slots),
        "--positive-range-protocol", POSITIVE_RANGE_PROTOCOL,
        "--memento-size", str(MEMENTO_BITS),
        "--seed", str(SEED),
    ]


def run_xhot_case(
    bin_path: Path,
    cwd: Path,
    *,
    m_exp: int,
    x: int,
    ratio: float,
    schedule: str,
    zipf_s: float,
    query_count: int,
    case_group: str,
) -> dict[str, Any]:
    if m_exp > 62:
        return {
            "case_group": case_group,
            "status": "skipped",
            "skip_reason": "m_exp_gt_62_unsupported_for_uint64",
            "m_exp": m_exp,
            "x": x,
            "attack_ratio": ratio,
            "schedule": schedule,
            "zipf_s": zipf_s,
        }

    if m_exp > MAX_RUNNABLE_M_EXP:
        return {
            "case_group": case_group,
            "status": "skipped",
            "skip_reason": "m_exp_exceeds_runtime_guard",
            "m_exp": m_exp,
            "x": x,
            "attack_ratio": ratio,
            "schedule": schedule,
            "zipf_s": zipf_s,
        }

    m_slots = 1 << m_exp
    n_inserts = derive_n(m_slots)
    effective_query_count = query_count if query_count == 0 else max(query_count, n_inserts)

    cmd = make_common_prefix(bin_path, m_slots, n_inserts, effective_query_count)
    cmd += [
        "--mode", "x-hot",
        "--hot-slots", str(x),
        "--xhot-schedule", schedule,
        "--attack-insert-ratio", str(ratio),
    ]
    if schedule == "zipf":
        cmd += ["--zipf-s", str(zipf_s)]
    cmd += [str(BPK)]

    kv = run_bench(cmd, cwd)
    insert_ms = to_float(kv, "insert_ms")
    actual_inserts = to_int(kv, "actual_inserts")
    insert_qps = (1000.0 * actual_inserts / insert_ms) if insert_ms > 0 else 0.0

    return {
        "case_group": case_group,
        "status": "ok",
        "skip_reason": "",
        "m_exp": m_exp,
        "m_slots": m_slots,
        "n_inserts": n_inserts,
        "x": x,
        "attack_ratio": ratio,
        "schedule": schedule,
        "zipf_s": zipf_s,
        "query_count": effective_query_count,
        "actual_inserts": actual_inserts,
        "insert_ms": insert_ms,
        "insert_qps": insert_qps,
        "point_pos_qps": to_float(kv, "point_pos_qps"),
        "short_range_pos_qps": to_float(kv, "short_range_pos_qps"),
        "long_range_pos_qps": to_float(kv, "long_range_pos_qps"),
        "final_saturated_ratio": to_float(kv, "final_saturated_ratio"),
        "final_saturated_blocks": to_int(kv, "final_saturated_blocks"),
        "nblocks": to_int(kv, "nblocks"),
    }


def run_random_case(
    bin_path: Path,
    cwd: Path,
    *,
    m_exp: int,
    query_count: int,
    case_group: str,
) -> dict[str, Any]:
    if m_exp > 62:
        return {
            "case_group": case_group,
            "status": "skipped",
            "skip_reason": "m_exp_gt_62_unsupported_for_uint64",
            "m_exp": m_exp,
        }

    if m_exp > MAX_RUNNABLE_M_EXP:
        return {
            "case_group": case_group,
            "status": "skipped",
            "skip_reason": "m_exp_exceeds_runtime_guard",
            "m_exp": m_exp,
        }

    m_slots = 1 << m_exp
    n_inserts = derive_n(m_slots)
    effective_query_count = query_count if query_count == 0 else max(query_count, n_inserts)

    cmd = make_common_prefix(bin_path, m_slots, n_inserts, effective_query_count)
    cmd += [
        "--mode", "random",
        "--hot-slots", "0",
        str(BPK),
    ]

    kv = run_bench(cmd, cwd)
    insert_ms = to_float(kv, "insert_ms")
    actual_inserts = to_int(kv, "actual_inserts")
    insert_qps = (1000.0 * actual_inserts / insert_ms) if insert_ms > 0 else 0.0

    return {
        "case_group": case_group,
        "status": "ok",
        "skip_reason": "",
        "m_exp": m_exp,
        "m_slots": m_slots,
        "n_inserts": n_inserts,
        "x": 0,
        "attack_ratio": 0.0,
        "schedule": "random",
        "zipf_s": None,
        "query_count": effective_query_count,
        "actual_inserts": actual_inserts,
        "insert_ms": insert_ms,
        "insert_qps": insert_qps,
        "point_pos_qps": to_float(kv, "point_pos_qps"),
        "short_range_pos_qps": to_float(kv, "short_range_pos_qps"),
        "long_range_pos_qps": to_float(kv, "long_range_pos_qps"),
        "final_saturated_ratio": to_float(kv, "final_saturated_ratio"),
        "final_saturated_blocks": to_int(kv, "final_saturated_blocks"),
        "nblocks": to_int(kv, "nblocks"),
    }


def run_many(tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]], jobs: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with cf.ThreadPoolExecutor(max_workers=jobs) as ex:
        futs = [ex.submit(fn, *fn_args, **fn_kwargs) for (fn, fn_args, fn_kwargs) in tasks]
        total = len(futs)
        for idx, fut in enumerate(cf.as_completed(futs), 1):
            rows.append(fut.result())
            if idx % 100 == 0 or idx == total:
                print(f"done {idx}/{total}")
    return rows


def save_df(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(out_csv)


def main() -> None:
    global MAX_RUNNABLE_M_EXP

    args = parse_args()
    MAX_RUNNABLE_M_EXP = args.max_runnable_m_exp

    selected = {x.strip() for x in args.experiments.split(",") if x.strip()}
    if "all" in selected:
        selected = {"q1", "q2", "i1", "i2", "i3"}

    root = Path(args.root)
    out_dir = root / args.out_dir
    bin_path = root / "build/bench/bench_memento_security"

    # ------------------------
    # Query experiment 1
    # ------------------------
    if "q1" in selected:
        print("[run] query_exp1_x_sweep")
        q1_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for ratio in QUERY_EXP1_ATTACK_RATIOS:
            for x in QUERY_EXP1_X_VALUES:
                q1_tasks.append(
                    (
                        run_xhot_case,
                        (bin_path, root),
                        {
                            "m_exp": QUERY_EXP1_M_EXP,
                            "x": x,
                            "ratio": ratio,
                            "schedule": "round-robin",
                            "zipf_s": 0.0,
                            "query_count": QUERY_COUNT,
                            "case_group": "query_exp1",
                        },
                    )
                )

        q1_rows = run_many(q1_tasks, args.jobs)
        q1_df = pd.DataFrame(q1_rows).sort_values(["attack_ratio", "x"]).reset_index(drop=True)
        save_df(q1_df, out_dir / "query_exp1_x_sweep.csv")

        q1_rb = run_random_case(
            bin_path,
            root,
            m_exp=QUERY_EXP1_M_EXP,
            query_count=QUERY_COUNT,
            case_group="query_exp1_random",
        )
        q1_rb_df = pd.DataFrame([q1_rb])
        save_df(q1_rb_df, out_dir / "query_exp1_random_baseline.csv")

    # ------------------------
    # Query experiment 2
    # ------------------------
    if "q2" in selected:
        print("[run] query_exp2_m_sweep")
        q2_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for m_exp in QUERY_EXP2_M_EXPS:
            for ratio in QUERY_EXP2_ATTACK_RATIOS:
                q2_tasks.append(
                    (
                        run_xhot_case,
                        (bin_path, root),
                        {
                            "m_exp": m_exp,
                            "x": QUERY_EXP2_X,
                            "ratio": ratio,
                            "schedule": "round-robin",
                            "zipf_s": 0.0,
                            "query_count": QUERY_COUNT,
                            "case_group": "query_exp2",
                        },
                    )
                )

        q2_rows = run_many(q2_tasks, args.jobs)
        q2_df = pd.DataFrame(q2_rows).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(q2_df, out_dir / "query_exp2_m_sweep.csv")

        q2_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=m_exp,
                query_count=QUERY_COUNT,
                case_group="query_exp2_random",
            )
            for m_exp in QUERY_EXP2_M_EXPS
        ]
        q2_rb_df = pd.DataFrame(q2_rb_rows).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(q2_rb_df, out_dir / "query_exp2_random_baseline.csv")

    # ------------------------
    # Insert experiment 1
    # ------------------------
    if "i1" in selected:
        print("[run] insert_exp1_x_sweep")
        i1_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for ratio in INSERT_EXP1_ATTACK_RATIOS:
            for x in INSERT_EXP1_X_VALUES:
                i1_tasks.append(
                    (
                        run_xhot_case,
                        (bin_path, root),
                        {
                            "m_exp": INSERT_EXP1_M_EXP,
                            "x": x,
                            "ratio": ratio,
                            "schedule": "zipf",
                            "zipf_s": INSERT_EXP1_ZIPF_S,
                            "query_count": INSERT_ONLY_QUERY_COUNT,
                            "case_group": "insert_exp1",
                        },
                    )
                )

        i1_rows = run_many(i1_tasks, args.jobs)
        i1_df = pd.DataFrame(i1_rows).sort_values(["attack_ratio", "x"]).reset_index(drop=True)
        save_df(i1_df, out_dir / "insert_exp1_x_sweep.csv")

    if "i1" in selected or "i1rb" in selected:
        i1_rb = run_random_case(
            bin_path,
            root,
            m_exp=INSERT_EXP1_M_EXP,
            query_count=INSERT_ONLY_QUERY_COUNT,
            case_group="insert_exp1_random",
        )
        i1_rb_df = pd.DataFrame([i1_rb])
        save_df(i1_rb_df, out_dir / "insert_exp1_random_baseline.csv")

    # ------------------------
    # Insert experiment 2
    # ------------------------
    if "i2" in selected:
        print("[run] insert_exp2_m_sweep")
        i2_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for m_exp in INSERT_EXP2_M_EXPS:
            for ratio in INSERT_EXP2_ATTACK_RATIOS:
                i2_tasks.append(
                    (
                        run_xhot_case,
                        (bin_path, root),
                        {
                            "m_exp": m_exp,
                            "x": INSERT_EXP2_X,
                            "ratio": ratio,
                            "schedule": "zipf",
                            "zipf_s": INSERT_EXP2_ZIPF_S,
                            "query_count": INSERT_ONLY_QUERY_COUNT,
                            "case_group": "insert_exp2",
                        },
                    )
                )

        i2_rows = run_many(i2_tasks, args.jobs)
        i2_df = pd.DataFrame(i2_rows).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(i2_df, out_dir / "insert_exp2_m_sweep.csv")

    if "i2" in selected or "i2rb" in selected:
        i2_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=m_exp,
                query_count=INSERT_ONLY_QUERY_COUNT,
                case_group="insert_exp2_random",
            )
            for m_exp in INSERT_EXP2_M_EXPS
        ]
        i2_rb_df = pd.DataFrame(i2_rb_rows).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(i2_rb_df, out_dir / "insert_exp2_random_baseline.csv")

    # ------------------------
    # Insert experiment 3
    # ------------------------
    if "i3" in selected:
        print("[run] insert_exp3_skew_sweep")
        i3_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for ratio in INSERT_EXP3_ATTACK_RATIOS:
            for skew in INSERT_EXP3_ZIPF_S_VALUES:
                i3_tasks.append(
                    (
                        run_xhot_case,
                        (bin_path, root),
                        {
                            "m_exp": INSERT_EXP3_M_EXP,
                            "x": INSERT_EXP3_X,
                            "ratio": ratio,
                            "schedule": "zipf",
                            "zipf_s": skew,
                            "query_count": INSERT_ONLY_QUERY_COUNT,
                            "case_group": "insert_exp3",
                        },
                    )
                )

        i3_rows = run_many(i3_tasks, args.jobs)
        i3_df = pd.DataFrame(i3_rows).sort_values(["attack_ratio", "zipf_s"]).reset_index(drop=True)
        save_df(i3_df, out_dir / "insert_exp3_skew_sweep.csv")

    if "i3" in selected or "i3rb" in selected:
        i3_rb = run_random_case(
            bin_path,
            root,
            m_exp=INSERT_EXP3_M_EXP,
            query_count=INSERT_ONLY_QUERY_COUNT,
            case_group="insert_exp3_random",
        )
        i3_rb_df = pd.DataFrame([i3_rb])
        save_df(i3_rb_df, out_dir / "insert_exp3_random_baseline.csv")

    config = {
        "BPK": BPK,
        "MEMENTO_BITS": MEMENTO_BITS,
        "TARGET_LOAD": TARGET_LOAD,
        "SEED": SEED,
        "POSITIVE_RANGE_PROTOCOL": POSITIVE_RANGE_PROTOCOL,
        "QUERY_COUNT": QUERY_COUNT,
        "INSERT_ONLY_QUERY_COUNT": INSERT_ONLY_QUERY_COUNT,
        "QUERY_EXP1_M_EXP": QUERY_EXP1_M_EXP,
        "QUERY_EXP1_ATTACK_RATIOS": QUERY_EXP1_ATTACK_RATIOS,
        "QUERY_EXP1_X_MIN": min(QUERY_EXP1_X_VALUES),
        "QUERY_EXP1_X_MAX": max(QUERY_EXP1_X_VALUES),
        "QUERY_EXP2_M_EXPS": QUERY_EXP2_M_EXPS,
        "QUERY_EXP2_X": QUERY_EXP2_X,
        "QUERY_EXP2_ATTACK_RATIOS": QUERY_EXP2_ATTACK_RATIOS,
        "INSERT_EXP1_M_EXP": INSERT_EXP1_M_EXP,
        "INSERT_EXP1_ATTACK_RATIOS": INSERT_EXP1_ATTACK_RATIOS,
        "INSERT_EXP1_ZIPF_S": INSERT_EXP1_ZIPF_S,
        "INSERT_EXP2_M_EXPS": INSERT_EXP2_M_EXPS,
        "INSERT_EXP2_X": INSERT_EXP2_X,
        "INSERT_EXP2_ATTACK_RATIOS": INSERT_EXP2_ATTACK_RATIOS,
        "INSERT_EXP2_ZIPF_S": INSERT_EXP2_ZIPF_S,
        "INSERT_EXP3_M_EXP": INSERT_EXP3_M_EXP,
        "INSERT_EXP3_X": INSERT_EXP3_X,
        "INSERT_EXP3_ATTACK_RATIOS": INSERT_EXP3_ATTACK_RATIOS,
        "INSERT_EXP3_ZIPF_S_VALUES": INSERT_EXP3_ZIPF_S_VALUES,
        "MAX_RUNNABLE_M_EXP": MAX_RUNNABLE_M_EXP,
    }
    cfg_path = out_dir / "experiment_config.json"
    cfg_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(cfg_path)


if __name__ == "__main__":
    main()
