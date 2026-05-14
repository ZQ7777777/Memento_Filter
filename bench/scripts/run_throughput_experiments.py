#!/usr/bin/env python3
from __future__ import annotations

import argparse
import concurrent.futures as cf
import threading
import time
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
REPEATS = 5
SEED_STEP = 1
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
# defense flags propagated to bench binary
DEF_ADAPTIVE_VERIFY = False
VERIFY_THRESHOLD = 16
DEF_KEEPSAKE_RLE = False
DEF_RECONSTRUCT = False
RECONSTRUCT_THRESHOLD = 32


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/zq/code2/Memento_Filter")
    ap.add_argument("--out-dir", default="paper_results/results/throughput")
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--max-runnable-m-exp", type=int, default=MAX_RUNNABLE_M_EXP)
    ap.add_argument("--repeats", type=int, default=REPEATS)
    ap.add_argument("--seed-base", type=int, default=SEED)
    ap.add_argument("--seed-step", type=int, default=SEED_STEP)
    ap.add_argument(
        "--experiments",
        default="all",
        help="comma-separated experiment ids: q1,q2,i1,i2,i3,i1rb,i2rb,i3rb,q2sq,i2sq or all",
    )
    ap.add_argument("--def-adaptive-verify", action="store_true", help="enable adaptive verification in bench")
    ap.add_argument("--verify-threshold", type=int, default=32, help="offset threshold for adaptive verification")
    ap.add_argument("--def-keepsake-rle", action="store_true", help="enable keepsake RLE compression in bench")
    ap.add_argument("--def-reconstruct", action="store_true", help="enable reconstruction defense in bench")
    ap.add_argument("--reconstruct-threshold", type=int, default=32, help="offset threshold for reconstruction")
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


def make_common_prefix(
    bin_path: Path,
    m_slots: int,
    n_inserts: int,
    query_count: int,
    seed: int,
    *,
    use_defense: bool = True,
) -> list[str]:
    cmd = [
        str(bin_path),
        "--requested-inserts", str(n_inserts),
        "--query-count", str(query_count),
        "--target-load", str(TARGET_LOAD),
        "--strict-load-factor",
        "--fixed-nslots", str(m_slots),
        "--positive-range-protocol", POSITIVE_RANGE_PROTOCOL,
        "--memento-size", str(MEMENTO_BITS),
        "--seed", str(seed),
    ]

    # propagate optional defense flags into bench invocation
    if use_defense and DEF_ADAPTIVE_VERIFY:
        cmd += ["--def-adaptive-verify", "--verify-threshold", str(VERIFY_THRESHOLD)]
    if use_defense and DEF_KEEPSAKE_RLE:
        cmd += ["--def-keepsake-rle"]
    if use_defense and DEF_RECONSTRUCT:
        cmd += ["--def-reconstruct", "--reconstruct-threshold", str(RECONSTRUCT_THRESHOLD)]

    return cmd


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
    seed: int,
    repeat: int,
    use_defense: bool = True,
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

    cmd = make_common_prefix(bin_path, m_slots, n_inserts, effective_query_count, seed, use_defense=use_defense)
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
        "seed": seed,
        "repeat": repeat,
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


def run_sq_case(
    bin_path: Path,
    cwd: Path,
    *,
    m_exp: int,
    ratio: float,
    query_count: int,
    case_group: str,
    seed: int,
    repeat: int,
    use_defense: bool = True,
) -> dict[str, Any]:
    if m_exp > 62:
        return {
            "case_group": case_group,
            "status": "skipped",
            "skip_reason": "m_exp_gt_62_unsupported_for_uint64",
            "m_exp": m_exp,
            "attack_ratio": ratio,
            "schedule": "sq",
        }

    if m_exp > MAX_RUNNABLE_M_EXP:
        return {
            "case_group": case_group,
            "status": "skipped",
            "skip_reason": "m_exp_exceeds_runtime_guard",
            "m_exp": m_exp,
            "attack_ratio": ratio,
            "schedule": "sq",
        }

    m_slots = 1 << m_exp
    n_inserts = derive_n(m_slots)
    effective_query_count = query_count if query_count == 0 else max(query_count, n_inserts)

    cmd = make_common_prefix(bin_path, m_slots, n_inserts, effective_query_count, seed, use_defense=use_defense)
    cmd += [
        "--mode", "sq",
        "--hot-slots", "0",
        "--attack-insert-ratio", str(ratio),
    ]
    cmd += [str(BPK)]

    kv = run_bench(cmd, cwd)
    insert_ms = to_float(kv, "insert_ms")
    actual_inserts = to_int(kv, "actual_inserts")
    insert_qps = (1000.0 * actual_inserts / insert_ms) if insert_ms > 0 else 0.0

    return {
        "case_group": case_group,
        "status": "ok",
        "skip_reason": "",
        "seed": seed,
        "repeat": repeat,
        "m_exp": m_exp,
        "m_slots": m_slots,
        "n_inserts": n_inserts,
        "x": 0,
        "attack_ratio": ratio,
        "schedule": "sq",
        "zipf_s": 0.0,
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
    seed: int,
    repeat: int,
    use_defense: bool = True,
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

    cmd = make_common_prefix(bin_path, m_slots, n_inserts, effective_query_count, seed, use_defense=use_defense)
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
        "seed": seed,
        "repeat": repeat,
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

        # Heartbeat thread to periodically print progress while tasks run
        stop_flag = threading.Event()

        def heartbeat():
            while not stop_flag.is_set():
                done = sum(1 for f in futs if f.done())
                print(f"progress: {done}/{total} tasks completed")
                stop_flag.wait(30.0)

        hb = threading.Thread(target=heartbeat, daemon=True)
        hb.start()

        try:
            for idx, fut in enumerate(cf.as_completed(futs), 1):
                rows.append(fut.result())
                if idx % 10 == 0 or idx == total:
                    print(f"done {idx}/{total}")
        finally:
            stop_flag.set()
            hb.join(timeout=1.0)
    return rows


def save_df(df: pd.DataFrame, out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(out_csv)


def aggregate_runs(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "actual_inserts",
        "insert_ms",
        "insert_qps",
        "point_pos_qps",
        "short_range_pos_qps",
        "long_range_pos_qps",
        "final_saturated_ratio",
        "final_saturated_blocks",
        "nblocks",
        "query_count",
    ]
    metric_cols = [c for c in metric_cols if c in df.columns]
    group_cols = [
        c for c in df.columns
        if c not in metric_cols and c not in {"seed", "repeat"}
    ]
    if not metric_cols:
        return df.copy()

    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    new_cols = []
    for col in agg.columns:
        if isinstance(col, tuple):
            base, stat = col
            if stat == "mean":
                new_cols.append(f"{base}_mean")
            elif stat == "std":
                new_cols.append(f"{base}_std")
            elif stat == "count":
                new_cols.append(f"{base}_count")
            elif stat == "":
                new_cols.append(base)
            else:
                new_cols.append(f"{base}_{stat}")
        else:
            new_cols.append(col)
    agg.columns = new_cols

    for metric in metric_cols:
        std_col = f"{metric}_std"
        count_col = f"{metric}_count"
        if std_col in agg.columns and count_col in agg.columns:
            agg[f"{metric}_ci95"] = 1.96 * agg[std_col] / agg[count_col].pow(0.5)
    return agg


def write_ttest_report(
    attack_raw: pd.DataFrame,
    baseline_raw: pd.DataFrame,
    group_cols: list[str],
    match_cols: list[str],
    out_csv: Path,
) -> None:
    try:
        from scipy import stats
    except ImportError:
        print("[warn] scipy not available; skipping t-test report")
        return

    metric_cols = [
        "insert_qps",
        "point_pos_qps",
        "short_range_pos_qps",
        "long_range_pos_qps",
    ]
    metric_cols = [c for c in metric_cols if c in attack_raw.columns and c in baseline_raw.columns]
    rows = []

    for keys, g in attack_raw.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        key_map = dict(zip(group_cols, keys))
        base_slice = baseline_raw
        for col in match_cols:
            if col in key_map:
                base_slice = base_slice[base_slice[col] == key_map[col]]
        if base_slice.empty:
            continue

        for metric in metric_cols:
            a = g[metric].dropna().to_numpy()
            b = base_slice[metric].dropna().to_numpy()
            if a.size < 2 or b.size < 2:
                p_val = float("nan")
            else:
                _, p_val = stats.ttest_ind(a, b, equal_var=False)
            row = {
                "metric": metric,
                "attack_mean": float(g[metric].mean()),
                "baseline_mean": float(base_slice[metric].mean()),
                "attack_n": int(a.size),
                "baseline_n": int(b.size),
                "p_value": float(p_val),
            }
            row.update(key_map)
            rows.append(row)

    if not rows:
        return
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(out_csv)


def summarize_uplift_by_ratio_metric(
    q_def: pd.DataFrame,
    q_base: pd.DataFrame,
    i_def: pd.DataFrame,
    i_base: pd.DataFrame,
    out_dir: Path,
) -> None:
    try:
        from scipy import stats
    except ImportError:
        stats = None

    query_rows: list[dict[str, Any]] = []
    insert_rows: list[dict[str, Any]] = []

    query_metrics = ["point_pos_qps", "short_range_pos_qps", "long_range_pos_qps"]
    for metric in query_metrics:
        dcol = f"{metric}_mean" if f"{metric}_mean" in q_def.columns else metric
        bcol = f"{metric}_mean" if f"{metric}_mean" in q_base.columns else metric
        if dcol not in q_def.columns or bcol not in q_base.columns:
            continue
        m = q_def[["attack_ratio", "m_exp", dcol]].merge(
            q_base[["attack_ratio", "m_exp", bcol]],
            on=["attack_ratio", "m_exp"],
            suffixes=("_def", "_base"),
        )
        if m.empty:
            continue
        for attack_ratio, g in m.groupby("attack_ratio", dropna=False):
            g = g[g[f"{dcol}_base"] > 0]
            if g.empty:
                continue
            g = g.sort_values("m_exp")
            uplift_pct = (g[f"{dcol}_def"] / g[f"{dcol}_base"] - 1.0) * 100.0
            if stats is not None and len(g) >= 2:
                _, p_value = stats.ttest_rel(g[f"{dcol}_def"].to_numpy(), g[f"{dcol}_base"].to_numpy())
            else:
                p_value = float("nan")
            query_rows.append(
                {
                    "attack_ratio": float(attack_ratio),
                    "metric": metric,
                    "avg_uplift_pct": float(uplift_pct.mean()),
                    "p_value": float(p_value),
                }
            )

    insert_metric = "insert_qps"
    dcol = f"{insert_metric}_mean" if f"{insert_metric}_mean" in i_def.columns else insert_metric
    bcol = f"{insert_metric}_mean" if f"{insert_metric}_mean" in i_base.columns else insert_metric
    if dcol in i_def.columns and bcol in i_base.columns:
        m = i_def[["attack_ratio", "m_exp", dcol]].merge(
            i_base[["attack_ratio", "m_exp", bcol]],
            on=["attack_ratio", "m_exp"],
            suffixes=("_def", "_base"),
        )
        if not m.empty:
            for attack_ratio, g in m.groupby("attack_ratio", dropna=False):
                g = g[g[f"{bcol}_base"] > 0]
                if g.empty:
                    continue
                g = g.sort_values("m_exp")
                uplift_pct = (g[f"{dcol}_def"] / g[f"{bcol}_base"] - 1.0) * 100.0
                if stats is not None and len(g) >= 2:
                    _, p_value = stats.ttest_rel(g[f"{dcol}_def"].to_numpy(), g[f"{bcol}_base"].to_numpy())
                else:
                    p_value = float("nan")
                insert_rows.append(
                    {
                        "attack_ratio": float(attack_ratio),
                        "metric": insert_metric,
                        "avg_uplift_pct": float(uplift_pct.mean()),
                        "p_value": float(p_value),
                    }
                )

    if not query_rows and not insert_rows:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    if query_rows:
        query_out = pd.DataFrame(query_rows).sort_values(["attack_ratio", "metric"]).reset_index(drop=True)
        query_csv = out_dir / "uplift_summary_query_by_attack_ratio_metric.csv"
        query_out.to_csv(query_csv, index=False)
        print(query_csv)
        print("[uplift][query] 攻击比例\t指标\t平均提升\tp-value")
        for _, r in query_out.iterrows():
            pv = "nan" if pd.isna(r["p_value"]) else f"{r['p_value']:.3g}"
            print(f"{r['attack_ratio']*100:.0f}%\t{r['metric']}\t{r['avg_uplift_pct']:.2f}%\t{pv}")

    if insert_rows:
        insert_out = pd.DataFrame(insert_rows).sort_values(["attack_ratio", "metric"]).reset_index(drop=True)
        insert_csv = out_dir / "uplift_summary_insert_by_attack_ratio_metric.csv"
        insert_out.to_csv(insert_csv, index=False)
        print(insert_csv)
        print("[uplift][insert] 攻击比例\t指标\t平均提升\tp-value")
        for _, r in insert_out.iterrows():
            pv = "nan" if pd.isna(r["p_value"]) else f"{r['p_value']:.3g}"
            print(f"{r['attack_ratio']*100:.0f}%\t{r['metric']}\t{r['avg_uplift_pct']:.2f}%\t{pv}")


def main() -> None:
    global MAX_RUNNABLE_M_EXP

    args = parse_args()
    # propagate defense flags into module globals used when building bench cmd
    global DEF_ADAPTIVE_VERIFY, VERIFY_THRESHOLD, DEF_KEEPSAKE_RLE, DEF_RECONSTRUCT, RECONSTRUCT_THRESHOLD
    DEF_ADAPTIVE_VERIFY = bool(args.def_adaptive_verify)
    VERIFY_THRESHOLD = int(args.verify_threshold)
    DEF_KEEPSAKE_RLE = bool(args.def_keepsake_rle)
    DEF_RECONSTRUCT = bool(args.def_reconstruct)
    RECONSTRUCT_THRESHOLD = int(args.reconstruct_threshold)
    MAX_RUNNABLE_M_EXP = args.max_runnable_m_exp
    repeats = max(1, int(args.repeats))
    seed_base = int(args.seed_base)
    seed_step = int(args.seed_step)
    seeds = [seed_base + i * seed_step for i in range(repeats)]

    selected = {x.strip() for x in args.experiments.split(",") if x.strip()}
    if "all" in selected:
        selected = {"q1", "q2", "i1", "i2", "i3"}

    root = Path(args.root)
    out_dir = root / args.out_dir
    bin_path = root / "build/bench/bench_memento_security"
    q2_df = None
    q2_nd_df = None
    i2_df = None
    i2_nd_df = None

    # ------------------------
    # Query experiment 1
    # ------------------------
    if "q1" in selected:
        print("[run] query_exp1_x_sweep")
        q1_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for ratio in QUERY_EXP1_ATTACK_RATIOS:
            for x in QUERY_EXP1_X_VALUES:
                for repeat_idx, seed in enumerate(seeds):
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
                                "seed": seed,
                                "repeat": repeat_idx,
                            },
                        )
                    )

        q1_rows = run_many(q1_tasks, args.jobs)
        q1_raw = pd.DataFrame(q1_rows).sort_values(["attack_ratio", "x", "repeat"]).reset_index(drop=True)
        save_df(q1_raw, out_dir / "query_exp1_x_sweep_raw.csv")

        q1_df = aggregate_runs(q1_raw).sort_values(["attack_ratio", "x"]).reset_index(drop=True)
        save_df(q1_df, out_dir / "query_exp1_x_sweep.csv")

        q1_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=QUERY_EXP1_M_EXP,
                query_count=QUERY_COUNT,
                case_group="query_exp1_random",
                seed=seed,
                repeat=repeat_idx,
            )
            for repeat_idx, seed in enumerate(seeds)
        ]
        q1_rb_raw = pd.DataFrame(q1_rb_rows)
        save_df(q1_rb_raw, out_dir / "query_exp1_random_baseline_raw.csv")
        q1_rb_df = aggregate_runs(q1_rb_raw).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(q1_rb_df, out_dir / "query_exp1_random_baseline.csv")

        write_ttest_report(
            q1_raw[q1_raw["status"] == "ok"],
            q1_rb_raw[q1_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            [],
            out_dir / "query_exp1_ttest.csv",
        )

    # ------------------------
    # Query experiment 2
    # ------------------------
    if "q2" in selected:
        print("[run] query_exp2_m_sweep")
        defense_enabled = DEF_ADAPTIVE_VERIFY or DEF_KEEPSAKE_RLE or DEF_RECONSTRUCT
        if defense_enabled:
            print("[run] query_exp2_paired_def_vs_no_def")
            q2_paired_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
            for m_exp in QUERY_EXP2_M_EXPS:
                for ratio in QUERY_EXP2_ATTACK_RATIOS:
                    for repeat_idx, seed in enumerate(seeds):
                        # Pair each defended datapoint with its no-defense counterpart.
                        q2_paired_tasks.append(
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
                                    "seed": seed,
                                    "repeat": repeat_idx,
                                },
                            )
                        )
                        q2_paired_tasks.append(
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
                                    "case_group": "query_exp2_no_defense",
                                    "seed": seed,
                                    "repeat": repeat_idx,
                                    "use_defense": False,
                                },
                            )
                        )

            q2_paired_rows = run_many(q2_paired_tasks, args.jobs)
            q2_paired_raw = pd.DataFrame(q2_paired_rows)
            q2_raw = q2_paired_raw[q2_paired_raw["case_group"] == "query_exp2"].sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
            q2_nd_raw = q2_paired_raw[q2_paired_raw["case_group"] == "query_exp2_no_defense"].sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        else:
            q2_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
            for m_exp in QUERY_EXP2_M_EXPS:
                for ratio in QUERY_EXP2_ATTACK_RATIOS:
                    for repeat_idx, seed in enumerate(seeds):
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
                                    "seed": seed,
                                    "repeat": repeat_idx,
                                },
                            )
                        )

            q2_rows = run_many(q2_tasks, args.jobs)
            q2_raw = pd.DataFrame(q2_rows).sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        save_df(q2_raw, out_dir / "query_exp2_m_sweep_raw.csv")

        q2_df = aggregate_runs(q2_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(q2_df, out_dir / "query_exp2_m_sweep.csv")

        q2_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=m_exp,
                query_count=QUERY_COUNT,
                case_group="query_exp2_random",
                seed=seed,
                repeat=repeat_idx,
            )
            for m_exp in QUERY_EXP2_M_EXPS
            for repeat_idx, seed in enumerate(seeds)
        ]
        q2_rb_raw = pd.DataFrame(q2_rb_rows).sort_values(["m_exp", "repeat"]).reset_index(drop=True)
        save_df(q2_rb_raw, out_dir / "query_exp2_random_baseline_raw.csv")

        q2_rb_df = aggregate_runs(q2_rb_raw).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(q2_rb_df, out_dir / "query_exp2_random_baseline.csv")

        if defense_enabled:
            save_df(q2_nd_raw, out_dir / "query_exp2_no_defense_raw.csv")
            q2_nd_df = aggregate_runs(q2_nd_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
            save_df(q2_nd_df, out_dir / "query_exp2_no_defense.csv")

        write_ttest_report(
            q2_raw[q2_raw["status"] == "ok"],
            q2_rb_raw[q2_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            ["m_exp"],
            out_dir / "query_exp2_ttest.csv",
        )

    # ------------------------
    # Query experiment 2 (sq vs x-hot)
    # ------------------------
    if "q2sq" in selected:
        print("[run] query_exp2_m_sweep_sq")
        q2_xhot_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        q2_sq_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for m_exp in QUERY_EXP2_M_EXPS:
            for ratio in QUERY_EXP2_ATTACK_RATIOS:
                for repeat_idx, seed in enumerate(seeds):
                    q2_xhot_tasks.append(
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
                                "case_group": "query_exp2_xhot",
                                "seed": seed,
                                "repeat": repeat_idx,
                            },
                        )
                    )
                    q2_sq_tasks.append(
                        (
                            run_sq_case,
                            (bin_path, root),
                            {
                                "m_exp": m_exp,
                                "ratio": ratio,
                                "query_count": QUERY_COUNT,
                                "case_group": "query_exp2_sq",
                                "seed": seed,
                                "repeat": repeat_idx,
                            },
                        )
                    )

        q2_xhot_rows = run_many(q2_xhot_tasks, args.jobs)
        q2_xhot_raw = pd.DataFrame(q2_xhot_rows).sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        save_df(q2_xhot_raw, out_dir / "query_exp2_m_sweep_xhot_raw.csv")
        q2_xhot_df = aggregate_runs(q2_xhot_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(q2_xhot_df, out_dir / "query_exp2_m_sweep_xhot.csv")

        q2_sq_rows = run_many(q2_sq_tasks, args.jobs)
        q2_sq_raw = pd.DataFrame(q2_sq_rows).sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        save_df(q2_sq_raw, out_dir / "query_exp2_m_sweep_sq_raw.csv")
        q2_sq_df = aggregate_runs(q2_sq_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(q2_sq_df, out_dir / "query_exp2_m_sweep_sq.csv")

        q2_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=m_exp,
                query_count=QUERY_COUNT,
                case_group="query_exp2_random",
                seed=seed,
                repeat=repeat_idx,
            )
            for m_exp in QUERY_EXP2_M_EXPS
            for repeat_idx, seed in enumerate(seeds)
        ]
        q2_rb_raw = pd.DataFrame(q2_rb_rows).sort_values(["m_exp", "repeat"]).reset_index(drop=True)
        save_df(q2_rb_raw, out_dir / "query_exp2_random_baseline_raw.csv")

        q2_rb_df = aggregate_runs(q2_rb_raw).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(q2_rb_df, out_dir / "query_exp2_random_baseline.csv")

        write_ttest_report(
            q2_xhot_raw[q2_xhot_raw["status"] == "ok"],
            q2_rb_raw[q2_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            ["m_exp"],
            out_dir / "query_exp2_xhot_ttest.csv",
        )
        write_ttest_report(
            q2_sq_raw[q2_sq_raw["status"] == "ok"],
            q2_rb_raw[q2_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            ["m_exp"],
            out_dir / "query_exp2_sq_ttest.csv",
        )

    # ------------------------
    # Insert experiment 1
    # ------------------------
    if "i1" in selected:
        print("[run] insert_exp1_x_sweep")
        i1_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for ratio in INSERT_EXP1_ATTACK_RATIOS:
            for x in INSERT_EXP1_X_VALUES:
                for repeat_idx, seed in enumerate(seeds):
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
                                "seed": seed,
                                "repeat": repeat_idx,
                            },
                        )
                    )

        i1_rows = run_many(i1_tasks, args.jobs)
        i1_raw = pd.DataFrame(i1_rows).sort_values(["attack_ratio", "x", "repeat"]).reset_index(drop=True)
        save_df(i1_raw, out_dir / "insert_exp1_x_sweep_raw.csv")

        i1_df = aggregate_runs(i1_raw).sort_values(["attack_ratio", "x"]).reset_index(drop=True)
        save_df(i1_df, out_dir / "insert_exp1_x_sweep.csv")

    if "i1" in selected or "i1rb" in selected:
        i1_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=INSERT_EXP1_M_EXP,
                query_count=INSERT_ONLY_QUERY_COUNT,
                case_group="insert_exp1_random",
                seed=seed,
                repeat=repeat_idx,
            )
            for repeat_idx, seed in enumerate(seeds)
        ]
        i1_rb_raw = pd.DataFrame(i1_rb_rows)
        save_df(i1_rb_raw, out_dir / "insert_exp1_random_baseline_raw.csv")
        i1_rb_df = aggregate_runs(i1_rb_raw).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(i1_rb_df, out_dir / "insert_exp1_random_baseline.csv")

        write_ttest_report(
            i1_raw[i1_raw["status"] == "ok"],
            i1_rb_raw[i1_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            [],
            out_dir / "insert_exp1_ttest.csv",
        )

    # ------------------------
    # Insert experiment 2
    # ------------------------
    if "i2" in selected:
        print("[run] insert_exp2_m_sweep")
        defense_enabled = DEF_ADAPTIVE_VERIFY or DEF_KEEPSAKE_RLE or DEF_RECONSTRUCT
        if defense_enabled:
            print("[run] insert_exp2_paired_def_vs_no_def")
            i2_paired_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
            for m_exp in INSERT_EXP2_M_EXPS:
                for ratio in INSERT_EXP2_ATTACK_RATIOS:
                    for repeat_idx, seed in enumerate(seeds):
                        i2_paired_tasks.append(
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
                                    "seed": seed,
                                    "repeat": repeat_idx,
                                },
                            )
                        )
                        i2_paired_tasks.append(
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
                                    "case_group": "insert_exp2_no_defense",
                                    "seed": seed,
                                    "repeat": repeat_idx,
                                    "use_defense": False,
                                },
                            )
                        )

            i2_paired_rows = run_many(i2_paired_tasks, args.jobs)
            i2_paired_raw = pd.DataFrame(i2_paired_rows)
            i2_raw = i2_paired_raw[i2_paired_raw["case_group"] == "insert_exp2"].sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
            i2_nd_raw = i2_paired_raw[i2_paired_raw["case_group"] == "insert_exp2_no_defense"].sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        else:
            i2_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
            for m_exp in INSERT_EXP2_M_EXPS:
                for ratio in INSERT_EXP2_ATTACK_RATIOS:
                    for repeat_idx, seed in enumerate(seeds):
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
                                    "seed": seed,
                                    "repeat": repeat_idx,
                                },
                            )
                        )

            i2_rows = run_many(i2_tasks, args.jobs)
            i2_raw = pd.DataFrame(i2_rows).sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        save_df(i2_raw, out_dir / "insert_exp2_m_sweep_raw.csv")

        i2_df = aggregate_runs(i2_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(i2_df, out_dir / "insert_exp2_m_sweep.csv")

    if "i2" in selected or "i2rb" in selected:
        i2_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=m_exp,
                query_count=INSERT_ONLY_QUERY_COUNT,
                case_group="insert_exp2_random",
                seed=seed,
                repeat=repeat_idx,
            )
            for m_exp in INSERT_EXP2_M_EXPS
            for repeat_idx, seed in enumerate(seeds)
        ]
        i2_rb_raw = pd.DataFrame(i2_rb_rows).sort_values(["m_exp", "repeat"]).reset_index(drop=True)
        save_df(i2_rb_raw, out_dir / "insert_exp2_random_baseline_raw.csv")
        i2_rb_df = aggregate_runs(i2_rb_raw).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(i2_rb_df, out_dir / "insert_exp2_random_baseline.csv")

        if defense_enabled:
            save_df(i2_nd_raw, out_dir / "insert_exp2_no_defense_raw.csv")
            i2_nd_df = aggregate_runs(i2_nd_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
            save_df(i2_nd_df, out_dir / "insert_exp2_no_defense.csv")

        write_ttest_report(
            i2_raw[i2_raw["status"] == "ok"],
            i2_rb_raw[i2_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            ["m_exp"],
            out_dir / "insert_exp2_ttest.csv",
        )

    if q2_df is not None and q2_nd_df is not None and i2_df is not None and i2_nd_df is not None:
        summarize_uplift_by_ratio_metric(
            q2_df,
            q2_nd_df,
            i2_df,
            i2_nd_df,
            out_dir,
        )

    # ------------------------
    # Insert experiment 2 (sq vs x-hot-zipf)
    # ------------------------
    if "i2sq" in selected:
        print("[run] insert_exp2_m_sweep_sq")
        i2_xhot_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        i2_sq_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for m_exp in INSERT_EXP2_M_EXPS:
            for ratio in INSERT_EXP2_ATTACK_RATIOS:
                for repeat_idx, seed in enumerate(seeds):
                    i2_xhot_tasks.append(
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
                                "case_group": "insert_exp2_xhot",
                                "seed": seed,
                                "repeat": repeat_idx,
                            },
                        )
                    )
                    i2_sq_tasks.append(
                        (
                            run_sq_case,
                            (bin_path, root),
                            {
                                "m_exp": m_exp,
                                "ratio": ratio,
                                "query_count": INSERT_ONLY_QUERY_COUNT,
                                "case_group": "insert_exp2_sq",
                                "seed": seed,
                                "repeat": repeat_idx,
                            },
                        )
                    )

        i2_xhot_rows = run_many(i2_xhot_tasks, args.jobs)
        i2_xhot_raw = pd.DataFrame(i2_xhot_rows).sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        save_df(i2_xhot_raw, out_dir / "insert_exp2_m_sweep_xhot_raw.csv")
        i2_xhot_df = aggregate_runs(i2_xhot_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(i2_xhot_df, out_dir / "insert_exp2_m_sweep_xhot.csv")

        i2_sq_rows = run_many(i2_sq_tasks, args.jobs)
        i2_sq_raw = pd.DataFrame(i2_sq_rows).sort_values(["m_exp", "attack_ratio", "repeat"]).reset_index(drop=True)
        save_df(i2_sq_raw, out_dir / "insert_exp2_m_sweep_sq_raw.csv")
        i2_sq_df = aggregate_runs(i2_sq_raw).sort_values(["m_exp", "attack_ratio"]).reset_index(drop=True)
        save_df(i2_sq_df, out_dir / "insert_exp2_m_sweep_sq.csv")

        i2_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=m_exp,
                query_count=INSERT_ONLY_QUERY_COUNT,
                case_group="insert_exp2_random",
                seed=seed,
                repeat=repeat_idx,
            )
            for m_exp in INSERT_EXP2_M_EXPS
            for repeat_idx, seed in enumerate(seeds)
        ]
        i2_rb_raw = pd.DataFrame(i2_rb_rows).sort_values(["m_exp", "repeat"]).reset_index(drop=True)
        save_df(i2_rb_raw, out_dir / "insert_exp2_random_baseline_raw.csv")
        i2_rb_df = aggregate_runs(i2_rb_raw).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(i2_rb_df, out_dir / "insert_exp2_random_baseline.csv")

        write_ttest_report(
            i2_xhot_raw[i2_xhot_raw["status"] == "ok"],
            i2_rb_raw[i2_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            ["m_exp"],
            out_dir / "insert_exp2_xhot_ttest.csv",
        )
        write_ttest_report(
            i2_sq_raw[i2_sq_raw["status"] == "ok"],
            i2_rb_raw[i2_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            ["m_exp"],
            out_dir / "insert_exp2_sq_ttest.csv",
        )

    # ------------------------
    # Insert experiment 3
    # ------------------------
    if "i3" in selected:
        print("[run] insert_exp3_skew_sweep")
        i3_tasks: list[tuple[Any, tuple[Any, ...], dict[str, Any]]] = []
        for ratio in INSERT_EXP3_ATTACK_RATIOS:
            for skew in INSERT_EXP3_ZIPF_S_VALUES:
                for repeat_idx, seed in enumerate(seeds):
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
                                "seed": seed,
                                "repeat": repeat_idx,
                            },
                        )
                    )

        i3_rows = run_many(i3_tasks, args.jobs)
        i3_raw = pd.DataFrame(i3_rows).sort_values(["attack_ratio", "zipf_s", "repeat"]).reset_index(drop=True)
        save_df(i3_raw, out_dir / "insert_exp3_skew_sweep_raw.csv")

        i3_df = aggregate_runs(i3_raw).sort_values(["attack_ratio", "zipf_s"]).reset_index(drop=True)
        save_df(i3_df, out_dir / "insert_exp3_skew_sweep.csv")

    if "i3" in selected or "i3rb" in selected:
        i3_rb_rows = [
            run_random_case(
                bin_path,
                root,
                m_exp=INSERT_EXP3_M_EXP,
                query_count=INSERT_ONLY_QUERY_COUNT,
                case_group="insert_exp3_random",
                seed=seed,
                repeat=repeat_idx,
            )
            for repeat_idx, seed in enumerate(seeds)
        ]
        i3_rb_raw = pd.DataFrame(i3_rb_rows)
        save_df(i3_rb_raw, out_dir / "insert_exp3_random_baseline_raw.csv")
        i3_rb_df = aggregate_runs(i3_rb_raw).sort_values(["m_exp"]).reset_index(drop=True)
        save_df(i3_rb_df, out_dir / "insert_exp3_random_baseline.csv")

        write_ttest_report(
            i3_raw[i3_raw["status"] == "ok"],
            i3_rb_raw[i3_rb_raw["status"] == "ok"],
            ["case_group", "m_exp", "x", "attack_ratio", "schedule", "zipf_s"],
            [],
            out_dir / "insert_exp3_ttest.csv",
        )

    config = {
        "BPK": BPK,
        "MEMENTO_BITS": MEMENTO_BITS,
        "TARGET_LOAD": TARGET_LOAD,
        "SEED_BASE": seed_base,
        "SEED_STEP": seed_step,
        "REPEATS": repeats,
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
