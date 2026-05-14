#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

JOBS="${JOBS:-24}"
REPEATS="${REPEATS:-5}"
VERIFY_THRESHOLD="${VERIFY_THRESHOLD:-32}"

PYTHON_BIN="${PYTHON_BIN:-python3}"
if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi

pushd "${ROOT_DIR}" >/dev/null

run_experiment() {
  local name="$1"
  shift
  local out_dir="paper_results/results/throughput_def/${name}"
  local fig_dir="paper_results/figures/throughput_def/${name}"

  echo "[run] ${name}"
  "${PYTHON_BIN}" -u bench/scripts/run_throughput_experiments.py \
    --root . \
    --experiments q2,i2 \
    --jobs "${JOBS}" \
    --repeats "${REPEATS}" \
    --out-dir "${out_dir}" \
    "$@"

  echo "[plot] ${name}"
  "${PYTHON_BIN}" -u bench/scripts/plot_throughput_experiments.py \
    --result-dir "${out_dir}" \
    --baseline-dir "${out_dir}" \
    --figure-dir "${fig_dir}" \
    --figures q2,i2,q2u,i2u \
    --formats pdf,svg
}

run_experiment def32_adaptive_v2 --def-adaptive-verify --verify-threshold "${VERIFY_THRESHOLD}"
run_experiment def32_rle_v2 --def-keepsake-rle

"${PYTHON_BIN}" - <<'PY'
import pandas as pd
from pathlib import Path

root = Path("paper_results/results/throughput_def")
rows = []
for defense, d in [("adaptive", root / "def32_adaptive_v2"), ("rle", root / "def32_rle_v2")]:
    for workload, fn in [
        ("query", "uplift_summary_query_by_attack_ratio_metric.csv"),
        ("insert", "uplift_summary_insert_by_attack_ratio_metric.csv"),
    ]:
        p = d / fn
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df.insert(0, "defense", defense)
        df.insert(1, "workload", workload)
        rows.append(df)

if rows:
    out = pd.concat(rows, ignore_index=True)
    out = out[["defense", "workload", "attack_ratio", "metric", "avg_uplift_pct", "p_value"]]
    out = out.sort_values(["defense", "workload", "attack_ratio", "metric"]).reset_index(drop=True)
    out_path = root / "uplift_pvalue_summary_def32_v2.csv"
    out.to_csv(out_path, index=False)
    print(out_path)
    print(out.to_string(index=False))
PY

echo "[done] results and figures are under paper_results/results/throughput_def and paper_results/figures/throughput_def"

popd >/dev/null