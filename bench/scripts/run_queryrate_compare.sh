#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BENCH_BIN="${BUILD_DIR}/bench/bench_memento_overflow"
OUT_DIR="${ROOT_DIR}/paper_results/results/overflow_sweep"

mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_CSV="${OUT_DIR}/queryrate_fp13_m10_n5000_q20000_${STAMP}.csv"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target bench_memento_overflow -j4

for x in $(seq 1 256); do
  "${BENCH_BIN}" 26.5 \
    --mode x-hot \
    --hot-slots "${x}" \
    --requested-inserts 5000 \
    --memento-size 10 \
    --query-count 20000 \
    --csv "${OUT_CSV}" >/dev/null
  printf '[+] x-hot x=%s done\n' "${x}"
done

"${BENCH_BIN}" 26.5 \
  --mode random \
  --requested-inserts 5000 \
  --memento-size 10 \
  --query-count 20000 \
  --csv "${OUT_CSV}" >/dev/null

printf '[+] csv: %s\n' "${OUT_CSV}"
