#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BUILD_DIR="${ROOT_DIR}/build"
BENCH_BIN="${BUILD_DIR}/bench/bench_memento_overflow"
OUT_DIR="${ROOT_DIR}/paper_results/results/overflow_sweep"

mkdir -p "${OUT_DIR}"

STAMP="$(date +%Y%m%d_%H%M%S)"
MATRIX_CSV="${OUT_DIR}/overflow_matrix_${STAMP}.csv"
THRESHOLD_CSV="${OUT_DIR}/overflow_threshold_m8_${STAMP}.csv"
XHOT_CSV="${OUT_DIR}/overflow_xhot_m8_${STAMP}.csv"

printf '[+] root=%s\n' "${ROOT_DIR}"
printf '[+] build_dir=%s\n' "${BUILD_DIR}"
printf '[+] matrix_csv=%s\n' "${MATRIX_CSV}"
printf '[+] threshold_csv=%s\n' "${THRESHOLD_CSV}"
printf '[+] xhot_csv=%s\n' "${XHOT_CSV}"

cmake -S "${ROOT_DIR}" -B "${BUILD_DIR}"
cmake --build "${BUILD_DIR}" --target bench_memento_overflow -j4

# ------------------------------------------------------------
# Sweep A: broad matrix (mode x memento_bits x bpk)
# ------------------------------------------------------------
for mode in single-prefix dual-hot; do
  for m in 4 6 8 10; do
    for bpk in 10 12 14; do
      printf '[+] matrix run: mode=%s m=%s bpk=%s\n' "${mode}" "${m}" "${bpk}"
      "${BENCH_BIN}" "${bpk}" \
        --mode "${mode}" \
        --memento-size "${m}" \
        --requested-inserts 8192 \
        --target-load 0.95 \
        --seed 1380 \
        --csv "${MATRIX_CSV}"
    done
  done
done

# ------------------------------------------------------------
# Sweep B: threshold curve around overflow for memento_bits=8
# ------------------------------------------------------------
for mode in single-prefix dual-hot; do
  for req in 64 96 128 160 192 208 224 240 248 252 256 260 320 384 512 768 1024; do
    printf '[+] threshold run: mode=%s req=%s\n' "${mode}" "${req}"
    "${BENCH_BIN}" 12 \
      --mode "${mode}" \
      --memento-size 8 \
      --requested-inserts "${req}" \
      --target-load 0.95 \
      --seed 1380 \
      --csv "${THRESHOLD_CSV}"
  done
done

# ------------------------------------------------------------
# Sweep C: contiguous x-hot (q..q+x-1) at memento_bits=8
# ------------------------------------------------------------
for x in 1 2 3 4 8 16 32; do
  printf '[+] xhot run: x=%s\n' "${x}"
  "${BENCH_BIN}" 12 \
    --mode x-hot \
    --hot-slots "${x}" \
    --memento-size 8 \
    --requested-inserts 8192 \
    --target-load 0.95 \
    --seed 1380 \
    --csv "${XHOT_CSV}"
done

printf '[+] done.\n'
printf '[+] matrix_csv: %s\n' "${MATRIX_CSV}"
printf '[+] threshold_csv: %s\n' "${THRESHOLD_CSV}"
printf '[+] xhot_csv: %s\n' "${XHOT_CSV}"
