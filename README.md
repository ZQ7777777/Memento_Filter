# Memento Filter–Oriented Attacks and Potential Countermeasures

This repository contains the source code, attack implementations, and experimental evaluation for the paper **"Memento Filter–Oriented Attacks and Potential Countermeasures"**. 

We explore performance degradation attacks against the Memento Filter, demonstrating how targeted adversarial insertions (such as `x-hot` and `x-hot-zipf` workloads) can severely degrade both query and insertion throughput by exploiting structural block saturations, both under white-box and black-box threat models.

## Reproducibility

To ease the reproducibility process, we provide Python scripts to automate the execution of the attack benchmarks and the generation of the figures used in the paper's experimental section.

### Prerequisites
- CMake >= 3.10
- A modern C++ compiler (C++17 or later)
- Python 3 >= 3.7
- Pandas and Matplotlib for plotting (`pip install pandas matplotlib`)
- SciPy for statistical tests (`pip install scipy`)
- Inkscape (Optional, used for `.emf` vector figure export)

### 1. Compilation

First, clone the repository and build the benchmarking targets:

```bash
git clone https://github.com/<your-username>/Memento_Filter.git
cd Memento_Filter
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make bench_memento_security -j$(nproc)
cd ..
```

### 2. Running the Latest Defense Experiments

The latest defense comparison uses paired `q2/i2` runs for both `adaptive` and `rle`, with `repeats=5`. A single shell script now runs the experiments, generates the figures, and writes a combined uplift + p-value summary:

```bash
bash bench/scripts/run_latest_defense_experiments.sh
```

By default, the script writes results to:

```bash
paper_results/results/throughput_def/def32_adaptive_v2
paper_results/results/throughput_def/def32_rle_v2
```

and figures to:

```bash
paper_results/figures/throughput_def/def32_adaptive_v2
paper_results/figures/throughput_def/def32_rle_v2
```

If you want to override the runtime, you can set environment variables before launching the script:

```bash
JOBS=24 REPEATS=5 VERIFY_THRESHOLD=32 bash bench/scripts/run_latest_defense_experiments.sh
```

### 3. Manual Experiment Runner

If you prefer to invoke the Python runner directly, use relative paths from the repository root:

```bash
python3 bench/scripts/run_throughput_experiments.py \
    --root . \
    --experiments q2,i2 \
    --jobs 24 \
    --repeats 5 \
    --out-dir paper_results/results/throughput_def/def32_adaptive_v2 \
    --def-adaptive-verify \
    --verify-threshold 32
```

To render figures for a result directory:

```bash
python3 bench/scripts/plot_throughput_experiments.py \
    --result-dir paper_results/results/throughput_def/def32_adaptive_v2 \
    --baseline-dir paper_results/results/throughput_def/def32_adaptive_v2 \
    --figure-dir paper_results/figures/throughput_def/def32_adaptive_v2 \
    --figures q2,i2,q2u,i2u \
    --formats pdf,svg
```

### 4. Notes on Statistical Outputs

The throughput runner emits both raw and aggregated CSV files. Aggregated files include mean, standard deviation, and 95% confidence intervals (`*_mean`, `*_std`, `*_ci95`).
It also writes per-experiment t-test reports (for example, `query_exp2_ttest.csv` and `insert_exp2_ttest.csv`) and uplift summaries by attack ratio and metric with p-values.

## Repository Structure

- `bench/filters_benchmark/bench_memento_security.cpp`: The core C++ benchmarking implementation of the `x-hot` and `x-hot-zipf` adversarial workloads and throughput timing logic.
- `bench/scripts/`: Contains the automation (`run_throughput_experiments.py`), one-click defense wrapper (`run_latest_defense_experiments.sh`), and visualization (`plot_throughput_experiments.py`) scripts.
- `include/` and `src/`: The Memento Filter internal data structure files.

## Original Memento Filter
For the original Memento Filter implementation, please refer to the base repository or the paper *Memento Filter: A Fast, Dynamic, and Robust Range Filter*.

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.
