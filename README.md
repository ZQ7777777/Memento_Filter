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

### 2. Running and Rendering the Main Experimental Figures

The paper's core experimental results are presented in five key figures. To run the main experiments and generate all figures in one command:

#### Generate Primary Experimental Data

First, run the main throughput experiments (for query and insert operations across various attack ratios and parameters):

```bash
python3 bench/scripts/run_throughput_experiments.py \
    --root . \
    --experiments q1,q2,i1,i2,i3 \
    --jobs 24 \
    --repeats 5 \
    --out-dir paper_results/results/throughput_full
```

This generates raw and aggregated CSV files in `paper_results/results/throughput_full/`.

#### Render the Five Main Figures

Once data is available, render all five figures with confidence-interval bands for attack-ratio curves:

```bash
python3 bench/scripts/plot_throughput_experiments.py \
    --result-dir paper_results/results/throughput_full \
    --baseline-dir paper_results/results/throughput_full \
    --figure-dir paper_results/figures/throughput_full \
    --figures i1,i2,i3,q1sat,q1,q2 \
    --formats pdf,svg,emf
```

This produces the five main figures:

1. **`insert_exp1_throughput_vs_x`** — Insert throughput degradation across x-sweep attack ratios (1%, 5%, 10%, 100%)
2. **`insert_exp2_throughput_vs_m`** — Insert throughput vs. filter size m, with shaded CI bands for all attack-ratio curves
3. **`insert_exp3_throughput_vs_skewness`** — Insert throughput under varying Zipf skewness
4. **`query_exp1_saturation_vs_x`** — Filter saturation ratio (blocked fragments) as a function of x-sweep attacks
5. **`query_exp12_combined`** — Combined view: first row shows query throughput x-sweep; second row shows m-sweep comparison; both use shaded CI bands for attack-ratio curves

**Note on visualization:** 
- All main attack-ratio curves use shaded confidence-interval bands (darker gray, 28% opacity) to reduce visual clutter and clearly distinguish uncertainty from data variability.
- This design choice improves readability and emphasizes the attack severity across different experimental settings.

### 3. Running the Latest Defense Experiments

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
