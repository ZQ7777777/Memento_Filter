# On the Security of Memento Filter: Attacks and Potential Countermeasures

This repository contains the source code, attack implementations, and experimental evaluation for the paper **"On the Security of Memento Filter: Attacks and Potential Countermeasures"**. 

We explore performance degradation attacks against the Memento Filter, demonstrating how targeted adversarial insertions (such as `x-hot` and `x-hot-zipf` workloads) can severely degrade both query and insertion throughput by exploiting structural block saturations, both under white-box and black-box threat models.

## Reproducibility

To ease the reproducibility process, we provide Python scripts to automate the execution of the attack benchmarks and the generation of the figures used in the paper's experimental section.

### Prerequisites
- CMake >= 3.10
- A modern C++ compiler (C++17 or later)
- Python 3 >= 3.7
- Pandas and Matplotlib for plotting (`pip install pandas matplotlib`)
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

### 2. Running the Attacks

To run the full suite of throughput degradation experiments (sweeping over attack ratios, memory sizes `m`, target ranges `x`, and Zipf skewness `s`):

```bash
mkdir -p paper_results/results/throughput_full
python3 bench/scripts/run_throughput_experiments.py \
    --experiments all \
    --jobs 24 \
    --out-dir paper_results/results/throughput_full
```
*Note: You can adjust the `--jobs` flag based on your machine's available CPU cores.*

### 3. Generating Figures

After the benchmarks are complete, generate the publication-ready figures (available in PDF, SVG, and EMF formats) using the automated plotting script:

```bash
mkdir -p paper_results/figures/throughput_full
python3 bench/scripts/plot_throughput_experiments.py \
    --result-dir paper_results/results/throughput_full \
    --figure-dir paper_results/figures/throughput_full \
    --figures all \
    --formats pdf,svg,emf
```
The generated figures (e.g., `query_exp12_combined.pdf`, `insert_exp1_throughput_vs_x.pdf`) directly correspond to the plots evaluating query and insertion degradation in the paper.

## Repository Structure

- `bench/filters_benchmark/bench_memento_security.cpp`: The core C++ benchmarking implementation of the `x-hot` and `x-hot-zipf` adversarial workloads and throughput timing logic.
- `bench/scripts/`: Contains the automation (`run_throughput_experiments.py`) and visualization (`plot_throughput_experiments.py`) Python scripts.
- `include/` and `src/`: The Memento Filter internal data structure files.

## Original Memento Filter
For the original Memento Filter implementation, please refer to the base repository or the paper *Memento Filter: A Fast, Dynamic, and Robust Range Filter*.

## License

This project is licensed under the GPLv3 License - see the [LICENSE](LICENSE) file for details.
