#!/bin/bash

#
# Automated experiment script for different query scales
# Usage: bash auto_experiment.sh
#

set -e  # Exit on any error

# Configuration
N_KEYS=200000
QUERY_MULTIPLIERS=(1 5 10 15 20 25 30 35 40 45 50)
BASE_DIR="/home/zq/code2"
MEMENTO_BUILD_PATH="${BASE_DIR}/Memento_Filter/build"
REAL_DATASETS_PATH="${BASE_DIR}/paper_results/real_datasets"
RESULTS_BASE_DIR="${BASE_DIR}/paper_results"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

function log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

function log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

function log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

function log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

function print_header() {
    echo -e "${BLUE}===============================================${NC}"
    echo -e "${BLUE}  Automated Memento Filter Experiment Suite  ${NC}"
    echo -e "${BLUE}===============================================${NC}"
    echo -e "N_KEYS: ${N_KEYS}"
    echo -e "Query multipliers: ${QUERY_MULTIPLIERS[*]}"
    echo -e "Base directory: ${BASE_DIR}"
    echo ""
}

function cleanup_previous_results() {
    local suffix=$1
    log_info "Cleaning up previous results for suffix: $suffix"
    
    # Remove previous workloads and results with this suffix
    if [ -d "${RESULTS_BASE_DIR}/workloads${suffix}" ]; then
        rm -rf "${RESULTS_BASE_DIR}/workloads${suffix}"
        log_info "Removed old workloads${suffix}"
    fi
    
    if [ -d "${RESULTS_BASE_DIR}/results${suffix}" ]; then
        rm -rf "${RESULTS_BASE_DIR}/results${suffix}"
        log_info "Removed old results${suffix}"
    fi
    
    if [ -d "${RESULTS_BASE_DIR}/figures${suffix}" ]; then
        rm -rf "${RESULTS_BASE_DIR}/figures${suffix}"
        log_info "Removed old figures${suffix}"
    fi
}

function build_memento() {
    log_info "Building Memento Filter..."
    cd "${MEMENTO_BUILD_PATH}"
    if ! make -j8; then
        log_error "Failed to build Memento Filter"
        exit 1
    fi
    log_success "Memento Filter built successfully"
}

function generate_datasets() {
    local n_queries=$1
    local suffix=$2
    
    log_info "Generating datasets for N_QUERIES=${n_queries} (suffix: ${suffix})"
    
    cd "${RESULTS_BASE_DIR}"
    
    # Temporarily modify the generate_datasets.sh script
    local temp_script="/tmp/generate_datasets_temp.sh"
    cp "${BASE_DIR}/Memento_Filter/bench/scripts/generate_datasets.sh" "$temp_script"
    
    # Modify the N_KEYS and N_QUERIES values in the temp script
    sed -i "s/N_KEYS=200000/N_KEYS=${N_KEYS}/" "$temp_script"
    sed -i "s/N_QUERIES=10000000/N_QUERIES=${n_queries}/" "$temp_script"
    
    # Make it executable
    chmod +x "$temp_script"
    
    # Run the modified script
    if ! bash "$temp_script" "${MEMENTO_BUILD_PATH}" "${REAL_DATASETS_PATH}" -f fpr; then
        log_error "Failed to generate datasets for N_QUERIES=${n_queries}"
        rm -f "$temp_script"
        exit 1
    fi
    
    # Rename the workloads directory
    if [ -d "workloads" ]; then
        mv "workloads" "workloads${suffix}"
        log_success "Generated datasets saved to workloads${suffix}"
    else
        log_error "Workloads directory not found after generation"
        rm -f "$temp_script"
        exit 1
    fi
    
    rm -f "$temp_script"
}

function execute_tests() {
    local suffix=$1
    
    log_info "Executing tests for suffix: ${suffix}"
    
    cd "${RESULTS_BASE_DIR}"
    
    if ! bash "${BASE_DIR}/Memento_Filter/bench/scripts/execute_tests.sh" "${MEMENTO_BUILD_PATH}" "workloads${suffix}" -f fpr; then
        log_error "Failed to execute tests for suffix: ${suffix}"
        exit 1
    fi
    
    # Rename the results directory
    if [ -d "results" ]; then
        mv "results" "results${suffix}"
        log_success "Test results saved to results${suffix}"
    else
        log_error "Results directory not found after test execution"
        exit 1
    fi
}

function generate_plots() {
    local suffix=$1
    
    log_info "Generating plots for suffix: ${suffix}"
    
    cd "${RESULTS_BASE_DIR}"
    
    # Create figures directory with suffix
    mkdir -p "figures${suffix}"
    
    if ! python3 "${BASE_DIR}/Memento_Filter/bench/scripts/plot.py" \
        --result_dir "./results${suffix}" \
        --figure_dir "./figures${suffix}" \
        --figures all; then
        log_error "Failed to generate plots for suffix: ${suffix}"
        exit 1
    fi
    
    log_success "Plots generated in figures${suffix}"
}

function run_single_experiment() {
    local multiplier=$1
    local n_queries=$((N_KEYS * multiplier))
    local suffix="_tiny_qs_${multiplier}"
    
    echo ""
    log_info "=========================================="
    log_info "Running experiment ${multiplier}x: N_KEYS=${N_KEYS}, N_QUERIES=${n_queries}"
    log_info "Suffix: ${suffix}"
    log_info "=========================================="
    
    # Step 1: Cleanup previous results
    cleanup_previous_results "$suffix"
    
    # Step 2: Build Memento Filter
    build_memento
    
    # Step 3: Generate datasets
    generate_datasets "$n_queries" "$suffix"
    
    # Step 4: Execute tests
    execute_tests "$suffix"
    
    # Step 5: Generate plots
    generate_plots "$suffix"
    
    log_success "Completed experiment ${multiplier}x successfully!"
}

function main() {
    print_header
    
    # Check if required directories exist
    if [ ! -d "$MEMENTO_BUILD_PATH" ]; then
        log_error "Memento build path does not exist: $MEMENTO_BUILD_PATH"
        exit 1
    fi
    
    if [ ! -d "$REAL_DATASETS_PATH" ]; then
        log_error "Real datasets path does not exist: $REAL_DATASETS_PATH"
        exit 1
    fi
    
    # Record start time
    local start_time=$(date +%s)
    
    # Run experiments for each multiplier
    local total_experiments=${#QUERY_MULTIPLIERS[@]}
    local current_experiment=0
    
    for multiplier in "${QUERY_MULTIPLIERS[@]}"; do
        current_experiment=$((current_experiment + 1))
        
        log_info "Progress: ${current_experiment}/${total_experiments} experiments"
        
        if ! run_single_experiment "$multiplier"; then
            log_error "Experiment ${multiplier}x failed"
            exit 1
        fi
        
        # Calculate and display progress
        local elapsed=$(($(date +%s) - start_time))
        local avg_time_per_exp=$((elapsed / current_experiment))
        local remaining_experiments=$((total_experiments - current_experiment))
        local estimated_remaining=$((avg_time_per_exp * remaining_experiments))
        
        log_info "Elapsed: ${elapsed}s, Estimated remaining: ${estimated_remaining}s"
    done
    
    # Final summary
    local total_time=$(($(date +%s) - start_time))
    echo ""
    log_success "==============================================="
    log_success "All experiments completed successfully!"
    log_success "Total time: ${total_time} seconds"
    log_success "Results saved in:"
    for multiplier in "${QUERY_MULTIPLIERS[@]}"; do
        echo "  - workloads_tiny_qs_${multiplier}"
        echo "  - results_tiny_qs_${multiplier}"
        echo "  - figures_tiny_qs_${multiplier}"
    done
    log_success "==============================================="
}

# Handle Ctrl+C gracefully
trap 'log_error "Script interrupted by user"; exit 1' INT

# Run main function
main "$@"
