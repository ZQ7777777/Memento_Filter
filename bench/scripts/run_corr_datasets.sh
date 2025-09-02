#!/bin/bash

#
# Automated script to run correlation tests on multiple datasets
# This script processes corr_test_tiny_qs_{1,5,10,15,20,25,30,35,40,45,50} datasets
# by renaming each to corr_test, running the experiment, and storing results with dataset suffix
#

# Configuration
DATASETS=("1" "5" "10" "15" "20" "25" "30" "35" "40" "45" "50")
WORKLOADS_DIR="./workloads"
RESULTS_DIR="./results"
MEMENTO_BUILD_PATH="../Memento_Filter/build"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to backup existing corr_test if it exists
backup_corr_test() {
    if [ -d "$WORKLOADS_DIR/corr_test" ]; then
        backup_name="corr_test_backup_$(date +%Y%m%d_%H%M%S)"
        print_warning "Existing corr_test found, backing up to $backup_name"
        mv "$WORKLOADS_DIR/corr_test" "$WORKLOADS_DIR/$backup_name"
    fi
}

# Function to restore original corr_test from backup
restore_corr_test() {
    # Find the most recent backup
    local backup=$(ls -t "$WORKLOADS_DIR"/corr_test_backup_* 2>/dev/null | head -n 1)
    if [ -n "$backup" ]; then
        print_status "Restoring original corr_test from backup"
        rm -rf "$WORKLOADS_DIR/corr_test" 2>/dev/null
        mv "$backup" "$WORKLOADS_DIR/corr_test"
    fi
}

# Function to run experiment for a specific dataset
run_experiment() {
    local dataset_suffix=$1
    local source_dataset="corr_test_tiny_qs_$dataset_suffix"
    local target_dataset="corr_test"
    
    print_status "Processing dataset: $source_dataset"
    
    # Check if source dataset exists
    if [ ! -d "$WORKLOADS_DIR/$source_dataset" ]; then
        print_error "Dataset $source_dataset not found in $WORKLOADS_DIR"
        return 1
    fi
    
    # Remove existing corr_test if it exists
    if [ -d "$WORKLOADS_DIR/$target_dataset" ]; then
        rm -rf "$WORKLOADS_DIR/$target_dataset"
    fi
    
    # Create symlink or copy the dataset
    print_status "Linking $source_dataset to $target_dataset"
    ln -s "$source_dataset" "$WORKLOADS_DIR/$target_dataset"
    
    # Build the project
    print_status "Building Memento Filter..."
    cd "$MEMENTO_BUILD_PATH"
    if ! make -j8; then
        print_error "Build failed for dataset $source_dataset"
        rm -rf "$WORKLOADS_DIR/$target_dataset"
        return 1
    fi
    
    # Run the experiment
    print_status "Running correlation experiment for $source_dataset..."
    cd /home/zq/code2/paper_results
    
    # Execute the test
    if ! bash ../Memento_Filter/bench/scripts/execute_tests.sh ../Memento_Filter/build workloads -f correlated; then
        print_error "Experiment failed for dataset $source_dataset"
        rm -rf "$WORKLOADS_DIR/$target_dataset"
        return 1
    fi
    
    # Find the most recent results directory
    local latest_result=$(ls -t "$RESULTS_DIR"/corr_test/ | head -n 1)
    if [ -n "$latest_result" ]; then
        local new_result_name="${latest_result%.*}_qs_$dataset_suffix.${latest_result##*.}"
        print_status "Renaming results from $latest_result to $new_result_name"
        mv "$RESULTS_DIR/corr_test/$latest_result" "$RESULTS_DIR/corr_test/$new_result_name"
        print_success "Results saved as: $new_result_name"
    else
        print_warning "No results found for dataset $source_dataset"
    fi
    
    # Clean up the symlink
    rm -rf "$WORKLOADS_DIR/$target_dataset"
    
    print_success "Completed processing dataset: $source_dataset"
    echo "----------------------------------------"
}

# Main execution function
main() {
    print_status "Starting automated correlation test suite"
    print_status "Processing ${#DATASETS[@]} datasets: ${DATASETS[*]}"
    echo "========================================"
    
    # Backup existing corr_test
    backup_corr_test
    
    # Create results directory if it doesn't exist
    mkdir -p "$RESULTS_DIR/corr_test"
    
    # Track success and failure counts
    local success_count=0
    local failure_count=0
    local failed_datasets=()
    
    # Process each dataset
    for dataset_suffix in "${DATASETS[@]}"; do
        if run_experiment "$dataset_suffix"; then
            ((success_count++))
        else
            ((failure_count++))
            failed_datasets+=("$dataset_suffix")
        fi
    done
    
    # Restore original corr_test
    restore_corr_test
    
    # Print summary
    echo "========================================"
    print_status "Experiment Summary:"
    print_success "Successful experiments: $success_count"
    if [ $failure_count -gt 0 ]; then
        print_error "Failed experiments: $failure_count"
        print_error "Failed datasets: ${failed_datasets[*]}"
    else
        print_success "All experiments completed successfully!"
    fi
    
    print_status "Results are stored in: $RESULTS_DIR/corr_test/"
    print_status "Each result folder is suffixed with the corresponding query scale (qs_X)"
}

# Check prerequisites
check_prerequisites() {
    # Check if directories exist
    if [ ! -d "$WORKLOADS_DIR" ]; then
        print_error "Workloads directory not found: $WORKLOADS_DIR"
        exit 1
    fi
    
    if [ ! -d "$MEMENTO_BUILD_PATH" ]; then
        print_error "Memento build directory not found: $MEMENTO_BUILD_PATH"
        exit 1
    fi
    
    # Check if at least one dataset exists
    local found_dataset=false
    for dataset_suffix in "${DATASETS[@]}"; do
        if [ -d "$WORKLOADS_DIR/corr_test_tiny_qs_$dataset_suffix" ]; then
            found_dataset=true
            break
        fi
    done
    
    if [ "$found_dataset" = false ]; then
        print_error "No corr_test_tiny_qs_* datasets found in $WORKLOADS_DIR"
        exit 1
    fi
    
    print_success "Prerequisites check passed"
}

# Help function
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Automated script to run correlation tests on multiple tiny query scale datasets.

This script processes the following datasets:
$(printf "  - corr_test_tiny_qs_%s\n" "${DATASETS[@]}")

For each dataset, it:
1. Renames the dataset to 'corr_test'
2. Builds the Memento Filter
3. Runs the correlation experiment
4. Renames the results with dataset suffix
5. Cleans up and moves to the next dataset

OPTIONS:
    -h, --help     Show this help message and exit

EXAMPLES:
    $0                    # Run all datasets
    
Results will be stored in: $RESULTS_DIR/corr_test/
Each result folder will be suffixed with the query scale (e.g., *_qs_10)

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Run the script
check_prerequisites
main

print_status "Script execution completed."
