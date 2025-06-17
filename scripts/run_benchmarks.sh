#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
RESULTS_DIR="$PROJECT_ROOT/results"

# Run CPU benchmarks with different thread counts
for nthreads in 1 2 4 8 16 32 64; do
    julia --project=. -t $nthreads ${SCRIPT_DIR}/benchmark.jl cpu ${RESULTS_DIR}/cpu_${nthreads}_threads.json --max-p=5
done
