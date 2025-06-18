#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
RESULTS_DIR="$PROJECT_ROOT/results"

# Run CPU benchmarks with different thread counts
for nthreads in 1 2 4 8 12 16 24 32 48; do
    julia --project=. -t $nthreads ${SCRIPT_DIR}/benchmark.jl cpu ${RESULTS_DIR}/cpu_${nthreads}_threads.json --n-iter 512 --max-p 12 --samples 20
done
