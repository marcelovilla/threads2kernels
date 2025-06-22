#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for nthreads in 128 64 32 16 8 4 2 1; do
    julia --project=. -t $nthreads ${SCRIPT_DIR}/cpu_benchmark.jl
done
