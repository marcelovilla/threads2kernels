#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for nthreads in 1 2 4 8; do
    julia --project=. -t $nthreads ${SCRIPT_DIR}/cpu_benchmark.jl cpu
done
