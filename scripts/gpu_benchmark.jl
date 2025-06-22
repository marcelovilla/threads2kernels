using CUDA

include(joinpath(dirname(@__DIR__), "src/benchmark.jl"))

problem_sizes = Int[4e6, 8e6, 16e6, 32e6, 64e6, 128e6, 256e6, 512e6]

backend = CUDABackend()

run_benchmark_suite(
    backend,
    joinpath(dirname(@__DIR__), "results/gpu.json"),
    problem_sizes;
    n_iter = 500,
    seconds = Inf,
    samples = 10,
    evals = 1,
    suite_key = "cuda",
)
