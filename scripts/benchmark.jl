using BenchmarkTools
using Comonicon
using KernelAbstractions

include("../src/jacobi_ka.jl")

# N = 2^14 is probably a reasonable upper limit in terms of memory usage. A grid of this
# size and of type Float64 requires 2^14 * 2^14 * 8 bytes ≈ 2GB. We need to take into 
# account that we will allocate three of such grids: two for the Jacobi method and one 
# for the right-hand side.
# 
# We set seconds=Inf as a default to run the benchmarks for a fixed number of samples. 
# Otherwise, the benchmarks will end as soon as the time budget is reached and at least 
# one sample has finished.
"""
Run the Jacobi method benchmark with the specified backend for different grid sizes.

# Arguments

- `backend_name`: The backend to use, either "cpu" or "cuda".
- `output_name`: The name of the output file to save the benchmark results.

# Options

- `--n_iter <4096::Int>`: The number of iterations for the Jacobi method. Default=4096
- `--min_p <2::Int>`: The minimum power of 2 for the grid size (2^p). Default=2
- `--max_p <14::Int>`: The maximum power of 2 for the grid size (2^p). Default=14
- `--seconds <Inf::Float64>`: The time budget for the benchmark. Default=Inf
- `--samples <50::Int>`: The number of samples to run for each grid size. Default=50
- `--evals <1::Int>`: The number of evaluations for each sample. Default=1
"""
Comonicon.@main function foobar(
    backend_name,
    output_name;
    n_iter::Int = 4096,
    min_p::Int = 2,
    max_p::Int = 14,
    seconds::Float64 = Inf,
    samples::Int = 50,
    evals::Int = 1,
)
    if backend_name == "cpu"
        backend = CPU()
    elseif backend_name == "cuda"
        backend = CUDABackend()
    else
        error("Only CPU and CUDA backends are supported. Please specify 'cpu' or 'cuda'.")
    end

    # The heat equation right-hand side function
    f(x, y) = -2π^2 * sin(π * x) * sin(π * y)

    suite = BenchmarkGroup()
    for p = min_p:max_p
        N = 2^p
        u₀, rhs, h, stencil_kernel = setup(N, f, backend)
        # Make sure to copy u₀ at the start of every sample, so that it always starts from
        # the same initial condition. Otherwise, it will grab the solution from the previous
        # sample as jacobi! modifies u in place.
        suite[N] = @benchmarkable jacobi!(u, $rhs, $h, $stencil_kernel, $n_iter) setup =
            (u = copy($u₀))
    end

    results = run(suite, seconds = seconds, samples = samples, evals = evals)
    BenchmarkTools.save(output_name, results)
end
