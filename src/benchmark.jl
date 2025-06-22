using BenchmarkTools
using Printf

include(joinpath(@__DIR__, "jacobi_ka.jl"))

function run_benchmark_suite(
    backend,
    output_path,
    problem_sizes;
    n_iter = 1,
    seconds = Inf,
    samples = 20,
    evals = 1,
    suite_key = "default",
)

    # Instead of creating a new file for each run, we load the existing results and append
    # the new results to it.
    results = if isfile(output_path)
        BenchmarkTools.load(output_path)[1]
    else
        BenchmarkGroup()
    end

    # The heat equation right-hand side function
    f(x, y) = -2π^2 * sin(π * x) * sin(π * y)

    suite = BenchmarkGroup()
    for problem_size in problem_sizes
        N = isqrt(problem_size)
        u₀, rhs, h, stencil_kernel = setup(N, f, backend)
        # Make sure to copy u₀ at the start of every sample
        key = @sprintf("%.2E", problem_size)
        suite[key] = @benchmarkable jacobi!(u, $rhs, $h, $stencil_kernel, $n_iter) setup =
            (u = copy($u₀))
    end

    results[suite_key] = run(suite; seconds, samples, evals)
    BenchmarkTools.save(output_path, results)
end
