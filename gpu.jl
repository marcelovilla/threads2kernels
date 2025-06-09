using BenchmarkTools
using CUDA
using KernelAbstractions

# Both kernel functions are meant to be run only on the interior points of the grid.
# KernelAbstractions does not seem to support arbitrary indexing, so we just add an offset
# of 1 to the indices and skip the boundaries when specifying the ndrange.

@kernel function rhs_kernel!(rhs, @Const(f), @Const(h))
    i, j = @index(Global, NTuple)
    @inbounds rhs[i+1, j+1] = f(i * h, j * h)
end

@kernel function stencil_kernel!(u, u_new, @Const(rhs), @Const(h))
    i, j = @index(Global, NTuple)
    @inbounds u_new[i+1, j+1] = 0.25 * (u[i+2, j+1] + u[i, j+1] + u[i+1,j+2] + u[i+1, j] - h^2 * rhs[i+1, j+1])
end

function jacobi!(u, rhs, h, kernel; n_iter=4096)
    M, N = size(u)
    u_new = copy(u)
    for _ in 1:n_iter
        # Run the kernel on the interior points of the grid, skipping the boundaries
        kernel(u, u_new, rhs, h, ndrange=(M-2, N-2))
        u, u_new = u_new, u
    end
end

function setup(N, f, backend)
    h = 1 / (N + 1)
    u = KernelAbstractions.zeros(backend, Float64, N, N);

    # Only interior points are computed and boundaries are left as zero
    rhs = KernelAbstractions.zeros(backend, Float64, N, N)
    rhs_kernel = rhs_kernel!(backend)
    rhs_kernel(rhs, f, h, ndrange=(N-2, N-2))

    stencil_kernel = stencil_kernel!(backend)

    return u, rhs, h, stencil_kernel
end

# The heat equation right-hand side function
f(x, y) = -2π^2 * sin(π * x) * sin(π * y)

# Alternatively, fallback to CPU as a backend
# backend = CPU()
backend = CUDABackend()

suite = BenchmarkGroup()
# N = 2^14 is probably a reasonable upper limit in terms of memory usage. A grid of this
# size and of type Float64 requires 2^14 * 2^14 * 8 bytes ≈ 2GB. We need to take into 
# account that we will allocate three of such grids: two for the Jacobi method and one 
# for the right-hand side.
for p in 2:14
    N = 2^p
    u₀, rhs, h, stencil_kernel = setup(N, f, backend)
    suite[N] = @benchmarkable jacobi!(u, $rhs, $h, $stencil_kernel) setup=(u = copy($u₀))
end

# We set seconds=Inf to run the benchmarks for a fixed number of samples. Otherwise, the
# benchmarks will end as soon as the time budget is reached and at least one sample has
# finished.
results = run(suite, seconds=Inf, samples=50, evals=1)

BenchmarkTools.save("gpu.json", results)
