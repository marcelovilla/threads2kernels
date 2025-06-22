using KernelAbstractions

# Both kernel functions are meant to be run only on the interior points of the grid.
# KernelAbstractions does not seem to support arbitrary indexing, so we just add an offset
# of 1 to the indices and skip the boundaries when specifying the ndrange.

@kernel function rhs_kernel!(rhs, @Const(f), @Const(h))
    i, j = @index(Global, NTuple)
    @inbounds rhs[i+1, j+1] = f((i + 1) * h, (j + 1) * h)
end

@kernel function stencil_kernel!(u, u_new, @Const(rhs), @Const(h))
    i, j = @index(Global, NTuple)
    @inbounds u_new[i+1, j+1] =
        0.25 * (u[i+2, j+1] + u[i, j+1] + u[i+1, j+2] + u[i+1, j] - h^2 * rhs[i+1, j+1])
end

function jacobi!(u, rhs, h, kernel, n_iter)
    M, N = size(u)
    u_new = copy(u)
    for _ = 1:n_iter
        # Run the kernel on the interior points of the grid, skipping the boundaries
        kernel(u, u_new, rhs, h, ndrange = (M - 2, N - 2))
        u, u_new = u_new, u
    end
end

function setup(N, f, backend)
    u = KernelAbstractions.zeros(backend, Float64, N, N)
    rhs = KernelAbstractions.zeros(backend, Float64, N, N)

    # Only interior points are computed and boundaries are left as zero
    h = 1 / (N + 1)
    rhs_kernel = rhs_kernel!(backend)
    rhs_kernel(rhs, f, h, ndrange = (N - 2, N - 2))

    stencil_kernel = stencil_kernel!(backend)

    return u, rhs, h, stencil_kernel
end
