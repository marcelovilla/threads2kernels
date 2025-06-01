using Base.Threads
using BenchmarkTools

"""
    stencil!(u, u_new, rhs, h)

Perform a single iteration of the Jacobi method on a discretized grid
using a five-point stencil.

# Arguments
- `u`: The current solution grid.
- `u_new`: The grid to store the updated solution.
- `rhs`: The right-hand side grid.
- `h`: The grid spacing.
"""
function stencil!(u, u_new, rhs, h)
    M, N = size(u)
    @threads for j in 2:N-1
        for i in 2:M-1
            u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i,j+1] + u[i, j-1] - h^2 * rhs[i, j])
        end
    end
end

"""
    jacobi!(u, rhs, h; k=4096)

Perform the Jacobi method on a discretized grid with a fixed number of
iterations.

# Arguments
- `u`: The current solution grid.
- `rhs`: The right-hand side grid.
- `h`: The grid spacing.
- `n_iter`: The number of iterations to perform (default is 4096).
"""
function jacobi!(u, rhs, h; n_iter=4096)
    u_new = copy(u)
    for _ in 1:n_iter
        stencil!(u, u_new, rhs, h)
        u, u_new = u_new, u
    end
end

"""
    setup(N, f)

Set up the initial conditions for the Jacobi method. The grids are 
initialized with Dirichlet boundary conditions u(x, y) = 0.

# Arguments
- `N`: The size of the grid (N x N).
- `f`: A function to compute the right-hand side values.
"""
function setup(N, f)
    h = 1 / (N + 1)
    u = zeros(Float64, N, N)

    # x and y coordinates are the same on a square grid
    coords = h * (2:N-1)

    # Only interior points are computed and boundaries are left as zero
    rhs = zeros(Float64, N, N)
    rhs[2:N-1, 2:N-1] .= f.(coords', coords)

    return u, rhs, h
end


# The heat equation right-hand side function
f(x, y) = -2π^2 * sin(π * x) * sin(π * y)

N = parse(Int, ARGS[1])
u, rhs, h = setup(N, f)
result = @benchmark jacobi($u, $rhs, $h)

show(stdout, MIME"text/plain"(), result)
