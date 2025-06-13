using Base.Threads

function stencil!(u, u_new, rhs, h)
    M, N = size(u)
    @threads for j = 2:N-1
        for i = 2:M-1
            @inbounds u_new[i, j] =
                0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] - h^2 * rhs[i, j])
        end
    end
end

function jacobi!(u, rhs, h, n_iter)
    u_new = copy(u)
    for _ = 1:n_iter
        stencil!(u, u_new, rhs, h)
        u, u_new = u_new, u
    end
end

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
