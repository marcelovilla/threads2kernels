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
    u = zeros(Float64, N, N)
    rhs = zeros(Float64, N, N)

    # boundaries are left as zero, so we only compute the interior points
    h = 1 / (N + 1)
    interior_points = h * (2:N-1)
    rhs[2:N-1, 2:N-1] .= f.(interior_points', interior_points)

    return u, rhs, h
end
