using KernelAbstractions

@kernel function stencil_kernel!(u, u_new)
    M, N = size(u)
    i, j = @index(Global, NTuple)
    if i > 1 && j > 1 && i < M && j < N
        @inbounds u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i,j+1] + u[i, j-1])
    end
end

backend = CPU()

u = KernelAbstractions.zeros(backend, Float32, 100, 100);
u[1,:] = u[end,:] = u[:,1] = u[:,end] .= 10.0f0
u_new = copy(u);

kernel = stencil_kernel!(backend)
kernel(u, u_new, ndrange=(size(u, 1), size(u, 2)))
