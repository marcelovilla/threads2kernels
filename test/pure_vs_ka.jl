# Test the pure version of Jacobi against the KernelAbstractions version
using KernelAbstractions
using Test

include("../src/jacobi_ka.jl")
include("../src/jacobi_pure.jl")
include("./common.jl")

@testset for p in p_range
    N = 2^p

    # We benefit from Julia's multiple dispatch here as the signature for the functions are
    # different between the pure and the KernelAbstractions versions.

    # Run the pure version
    uₚ, rhsₚ, hₚ = setup(N, f)
    jacobi!(uₚ, rhsₚ, hₚ, n_iter)

    # Run the KernelAbstractions version
    uₖ, rhsₖ, hₖ, stencil_kernel = setup(N, f, CPU())
    jacobi!(uₖ, rhsₖ, hₖ, stencil_kernel, n_iter)

    @test rhsₚ ≈ rhsₖ
    @test uₚ ≈ uₖ
end
