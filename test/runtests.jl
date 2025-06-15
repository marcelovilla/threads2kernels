using Test

@testset "Tests" begin

    @testset "Pure vs KA implementation" begin
        include("./pure_vs_ka.jl")
    end
end
