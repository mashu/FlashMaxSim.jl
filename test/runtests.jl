using Test
using Random
using FlashMaxSim
using Zygote
using LinearAlgebra: dot, norm

cosine(a, b) = begin
    av = vec(Float64.(a)); bv = vec(Float64.(b))
    na, nb = norm(av), norm(bv)
    (na == 0 && nb == 0) && return 1.0
    (na == 0 || nb == 0) && return NaN
    dot(av, bv) / (na * nb)
end

@testset "FlashMaxSim" begin
    include("test_forward.jl")
    include("test_grad.jl")
    include("test_paper_correctness.jl")
    if get(ENV, "FLASHMAXSIM_CUDA", "1") != "0"
        include("test_cuda.jl")
    end
end
