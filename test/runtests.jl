using Test
using Random
using FlashMaxSim
using Zygote
using LinearAlgebra: dot, norm
using ChainRulesCore: rrule, ZeroTangent, Thunk
using KernelAbstractions

cosine(a, b) = begin
    av = vec(Float64.(a)); bv = vec(Float64.(b))
    na, nb = norm(av), norm(bv)
    (na == 0 && nb == 0) && return 1.0
    (na == 0 || nb == 0) && return NaN
    dot(av, bv) / (na * nb)
end

relnorm(a, b) = begin
    av = vec(Float64.(a)); bv = vec(Float64.(b))
    na, nb = norm(av), norm(bv)
    (na == 0 && nb == 0) && return 1.0
    nb == 0 && return Inf
    na / nb
end

run_cuda = false
if get(ENV, "FLASHMAXSIM_CUDA", "1") != "0" && Base.find_package("CUDA") !== nothing
    using CUDA
    run_cuda = CUDA.functional()
    run_cuda || @info "CUDA.jl loaded but no functional device — skip GPU tests"
end

@testset "FlashMaxSim" begin
    include("test_forward.jl")
    include("test_grad.jl")
    include("test_paper_correctness.jl")
    include("test_regressions.jl")
    if run_cuda
        include("test_cuda.jl")
    end
end
