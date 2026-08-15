# Float16 scores stay in FP32 so near-duplicate ("sister") documents remain separable.

l2cols(x) = x ./ sqrt.(sum(abs2, x; dims = 1) .+ floatmin(eltype(x)))

"""Perturb one document token by mixing in an orthogonal direction, then L2-normalize."""
function sister_token(dcol::AbstractVector, noise::AbstractVector, α)
    n = noise .- dcol .* sum(noise .* dcol)
    n ./= sqrt(sum(abs2, n))
    v = (1 - α) .* dcol .+ α .* n
    v ./= sqrt(sum(abs2, v))
    v
end

@testset "Float16 features return Float32 scores" begin
    @test score_eltype(Float16) === Float32
    @test score_eltype(Float32) === Float32
    @test score_eltype(Float64) === Float64
    q = l2cols(randn(Float16, 16, 8))
    d = l2cols(randn(Float16, 16, 10))
    s = maxsim(q, d)
    @test s isa Float32
    @test s ≈ maxsim_dense(q, d) rtol=1e-5 atol=1e-5
    @test maxsim_dense(q, d) isa Float32
end

@testset "sister documents: SNP-scale ranking gap survives Float16 features" begin
    Random.seed!(2026)
    dim, Tq, Td = 128, 32, 40
    q64 = l2cols(randn(Float64, dim, Tq))
    d64 = l2cols(randn(Float64, dim, Td))
    d64[:, 1:Tq] .= q64   # exact token copies → MaxSim ≈ Tq before Float16 rounding

    α = 0.08
    d_sis64 = copy(d64)
    d_sis64[:, 1] = sister_token(d64[:, 1], randn(Float64, dim), α)

    gap64 = maxsim_dense(q64, d64) - maxsim_dense(q64, d_sis64)
    @test 1e-4 < gap64 < 0.02   # below Float16 ulp at score ~32 (0.03125)

    q16, d16, sis16 = Float16.(q64), Float16.(d64), Float16.(d_sis64)
    sa = maxsim(q16, d16)
    sb = maxsim(q16, sis16)
    @test sa isa Float32
    @test sb isa Float32
    @test sa != sb
    @test sa > sb
    @test sign(sa - sb) == sign(gap64)
    # The gap sits inside one Float16 ulp of the score — a Float16 return would
    # have been unable to tell the sisters apart.
    @test abs(sa - sb) < Float64(eps(Float16(sa)))

    backend = KernelAbstractions.CPU()
    sa_k, _ = FlashMaxSim.pair_forward_ka(backend, q16, d16, trues(Tq), trues(Td), Float16(0))
    sb_k, _ = FlashMaxSim.pair_forward_ka(backend, q16, sis16, trues(Tq), trues(Td), Float16(0))
    @test eltype(sa_k) === Float32
    @test only(sa_k) > only(sb_k)

    Q = reshape(q16, dim, Tq, 1)
    D = cat(reshape(d16, dim, Td, 1), reshape(sis16, dim, Td, 1); dims=3)
    paired = maxsim(cat(Q, Q; dims=3), D)
    @test eltype(paired) === Float32
    @test paired[1] > paired[2]

    idxs = reshape(Int32[1, 2], 2, 1)
    gallery = D
    Sc = maxsim(Q, gallery, idxs)
    @test eltype(Sc) === Float32
    @test Sc[1, 1] > Sc[2, 1]
    @test Sc[1, 1] != Sc[2, 1]

    S_ib = maxsim(Q, gallery, InBatch())
    @test eltype(S_ib) === Float32
    @test S_ib[1, 1] > S_ib[2, 1]

    P = pack_docs([d16, sis16])
    sp = maxsim(q16, P)
    @test eltype(sp) === Float32
    @test sp[1] > sp[2]
end

@testset "sister ranking: order matches Float64 oracle" begin
    Random.seed!(2027)
    dim, Tq, Td = 128, 32, 32
    q64 = l2cols(randn(Float64, dim, Tq))
    d0 = l2cols(randn(Float64, dim, Td))
    d0[:, 1:Tq] .= q64
    alphas = (0.0, 0.04, 0.08, 0.12, 0.16)
    docs64 = Vector{Matrix{Float64}}(undef, length(alphas))
    @inbounds for (i, α) in enumerate(alphas)
        d = copy(d0)
        if α > 0
            d[:, 1] = sister_token(d0[:, 1], randn(Float64, dim), α)
        end
        docs64[i] = d
    end
    scores64 = [maxsim_dense(q64, d) for d in docs64]
    q16 = Float16.(q64)
    scores16 = [maxsim(q16, Float16.(d)) for d in docs64]
    @test eltype(scores16[1]) === Float32
    @test issorted(scores16; rev = true)
    @test sortperm(scores16; rev = true) == sortperm(scores64; rev = true)
    @test length(unique(scores16)) == length(scores16)
    # Truncating the published scores to Float16 would collapse near-duplicates.
    @test length(unique(Float16.(scores16))) < length(scores16)
    # Adjacent sisters differ by less than a Float16 score ulp, yet stay ordered.
    @test all(abs(scores64[i] - scores64[i + 1]) < Float64(eps(Float16(32)))
              for i in 1:(length(scores64) - 1))
end
