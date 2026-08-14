# Mathematical oracle: scores vs dense MaxSim; gradients vs closed-form Eqs. 2–3.
#
# Independent of AtomicUnified vs InvGrid and of CPU vs GPU. A systematically
# wrong kernel that agrees with itself (same bug on both modes) fails here.

"""Paper §4.2 Eqs. 2–3 for one pair: ∇q_t = δ d_{u★}, ∇d_u = Σ_{t: u★=u} δ q_t."""
function closed_form_pair(δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                          qmask::AbstractVector{Bool},
                          argmax_u::AbstractVector{<:Integer}) where {T}
    dq = zeros(T, size(q))
    dd = zeros(T, size(d))
    Td = size(d, 2)
    dim, Tq = size(q)
    @inbounds for t in 1:Tq
        qmask[t] || continue
        u = Int(argmax_u[t])
        (1 <= u <= Td) || continue
        for k in 1:dim
            dq[k, t] = δ * d[k, u]
            dd[k, u] += δ * q[k, t]
        end
    end
    dq, dd
end

function closed_form_inbatch(Δ::AbstractMatrix{T}, Q, D, qmask, args, inv_n) where {T}
    dQ = zeros(T, size(Q))
    dD = zeros(T, size(D))
    dim, Tq, Bq = size(Q)
    Td, Bd = size(D, 2), size(D, 3)
    @inbounds for j in 1:Bd, i in 1:Bq
        δ = T(Δ[j, i]) * inv_n[i]
        δ == zero(T) && continue
        for t in 1:Tq
            qmask[t, i] || continue
            u = Int(args[t, j, i])
            (1 <= u <= Td) || continue
            for k in 1:dim
                dQ[k, t, i] += δ * D[k, u, j]
                dD[k, u, j] += δ * Q[k, t, i]
            end
        end
    end
    dQ, dD
end

function closed_form_candidates(Δ::AbstractMatrix{T}, Q, gallery, idxs, qmask, args, inv_n) where {T}
    dQ = zeros(T, size(Q))
    dG = zeros(T, size(gallery))
    dim, Tq, B = size(Q)
    Td, N = size(gallery, 2), size(gallery, 3)
    C = size(idxs, 1)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        δ = T(Δ[c, b]) * inv_n[b]
        δ == zero(T) && continue
        for t in 1:Tq
            qmask[t, b] || continue
            u = Int(args[t, c, b])
            (1 <= u <= Td) || continue
            for k in 1:dim
                dQ[k, t, b] += δ * gallery[k, u, j]
                dG[k, u, j] += δ * Q[k, t, b]
            end
        end
    end
    dQ, dG
end

function unit_l2_cols!(X)
    X ./= sqrt.(sum(abs2, X; dims = 1) .+ eps(eltype(X)))
    X
end

@testset "math oracle: pair score + closed-form grads" begin
    Random.seed!(101)
    T = Float32
    for (dim, Tq, Td) in ((8, 5, 6), (16, 12, 20), (32, 24, 31))
        q = unit_l2_cols!(randn(T, dim, Tq))
        d = unit_l2_cols!(randn(T, dim, Td))
        qm = [isodd(t) for t in 1:Tq]
        dm = [t % 3 != 0 for t in 1:Td]
        s_ref = maxsim_dense(q, d, qm, dm)
        for mode in (AtomicUnified(), InvGrid())
            cfg = MaxSim{T}(T(-1.0f4), false, mode)
            s, arg = FlashMaxSim.pair_forward(q, d, qm, dm, cfg.neg)
            @test s ≈ s_ref rtol=1e-5 atol=1e-5
            @test maxsim(cfg, q, d, qm, dm) ≈ s_ref rtol=1e-5 atol=1e-5

            δ = T(1.25)
            dq_cf, dd_cf = closed_form_pair(δ, q, d, qm, arg)
            dq, dd = FlashMaxSim.pair_pullback(δ, q, d, qm, arg, mode)
            @test dq ≈ dq_cf
            @test dd ≈ dd_cf

            gq, gd = gradient((q, d) -> maxsim(cfg, q, d, qm, dm), q, d)
            dq1, dd1 = closed_form_pair(one(T), q, d, qm, arg)
            @test cosine(gq, dq1) ≥ 0.999
            @test cosine(gd, dd1) ≥ 0.999
            @test gq ≈ dq1 rtol=1e-5 atol=1e-5
            @test gd ≈ dd1 rtol=1e-5 atol=1e-5
        end
    end
end

@testset "math oracle: in-batch / candidates vs closed-form" begin
    Random.seed!(102)
    T = Float32
    Q = unit_l2_cols!(randn(T, 8, 4, 3))
    D = unit_l2_cols!(randn(T, 8, 5, 3))
    qm, dm = trues(4, 3), trues(5, 3)
    ones3 = ones(T, 3)
    Δ = randn(T, 3, 3)
    for mode in (AtomicUnified(), InvGrid())
        cfg = MaxSim{T}(T(-1.0f4), false, mode)
        S, args = FlashMaxSim.inbatch_forward(Q, D, qm, dm, cfg.neg)
        @test S ≈ maxsim_dense(Q, D, qm, dm, InBatch()) rtol=1e-5 atol=1e-5
        dQ_cf, dD_cf = closed_form_inbatch(Δ, Q, D, qm, args, ones3)
        dQ, dD = FlashMaxSim.inbatch_pullback(Δ, Q, D, qm, args, ones3, mode)
        @test dQ ≈ dQ_cf rtol=1e-5 atol=1e-5
        @test dD ≈ dD_cf rtol=1e-5 atol=1e-5
        @test cosine(dQ, dQ_cf) ≥ 0.999
        @test cosine(dD, dD_cf) ≥ 0.999
    end

    G = unit_l2_cols!(randn(T, 8, 5, 7))
    idxs = Int32[1 2 3; 4 0 6]
    dmG = trues(5, 7)
    Δc = randn(T, 2, 3)
    for mode in (AtomicUnified(), InvGrid())
        cfg = MaxSim{T}(T(-1.0f4), false, mode)
        Sc, args = FlashMaxSim.candidates_forward(Q, G, idxs, qm, dmG, cfg.neg)
        @test Sc ≈ maxsim_dense(Q, G, idxs, qm, dmG) rtol=1e-5 atol=1e-5
        @test Sc[2, 2] == cfg.neg
        dQ_cf, dG_cf = closed_form_candidates(Δc, Q, G, idxs, qm, args, ones3)
        dQ, dG = FlashMaxSim.candidates_pullback(Δc, Q, G, idxs, qm, args, ones3, mode)
        @test dQ ≈ dQ_cf rtol=1e-5 atol=1e-5
        @test dG ≈ dG_cf rtol=1e-5 atol=1e-5
    end
end

@testset "math oracle: AtomicUnified == InvGrid (same tape)" begin
    Random.seed!(103)
    T = Float32
    q, d = randn(T, 16, 10), randn(T, 16, 14)
    qm, dm = trues(10), trues(14)
    _, arg = FlashMaxSim.pair_forward(q, d, qm, dm, T(-1.0f4))
    dq_a, dd_a = FlashMaxSim.pair_pullback(one(T), q, d, qm, arg, AtomicUnified())
    dq_i, dd_i = FlashMaxSim.pair_pullback(one(T), q, d, qm, arg, InvGrid())
    @test dq_a == dq_i
    @test dd_a ≈ dd_i
end

if run_cuda
    @testset "math oracle: GPU vs closed-form (paper cosine protocol)" begin
        Random.seed!(104)
        T = Float32
        dim, Tq, Td, B = 16, 8, 11, 3
        q = unit_l2_cols!(randn(T, dim, Tq))
        d = unit_l2_cols!(randn(T, dim, Td))
        qm, dm = trues(Tq), trues(Td)
        qg, dg = CuArray(q), CuArray(d)
        qmg, dmg = CuArray(collect(qm)), CuArray(collect(dm))
        s_ref = maxsim_dense(q, d, qm, dm)
        _, arg_h = FlashMaxSim.pair_forward(q, d, qm, dm, T(-1.0f4))
        dq_cf, dd_cf = closed_form_pair(one(T), q, d, qm, arg_h)
        for mode in (AtomicUnified(), InvGrid())
            cfg = MaxSim{T}(T(-1.0f4), false, mode)
            @test only(Array(maxsim(cfg, qg, dg, qmg, dmg))) ≈ s_ref rtol=1e-5 atol=1e-5
            gq, gd = gradient((q, d) -> sum(maxsim(cfg, q, d, qmg, dmg)), qg, dg)
            @test cosine(Array(gq), dq_cf) ≥ 0.999
            @test cosine(Array(gd), dd_cf) ≥ 0.999
        end

        Q = unit_l2_cols!(randn(T, dim, Tq, B))
        D = unit_l2_cols!(randn(T, dim, Td, B))
        Qg, Dg = CuArray(Q), CuArray(D)
        qmgB = CuArray(fill(true, Tq, B))
        dmgB = CuArray(fill(true, Td, B))
        onesB = ones(T, B)
        Δ = randn(T, B, B)
        S, args = FlashMaxSim.inbatch_forward(Q, D, trues(Tq, B), trues(Td, B), T(-1.0f4))
        dQ_cf, dD_cf = closed_form_inbatch(Δ, Q, D, trues(Tq, B), args, onesB)
        for mode in (AtomicUnified(), InvGrid())
            dQ, dD = FlashMaxSim.inbatch_pullback(CuArray(Δ), Qg, Dg, qmgB,
                                                  CuArray(args), CuArray(onesB), mode)
            @test cosine(Array(dQ), dQ_cf) ≥ 0.999
            @test cosine(Array(dD), dD_cf) ≥ 0.999
        end
    end
end
