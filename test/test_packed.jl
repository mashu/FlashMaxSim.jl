@testset "pack_docs / pack_pairs" begin
    Random.seed!(30)
    T = Float32
    dim = 8
    docs = [randn(T, dim, 5), randn(T, dim, 0), randn(T, dim, 3)]
    packed, cu = pack_docs(docs)
    @test cu == Int32[1, 6, 6, 9]
    @test size(packed) == (dim, 8)
    @test packed[:, 1:5] == docs[1]
    @test packed[:, 6:8] == docs[3]

    qs = [randn(T, dim, 4), randn(T, dim, 2)]
    ds = [randn(T, dim, 5), randn(T, dim, 3)]
    Qp, Dp, cu_q, cu_d = pack_pairs(qs, ds)
    @test cu_q == Int32[1, 5, 7]
    @test cu_d == Int32[1, 6, 9]
    @test Qp[:, 1:4] == qs[1]
end

@testset "packed MaxSim matches unpacked pairs" begin
    Random.seed!(31)
    T = Float32
    dim, Tq, B = 16, 7, 5
    q = randn(T, dim, Tq)
    docs = [randn(T, dim, rand(2:9)) for _ in 1:B]
    packed, cu = pack_docs(docs)
    s = maxsim(q, packed, cu)
    @test size(s) == (B,)
    for b in 1:B
        @test s[b] ≈ maxsim(q, docs[b]) rtol=1e-5 atol=1e-5
    end
    sn = maxsim(q, packed, cu; normalize = true)
    nq = T(size(q, 2))
    @test sn ≈ s ./ nq rtol=1e-5 atol=1e-5

    docs_e = [randn(T, dim, 4), randn(T, dim, 0), randn(T, dim, 6)]
    pe, cue = pack_docs(docs_e)
    se = maxsim(q, pe, cue)
    @test se[2] == zero(T)
    @test se[1] ≈ maxsim(q, docs_e[1]) rtol=1e-5
end

@testset "varlen MaxSim matches unpacked pairs" begin
    Random.seed!(32)
    T = Float32
    dim, N = 12, 4
    qs = [randn(T, dim, rand(3:8)) for _ in 1:N]
    ds = [randn(T, dim, rand(4:10)) for _ in 1:N]
    Qp, Dp, cu_q, cu_d = pack_pairs(qs, ds)
    s = maxsim(Qp, Dp, cu_q, cu_d)
    @test size(s) == (N,)
    for n in 1:N
        @test s[n] ≈ maxsim(qs[n], ds[n]) rtol=1e-5 atol=1e-5
    end
    sn = maxsim(Qp, Dp, cu_q, cu_d; normalize = true)
    for n in 1:N
        @test sn[n] ≈ s[n] / T(size(qs[n], 2)) rtol=1e-5 atol=1e-5
    end
end

@testset "packed / varlen gradients vs unpacked" begin
    Random.seed!(33)
    T = Float32
    dim, Tq = 8, 5
    q = randn(T, dim, Tq)
    docs = [randn(T, dim, 6), randn(T, dim, 4), randn(T, dim, 7)]
    packed, cu = pack_docs(docs)
    gq, gP = gradient((q, p) -> sum(maxsim(q, p, cu)), q, packed)
    gq_ref = zeros(T, dim, Tq)
    gP_ref = zeros(T, size(packed)...)
    off = 1
    for b in 1:3
        Td = size(docs[b], 2)
        gqi, gdi = gradient((q, d) -> maxsim(q, d), q, docs[b])
        gq_ref .+= gqi
        gP_ref[:, off:off + Td - 1] .+= gdi
        off += Td
    end
    @test cosine(gq, gq_ref) ≥ 0.999
    @test cosine(gP, gP_ref) ≥ 0.999

    qs = [randn(T, dim, 4), randn(T, dim, 6)]
    ds = [randn(T, dim, 5), randn(T, dim, 3)]
    Qp, Dp, cu_q, cu_d = pack_pairs(qs, ds)
    gQ, gD = gradient((Q, D) -> sum(maxsim(Q, D, cu_q, cu_d)), Qp, Dp)
    gQ_ref = zeros(T, size(Qp)...)
    gD_ref = zeros(T, size(Dp)...)
    gQ_ref[:, 1:4] .= gradient((q, d) -> maxsim(q, d), qs[1], ds[1])[1]
    gD_ref[:, 1:5] .= gradient((q, d) -> maxsim(q, d), qs[1], ds[1])[2]
    gQ_ref[:, 5:10] .= gradient((q, d) -> maxsim(q, d), qs[2], ds[2])[1]
    gD_ref[:, 6:8] .= gradient((q, d) -> maxsim(q, d), qs[2], ds[2])[2]
    @test cosine(gQ, gQ_ref) ≥ 0.999
    @test cosine(gD, gD_ref) ≥ 0.999

    cfg_i = MaxSim{T}(T(-1.0f4), false, InvGrid())
    gq_i, gP_i = gradient((q, p) -> sum(maxsim(cfg_i, q, p, cu, trues(Tq))), q, packed)
    @test cosine(gq_i, gq) ≥ 0.999
    @test cosine(gP_i, gP) ≥ 0.999
end

@testset "packed KA CPU vs host" begin
    Random.seed!(34)
    T = Float32
    backend = KernelAbstractions.CPU()
    dim, Tq = 8, 6
    q = randn(T, dim, Tq)
    docs = [randn(T, dim, 5), randn(T, dim, 3)]
    packed, cu = pack_docs(docs)
    qm = trues(Tq)
    s_h, a_h = FlashMaxSim.packed_forward_host(q, packed, cu, qm, zero(T))
    s_k, a_k = FlashMaxSim.packed_forward_ka(backend, q, packed, cu, qm, zero(T))
    @test a_k == a_h
    @test s_k ≈ s_h rtol=1e-5 atol=1e-5
    inv_n = ones(T, 1)
    Δ = randn(T, 2)
    dq_h, dP_h = FlashMaxSim.packed_pullback(KernelAbstractions.CPU(), Δ, q, packed, cu, qm, a_h, inv_n,
                                            AtomicUnified())
    dq_k, dP_k = FlashMaxSim.packed_pullback_ka(backend, Δ, q, packed, cu, qm, a_k, inv_n)
    @test dq_k ≈ dq_h rtol=1e-5 atol=1e-5
    @test dP_k ≈ dP_h rtol=1e-5 atol=1e-5
end
