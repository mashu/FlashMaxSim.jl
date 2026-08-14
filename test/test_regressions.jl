# Regression tests for cotangent dispatch, candidate sentinels, scatter bounds.

@testset "ZeroTangent / unused MaxSim output" begin
    T = Float32
    q, d = randn(T, 8, 4), randn(T, 8, 5)
    qm, dm = trues(4), trues(5)
    cfg = MaxSim{T}()
    _, back = rrule(maxsim, cfg, q, d, qm, dm)
    dq, dd = back(ZeroTangent())[3:4]
    @test all(iszero, dq) && all(iszero, dd)
    dq, dd = back(Thunk(() -> one(T)))[3:4]
    @test all(isfinite, dq) && all(isfinite, dd)

    Q, D = randn(T, 8, 4, 2), randn(T, 8, 5, 2)
    _, backb = rrule(maxsim, cfg, Q, D, trues(4, 2), trues(5, 2))
    dQ, dD = backb(ZeroTangent())[3:4]
    @test all(iszero, dQ) && all(iszero, dD)

    g = gradient(q -> (maxsim(q, d); one(T)), q)[1]
    @test g === nothing
    gB = gradient(Q -> (maxsim(Q, D); one(T)), Q)[1]
    @test gB === nothing
end

@testset "candidate normalize keeps neg sentinel" begin
    Random.seed!(7)
    T = Float32
    Q = randn(T, 8, 4, 2)
    gallery = randn(T, 8, 5, 4)
    idxs = Int32[1 2; 0 1]
    qm, dm = trues(4, 2), trues(5, 4)
    neg = MaxSim{T}().neg
    Sc = maxsim(Q, gallery, idxs, qm, dm; normalize = true)
    Sd = maxsim_dense(Q, gallery, idxs, qm, dm; normalize = true)
    @test Sc[2, 1] == neg
    @test Sd[2, 1] == neg
    @test Sc ≈ Sd rtol=1e-5 atol=1e-5
    @test Sc[2, 1] != neg / T(4)
    gQ, gG = gradient((Q, G) -> sum(maxsim(Q, G, idxs, qm, dm; normalize = true)), Q, gallery)
    @test all(isfinite, gQ) && all(isfinite, gG)
end

@testset "AtomicUnified dest bounds and length checks" begin
    T = Float32
    q = randn(T, 4, 3)
    dd = zeros(T, 4, 2)
    δ = T[1, 2, 3]
    arg = Int32[1, 0, 99]
    FlashMaxSim.accumulate_doc!(AtomicUnified(), dd, q, δ, arg)
    @test all(isfinite, dd)
    @test dd[:, 2] == zeros(T, 4)
    expected = δ[1] .* q[:, 1]
    @test dd[:, 1] ≈ expected

    dd_i = zeros(T, 4, 2)
    FlashMaxSim.accumulate_doc!(InvGrid(), dd_i, q, δ, arg)
    @test dd_i ≈ dd

    @test_throws DimensionMismatch FlashMaxSim.accumulate_doc!(
        AtomicUnified(), dd, q, T[1, 2], arg)
    @test_throws DimensionMismatch FlashMaxSim.accumulate_doc!(
        InvGrid(), dd, q, δ, Int32[1, 2])
end

@testset "empty doc does not emit neg into the sum" begin
    Random.seed!(11)
    T = Float32
    q = randn(T, 8, 5)
    d = randn(T, 8, 6)
    qm = trues(5)
    dm = falses(6)
    neg = MaxSim{T}().neg
    @test maxsim(q, d, qm, dm) == zero(T)
    @test maxsim(q, d, qm, dm; normalize = true) == zero(T)
    @test maxsim_dense(q, d, qm, dm) == zero(T)
    @test maxsim(q, d, qm, dm) != neg

    Q = randn(T, 8, 5, 3)
    D = randn(T, 8, 6, 3)
    qmb, dmb = trues(5, 3), falses(6, 3)
    S = maxsim(Q, D, qmb, dmb, InBatch(); normalize = true)
    @test all(iszero, S)
    @test all(isfinite, S)
    # InfoNCE-scale: normalized empty scores must not be −1e4
    @test maximum(abs, S) < T(1)
end

@testset "host DOC_TILE GEMM vs dense" begin
    Random.seed!(8)
    T = Float32
    q = randn(T, 16, 12)
    d = randn(T, 16, 130)
    qm = [isodd(t) for t in 1:12]
    dm = [t % 3 != 0 for t in 1:130]
    @test maxsim(q, d, qm, dm) ≈ maxsim_dense(q, d, qm, dm) rtol=1e-5 atol=1e-5
    @test maxsim(q, d, qm, dm; normalize = true) ≈
          maxsim_dense(q, d, qm, dm; normalize = true) rtol=1e-5 atol=1e-5
end

@testset "KA CPU kernels match BLAS host (same atol)" begin
    Random.seed!(9)
    T = Float32
    q = randn(T, 16, 12)
    d = randn(T, 16, 20)
    qm = collect(isodd(t) for t in 1:12)
    dm = collect(t % 3 != 0 for t in 1:20)
    neg = T(-1.0f4)
    s_h, arg_h = FlashMaxSim.pair_forward_host(q, d, qm, dm, neg)
    s_k, arg_k = FlashMaxSim.pair_forward_ka(q, d, qm, dm, neg)
    @test only(s_k) ≈ s_h rtol=1e-5 atol=1e-5
    @test arg_k == arg_h
    @test only(s_k) ≈ maxsim_dense(q, d, qm, dm) rtol=1e-5 atol=1e-5

    δ = T(1.5)
    backend = KernelAbstractions.CPU()
    for mode in (AtomicUnified(), InvGrid())
        dq_s, dd_s = FlashMaxSim.pair_pullback(backend, δ, q, d, qm, arg_h, mode)
        dq_k, dd_k = FlashMaxSim.pair_pullback_ka(backend, δ, q, d, qm, arg_k, mode)
        @test dq_k ≈ dq_s rtol=1e-5 atol=1e-5
        @test dd_k ≈ dd_s rtol=1e-5 atol=1e-5
    end
end

@testset "adversarial neg: host == KA == dense" begin
    Random.seed!(99)
    T = Float32
    q, d = randn(T, 8, 6), randn(T, 8, 7)
    qm, dm = trues(6), trues(7)
    for neg in (T(0), T(10), T(-1.0f4))
        s_h, a_h = FlashMaxSim.pair_forward_host(q, d, qm, dm, neg)
        s_k, a_k = FlashMaxSim.pair_forward_ka(q, d, qm, dm, neg)
        @test a_h == a_k
        @test s_h ≈ only(s_k) rtol=1e-5
        @test s_h ≈ maxsim_dense(q, d, qm, dm; neg) rtol=1e-5 atol=1e-5
    end
    # All-negative dots: true max is negative, not dropped / clamped to `neg`.
    qn = ones(T, 4, 3)
    dn = -ones(T, 4, 5)
    s_h, a_h = FlashMaxSim.pair_forward_host(qn, dn, trues(3), trues(5), T(0))
    s_k, a_k = FlashMaxSim.pair_forward_ka(qn, dn, trues(3), trues(5), T(0))
    @test a_h == a_k
    @test all(>(0), a_h)
    @test s_h ≈ only(s_k)
    @test s_h ≈ maxsim_dense(qn, dn, trues(3), trues(5); neg = T(0))
    @test s_h ≈ T(-12)  # 3 query tokens × (−4)
    @test maxsim(qn, dn; neg = T(0)) ≈ s_h
end

@testset "in-batch tile uses sizeof(T)" begin
    c32 = FlashMaxSim.inbatch_doc_chunk(Float32, 128, 32, 32, 10_000)
    c64 = FlashMaxSim.inbatch_doc_chunk(Float64, 128, 32, 32, 10_000)
    @test c32 == 128
    @test c64 == 64
end

@testset "NaN cotangent propagates on CPU" begin
    T = Float32
    q, d = randn(T, 4, 3), randn(T, 4, 5)
    cfg = MaxSim{T}()
    _, back = rrule(maxsim, cfg, q, d, trues(3), trues(5))
    dq, dd = back(T(NaN))[3:4]
    @test any(isnan, dq) && any(isnan, dd)

    Q, D = randn(T, 4, 3, 2), randn(T, 4, 5, 2)
    Δ = zeros(T, 2)
    Δ[1] = T(NaN)
    _, backb = rrule(maxsim, cfg, Q, D, trues(3, 2), trues(5, 2))
    dQ, dD = backb(Δ)[3:4]
    @test any(isnan, dQ) && any(isnan, dD)
end
