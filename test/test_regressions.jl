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
