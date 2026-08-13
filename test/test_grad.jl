@testset "pair gradients both backward modes" begin
    Random.seed!(10)
    T = Float32
    dim, Tq, Td = 8, 5, 6
    q = randn(T, dim, Tq)
    d = randn(T, dim, Td)
    qm, dm = trues(Tq), trues(Td)
    for mode in (AtomicUnified(), InvGrid())
        cfg = MaxSim{T}(T(-1.0f4), false, mode)
        gq, gd = gradient((q, d) -> maxsim(cfg, q, d, qm, dm), q, d)
        @test gq !== nothing && gd !== nothing
        @test all(isfinite, gq) && all(isfinite, gd)
        ε = T(1e-3)
        q2 = copy(q); q2[1, 1] += ε
        num = (maxsim(cfg, q2, d, qm, dm) - maxsim(cfg, q, d, qm, dm)) / ε
        @test gq[1, 1] ≈ num rtol=5e-2 atol=5e-2
        d2 = copy(d); d2[1, 1] += ε
        num_d = (maxsim(cfg, q, d2, qm, dm) - maxsim(cfg, q, d, qm, dm)) / ε
        @test gd[1, 1] ≈ num_d rtol=5e-2 atol=5e-2
    end
end

@testset "normalize + mask gradients" begin
    Random.seed!(12)
    T = Float32
    q = randn(T, 8, 5)
    d = randn(T, 8, 6)
    qm = BitVector([true, true, false, true, false])
    dm = trues(6)
    cfg = MaxSim{T}(T(-1.0f4), true, InvGrid())
    gq, gd = gradient((q, d) -> maxsim(cfg, q, d, qm, dm), q, d)
    @test all(isfinite, gq) && all(isfinite, gd)
    @test all(gq[:, 3] .== 0) && all(gq[:, 5] .== 0)
end

@testset "in-batch + candidate gradients" begin
    Random.seed!(11)
    T = Float32
    Q = randn(T, 8, 4, 3)
    D = randn(T, 8, 5, 3)
    qm, dm = trues(4, 3), trues(5, 3)
    cfg = MaxSim{T}(T(-1.0f4), true, InvGrid())
    gQ, gD = gradient((Q, D) -> sum(maxsim(cfg, Q, D, qm, dm, InBatch())), Q, D)
    @test gQ !== nothing && gD !== nothing
    @test all(isfinite, gQ) && all(isfinite, gD)

    G = randn(T, 8, 5, 7)
    idxs = Int32[1 2 3; 4 5 6]
    cfg2 = MaxSim{T}(T(-1.0f4), false, AtomicUnified())
    gQc, gG = gradient((Q, G) -> sum(maxsim(cfg2, Q, G, idxs, qm, trues(5, 7))), Q, G)
    @test gQc !== nothing && gG !== nothing
    @test all(isfinite, gQc) && all(isfinite, gG)
end

@testset "default-mask AD" begin
    Random.seed!(13)
    q, d = randn(Float32, 6, 4), randn(Float32, 6, 5)
    gq = gradient(q -> maxsim(q, d), q)[1]
    @test gq !== nothing && all(isfinite, gq)
end
