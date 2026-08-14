@testset "CUDA (KernelAbstractions backend)" begin
    CUDA.functional() || return
    CUDA.allowscalar(false)

    Random.seed!(20)
    T = Float32
    dim, Tq, Td = 32, 17, 23
    q = randn(T, dim, Tq)
    d = randn(T, dim, Td)
    q ./= sqrt.(sum(abs2, q; dims = 1) .+ eps(T))
    d ./= sqrt.(sum(abs2, d; dims = 1) .+ eps(T))
    qm, dm = trues(Tq), trues(Td)

    s_cpu = maxsim(q, d, qm, dm)
    qg, dg = CuArray(q), CuArray(d)
    qmg, dmg = CuArray(collect(qm)), CuArray(collect(dm))
    s_gpu = maxsim(qg, dg, qmg, dmg)
    @test s_gpu isa CuArray
    @test length(s_gpu) == 1
    @test only(Array(s_gpu)) ≈ s_cpu rtol=1e-5 atol=1e-5

    @test_throws ArgumentError maxsim(qg, dg, qm, dm)
    @test only(Array(maxsim(qg, dg))) ≈ s_cpu rtol=1e-5 atol=1e-5

    Q = randn(T, dim, Tq, 3)
    D = randn(T, dim, Td, 3)
    Sg = maxsim(CuArray(Q), CuArray(D), InBatch())
    @test Sg isa CuArray
    @test Array(Sg) ≈ maxsim(Q, D, InBatch()) rtol=1e-5 atol=1e-5
    paired_g = maxsim(CuArray(Q), CuArray(D))
    @test paired_g isa CuArray
    @test Array(paired_g) ≈ maxsim(Q, D) rtol=1e-5 atol=1e-5

    cfg = MaxSim{T}(T(-1.0f4), false, InvGrid())
    gq = gradient(x -> sum(maxsim(cfg, x, dg, qmg, dmg)), qg)[1]
    @test gq isa CuArray
    @test all(isfinite, Array(gq))
    gd = gradient(x -> sum(maxsim(cfg, qg, x, qmg, dmg)), dg)[1]
    @test gd isa CuArray
    @test all(isfinite, Array(gd))

    cfg_a = MaxSim{T}(T(-1.0f4), false, AtomicUnified())
    gqa = gradient(x -> sum(maxsim(cfg_a, x, dg, qmg, dmg)), qg)[1]
    @test cosine(Array(gq), Array(gqa)) ≥ 0.999
    @test 0.99 ≤ relnorm(Array(gq), Array(gqa)) ≤ 1.01

    gq_ref, gd_ref = gradient((q, d) -> maxsim_dense(q, d, qm, dm), q, d)
    @test cosine(Array(gq), gq_ref) ≥ 0.999
    @test cosine(Array(gd), gd_ref) ≥ 0.999

    Qg, Dg = CuArray(Q), CuArray(D)
    qmgB = CuArray(fill(true, Tq, 3))
    dmgB = CuArray(fill(true, Td, 3))
    q1, d1 = fill(true, Tq), fill(true, Td)
    gQ_gpu, gD_gpu = gradient((Q, D) -> sum(maxsim(Q, D, qmgB, dmgB)), Qg, Dg)
    gQ_cpu, gD_cpu = gradient((Q, D) -> sum(maxsim(Q, D)), Q, D)
    @test gQ_gpu isa CuArray && gD_gpu isa CuArray
    @test cosine(Array(gQ_gpu), gQ_cpu) ≥ 0.999
    @test cosine(Array(gD_gpu), gD_cpu) ≥ 0.999
    gQ_ref, gD_ref = gradient((Q, D) -> sum(maxsim_dense(Q[:, :, b], D[:, :, b], q1, d1)
                                           for b in 1:3), Q, D)
    @test cosine(Array(gQ_gpu), gQ_ref) ≥ 0.999
    @test cosine(Array(gD_gpu), gD_ref) ≥ 0.999

    gQi_gpu, gDi_gpu = gradient((Q, D) -> sum(maxsim(Q, D, qmgB, dmgB, InBatch())), Qg, Dg)
    gQi_cpu, gDi_cpu = gradient((Q, D) -> sum(maxsim(Q, D, InBatch())), Q, D)
    @test gQi_gpu isa CuArray && gDi_gpu isa CuArray
    @test cosine(Array(gQi_gpu), gQi_cpu) ≥ 0.999
    @test cosine(Array(gDi_gpu), gDi_cpu) ≥ 0.999
    gQi_ref, gDi_ref = gradient((Q, D) -> sum(maxsim_dense(Q[:, :, i], D[:, :, j], q1, d1)
                                             for j in 1:3, i in 1:3), Q, D)
    @test cosine(Array(gQi_gpu), gQi_ref) ≥ 0.999
    @test cosine(Array(gDi_gpu), gDi_ref) ≥ 0.999

    gallery = randn(T, dim, Td, 7)
    idxs = Int32[1 2 3; 4 5 6]
    idxg = CuArray(idxs)
    Gg = CuArray(gallery)
    dmgG = CuArray(fill(true, Td, 7))
    gQc_gpu, gG_gpu = gradient((Q, G) -> sum(maxsim(Q, G, idxg, qmgB, dmgG)), Qg, Gg)
    gQc_cpu, gG_cpu = gradient((Q, G) -> sum(maxsim(Q, G, idxs)), Q, gallery)
    @test gQc_gpu isa CuArray && gG_gpu isa CuArray
    @test cosine(Array(gQc_gpu), gQc_cpu) ≥ 0.999
    @test cosine(Array(gG_gpu), gG_cpu) ≥ 0.999
    gQc_ref, gG_ref = gradient((Q, G) -> sum(maxsim_dense(Q[:, :, b], G[:, :, Int(idxs[c, b])],
                                                         q1, d1)
                                            for c in 1:2, b in 1:3), Q, gallery)
    @test cosine(Array(gQc_gpu), gQc_ref) ≥ 0.999
    @test cosine(Array(gG_gpu), gG_ref) ≥ 0.999
    @test_throws ArgumentError maxsim(Qg, Gg, idxs, qmgB, dmgG)

    cfg_i = MaxSim{T}(T(-1.0f4), false, InvGrid())
    gQi_mode = gradient(x -> sum(maxsim(cfg_i, x, Dg, qmgB, dmgB)), Qg)[1]
    gQa_mode = gradient(x -> sum(maxsim(cfg_a, x, Dg, qmgB, dmgB)), Qg)[1]
    @test cosine(Array(gQi_mode), Array(gQa_mode)) ≥ 0.999

    neg = T(0)
    s_gpu = maxsim(qg, dg, qmg, dmg; neg)
    @test only(Array(s_gpu)) ≈ maxsim(q, d, qm, dm; neg) rtol=1e-5 atol=1e-5
    @test only(Array(s_gpu)) ≈ maxsim_dense(q, d, qm, dm; neg) rtol=1e-5 atol=1e-5
end
