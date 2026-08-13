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
    @test s_gpu ≈ s_cpu rtol=1e-5 atol=1e-5

    @test_throws ArgumentError maxsim(qg, dg, qm, dm)
    @test maxsim(qg, dg) ≈ s_cpu rtol=1e-5 atol=1e-5

    Q = randn(T, dim, Tq, 3)
    D = randn(T, dim, Td, 3)
    Sg = maxsim(CuArray(Q), CuArray(D), InBatch())
    @test Sg isa CuArray
    @test Array(Sg) ≈ maxsim(Q, D, InBatch()) rtol=1e-5 atol=1e-5
    paired_g = maxsim(CuArray(Q), CuArray(D))
    @test paired_g isa CuArray
    @test Array(paired_g) ≈ maxsim(Q, D) rtol=1e-5 atol=1e-5

    cfg = MaxSim{T}(T(-1.0f4), false, InvGrid())
    gq = gradient(x -> maxsim(cfg, x, dg, qmg, dmg), qg)[1]
    @test gq isa CuArray
    @test all(isfinite, Array(gq))
    gd = gradient(x -> maxsim(cfg, qg, x, qmg, dmg), dg)[1]
    @test gd isa CuArray
    @test all(isfinite, Array(gd))

    cfg_a = MaxSim{T}(T(-1.0f4), false, AtomicUnified())
    gqa = gradient(x -> maxsim(cfg_a, x, dg, qmg, dmg), qg)[1]
    @test cosine(Array(gq), Array(gqa)) ≥ 0.999
    @test 0.99 ≤ relnorm(Array(gq), Array(gqa)) ≤ 1.01
end
