@testset "CUDA (KernelAbstractions backend)" begin
    try
        using CUDA
    catch
        @info "CUDA not loadable — skip"
        return
    end
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
    @test s_gpu ≈ s_cpu rtol=1e-4 atol=1e-4

    @test_throws ArgumentError maxsim(qg, dg, qm, dm)
    @test Array(maxsim(qg, dg)) ≈ s_cpu rtol=1e-4 atol=1e-4

    Q = randn(T, dim, Tq, 3)
    D = randn(T, dim, Td, 3)
    @test Array(maxsim(CuArray(Q), CuArray(D), InBatch())) ≈
          maxsim(Q, D, InBatch()) rtol=1e-4 atol=1e-4
    @test Array(maxsim(CuArray(Q), CuArray(D))) ≈ maxsim(Q, D) rtol=1e-4 atol=1e-4

    cfg = MaxSim{T}(T(-1.0f4), false, InvGrid())
    gq = gradient(x -> maxsim(cfg, x, dg, qmg, dmg), qg)[1]
    @test gq !== nothing && all(isfinite, Array(gq))
    gd = gradient(x -> maxsim(cfg, qg, x, qmg, dmg), dg)[1]
    @test gd !== nothing && all(isfinite, Array(gd))
end
