# Paper-style correctness: fused vs dense scores; ∇ cosine vs dense Zygote path.

"""Dense MaxSim through Zygote (materializes S) — gradient oracle."""
function dense_grads(q, d, qmask, dmask; normalize = false)
    gradient((q, d) -> maxsim_dense(q, d, qmask, dmask; normalize), q, d)
end

@testset "gradient cosine vs dense (≥ 0.999, paper protocol)" begin
    Random.seed!(42)
    T = Float32
    for (dim, Tq, Td) in ((16, 8, 12), (32, 16, 24), (64, 32, 40))
        q = randn(T, dim, Tq)
        d = randn(T, dim, Td)
        q ./= sqrt.(sum(abs2, q; dims = 1) .+ eps(T))
        d ./= sqrt.(sum(abs2, d; dims = 1) .+ eps(T))
        qm, dm = trues(Tq), trues(Td)
        gq_ref, gd_ref = dense_grads(q, d, qm, dm)
        for mode in (AtomicUnified(), InvGrid(), AtomicUnified, InvGrid)
            cfg = MaxSim{T}(T(-1.0f4), false, mode)
            gq, gd = gradient((q, d) -> maxsim(cfg, q, d, qm, dm), q, d)
            @test cosine(gq, gq_ref) ≥ 0.999
            @test cosine(gd, gd_ref) ≥ 0.999
        end
        cfg_a = MaxSim{T}(T(-1.0f4), false, AtomicUnified())
        cfg_i = MaxSim{T}(T(-1.0f4), false, InvGrid())
        gqa, gda = gradient((q, d) -> maxsim(cfg_a, q, d, qm, dm), q, d)
        gqi, gdi = gradient((q, d) -> maxsim(cfg_i, q, d, qm, dm), q, d)
        @test cosine(gqa, gqi) ≥ 0.999
        @test cosine(gda, gdi) ≥ 0.999
    end
end

@testset "normalized gradient cosine" begin
    Random.seed!(43)
    T = Float32
    q = randn(T, 16, 10)
    d = randn(T, 16, 14)
    q ./= sqrt.(sum(abs2, q; dims = 1) .+ eps(T))
    d ./= sqrt.(sum(abs2, d; dims = 1) .+ eps(T))
    qm, dm = trues(10), trues(14)
    gq_ref, gd_ref = dense_grads(q, d, qm, dm; normalize = true)
    cfg = MaxSim{T}(T(-1.0f4), true, InvGrid())
    gq, gd = gradient((q, d) -> maxsim(cfg, q, d, qm, dm), q, d)
    @test cosine(gq, gq_ref) ≥ 0.999
    @test cosine(gd, gd_ref) ≥ 0.999
end

@testset "inverse-grid CSR roundtrip" begin
    argmax = Int32[3, 1, 3, 0, 2, 1]
    row_ptr, col_idx = FlashMaxSim.build_inverse_csr(argmax, 3)
    @test length(row_ptr) == 4
    @test Int(row_ptr[end]) == 5  # five active sources
    seg1 = Int.(col_idx[(Int(row_ptr[1]) + 1):Int(row_ptr[2])])
    @test sort(seg1) == [2, 6]
    # empty dest
    empty_arg = Int32[0, 0, 0]
    rp, ci = FlashMaxSim.build_inverse_csr(empty_arg, 2)
    @test Int(rp[end]) == 0
    @test isempty(ci)
end

@testset "InvGrid accumulate matches AtomicUnified" begin
    Random.seed!(44)
    T = Float32
    q = randn(T, 4, 5)
    δ = T[1, 0, 2, 0, 3]
    arg = Int32[2, 0, 1, 0, 2]
    dd_a = zeros(T, 4, 3)
    dd_i = zeros(T, 4, 3)
    FlashMaxSim.accumulate_doc!(AtomicUnified(), dd_a, q, δ, arg)
    FlashMaxSim.accumulate_doc!(InvGrid(), dd_i, q, δ, arg)
    @test dd_a ≈ dd_i
end
