# Runs every KernelAbstractions kernel on the CPU backend and checks it against
# the host reference. Without this, the paired / in-batch / candidate device
# kernels are unreachable in CI: the host methods win on `Array` inputs, and the
# pullbacks dispatch on `array_backend`, which is always `CPU()` on the host.
#
# Forward KA paths are reached by passing `view(X, :, :, :)` — a SubArray is not
# an `Array`, so the generic method is selected, but `get_backend` still says CPU.
# Backward KA paths are called through their `*_pullback_ka` entry points.

@testset "KA CPU parity: all layouts" begin
    Random.seed!(2024)
    T = Float32
    backend = KernelAbstractions.CPU()
    neg = T(-1.0f4)
    dim, Tq, Td, B, N, C = 8, 6, 7, 3, 5, 2

    Q = randn(T, dim, Tq, B)
    D = randn(T, dim, Td, B)
    G = randn(T, dim, Td, N)
    qm = collect(isodd(t) for t in 1:Tq, b in 1:B)
    dm = collect(u % 3 != 0 for u in 1:Td, b in 1:B)
    dmG = collect(u % 4 != 0 for u in 1:Td, j in 1:N)
    idxs = Int32[1 2 3; 4 0 5]
    inv_n = ones(T, B)

    Qv, Dv, Gv = view(Q, :, :, :), view(D, :, :, :), view(G, :, :, :)

    @testset "paired forward" begin
        s_h, a_h = FlashMaxSim.paired_forward(Q, D, qm, dm, neg)
        s_k, a_k = FlashMaxSim.paired_forward(Qv, Dv, qm, dm, neg)
        @test a_k == a_h
        @test s_k ≈ s_h rtol=1e-5 atol=1e-5
        @test s_k ≈ maxsim_dense(Q, D, qm, dm) rtol=1e-5 atol=1e-5
    end

    @testset "in-batch forward" begin
        S_h, A_h = FlashMaxSim.inbatch_forward(Q, D, qm, dm, neg)
        S_k, A_k = FlashMaxSim.inbatch_forward(Qv, Dv, qm, dm, neg)
        @test A_k == A_h
        @test S_k ≈ S_h rtol=1e-5 atol=1e-5
        @test S_k ≈ maxsim_dense(Q, D, qm, dm, InBatch()) rtol=1e-5 atol=1e-5
    end

    @testset "candidates forward" begin
        S_h, A_h = FlashMaxSim.candidates_forward(Q, G, idxs, qm, dmG, neg)
        S_k, A_k = FlashMaxSim.candidates_forward(Qv, Gv, idxs, qm, dmG, neg)
        @test A_k == A_h
        @test S_k ≈ S_h rtol=1e-5 atol=1e-5
        @test S_k[2, 2] == neg
        @test S_k ≈ maxsim_dense(Q, G, idxs, qm, dmG) rtol=1e-5 atol=1e-5
    end

    @testset "paired pullback" begin
        _, args = FlashMaxSim.paired_forward(Q, D, qm, dm, neg)
        Δ = randn(T, B)
        for mode in (AtomicUnified(), InvGrid())
            dQ_h, dD_h = FlashMaxSim.paired_pullback(backend, Δ, Q, D, qm, args, inv_n, mode)
            dQ_k, dD_k = FlashMaxSim.paired_pullback_ka(backend, Δ, Q, D, qm, args, inv_n, mode)
            @test dQ_k ≈ dQ_h rtol=1e-5 atol=1e-5
            @test dD_k ≈ dD_h rtol=1e-5 atol=1e-5
        end
    end

    @testset "in-batch pullback" begin
        _, args = FlashMaxSim.inbatch_forward(Q, D, qm, dm, neg)
        Δ = randn(T, B, B)
        for mode in (AtomicUnified(), InvGrid())
            dQ_h, dD_h = FlashMaxSim.inbatch_pullback(backend, Δ, Q, D, qm, args, inv_n, mode)
            dQ_k, dD_k = FlashMaxSim.inbatch_pullback_ka(backend, Δ, Q, D, qm, args, inv_n, mode)
            @test dQ_k ≈ dQ_h rtol=1e-5 atol=1e-5
            @test dD_k ≈ dD_h rtol=1e-5 atol=1e-5
        end
    end

    @testset "candidates pullback" begin
        _, args = FlashMaxSim.candidates_forward(Q, G, idxs, qm, dmG, neg)
        Δ = randn(T, C, B)
        for mode in (AtomicUnified(), InvGrid())
            dQ_h, dG_h = FlashMaxSim.candidates_pullback(backend, Δ, Q, G, idxs, qm, args, inv_n, mode)
            dQ_k, dG_k = FlashMaxSim.candidates_pullback_ka(backend, Δ, Q, G, idxs, qm, args, inv_n, mode)
            @test dQ_k ≈ dQ_h rtol=1e-5 atol=1e-5
            @test dG_k ≈ dG_h rtol=1e-5 atol=1e-5
        end
    end

    @testset "wrapped BitArray masks do not recurse" begin
        q, d = randn(T, 4, 3), randn(T, 4, 5)
        bm = trues(3, 1)
        @test FlashMaxSim.array_backend(reshape(bm, 3)) === backend
        @test FlashMaxSim.array_backend(PermutedDimsArray(bm, (2, 1))) === backend
        @test maxsim(q, d, vec(view(bm, :, 1)), trues(5)) isa T
    end

    @testset "host views take the BLAS pair path" begin
        s_view = maxsim(view(Q, :, :, 1), view(D, :, :, 1),
                        view(qm, :, 1), view(dm, :, 1))
        s_copy = maxsim(Q[:, :, 1], D[:, :, 1], qm[:, 1], dm[:, 1])
        @test s_view ≈ s_copy rtol=1e-6
    end
end

@testset "pair_pullback validates qmask" begin
    T = Float32
    q, d = randn(T, 4, 3), randn(T, 4, 5)
    arg = Int32[1, 2, 1]
    backend = KernelAbstractions.CPU()
    @test_throws DimensionMismatch FlashMaxSim.pair_pullback(
        backend, one(T), q, d, trues(2), arg, AtomicUnified())
    @test_throws DimensionMismatch FlashMaxSim.pair_pullback_ka(
        backend, one(T), q, d, trues(2), arg, AtomicUnified())
end

@testset "as_array_cotangent axes and scalar" begin
    T = Float32
    proto = ones(T, 3)
    @test_throws DimensionMismatch FlashMaxSim.as_array_cotangent(T, ones(T, 2), proto)
    @test_throws ArgumentError FlashMaxSim.as_array_cotangent(T, one(T), proto)
    proto1 = ones(T, 1)
    @test FlashMaxSim.as_array_cotangent(T, T(2), proto1) == fill(T(2), 1)
    Δ = T[1, 2, 3]
    @test FlashMaxSim.as_array_cotangent(T, Δ, proto) == Δ
end
