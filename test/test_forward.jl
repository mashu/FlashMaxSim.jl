@testset "pair matches dense (Prop. 1)" begin
    Random.seed!(1)
    T = Float32
    for (dim, Tq, Td) in ((8, 4, 5), (16, 12, 20), (32, 24, 31))
        q = randn(T, dim, Tq)
        d = randn(T, dim, Td)
        qm, dm = trues(Tq), trues(Td)
        @test maxsim(q, d, qm, dm) ≈ maxsim_dense(q, d, qm, dm) rtol=1e-5 atol=1e-5
        @test maxsim(q, d, qm, dm; normalize = true) ≈
              maxsim_dense(q, d, qm, dm; normalize = true) rtol=1e-5 atol=1e-5
    end
end

@testset "masks" begin
    Random.seed!(2)
    T = Float32
    q = randn(T, 16, 6)
    d = randn(T, 16, 8)
    qm = [true, true, false, true, false, true]
    dm = [true, false, true, true, false, true, true, false]
    @test maxsim(q, d, qm, dm) ≈ maxsim_dense(q, d, qm, dm) rtol=1e-5
end

@testset "layouts" begin
    Random.seed!(3)
    T = Float32
    dim, Tq, Td, B = 16, 8, 10, 4
    Q = randn(T, dim, Tq, B)
    D = randn(T, dim, Td, B)
    paired = maxsim(Q, D)
    @test size(paired) == (B,)
    for b in 1:B
        @test paired[b] ≈ maxsim(Q[:, :, b], D[:, :, b]) rtol=1e-5
    end
    S = maxsim(Q, D, InBatch())
    @test size(S) == (B, B)
    @test S[2, 3] ≈ maxsim(Q[:, :, 3], D[:, :, 2]) rtol=1e-5
    @test S ≈ maxsim_dense(Q, D, trues(Tq, B), trues(Td, B), InBatch()) rtol=1e-5

    gallery = randn(T, dim, Td, 12)
    idxs = Int32[1 2 3 4; 5 6 7 8; 0 1 2 3]
    Sc = maxsim(Q, gallery, idxs)
    @test size(Sc) == (3, B)
    @test Sc[1, 1] ≈ maxsim(Q[:, :, 1], gallery[:, :, 1]) rtol=1e-5
    @test Sc[3, 1] == MaxSim{T}().neg
    @test Sc ≈ maxsim_dense(Q, gallery, idxs, trues(Tq, B), trues(Td, 12)) rtol=1e-5
    Scn = maxsim(Q, gallery, idxs; normalize = true)
    @test Scn[3, 1] == MaxSim{T}().neg
    @test Scn ≈ maxsim_dense(Q, gallery, idxs, trues(Tq, B), trues(Td, 12); normalize = true) rtol=1e-5
end

@testset "MaxSim callable + Float64" begin
    cfg = MaxSim(normalize = true, backward = InvGrid)
    q, d = randn(Float32, 8, 4), randn(Float32, 8, 5)
    @test cfg(q, d) ≈ maxsim(q, d; normalize = true)

    Q = randn(Float32, 8, 4, 3)
    D = randn(Float32, 8, 5, 3)
    @test cfg(Q, D) ≈ maxsim(Q, D; normalize = true)
    @test cfg(Q, D, InBatch()) ≈ maxsim(Q, D, InBatch(); normalize = true)
    idxs = Int32[1 2 3; 2 3 1]
    G = randn(Float32, 8, 5, 4)
    @test cfg(Q, G, idxs) ≈ maxsim(Q, G, idxs; normalize = true)

    q64, d64 = randn(Float64, 8, 4), randn(Float64, 8, 5)
    cfg64 = MaxSim(Float64; normalize = false, backward = AtomicUnified())
    @test cfg64(q64, d64) ≈ maxsim_dense(q64, d64) rtol=1e-10
end

@testset "colocation" begin
    q, d = randn(Float32, 4, 3), randn(Float32, 4, 5)
    # BitArray masks are host-colocated with Array features
    @test maxsim(q, d, trues(3), trues(5)) isa Float32
    @test_throws DimensionMismatch maxsim(q, d, trues(2), trues(5))
end

@testset "layout shape checks" begin
    T = Float32
    Q = randn(T, 4, 3, 2)
    D = randn(T, 4, 5, 2)
    @test_throws DimensionMismatch maxsim(Q, D, trues(3, 2), trues(5, 3))
    @test_throws DimensionMismatch maxsim(Q, D, trues(2, 2), trues(5, 2))
    @test_throws DimensionMismatch maxsim(Q, randn(T, 5, 5, 2))
    @test_throws DimensionMismatch maxsim(Q, randn(T, 4, 5, 3))
    @test_throws DimensionMismatch maxsim(Q, D, trues(3, 2), trues(4, 2), InBatch())
    gallery = randn(T, 4, 5, 4)
    idxs = Int32[1 2; 3 4; 0 1]
    @test_throws DimensionMismatch maxsim(Q, gallery, idxs, trues(3, 3), trues(5, 4))
    @test_throws DimensionMismatch maxsim(Q, gallery, Int32[1 2 3; 0 1 2], trues(3, 2), trues(5, 4))
    @test_throws DimensionMismatch maxsim(Q, randn(T, 5, 5, 4), idxs)
end
