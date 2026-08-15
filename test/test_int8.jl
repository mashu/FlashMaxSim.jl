function spearman(a, b)
    function ranks(x)
        p = sortperm(vec(Float64.(x)))
        r = similar(p, Float64)
        r[p] = 1:length(p)
        r
    end
    ra, rb = ranks(a), ranks(b)
    ra .-= sum(ra) / length(ra)
    rb .-= sum(rb) / length(rb)
    na, nb = sqrt(sum(abs2, ra)), sqrt(sum(abs2, rb))
    (na == 0 || nb == 0) && return 1.0
    sum(ra .* rb) / (na * nb)
end

function topk_overlap(a, b, k)
    pa = partialsortperm(vec(Float64.(a)), 1:k; rev = true)
    pb = partialsortperm(vec(Float64.(b)), 1:k; rev = true)
    length(intersect(pa, pb)) / k
end

@testset "INT8 deferred dequant matches dequantized FP MaxSim" begin
    Random.seed!(40)
    T = Float32
    dim, Tq, Td = 16, 8, 11
    q = randn(T, dim, Tq)
    d = randn(T, dim, Td)
    d_idx = quantize_int8_symmetric(d)
    q_idx = quantize_int8_symmetric(q)
    q_hat = FlashMaxSim.dequant_int8(q_idx)
    d_hat = FlashMaxSim.dequant_int8(d_idx)
    s8 = maxsim(q, d_idx)
    s_hat = maxsim(q_hat, d_hat)
    @test s8 ≈ s_hat rtol=1e-5 atol=1e-5
    @test maxsim(q, d_idx; normalize = true) ≈ s_hat / T(Tq) rtol=1e-5

    Q = randn(T, dim, Tq, 3)
    D = randn(T, dim, Td, 3)
    D_idx = quantize_int8_symmetric(D)
    Q_idx = quantize_int8_symmetric(Q)
    s8b = maxsim(Q, D_idx)
    s_hatb = maxsim(FlashMaxSim.dequant_int8(Q_idx), FlashMaxSim.dequant_int8(D_idx))
    @test s8b ≈ s_hatb rtol=1e-5 atol=1e-5
end

@testset "INT8 Spearman / top-20 vs FP32" begin
    Random.seed!(41)
    T = Float32
    dim, Tq, Td, N = 32, 12, 20, 64
    q = randn(T, dim, Tq)
    q ./= sqrt.(sum(abs2, q; dims = 1) .+ eps(T))
    gallery = randn(T, dim, Td, N)
    gallery ./= sqrt.(sum(abs2, gallery; dims = 1) .+ eps(T))
    s_fp = [maxsim(q, gallery[:, :, j]) for j in 1:N]
    idx = [quantize_int8_symmetric(gallery[:, :, j]) for j in 1:N]
    s_i8 = [maxsim(q, idx[j]) for j in 1:N]
    @test spearman(s_fp, s_i8) ≥ 0.95
    @test topk_overlap(s_fp, s_i8, 20) ≥ 0.7
end

@testset "INT8 KA CPU vs host" begin
    Random.seed!(42)
    T = Float32
    backend = KernelAbstractions.CPU()
    q = randn(T, 8, 5)
    d = randn(T, 8, 7)
    d_idx = quantize_int8_symmetric(d)
    qq = quantize_int8_symmetric(q)
    qm, dm = trues(5), trues(7)
    s_h, a_h = FlashMaxSim.int8_pair_forward_host(qq.codes, qq.scales, d_idx.codes,
                                                  d_idx.scales, qm, dm)
    s_k, a_k = FlashMaxSim.int8_pair_forward_ka(backend, qq.codes, qq.scales,
                                                d_idx.codes, d_idx.scales, qm, dm)
    @test a_k == a_h
    @test only(s_k) ≈ s_h rtol=1e-5 atol=1e-5
end

@testset "INT8 is forward-only" begin
    q = randn(Float32, 4, 3)
    d = quantize_int8_symmetric(randn(Float32, 4, 5))
    @test_throws ArgumentError gradient(q -> maxsim(q, d), q)
end
