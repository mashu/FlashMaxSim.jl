# FlashMaxSimCUDAExt — WMMA tensor-core scans (sm ≥ 7.0 F16; sm ≥ 7.5 INT8).
#
# Overrides launch_*_scan! for CuArray{Float16} / CuArray{Int8}. Float32 stays
# on the KernelAbstractions SRAM tile path (no silent downcast).
# Four warps per block (64 query tokens); K / Tq / Td padded to 16 in smem.

module FlashMaxSimCUDAExt

using FlashMaxSim
using CUDA
using CUDA: CUDABackend, CuArray, CuDynamicSharedArray, WMMA

const WMMA_TILE = 16
const WMMA_WARPS = 4
const WMMA_THREADS = WMMA_WARPS * 32
const WMMA_QSTRIDE = WMMA_WARPS * WMMA_TILE
const WMMA_F16_TILE = WMMA_TILE * WMMA_TILE * sizeof(Float16)
const WMMA_F32_TILE = WMMA_TILE * WMMA_TILE * sizeof(Float32)
const WMMA_I8_TILE = WMMA_TILE * WMMA_TILE * sizeof(Int8)
const WMMA_I32_TILE = WMMA_TILE * WMMA_TILE * sizeof(Int32)
const WMMA_F16_SMEM = WMMA_WARPS * WMMA_F16_TILE + WMMA_F16_TILE + WMMA_WARPS * WMMA_F32_TILE
const WMMA_I8_SMEM = WMMA_WARPS * WMMA_I8_TILE + WMMA_I8_TILE + WMMA_WARPS * WMMA_I32_TILE

function FlashMaxSim.tensor_cores_active(::CUDABackend, ::Type{Float16})
    CUDA.functional() && CUDA.capability(CUDA.device()) >= v"7.0"
end

function FlashMaxSim.tensor_cores_active(::CUDABackend, ::Type{Int8})
    CUDA.functional() && CUDA.capability(CUDA.device()) >= v"7.5"
end

@inline function wmma_ids()
    tid = Int(threadIdx().x)
    warp = (tid - 1) >> 5
    lane = ((tid - 1) & 31) + 1
    tid, warp, lane
end

@inline function wmma_f16_panels(warp)
    As = CuDynamicSharedArray(Float16, (WMMA_TILE, WMMA_TILE), warp * WMMA_F16_TILE)
    Bs = CuDynamicSharedArray(Float16, (WMMA_TILE, WMMA_TILE), WMMA_WARPS * WMMA_F16_TILE)
    Cs = CuDynamicSharedArray(Float32, (WMMA_TILE, WMMA_TILE),
                              WMMA_WARPS * WMMA_F16_TILE + WMMA_F16_TILE + warp * WMMA_F32_TILE)
    As, Bs, Cs
end

@inline function wmma_i8_panels(warp)
    As = CuDynamicSharedArray(Int8, (WMMA_TILE, WMMA_TILE), warp * WMMA_I8_TILE)
    Bs = CuDynamicSharedArray(Int8, (WMMA_TILE, WMMA_TILE), WMMA_WARPS * WMMA_I8_TILE)
    Cs = CuDynamicSharedArray(Int32, (WMMA_TILE, WMMA_TILE),
                              WMMA_WARPS * WMMA_I8_TILE + WMMA_I8_TILE + warp * WMMA_I32_TILE)
    As, Bs, Cs
end

# ---- F16 launch --------------------------------------------------------------

function FlashMaxSim.launch_pair_scan!(backend::CUDABackend, argmax_u, partial,
                                       q::CuArray{Float16}, d::CuArray{Float16},
                                       qmask, dmask)
    if FlashMaxSim.tensor_cores_active(backend, Float16)
        launch_pair_wmma!(argmax_u, partial, q, d, qmask, dmask)
    else
        FlashMaxSim.launch_grouped!(FlashMaxSim.pair_tile_kernel!, backend,
                                    FlashMaxSim.query_tile_group(size(q, 2)), size(q, 2),
                                    argmax_u, partial, q, d, qmask, dmask)
    end
    nothing
end

function FlashMaxSim.launch_paired_scan!(backend::CUDABackend, args, partial,
                                         Q::CuArray{Float16,3}, D::CuArray{Float16,3},
                                         qmask, dmask)
    if FlashMaxSim.tensor_cores_active(backend, Float16)
        launch_paired_wmma!(args, partial, Q, D, qmask, dmask)
    else
        nd = (size(Q, 2), size(Q, 3))
        FlashMaxSim.launch_grouped!(FlashMaxSim.paired_tile_kernel!, backend,
                                    FlashMaxSim.query_tile_group(nd), nd,
                                    args, partial, Q, D, qmask, dmask)
    end
    nothing
end

function FlashMaxSim.launch_packed_scan!(backend::CUDABackend, args, partial,
                                         q::CuArray{Float16}, packed::CuArray{Float16},
                                         cu, qmask)
    if FlashMaxSim.tensor_cores_active(backend, Float16)
        launch_packed_wmma!(args, partial, q, packed, cu, qmask)
    else
        nd = (size(q, 2), length(cu) - 1)
        FlashMaxSim.launch_grouped!(FlashMaxSim.packed_tile_kernel!, backend,
                                    FlashMaxSim.query_tile_group(nd), nd,
                                    args, partial, q, packed, cu, qmask)
    end
    nothing
end

function FlashMaxSim.launch_varlen_scan!(backend::CUDABackend, args, partial,
                                         Qp::CuArray{Float16}, Dp::CuArray{Float16},
                                         cu_q, cu_d)
    if FlashMaxSim.tensor_cores_active(backend, Float16)
        launch_varlen_wmma!(args, partial, Qp, Dp, cu_q, cu_d)
    else
        nd = (size(args, 1), size(args, 2))
        FlashMaxSim.launch_grouped!(FlashMaxSim.varlen_tile_kernel!, backend,
                                    FlashMaxSim.query_tile_group(nd), nd,
                                    args, partial, Qp, Dp, cu_q, cu_d)
    end
    nothing
end

function FlashMaxSim.launch_candidates_scan!(backend::CUDABackend, args, partial,
                                             Q::CuArray{Float16,3}, gallery::CuArray{Float16,3},
                                             idx, qmask, dmask, N)
    if FlashMaxSim.tensor_cores_active(backend, Float16)
        launch_candidates_wmma!(args, partial, Q, gallery, idx, qmask, dmask, N)
    else
        nd = (size(Q, 2), size(idx, 1), size(Q, 3))
        FlashMaxSim.launch_grouped!(FlashMaxSim.candidates_tile_kernel!, backend,
                                    FlashMaxSim.query_tile_group(nd), nd,
                                    args, partial, Q, gallery, idx, qmask, dmask, N)
    end
    nothing
end

function FlashMaxSim.inbatch_device_scan!(backend::CUDABackend, S::CuArray{Float16},
                                          args, Q::CuArray{Float16,3}, D::CuArray{Float16,3},
                                          qmask, dmask)
    if FlashMaxSim.tensor_cores_active(backend, Float16)
        Tq, Bq, Bd = size(Q, 2), size(Q, 3), size(D, 3)
        partial = similar(Q, Float16, Tq, Bd, Bq)
        fill!(partial, zero(Float16))
        launch_inbatch_wmma!(args, partial, Q, D, qmask, dmask)
        S .= dropdims(sum(partial; dims = 1); dims = 1)
    else
        FlashMaxSim.inbatch_gemm_scan!(backend, S, args, Q, D, qmask, dmask)
    end
    nothing
end

function FlashMaxSim.launch_int8_pair_scan!(backend::CUDABackend, args, partial,
                                            qc::CuArray{Int8}, qs, dc::CuArray{Int8}, ds,
                                            qmask, dmask)
    if FlashMaxSim.tensor_cores_active(backend, Int8)
        launch_int8_pair_wmma!(args, partial, qc, qs, dc, ds, qmask, dmask)
    else
        FlashMaxSim.launch_grouped!(FlashMaxSim.int8_pair_tile_kernel!, backend,
                                    FlashMaxSim.query_tile_group(size(qc, 2)), size(qc, 2),
                                    args, partial, qc, qs, dc, ds, qmask, dmask)
    end
    nothing
end

function FlashMaxSim.launch_int8_paired_scan!(backend::CUDABackend, args, partial,
                                              Qc::CuArray{Int8,3}, Qs, Dc::CuArray{Int8,3}, Ds,
                                              qmask, dmask)
    if FlashMaxSim.tensor_cores_active(backend, Int8)
        launch_int8_paired_wmma!(args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    else
        nd = (size(Qc, 2), size(Qc, 3))
        FlashMaxSim.launch_grouped!(FlashMaxSim.int8_paired_tile_kernel!, backend,
                                    FlashMaxSim.query_tile_group(nd), nd,
                                    args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    end
    nothing
end

function launch_pair_wmma!(argmax_u, partial, q, d, qmask, dmask)
    Tq = size(q, 2)
    iszero(Tq) && return
    @cuda threads = WMMA_THREADS blocks = cld(Tq, WMMA_QSTRIDE) shmem = WMMA_F16_SMEM pair_wmma_kernel!(
        argmax_u, partial, q, d, qmask, dmask)
    nothing
end

function launch_paired_wmma!(args, partial, Q, D, qmask, dmask)
    Tq, B = size(Q, 2), size(Q, 3)
    (iszero(Tq) || iszero(B)) && return
    @cuda threads = WMMA_THREADS blocks = (cld(Tq, WMMA_QSTRIDE), B) shmem = WMMA_F16_SMEM paired_wmma_kernel!(
        args, partial, Q, D, qmask, dmask)
    nothing
end

function launch_packed_wmma!(args, partial, q, packed, cu, qmask)
    Tq = size(q, 2)
    B = length(cu) - 1
    (iszero(Tq) || iszero(B)) && return
    @cuda threads = WMMA_THREADS blocks = (cld(Tq, WMMA_QSTRIDE), B) shmem = WMMA_F16_SMEM packed_wmma_kernel!(
        args, partial, q, packed, cu, qmask)
    nothing
end

function launch_varlen_wmma!(args, partial, Qp, Dp, cu_q, cu_d)
    max_q, N = size(args, 1), size(args, 2)
    (iszero(max_q) || iszero(N)) && return
    @cuda threads = WMMA_THREADS blocks = (cld(max_q, WMMA_QSTRIDE), N) shmem = WMMA_F16_SMEM varlen_wmma_kernel!(
        args, partial, Qp, Dp, cu_q, cu_d)
    nothing
end

function launch_candidates_wmma!(args, partial, Q, gallery, idx, qmask, dmask, N)
    Tq, C, B = size(Q, 2), size(idx, 1), size(Q, 3)
    (iszero(Tq) || iszero(C) || iszero(B)) && return
    @cuda threads = WMMA_THREADS blocks = (cld(Tq, WMMA_QSTRIDE), C, B) shmem = WMMA_F16_SMEM candidates_wmma_kernel!(
        args, partial, Q, gallery, idx, qmask, dmask, Int32(N))
    nothing
end

function launch_inbatch_wmma!(args, partial, Q, D, qmask, dmask)
    Tq, Bq, Bd = size(Q, 2), size(Q, 3), size(D, 3)
    (iszero(Tq) || iszero(Bq) || iszero(Bd)) && return
    @cuda threads = WMMA_THREADS blocks = (cld(Tq, WMMA_QSTRIDE), Bq, Bd) shmem = WMMA_F16_SMEM inbatch_wmma_kernel!(
        args, partial, Q, D, qmask, dmask)
    nothing
end

function launch_int8_pair_wmma!(args, partial, qc, qs, dc, ds, qmask, dmask)
    Tq = size(qc, 2)
    iszero(Tq) && return
    @cuda threads = WMMA_THREADS blocks = cld(Tq, WMMA_QSTRIDE) shmem = WMMA_I8_SMEM int8_pair_wmma_kernel!(
        args, partial, qc, qs, dc, ds, qmask, dmask)
    nothing
end

function launch_int8_paired_wmma!(args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    Tq, B = size(Qc, 2), size(Qc, 3)
    (iszero(Tq) || iszero(B)) && return
    @cuda threads = WMMA_THREADS blocks = (cld(Tq, WMMA_QSTRIDE), B) shmem = WMMA_I8_SMEM int8_paired_wmma_kernel!(
        args, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    nothing
end

# ---- F16 kernels -------------------------------------------------------------

function pair_wmma_kernel!(argmax_out, partial, q, d, qmask, dmask)
    conf = WMMA.Config{16, 16, 16, Float32}
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    As, Bs, Cs = wmma_f16_panels(warp)
    dim = size(q, 1)
    Tq = size(q, 2)
    Td = size(d, 2)
    mx = 0.0f0
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? d[kg, ug] : Float16(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? q[kg, tg] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq && qmask[tg]
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug] || continue
                    s = Cs[lane, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= Tq
            @inbounds partial[tg] = arg == Int32(0) ? Float16(0) : Float16(mx)
            @inbounds argmax_out[tg] = arg
        end
    end
    return
end

function paired_wmma_kernel!(argmax_out, partial, Q, D, qmask, dmask)
    conf = WMMA.Config{16, 16, 16, Float32}
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    b = Int(blockIdx().y)
    As, Bs, Cs = wmma_f16_panels(warp)
    dim = size(Q, 1)
    Tq = size(Q, 2)
    Td = size(D, 2)
    mx = 0.0f0
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? D[kg, ug, b] : Float16(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? Q[kg, tg, b] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq && qmask[tg, b]
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug, b] || continue
                    s = Cs[lane, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= Tq
            @inbounds partial[tg, b] = arg == Int32(0) ? Float16(0) : Float16(mx)
            @inbounds argmax_out[tg, b] = arg
        end
    end
    return
end

function packed_wmma_kernel!(argmax_out, partial, q, packed, cu, qmask)
    conf = WMMA.Config{16, 16, 16, Float32}
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    b = Int(blockIdx().y)
    As, Bs, Cs = wmma_f16_panels(warp)
    dim = size(q, 1)
    Tq = size(q, 2)
    a = Int(cu[b])
    Td = Int(cu[b + 1]) - a
    mx = 0.0f0
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? packed[kg, a + ug - 1] : Float16(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? q[kg, tg] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq && qmask[tg]
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    s = Cs[lane, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= Tq
            @inbounds partial[tg, b] = arg == Int32(0) ? Float16(0) : Float16(mx)
            @inbounds argmax_out[tg, b] = arg
        end
    end
    return
end

function varlen_wmma_kernel!(argmax_out, partial, Qp, Dp, cu_q, cu_d)
    conf = WMMA.Config{16, 16, 16, Float32}
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    n = Int(blockIdx().y)
    As, Bs, Cs = wmma_f16_panels(warp)
    dim = size(Qp, 1)
    qa = Int(cu_q[n])
    Tq = Int(cu_q[n + 1]) - qa
    da = Int(cu_d[n])
    Td = Int(cu_d[n + 1]) - da
    mx = 0.0f0
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? Dp[kg, da + ug - 1] : Float16(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? Qp[kg, qa + tg - 1] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    s = Cs[lane, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= size(partial, 1)
            @inbounds partial[tg, n] = (tg <= Tq && arg != Int32(0)) ? Float16(mx) : Float16(0)
            @inbounds argmax_out[tg, n] = tg <= Tq ? arg : Int32(0)
        end
    end
    return
end

function candidates_wmma_kernel!(argmax_out, partial, Q, gallery, idx, qmask, dmask, N)
    conf = WMMA.Config{16, 16, 16, Float32}
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    c = Int(blockIdx().y)
    b = Int(blockIdx().z)
    As, Bs, Cs = wmma_f16_panels(warp)
    dim = size(Q, 1)
    Tq = size(Q, 2)
    Td = size(gallery, 2)
    j = Int(idx[c, b])
    in_gal = 1 <= j <= Int(N)
    mx = 0.0f0
    arg = Int32(0)
    Td_loop = in_gal ? Td : 0
    @inbounds for u0 in 0:WMMA_TILE:(Td_loop - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? gallery[kg, ug, j] : Float16(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? Q[kg, tg, b] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq && qmask[tg, b]
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug, j] || continue
                    s = Cs[lane, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= Tq
            @inbounds partial[tg, c, b] = (!in_gal || arg == Int32(0)) ? Float16(0) : Float16(mx)
            @inbounds argmax_out[tg, c, b] = arg
        end
    end
    return
end

function inbatch_wmma_kernel!(argmax_out, partial, Q, D, qmask, dmask)
    conf = WMMA.Config{16, 16, 16, Float32}
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    i = Int(blockIdx().y)
    j = Int(blockIdx().z)
    As, Bs, Cs = wmma_f16_panels(warp)
    dim = size(Q, 1)
    Tq = size(Q, 2)
    Td = size(D, 2)
    mx = 0.0f0
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? D[kg, ug, j] : Float16(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? Q[kg, tg, i] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq && qmask[tg, i]
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug, j] || continue
                    s = Cs[lane, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= Tq
            @inbounds partial[tg, j, i] = arg == Int32(0) ? Float16(0) : Float16(mx)
            @inbounds argmax_out[tg, j, i] = arg
        end
    end
    return
end

# ---- INT8 kernels (s8×s8→s32 tensor cores, deferred dequant) -----------------

function int8_pair_wmma_kernel!(argmax_out, partial, qc, qs, dc, ds, qmask, dmask)
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    As, Bs, Cs = wmma_i8_panels(warp)
    dim = size(qc, 1)
    Tq = size(qc, 2)
    Td = size(dc, 2)
    T = eltype(qs)
    mx = zero(T)
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = ntuple(_ -> Int32(0), Val(8))
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? dc[kg, ug] : Int8(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? qc[kg, tg] : Int8(0)
            end
            sync_threads()
            a_frag = WMMA.llvm_wmma_load_a_col_m16n16k16_shared_stride_s8(pointer(As), Int32(WMMA_TILE))
            b_frag = WMMA.llvm_wmma_load_b_col_m16n16k16_shared_stride_s8(pointer(Bs), Int32(WMMA_TILE))
            c_frag = WMMA.llvm_wmma_mma_col_col_m16n16k16_s8(a_frag, b_frag, c_frag)
            sync_threads()
        end
        WMMA.llvm_wmma_store_d_col_m16n16k16_shared_stride_s32(pointer(Cs), c_frag, Int32(WMMA_TILE))
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq && qmask[tg]
                qt = T(qs[tg])
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug] || continue
                    s = qt * T(ds[ug]) * T(Cs[lane, uloc])
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= Tq
            @inbounds partial[tg] = arg == Int32(0) ? zero(T) : mx
            @inbounds argmax_out[tg] = arg
        end
    end
    return
end

function int8_paired_wmma_kernel!(argmax_out, partial, Qc, Qs, Dc, Ds, qmask, dmask)
    tid, warp, lane = wmma_ids()
    t0 = (Int(blockIdx().x) - 1) * WMMA_QSTRIDE + warp * WMMA_TILE
    b = Int(blockIdx().y)
    As, Bs, Cs = wmma_i8_panels(warp)
    dim = size(Qc, 1)
    Tq = size(Qc, 2)
    Td = size(Dc, 2)
    T = eltype(Qs)
    mx = zero(T)
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = ntuple(_ -> Int32(0), Val(8))
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:WMMA_THREADS:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? Dc[kg, ug, b] : Int8(0)
            end
            for e in lane:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? Qc[kg, tg, b] : Int8(0)
            end
            sync_threads()
            a_frag = WMMA.llvm_wmma_load_a_col_m16n16k16_shared_stride_s8(pointer(As), Int32(WMMA_TILE))
            b_frag = WMMA.llvm_wmma_load_b_col_m16n16k16_shared_stride_s8(pointer(Bs), Int32(WMMA_TILE))
            c_frag = WMMA.llvm_wmma_mma_col_col_m16n16k16_s8(a_frag, b_frag, c_frag)
            sync_threads()
        end
        WMMA.llvm_wmma_store_d_col_m16n16k16_shared_stride_s32(pointer(Cs), c_frag, Int32(WMMA_TILE))
        sync_threads()
        if lane <= WMMA_TILE
            tg = t0 + lane
            if tg <= Tq && qmask[tg, b]
                qt = T(Qs[tg, b])
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug, b] || continue
                    s = qt * T(Ds[ug, b]) * T(Cs[lane, uloc])
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if lane <= WMMA_TILE
        tg = t0 + lane
        if tg <= Tq
            @inbounds partial[tg, b] = arg == Int32(0) ? zero(T) : mx
            @inbounds argmax_out[tg, b] = arg
        end
    end
    return
end

end # module
