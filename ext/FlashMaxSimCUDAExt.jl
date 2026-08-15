# FlashMaxSimCUDAExt — WMMA tensor-core Float16 pair / paired scans (sm ≥ 7.0).
#
# Overrides `launch_pair_scan!` / `launch_paired_scan!` for `CuArray{Float16}`.
# Float32 stays on the KernelAbstractions SRAM tile path (no silent downcast).
# One warp per 16 query tokens; K / Tq / Td padded to 16 in shared memory.

module FlashMaxSimCUDAExt

using FlashMaxSim
using CUDA
using CUDA: CUDABackend, CuArray, CuDynamicSharedArray, WMMA

const WMMA_TILE = 16
const WMMA_AS_BYTES = WMMA_TILE * WMMA_TILE * sizeof(Float16)
const WMMA_BS_BYTES = WMMA_AS_BYTES
const WMMA_CS_BYTES = WMMA_TILE * WMMA_TILE * sizeof(Float32)
const WMMA_SMEM = WMMA_AS_BYTES + WMMA_BS_BYTES + WMMA_CS_BYTES

function FlashMaxSim.tensor_cores_active(::CUDABackend, ::Type{Float16})
    CUDA.functional() && CUDA.capability(CUDA.device()) >= v"7.0"
end

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

function launch_pair_wmma!(argmax_u, partial, q, d, qmask, dmask)
    Tq = size(q, 2)
    iszero(Tq) && return
    @cuda threads = 32 blocks = cld(Tq, WMMA_TILE) shmem = WMMA_SMEM pair_wmma_kernel!(
        argmax_u, partial, q, d, qmask, dmask)
    nothing
end

function launch_paired_wmma!(args, partial, Q, D, qmask, dmask)
    Tq, B = size(Q, 2), size(Q, 3)
    (iszero(Tq) || iszero(B)) && return
    @cuda threads = 32 blocks = (cld(Tq, WMMA_TILE), B) shmem = WMMA_SMEM paired_wmma_kernel!(
        args, partial, Q, D, qmask, dmask)
    nothing
end

function pair_wmma_kernel!(argmax_out, partial, q, d, qmask, dmask)
    conf = WMMA.Config{16, 16, 16, Float32}
    t0 = (Int(blockIdx().x) - 1) * WMMA_TILE
    tid = Int(threadIdx().x)
    As = CuDynamicSharedArray(Float16, (WMMA_TILE, WMMA_TILE), 0)
    Bs = CuDynamicSharedArray(Float16, (WMMA_TILE, WMMA_TILE), WMMA_AS_BYTES)
    Cs = CuDynamicSharedArray(Float32, (WMMA_TILE, WMMA_TILE), WMMA_AS_BYTES + WMMA_BS_BYTES)
    dim = size(q, 1)
    Tq = size(q, 2)
    Td = size(d, 2)
    mx = 0.0f0
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? q[kg, tg] : Float16(0)
            end
            for e in tid:32:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? d[kg, ug] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if tid <= WMMA_TILE
            tg = t0 + tid
            if tg <= Tq && qmask[tg]
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug] || continue
                    s = Cs[tid, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if tid <= WMMA_TILE
        tg = t0 + tid
        if tg <= Tq
            @inbounds partial[tg] = arg == Int32(0) ? Float16(0) : Float16(mx)
            @inbounds argmax_out[tg] = arg
        end
    end
    return
end

function paired_wmma_kernel!(argmax_out, partial, Q, D, qmask, dmask)
    conf = WMMA.Config{16, 16, 16, Float32}
    t0 = (Int(blockIdx().x) - 1) * WMMA_TILE
    b = Int(blockIdx().y)
    tid = Int(threadIdx().x)
    As = CuDynamicSharedArray(Float16, (WMMA_TILE, WMMA_TILE), 0)
    Bs = CuDynamicSharedArray(Float16, (WMMA_TILE, WMMA_TILE), WMMA_AS_BYTES)
    Cs = CuDynamicSharedArray(Float32, (WMMA_TILE, WMMA_TILE), WMMA_AS_BYTES + WMMA_BS_BYTES)
    dim = size(Q, 1)
    Tq = size(Q, 2)
    Td = size(D, 2)
    mx = 0.0f0
    arg = Int32(0)
    @inbounds for u0 in 0:WMMA_TILE:(Td - 1)
        c_frag = WMMA.fill_c(0.0f0, conf)
        for k0 in 0:WMMA_TILE:(dim - 1)
            for e in tid:32:(WMMA_TILE * WMMA_TILE)
                tloc = ((e - 1) % WMMA_TILE) + 1
                kloc = ((e - 1) ÷ WMMA_TILE) + 1
                tg = t0 + tloc
                kg = k0 + kloc
                As[tloc, kloc] = (tg <= Tq && kg <= dim) ? Q[kg, tg, b] : Float16(0)
            end
            for e in tid:32:(WMMA_TILE * WMMA_TILE)
                kloc = ((e - 1) % WMMA_TILE) + 1
                uloc = ((e - 1) ÷ WMMA_TILE) + 1
                kg = k0 + kloc
                ug = u0 + uloc
                Bs[kloc, uloc] = (kg <= dim && ug <= Td) ? D[kg, ug, b] : Float16(0)
            end
            sync_threads()
            a_frag = WMMA.load_a(pointer(As), WMMA_TILE, WMMA.ColMajor, conf)
            b_frag = WMMA.load_b(pointer(Bs), WMMA_TILE, WMMA.ColMajor, conf)
            c_frag = WMMA.mma(a_frag, b_frag, c_frag, conf)
            sync_threads()
        end
        WMMA.store_d(pointer(Cs), c_frag, WMMA_TILE, WMMA.ColMajor, conf)
        sync_threads()
        if tid <= WMMA_TILE
            tg = t0 + tid
            if tg <= Tq && qmask[tg, b]
                for uloc in 1:WMMA_TILE
                    ug = u0 + uloc
                    ug > Td && continue
                    dmask[ug, b] || continue
                    s = Cs[tid, uloc]
                    if arg == Int32(0) || s > mx
                        mx = s
                        arg = Int32(ug)
                    end
                end
            end
        end
        sync_threads()
    end
    if tid <= WMMA_TILE
        tg = t0 + tid
        if tg <= Tq
            @inbounds partial[tg, b] = arg == Int32(0) ? Float16(0) : Float16(mx)
            @inbounds argmax_out[tg, b] = arg
        end
    end
    return
end

end # module
