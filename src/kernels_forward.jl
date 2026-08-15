# kernels_forward.jl — Fused MaxSim forward (paper Alg. 1), on-device.
#
# GPU: SRAM-tiled Q/D GEMM + online row-max (no HBM similarity matrix).
# CPU KA: scalar token loop (BLAS host path is separate). First valid document
# token always wins (`arg == 0 || s > mx`). Empty / fully-masked docs contribute 0.

"""Query-token workgroup width / document tile / embedding tile (Alg. 1 / §F.9).

`TILE_K` is the split-d inner tile: SRAM holds `(TILE_K, TILE_Q/D)`, not the
full embedding, so occupancy does not cliff at `d>512`. Float16 dots accumulate
in FP32 (`dot_accum`)."""
const TILE_Q = 32
const TILE_D = 32
const TILE_K = 32

dot_accum(::Type{Float16}) = Float32
dot_accum(::Type{T}) where {T<:AbstractFloat} = T

"""Eltype of a MaxSim **score**. Float16 features accumulate and return Float32
so ranking gaps below a Float16 ulp (≈ 0.03 at score 32) stay visible."""
score_eltype(::Type{T}) where {T<:AbstractFloat} = dot_accum(T)

query_tile_group(::Integer) = TILE_Q
query_tile_group(::Tuple{Int}) = (TILE_Q,)
query_tile_group(::Tuple{Int,Int}) = (TILE_Q, 1)
query_tile_group(::Tuple{Int,Int,Int}) = (TILE_Q, 1, 1)

# ---- CPU KA scalar scan ------------------------------------------------------

@kernel function pair_token_kernel!(argmax_out, partial,
                                    @Const(q), @Const(d),
                                    @Const(qmask), @Const(dmask))
    t = @index(Global)
    dim = size(q, 1)
    Td = size(d, 2)
    Tq = size(q, 2)
    AT = dot_accum(eltype(q))
    PT = eltype(partial)
    if t <= Tq
        mx = zero(AT)
        arg = Int32(0)
        if @inbounds qmask[t]
            @inbounds for u in 1:Td
                dmask[u] || continue
                s = zero(AT)
                for k in 1:dim
                    s += convert(AT, q[k, t]) * convert(AT, d[k, u])
                end
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
        @inbounds partial[t] = arg == Int32(0) ? zero(PT) : PT(mx)
        @inbounds argmax_out[t] = arg
    end
end

@kernel function paired_token_kernel!(argmax_out, partial,
                                      @Const(Q), @Const(D),
                                      @Const(qmask), @Const(dmask))
    t, b = @index(Global, NTuple)
    dim = size(Q, 1)
    Td = size(D, 2)
    AT = dot_accum(eltype(Q))
    PT = eltype(partial)
    mx = zero(AT)
    arg = Int32(0)
    if @inbounds qmask[t, b]
        @inbounds for u in 1:Td
            dmask[u, b] || continue
            s = zero(AT)
            for k in 1:dim
                s += convert(AT, Q[k, t, b]) * convert(AT, D[k, u, b])
            end
            if arg == Int32(0) || s > mx
                mx = s
                arg = Int32(u)
            end
        end
    end
    @inbounds partial[t, b] = arg == Int32(0) ? zero(PT) : PT(mx)
    @inbounds argmax_out[t, b] = arg
end

@kernel function candidates_token_kernel!(argmax_out, partial,
                                          @Const(Q), @Const(gallery), @Const(idxs),
                                          @Const(qmask), @Const(dmask), N)
    t, c, b = @index(Global, NTuple)
    j = Int(idxs[c, b])
    PT = eltype(partial)
    if 1 <= j <= N
        dim = size(Q, 1)
        Td = size(gallery, 2)
        AT = dot_accum(eltype(Q))
        mx = zero(AT)
        arg = Int32(0)
        if @inbounds qmask[t, b]
            @inbounds for u in 1:Td
                dmask[u, j] || continue
                s = zero(AT)
                for k in 1:dim
                    s += convert(AT, Q[k, t, b]) * convert(AT, gallery[k, u, j])
                end
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
        @inbounds partial[t, c, b] = arg == Int32(0) ? zero(PT) : PT(mx)
        @inbounds argmax_out[t, c, b] = arg
    else
        @inbounds partial[t, c, b] = zero(PT)
        @inbounds argmax_out[t, c, b] = Int32(0)
    end
end

# ---- GPU SRAM-tiled scan (paper Alg. 1 / §F.9 split-d) -----------------------
# One workgroup = TILE_Q query tokens of one (query, document) pair. Document
# tiles stream through SRAM; the embedding dim is split on TILE_K so the tile
# fits on-chip at any d. Padded lanes stay live through @synchronize.

@kernel unsafe_indices=true function pair_tile_kernel!(argmax_out, partial,
                                                       @Const(q), @Const(d),
                                                       @Const(qmask), @Const(dmask))
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)
    gs = @uniform prod(@groupsize())
    t = (gid - 1) * gs + lid
    AT = dot_accum(eltype(q))
    PT = eltype(partial)
    dim = @uniform size(q, 1)
    Td = @uniform size(d, 2)
    Tq = @uniform size(q, 2)
    valid = t <= Tq && @inbounds(qmask[t])
    Qs = @localmem eltype(q) (TILE_K + 1, TILE_Q)
    Ds = @localmem eltype(q) (TILE_K + 1, TILE_D)
    acc = @private dot_accum(eltype(q)) (TILE_D,)
    mx_s = @private dot_accum(eltype(q)) (1,)
    arg_s = @private Int32 (1,)
    mx_s[1] = zero(AT)
    arg_s[1] = Int32(0)
    ncell = TILE_K * TILE_D
    @inbounds for u0 in 1:TILE_D:Td
        for uu in 1:TILE_D
            acc[uu] = zero(AT)
        end
        for k0 in 1:TILE_K:dim
            for e0 in 0:gs:(ncell - 1)
                e = e0 + lid
                if e <= ncell
                    kk = ((e - 1) % TILE_K) + 1
                    uu = ((e - 1) ÷ TILE_K) + 1
                    k = k0 + kk - 1
                    u = u0 + uu - 1
                    Ds[kk, uu] = (k <= dim && u <= Td) ? d[k, u] : zero(eltype(q))
                end
            end
            for kk in 1:TILE_K
                k = k0 + kk - 1
                Qs[kk, lid] = (valid && k <= dim) ? q[k, t] : zero(eltype(q))
            end
            @synchronize()
            for uu in 1:TILE_D
                s = acc[uu]
                for kk in 1:TILE_K
                    s += convert(AT, Qs[kk, lid]) * convert(AT, Ds[kk, uu])
                end
                acc[uu] = s
            end
            @synchronize()
        end
        if valid
            for uu in 1:TILE_D
                u = u0 + uu - 1
                u > Td && continue
                dmask[u] || continue
                s = acc[uu]
                if arg_s[1] == Int32(0) || s > mx_s[1]
                    mx_s[1] = s
                    arg_s[1] = Int32(u)
                end
            end
        end
    end
    if t <= Tq
        arg = arg_s[1]
        @inbounds partial[t] = arg == Int32(0) ? zero(PT) : PT(mx_s[1])
        @inbounds argmax_out[t] = arg
    end
end

@kernel unsafe_indices=true function paired_tile_kernel!(argmax_out, partial,
                                                         @Const(Q), @Const(D),
                                                         @Const(qmask), @Const(dmask))
    gt, gb = @index(Group, NTuple)
    lt, lb = @index(Local, NTuple)
    tgs = @uniform @groupsize()[1]
    bgs = @uniform @groupsize()[2]
    t = (gt - 1) * tgs + lt
    b = (gb - 1) * bgs + lb
    AT = dot_accum(eltype(Q))
    PT = eltype(partial)
    dim = @uniform size(Q, 1)
    Td = @uniform size(D, 2)
    Tq = @uniform size(Q, 2)
    B = @uniform size(Q, 3)
    live = b <= B
    valid = live && t <= Tq && @inbounds(qmask[t, b])
    Qs = @localmem eltype(Q) (TILE_K + 1, TILE_Q)
    Ds = @localmem eltype(Q) (TILE_K + 1, TILE_D)
    acc = @private dot_accum(eltype(Q)) (TILE_D,)
    mx_s = @private dot_accum(eltype(Q)) (1,)
    arg_s = @private Int32 (1,)
    mx_s[1] = zero(AT)
    arg_s[1] = Int32(0)
    ncell = TILE_K * TILE_D
    gs = @uniform prod(@groupsize())
    lid = @index(Local, Linear)
    @inbounds for u0 in 1:TILE_D:Td
        for uu in 1:TILE_D
            acc[uu] = zero(AT)
        end
        for k0 in 1:TILE_K:dim
            for e0 in 0:gs:(ncell - 1)
                e = e0 + lid
                if e <= ncell
                    kk = ((e - 1) % TILE_K) + 1
                    uu = ((e - 1) ÷ TILE_K) + 1
                    k = k0 + kk - 1
                    u = u0 + uu - 1
                    Ds[kk, uu] = (live && k <= dim && u <= Td) ? D[k, u, b] : zero(eltype(Q))
                end
            end
            for kk in 1:TILE_K
                k = k0 + kk - 1
                Qs[kk, lt] = (valid && k <= dim) ? Q[k, t, b] : zero(eltype(Q))
            end
            @synchronize()
            for uu in 1:TILE_D
                s = acc[uu]
                for kk in 1:TILE_K
                    s += convert(AT, Qs[kk, lt]) * convert(AT, Ds[kk, uu])
                end
                acc[uu] = s
            end
            @synchronize()
        end
        if valid
            for uu in 1:TILE_D
                u = u0 + uu - 1
                u > Td && continue
                dmask[u, b] || continue
                s = acc[uu]
                if arg_s[1] == Int32(0) || s > mx_s[1]
                    mx_s[1] = s
                    arg_s[1] = Int32(u)
                end
            end
        end
    end
    if live && t <= Tq
        arg = arg_s[1]
        @inbounds partial[t, b] = arg == Int32(0) ? zero(PT) : PT(mx_s[1])
        @inbounds argmax_out[t, b] = arg
    end
end

@kernel unsafe_indices=true function candidates_tile_kernel!(argmax_out, partial,
                                                             @Const(Q), @Const(gallery),
                                                             @Const(idxs),
                                                             @Const(qmask), @Const(dmask), N)
    gt, gc, gb = @index(Group, NTuple)
    lt, lc, lb = @index(Local, NTuple)
    tgs = @uniform @groupsize()[1]
    cgs = @uniform @groupsize()[2]
    bgs = @uniform @groupsize()[3]
    t = (gt - 1) * tgs + lt
    c = (gc - 1) * cgs + lc
    b = (gb - 1) * bgs + lb
    AT = dot_accum(eltype(Q))
    PT = eltype(partial)
    dim = @uniform size(Q, 1)
    Td = @uniform size(gallery, 2)
    Tq = @uniform size(Q, 2)
    C = @uniform size(idxs, 1)
    B = @uniform size(Q, 3)
    live = c <= C && b <= B
    j = 0
    if live
        j = Int(idxs[c, b])
    end
    in_gal = live && 1 <= j <= N
    valid = in_gal && t <= Tq && @inbounds(qmask[t, b])
    Qs = @localmem eltype(Q) (TILE_K + 1, TILE_Q)
    Ds = @localmem eltype(Q) (TILE_K + 1, TILE_D)
    acc = @private dot_accum(eltype(Q)) (TILE_D,)
    mx_s = @private dot_accum(eltype(Q)) (1,)
    arg_s = @private Int32 (1,)
    mx_s[1] = zero(AT)
    arg_s[1] = Int32(0)
    ncell = TILE_K * TILE_D
    gs = @uniform prod(@groupsize())
    lid = @index(Local, Linear)
    if in_gal
        @inbounds for u0 in 1:TILE_D:Td
            for uu in 1:TILE_D
                acc[uu] = zero(AT)
            end
            for k0 in 1:TILE_K:dim
                for e0 in 0:gs:(ncell - 1)
                    e = e0 + lid
                    if e <= ncell
                        kk = ((e - 1) % TILE_K) + 1
                        uu = ((e - 1) ÷ TILE_K) + 1
                        k = k0 + kk - 1
                        u = u0 + uu - 1
                        Ds[kk, uu] = (k <= dim && u <= Td) ? gallery[k, u, j] : zero(eltype(Q))
                    end
                end
                for kk in 1:TILE_K
                    k = k0 + kk - 1
                    Qs[kk, lt] = (valid && k <= dim) ? Q[k, t, b] : zero(eltype(Q))
                end
                @synchronize()
                for uu in 1:TILE_D
                    s = acc[uu]
                    for kk in 1:TILE_K
                        s += convert(AT, Qs[kk, lt]) * convert(AT, Ds[kk, uu])
                    end
                    acc[uu] = s
                end
                @synchronize()
            end
            if valid
                for uu in 1:TILE_D
                    u = u0 + uu - 1
                    u > Td && continue
                    dmask[u, j] || continue
                    s = acc[uu]
                    if arg_s[1] == Int32(0) || s > mx_s[1]
                        mx_s[1] = s
                        arg_s[1] = Int32(u)
                    end
                end
            end
        end
    end
    if live && t <= Tq
        arg = arg_s[1]
        @inbounds partial[t, c, b] = (!in_gal || arg == Int32(0)) ? zero(PT) : PT(mx_s[1])
        @inbounds argmax_out[t, c, b] = arg
    end
end

"""Sum `x` into `out[1]` (serial; `Tq` is small). Stays on-device."""
@kernel function reduce_sum1_kernel!(out, @Const(x))
    _ = @index(Global)
    s = zero(eltype(out))
    @inbounds for i in 1:length(x)
        s += x[i]
    end
    @inbounds out[1] = s
end

"""`out[1] = max(count(mask), 1)` as `eltype(out)` — device query-length scale."""
@kernel function count_true1_kernel!(out, @Const(mask))
    _ = @index(Global)
    c = zero(eltype(out))
    @inbounds for i in 1:length(mask)
        if mask[i]
            c += one(eltype(out))
        end
    end
    @inbounds out[1] = max(c, one(eltype(out)))
end

"""Accumulate MaxSim for one doc-chunk from a precomputed `(Td, C, Tq, Bq)` tile."""
@kernel function inbatch_accumulate_kernel!(S, args, @Const(M4),
                                            @Const(qmask), @Const(dm), j0)
    t, c, i = @index(Global, NTuple)
    Td = size(M4, 1)
    mx = zero(eltype(M4))
    arg = Int32(0)
    if @inbounds qmask[t, i]
        @inbounds for u in 1:Td
            dm[u, c] || continue
            s = M4[u, c, t, i]
            if arg == Int32(0) || s > mx
                mx = s
                arg = Int32(u)
            end
        end
        if arg != Int32(0)
            j = Int(j0) + c - 1
            @inbounds args[t, j, i] = arg
            @atomic S[j, i] += mx
        end
    end
end
