# kernels_int8.jl — INT8×INT8 MaxSim with deferred per-token dequant (paper Alg. 4).

@kernel function int8_pair_token_kernel!(argmax_out, partial,
                                         @Const(qc), @Const(qs),
                                         @Const(dc), @Const(ds),
                                         @Const(qmask), @Const(dmask))
    t = @index(Global)
    dim = size(qc, 1)
    Td = size(dc, 2)
    Tq = size(qc, 2)
    T = eltype(qs)
    if t <= Tq
        mx = zero(T)
        arg = Int32(0)
        if @inbounds qmask[t]
            qt = T(qs[t])
            @inbounds for u in 1:Td
                dmask[u] || continue
                acc = Int32(0)
                for k in 1:dim
                    acc += Int32(qc[k, t]) * Int32(dc[k, u])
                end
                s = qt * T(ds[u]) * T(acc)
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
        @inbounds partial[t] = arg == Int32(0) ? zero(T) : mx
        @inbounds argmax_out[t] = arg
    end
end

@kernel unsafe_indices=true function int8_pair_tile_kernel!(argmax_out, partial,
                                                            @Const(qc), @Const(qs),
                                                            @Const(dc), @Const(ds),
                                                            @Const(qmask), @Const(dmask))
    gid = @index(Group, Linear)
    lid = @index(Local, Linear)
    gs = @uniform prod(@groupsize())
    t = (gid - 1) * gs + lid
    T = eltype(qs)
    dim = @uniform size(qc, 1)
    Td = @uniform size(dc, 2)
    Tq = @uniform size(qc, 2)
    valid = t <= Tq && @inbounds(qmask[t])
    Qs = @localmem Int8 (TILE_K + 1, TILE_Q)
    Ds = @localmem Int8 (TILE_K + 1, TILE_D)
    acc = @private Int32 (TILE_D,)
    mx = zero(T)
    arg = Int32(0)
    ncell = TILE_K * TILE_D
    qt = valid ? T(qs[t]) : zero(T)
    @inbounds for u0 in 1:TILE_D:Td
        for uu in 1:TILE_D
            acc[uu] = Int32(0)
        end
        for k0 in 1:TILE_K:dim
            for e0 in 0:gs:(ncell - 1)
                e = e0 + lid
                if e <= ncell
                    kk = ((e - 1) % TILE_K) + 1
                    uu = ((e - 1) ÷ TILE_K) + 1
                    k = k0 + kk - 1
                    u = u0 + uu - 1
                    Ds[kk, uu] = (k <= dim && u <= Td) ? dc[k, u] : Int8(0)
                end
            end
            for kk in 1:TILE_K
                k = k0 + kk - 1
                Qs[kk, lid] = (valid && k <= dim) ? qc[k, t] : Int8(0)
            end
            @synchronize()
            for uu in 1:TILE_D
                s = acc[uu]
                for kk in 1:TILE_K
                    s += Int32(Qs[kk, lid]) * Int32(Ds[kk, uu])
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
                s = qt * T(ds[u]) * T(acc[uu])
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
    end
    if t <= Tq
        @inbounds partial[t] = arg == Int32(0) ? zero(T) : mx
        @inbounds argmax_out[t] = arg
    end
end

@kernel function int8_paired_token_kernel!(argmax_out, partial,
                                           @Const(Qc), @Const(Qs),
                                           @Const(Dc), @Const(Ds),
                                           @Const(qmask), @Const(dmask))
    t, b = @index(Global, NTuple)
    dim = size(Qc, 1)
    Td = size(Dc, 2)
    Tq = size(Qc, 2)
    T = eltype(Qs)
    mx = zero(T)
    arg = Int32(0)
    if t <= Tq && @inbounds(qmask[t, b])
        qt = T(Qs[t, b])
        @inbounds for u in 1:Td
            dmask[u, b] || continue
            acc = Int32(0)
            for k in 1:dim
                acc += Int32(Qc[k, t, b]) * Int32(Dc[k, u, b])
            end
            s = qt * T(Ds[u, b]) * T(acc)
            if arg == Int32(0) || s > mx
                mx = s
                arg = Int32(u)
            end
        end
    end
    @inbounds partial[t, b] = arg == Int32(0) ? zero(T) : mx
    @inbounds argmax_out[t, b] = arg
end

@kernel unsafe_indices=true function int8_paired_tile_kernel!(argmax_out, partial,
                                                              @Const(Qc), @Const(qscales),
                                                              @Const(Dc), @Const(dscales),
                                                              @Const(qmask), @Const(dmask))
    gt, gb = @index(Group, NTuple)
    lt, lb = @index(Local, NTuple)
    tgs = @uniform @groupsize()[1]
    bgs = @uniform @groupsize()[2]
    t = (gt - 1) * tgs + lt
    b = (gb - 1) * bgs + lb
    T = eltype(qscales)
    dim = @uniform size(Qc, 1)
    Td = @uniform size(Dc, 2)
    Tq = @uniform size(Qc, 2)
    B = @uniform size(Qc, 3)
    live = b <= B
    valid = live && t <= Tq && @inbounds(qmask[t, b])
    Qs = @localmem Int8 (TILE_K + 1, TILE_Q)
    Ds = @localmem Int8 (TILE_K + 1, TILE_D)
    acc = @private Int32 (TILE_D,)
    mx = zero(T)
    arg = Int32(0)
    ncell = TILE_K * TILE_D
    gs = @uniform prod(@groupsize())
    lid = @index(Local, Linear)
    qt = valid ? T(qscales[t, b]) : zero(T)
    @inbounds for u0 in 1:TILE_D:Td
        for uu in 1:TILE_D
            acc[uu] = Int32(0)
        end
        for k0 in 1:TILE_K:dim
            for e0 in 0:gs:(ncell - 1)
                e = e0 + lid
                if e <= ncell
                    kk = ((e - 1) % TILE_K) + 1
                    uu = ((e - 1) ÷ TILE_K) + 1
                    k = k0 + kk - 1
                    u = u0 + uu - 1
                    Ds[kk, uu] = (live && k <= dim && u <= Td) ? Dc[k, u, b] : Int8(0)
                end
            end
            for kk in 1:TILE_K
                k = k0 + kk - 1
                Qs[kk, lt] = (valid && k <= dim) ? Qc[k, t, b] : Int8(0)
            end
            @synchronize()
            for uu in 1:TILE_D
                s = acc[uu]
                for kk in 1:TILE_K
                    s += Int32(Qs[kk, lt]) * Int32(Ds[kk, uu])
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
                s = qt * T(dscales[u, b]) * T(acc[uu])
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
    end
    if live && t <= Tq
        @inbounds partial[t, b] = arg == Int32(0) ? zero(T) : mx
        @inbounds argmax_out[t, b] = arg
    end
end
