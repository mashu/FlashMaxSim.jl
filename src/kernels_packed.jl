# kernels_packed.jl — cu_seqlens packed / varlen MaxSim (no padding tokens).
#
# `cu` is 1-based CSR: document `b` owns columns `cu[b]:(cu[b+1]-1)`.
# Argmax is local to that slice (1-based within the document).

@kernel function packed_token_kernel!(argmax_out, partial,
                                      @Const(q), @Const(packed), @Const(cu),
                                      @Const(qmask))
    t, b = @index(Global, NTuple)
    dim = size(q, 1)
    Tq = size(q, 2)
    a = Int(cu[b])
    z = Int(cu[b + 1]) - 1
    Td = z - a + 1
    mx = zero(eltype(q))
    arg = Int32(0)
    if t <= Tq && @inbounds(qmask[t]) && Td > 0
        @inbounds for u in 1:Td
            s = zero(eltype(q))
            for k in 1:dim
                s += q[k, t] * packed[k, a + u - 1]
            end
            if arg == Int32(0) || s > mx
                mx = s
                arg = Int32(u)
            end
        end
    end
    @inbounds partial[t, b] = arg == Int32(0) ? zero(eltype(q)) : mx
    @inbounds argmax_out[t, b] = arg
end

@kernel unsafe_indices=true function packed_tile_kernel!(argmax_out, partial,
                                                         @Const(q), @Const(packed),
                                                         @Const(cu), @Const(qmask))
    gt, gb = @index(Group, NTuple)
    lt, lb = @index(Local, NTuple)
    tgs = @uniform @groupsize()[1]
    bgs = @uniform @groupsize()[2]
    t = (gt - 1) * tgs + lt
    b = (gb - 1) * bgs + lb
    T = eltype(q)
    AT = dot_accum(T)
    dim = @uniform size(q, 1)
    Tq = @uniform size(q, 2)
    B = @uniform (length(cu) - 1)
    live = b <= B
    a = 1
    Td = 0
    if live
        a = Int(cu[b])
        Td = Int(cu[b + 1]) - a
    end
    valid = live && t <= Tq && @inbounds(qmask[t]) && Td > 0
    Qs = @localmem T (TILE_K + 1, TILE_Q)
    Ds = @localmem T (TILE_K + 1, TILE_D)
    acc = @private AT (TILE_D,)
    mx = zero(AT)
    arg = Int32(0)
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
                    Ds[kk, uu] = (live && k <= dim && u <= Td) ?
                                 packed[k, a + u - 1] : zero(T)
                end
            end
            for kk in 1:TILE_K
                k = k0 + kk - 1
                Qs[kk, lt] = (valid && k <= dim) ? q[k, t] : zero(T)
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
                s = acc[uu]
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
    end
    if live && t <= Tq
        @inbounds partial[t, b] = arg == Int32(0) ? zero(T) : T(mx)
        @inbounds argmax_out[t, b] = arg
    end
end

@kernel function varlen_token_kernel!(argmax_out, partial,
                                      @Const(Qp), @Const(Dp),
                                      @Const(cu_q), @Const(cu_d))
    t, n = @index(Global, NTuple)
    dim = size(Qp, 1)
    qa = Int(cu_q[n])
    Tq = Int(cu_q[n + 1]) - qa
    da = Int(cu_d[n])
    Td = Int(cu_d[n + 1]) - da
    mx = zero(eltype(Qp))
    arg = Int32(0)
    if t <= Tq && Td > 0
        @inbounds for u in 1:Td
            s = zero(eltype(Qp))
            for k in 1:dim
                s += Qp[k, qa + t - 1] * Dp[k, da + u - 1]
            end
            if arg == Int32(0) || s > mx
                mx = s
                arg = Int32(u)
            end
        end
    end
    @inbounds partial[t, n] = arg == Int32(0) ? zero(eltype(Qp)) : mx
    @inbounds argmax_out[t, n] = arg
end

@kernel unsafe_indices=true function varlen_tile_kernel!(argmax_out, partial,
                                                         @Const(Qp), @Const(Dp),
                                                         @Const(cu_q), @Const(cu_d))
    gt, gn = @index(Group, NTuple)
    lt, ln = @index(Local, NTuple)
    tgs = @uniform @groupsize()[1]
    ngs = @uniform @groupsize()[2]
    t = (gt - 1) * tgs + lt
    n = (gn - 1) * ngs + ln
    T = eltype(Qp)
    AT = dot_accum(T)
    dim = @uniform size(Qp, 1)
    N = @uniform (length(cu_q) - 1)
    live = n <= N
    qa = 1
    da = 1
    Tq = 0
    Td = 0
    if live
        qa = Int(cu_q[n])
        Tq = Int(cu_q[n + 1]) - qa
        da = Int(cu_d[n])
        Td = Int(cu_d[n + 1]) - da
    end
    valid = live && t <= Tq && Td > 0
    Qs = @localmem T (TILE_K + 1, TILE_Q)
    Ds = @localmem T (TILE_K + 1, TILE_D)
    acc = @private AT (TILE_D,)
    mx = zero(AT)
    arg = Int32(0)
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
                    Ds[kk, uu] = (live && k <= dim && u <= Td) ?
                                 Dp[k, da + u - 1] : zero(T)
                end
            end
            for kk in 1:TILE_K
                k = k0 + kk - 1
                Qs[kk, lt] = (valid && k <= dim) ? Qp[k, qa + t - 1] : zero(T)
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
                s = acc[uu]
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
    end
    if live && t <= size(partial, 1)
        @inbounds partial[t, n] = (t <= Tq && arg != Int32(0)) ? T(mx) : zero(T)
        @inbounds argmax_out[t, n] = t <= Tq ? arg : Int32(0)
    end
end

@kernel function unified_packed_atomic_kernel!(dq, dP, @Const(q), @Const(packed),
                                               @Const(cu), @Const(qmask),
                                               @Const(args), @Const(Δ),
                                               @Const(inv_n))
    k, t = @index(Global, NTuple)
    acc = zero(eltype(dq))
    @inbounds if qmask[t]
        qkt = q[k, t]
        inv = inv_n[1]
        B = length(cu) - 1
        for b in 1:B
            a = Int(cu[b])
            Td = Int(cu[b + 1]) - a
            u = Int(args[t, b])
            if 1 <= u <= Td
                δt = Δ[b] * inv
                acc += δt * packed[k, a + u - 1]
                @atomic dP[k, a + u - 1] += δt * qkt
            end
        end
    end
    @inbounds dq[k, t] = acc
end

@kernel function unified_varlen_atomic_kernel!(dQp, dDp, @Const(Qp), @Const(Dp),
                                               @Const(cu_q), @Const(cu_d),
                                               @Const(args), @Const(Δ),
                                               @Const(inv_n))
    k, t, n = @index(Global, NTuple)
    qa = Int(cu_q[n])
    Tq = Int(cu_q[n + 1]) - qa
    da = Int(cu_d[n])
    Td = Int(cu_d[n + 1]) - da
    @inbounds if t <= Tq
        u = Int(args[t, n])
        if 1 <= u <= Td
            δt = Δ[n] * inv_n[n]
            dQp[k, qa + t - 1] = δt * Dp[k, da + u - 1]
            @atomic dDp[k, da + u - 1] += δt * Qp[k, qa + t - 1]
        end
    end
end

@kernel function mul_length1_kernel!(out, @Const(a), @Const(s))
    i = @index(Global)
    @inbounds out[i] = a[i] * s[1]
end

@kernel function varlen_inv_n_kernel!(inv_n, @Const(cu_q))
    n = @index(Global)
    Tq = Int(cu_q[n + 1]) - Int(cu_q[n])
    T = eltype(inv_n)
    @inbounds inv_n[n] = one(T) / T(max(Tq, 1))
end

# ---- InvGrid packed / varlen (CSR dest = packed column) ----------------------

@kernel function gather_packed_kernel!(dq, @Const(packed), @Const(cu),
                                       @Const(qmask), @Const(args),
                                       @Const(Δ), @Const(inv_n))
    k, t = @index(Global, NTuple)
    acc = zero(eltype(dq))
    @inbounds if qmask[t]
        inv = inv_n[1]
        B = length(cu) - 1
        for b in 1:B
            a = Int(cu[b])
            Td = Int(cu[b + 1]) - a
            u = Int(args[t, b])
            if 1 <= u <= Td
                acc += (Δ[b] * inv) * packed[k, a + u - 1]
            end
        end
    end
    @inbounds dq[k, t] = acc
end

@kernel function csr_count_packed_kernel!(row_ptr, @Const(cu), @Const(qmask),
                                          @Const(args), n_dest)
    t, b = @index(Global, NTuple)
    @inbounds if qmask[t]
        a = Int(cu[b])
        Td = Int(cu[b + 1]) - a
        u = Int(args[t, b])
        if 1 <= u <= Td
            dest = a + u - 1
            if 1 <= dest <= n_dest
                @atomic row_ptr[dest + 1] += Int32(1)
            end
        end
    end
end

@kernel function csr_fill_packed_kernel!(col_idx, cursor, @Const(cu),
                                         @Const(qmask), @Const(args),
                                         n_dest, Tq)
    t, b = @index(Global, NTuple)
    @inbounds if qmask[t]
        a = Int(cu[b])
        Td = Int(cu[b + 1]) - a
        u = Int(args[t, b])
        if 1 <= u <= Td
            dest = a + u - 1
            if 1 <= dest <= n_dest
                pos = @atomic cursor[dest] += Int32(1)
                col_idx[pos] = Int32(t + (b - 1) * Tq)
            end
        end
    end
end

@kernel function scatter_packed_csr_kernel!(dP, @Const(q), @Const(row_ptr),
                                            @Const(col_idx), @Const(Δ),
                                            @Const(inv_n), Tq)
    k, dest = @index(Global, NTuple)
    acc = zero(eltype(dP))
    @inbounds begin
        a = Int(row_ptr[dest]) + 1
        stop = Int(row_ptr[dest + 1])
        inv = inv_n[1]
        for p in a:stop
            s = Int(col_idx[p])
            t = ((s - 1) % Tq) + 1
            b = ((s - 1) ÷ Tq) + 1
            acc += (Δ[b] * inv) * q[k, t]
        end
        dP[k, dest] = acc
    end
end

@kernel function gather_varlen_kernel!(dQp, @Const(Dp), @Const(cu_q), @Const(cu_d),
                                       @Const(args), @Const(Δ), @Const(inv_n))
    k, t, n = @index(Global, NTuple)
    qa = Int(cu_q[n])
    Tq = Int(cu_q[n + 1]) - qa
    da = Int(cu_d[n])
    Td = Int(cu_d[n + 1]) - da
    @inbounds if t <= Tq
        u = Int(args[t, n])
        if 1 <= u <= Td
            dQp[k, qa + t - 1] = (Δ[n] * inv_n[n]) * Dp[k, da + u - 1]
        end
    end
end

@kernel function csr_count_varlen_kernel!(row_ptr, @Const(cu_q), @Const(cu_d),
                                          @Const(args), n_dest)
    t, n = @index(Global, NTuple)
    Tq = Int(cu_q[n + 1]) - Int(cu_q[n])
    @inbounds if t <= Tq
        da = Int(cu_d[n])
        Td = Int(cu_d[n + 1]) - da
        u = Int(args[t, n])
        if 1 <= u <= Td
            dest = da + u - 1
            if 1 <= dest <= n_dest
                @atomic row_ptr[dest + 1] += Int32(1)
            end
        end
    end
end

@kernel function csr_fill_varlen_kernel!(col_idx, cursor, @Const(cu_q),
                                         @Const(cu_d), @Const(args),
                                         n_dest, max_q)
    t, n = @index(Global, NTuple)
    Tq = Int(cu_q[n + 1]) - Int(cu_q[n])
    @inbounds if t <= Tq
        da = Int(cu_d[n])
        Td = Int(cu_d[n + 1]) - da
        u = Int(args[t, n])
        if 1 <= u <= Td
            dest = da + u - 1
            if 1 <= dest <= n_dest
                pos = @atomic cursor[dest] += Int32(1)
                col_idx[pos] = Int32(t + (n - 1) * max_q)
            end
        end
    end
end

@kernel function scatter_varlen_csr_kernel!(dDp, @Const(Qp), @Const(cu_q),
                                            @Const(row_ptr), @Const(col_idx),
                                            @Const(Δ), @Const(inv_n), max_q)
    k, dest = @index(Global, NTuple)
    acc = zero(eltype(dDp))
    @inbounds begin
        a = Int(row_ptr[dest]) + 1
        stop = Int(row_ptr[dest + 1])
        for p in a:stop
            s = Int(col_idx[p])
            t = ((s - 1) % max_q) + 1
            n = ((s - 1) ÷ max_q) + 1
            qa = Int(cu_q[n])
            acc += (Δ[n] * inv_n[n]) * Qp[k, qa + t - 1]
        end
        dDp[k, dest] = acc
    end
end
