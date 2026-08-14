# kernels_backward.jl — Sparse ∇Q gather / ∇D scatter on-device (paper Eqs. 2–3).
#
# InvGrid uses a real inverse-grid CSR (paper Alg. 3): atomic count → exclusive
# prefix → atomic fill of `col_idx` → destination-owned accumulate. No
# O(Tq)·O(Bq)·O(N) dest-parallel scans.

# ---- gather (∇Q) -------------------------------------------------------------

@kernel function gather_pair_kernel!(dq, @Const(d), @Const(qmask),
                                     @Const(argmax_u), @Const(δ), Td)
    k, t = @index(Global, NTuple)
    @inbounds if qmask[t]
        u = Int(argmax_u[t])
        if 1 <= u <= Td
            dq[k, t] = δ[1] * d[k, u]
        end
    end
end

@kernel function gather_paired_kernel!(dQ, @Const(D), @Const(qmask), @Const(args),
                                       @Const(Δ), @Const(inv_n), Td)
    k, t, b = @index(Global, NTuple)
    @inbounds if qmask[t, b]
        u = Int(args[t, b])
        if 1 <= u <= Td
            dQ[k, t, b] = (Δ[b] * inv_n[b]) * D[k, u, b]
        end
    end
end

@kernel function gather_inbatch_kernel!(dQ, @Const(D), @Const(qmask), @Const(args),
                                        @Const(Δ), @Const(inv_n), Td, Bd)
    k, t, i = @index(Global, NTuple)
    acc = zero(eltype(dQ))
    @inbounds if qmask[t, i]
        δscale = inv_n[i]
        for j in 1:Bd
            u = Int(args[t, j, i])
            if 1 <= u <= Td
                acc += (Δ[j, i] * δscale) * D[k, u, j]
            end
        end
    end
    @inbounds dQ[k, t, i] = acc
end

@kernel function gather_candidates_kernel!(dQ, @Const(gallery), @Const(idxs),
                                           @Const(qmask), @Const(args),
                                           @Const(Δ), @Const(inv_n), Td, C, N)
    k, t, b = @index(Global, NTuple)
    acc = zero(eltype(dQ))
    @inbounds if qmask[t, b]
        δscale = inv_n[b]
        for c in 1:C
            j = Int(idxs[c, b])
            (1 <= j <= N) || continue
            u = Int(args[t, c, b])
            if 1 <= u <= Td
                acc += (Δ[c, b] * δscale) * gallery[k, u, j]
            end
        end
    end
    @inbounds dQ[k, t, b] = acc
end

# ---- atomic-unified scatter (∇D) --------------------------------------------

@kernel function scatter_pair_atomic_kernel!(dd, @Const(q), @Const(qmask),
                                             @Const(argmax_u), @Const(δ), Td)
    k, t = @index(Global, NTuple)
    @inbounds if qmask[t]
        u = Int(argmax_u[t])
        if 1 <= u <= Td
            @atomic dd[k, u] += δ[1] * q[k, t]
        end
    end
end

@kernel function scatter_paired_atomic_kernel!(dD, @Const(Q), @Const(qmask),
                                               @Const(args), @Const(Δ),
                                               @Const(inv_n), Td)
    k, t, b = @index(Global, NTuple)
    @inbounds if qmask[t, b]
        u = Int(args[t, b])
        if 1 <= u <= Td
            @atomic dD[k, u, b] += (Δ[b] * inv_n[b]) * Q[k, t, b]
        end
    end
end

@kernel function scatter_inbatch_atomic_kernel!(dD, @Const(Q), @Const(qmask),
                                                @Const(args), @Const(Δ),
                                                @Const(inv_n), Td, Bd)
    k, t, i = @index(Global, NTuple)
    @inbounds if qmask[t, i]
        δscale = inv_n[i]
        for j in 1:Bd
            u = Int(args[t, j, i])
            if 1 <= u <= Td
                @atomic dD[k, u, j] += (Δ[j, i] * δscale) * Q[k, t, i]
            end
        end
    end
end

@kernel function scatter_candidates_atomic_kernel!(dG, @Const(Q), @Const(idxs),
                                                   @Const(qmask), @Const(args),
                                                   @Const(Δ), @Const(inv_n),
                                                   Td, C, N)
    k, t, b = @index(Global, NTuple)
    @inbounds if qmask[t, b]
        δscale = inv_n[b]
        for c in 1:C
            j = Int(idxs[c, b])
            (1 <= j <= N) || continue
            u = Int(args[t, c, b])
            if 1 <= u <= Td
                @atomic dG[k, u, j] += (Δ[c, b] * δscale) * Q[k, t, b]
            end
        end
    end
end

# ---- InvGrid CSR build (Alg. 3 counting sort) --------------------------------
# `row_ptr` is length `n_dest+1` with 0-based exclusive ends (same as host).
# `@atomic x += v` returns the **new** value — used as 1-based `col_idx` index.
# Count/fill gate on `qmask` so masked sources are dropped even if argmax ≠ 0
# (matches AtomicUnified and the host `δ_src` path).

@kernel function csr_copy_kernel!(dst, @Const(src))
    I = @index(Global, Cartesian)
    @inbounds dst[I] = src[I]
end

@kernel function csr_count_pair_kernel!(row_ptr, @Const(argmax_u), @Const(qmask), Td)
    t = @index(Global)
    @inbounds if qmask[t]
        u = Int(argmax_u[t])
        if 1 <= u <= Td
            @atomic row_ptr[u + 1] += Int32(1)
        end
    end
end

@kernel function csr_prefix_pair_kernel!(row_ptr, Td)
    _ = @index(Global)
    @inbounds for u in 1:Td
        row_ptr[u + 1] += row_ptr[u]
    end
end

@kernel function csr_fill_pair_kernel!(col_idx, cursor, @Const(argmax_u),
                                       @Const(qmask), Td)
    t = @index(Global)
    @inbounds if qmask[t]
        u = Int(argmax_u[t])
        if 1 <= u <= Td
            pos = @atomic cursor[u] += Int32(1)
            col_idx[pos] = Int32(t)
        end
    end
end

@kernel function csr_count_paired_kernel!(row_ptr, @Const(args), @Const(qmask), Td)
    t, b = @index(Global, NTuple)
    @inbounds if qmask[t, b]
        u = Int(args[t, b])
        if 1 <= u <= Td
            @atomic row_ptr[u + 1, b] += Int32(1)
        end
    end
end

@kernel function csr_prefix_paired_kernel!(row_ptr, Td)
    b = @index(Global)
    @inbounds for u in 1:Td
        row_ptr[u + 1, b] += row_ptr[u, b]
    end
end

@kernel function csr_fill_paired_kernel!(col_idx, cursor, @Const(args),
                                         @Const(qmask), Td, Tq)
    t, b = @index(Global, NTuple)
    @inbounds if qmask[t, b]
        u = Int(args[t, b])
        if 1 <= u <= Td
            pos = @atomic cursor[u, b] += Int32(1)
            col_idx[(b - 1) * Tq + pos] = Int32(t)
        end
    end
end

@kernel function csr_count_inbatch_kernel!(row_ptr, @Const(args), @Const(qmask), Td)
    t, j, i = @index(Global, NTuple)
    @inbounds if qmask[t, i]
        u = Int(args[t, j, i])
        if 1 <= u <= Td
            @atomic row_ptr[u + 1, j] += Int32(1)
        end
    end
end

@kernel function csr_prefix_inbatch_kernel!(row_ptr, Td)
    j = @index(Global)
    @inbounds for u in 1:Td
        row_ptr[u + 1, j] += row_ptr[u, j]
    end
end

@kernel function csr_fill_inbatch_kernel!(col_idx, cursor, @Const(args),
                                          @Const(qmask), Td, Tq, Bq)
    t, j, i = @index(Global, NTuple)
    @inbounds if qmask[t, i]
        u = Int(args[t, j, i])
        if 1 <= u <= Td
            pos = @atomic cursor[u, j] += Int32(1)
            # pack source as 1-based linear (t, i) within the per-doc slab
            col_idx[(j - 1) * (Tq * Bq) + pos] = Int32(t + (i - 1) * Tq)
        end
    end
end

@kernel function csr_count_candidates_kernel!(row_ptr, @Const(idxs), @Const(args),
                                              @Const(qmask), Td, N)
    t, c, b = @index(Global, NTuple)
    @inbounds if qmask[t, b]
        j = Int(idxs[c, b])
        if 1 <= j <= N
            u = Int(args[t, c, b])
            if 1 <= u <= Td
                dest = u + (j - 1) * Td
                @atomic row_ptr[dest + 1] += Int32(1)
            end
        end
    end
end

@kernel function csr_prefix_candidates_kernel!(row_ptr, n_dest)
    _ = @index(Global)
    @inbounds for d in 1:n_dest
        row_ptr[d + 1] += row_ptr[d]
    end
end

@kernel function csr_fill_candidates_kernel!(col_idx, cursor, @Const(idxs),
                                             @Const(args), @Const(qmask),
                                             Td, N, Tq, C)
    t, c, b = @index(Global, NTuple)
    @inbounds if qmask[t, b]
        j = Int(idxs[c, b])
        if 1 <= j <= N
            u = Int(args[t, c, b])
            if 1 <= u <= Td
                dest = u + (j - 1) * Td
                pos = @atomic cursor[dest] += Int32(1)
                # pack (t, c, b) as 1-based linear source id
                col_idx[pos] = Int32(t + (c - 1) * Tq + (b - 1) * Tq * C)
            end
        end
    end
end

# ---- InvGrid CSR accumulate (∇D) --------------------------------------------

@kernel function scatter_pair_csr_kernel!(dd, @Const(q), @Const(row_ptr),
                                          @Const(col_idx), @Const(δ))
    k, u = @index(Global, NTuple)
    acc = zero(eltype(dd))
    @inbounds begin
        a = Int(row_ptr[u]) + 1
        b = Int(row_ptr[u + 1])
        δt = δ[1]
        for p in a:b
            t = Int(col_idx[p])
            acc += δt * q[k, t]
        end
        dd[k, u] = acc
    end
end

@kernel function scatter_paired_csr_kernel!(dD, @Const(Q), @Const(row_ptr),
                                            @Const(col_idx), @Const(Δ),
                                            @Const(inv_n), Tq)
    k, u, b = @index(Global, NTuple)
    acc = zero(eltype(dD))
    @inbounds begin
        base = (b - 1) * Tq
        a = base + Int(row_ptr[u, b]) + 1
        stop = base + Int(row_ptr[u + 1, b])
        δ = Δ[b] * inv_n[b]
        for p in a:stop
            t = Int(col_idx[p])
            acc += δ * Q[k, t, b]
        end
        dD[k, u, b] = acc
    end
end

@kernel function scatter_inbatch_csr_kernel!(dD, @Const(Q), @Const(row_ptr),
                                             @Const(col_idx), @Const(Δ),
                                             @Const(inv_n), Tq, Bq)
    k, u, j = @index(Global, NTuple)
    acc = zero(eltype(dD))
    @inbounds begin
        slab = Tq * Bq
        base = (j - 1) * slab
        a = base + Int(row_ptr[u, j]) + 1
        stop = base + Int(row_ptr[u + 1, j])
        for p in a:stop
            s = Int(col_idx[p])
            t = ((s - 1) % Tq) + 1
            i = ((s - 1) ÷ Tq) + 1
            acc += (Δ[j, i] * inv_n[i]) * Q[k, t, i]
        end
        dD[k, u, j] = acc
    end
end

@kernel function scatter_candidates_csr_kernel!(dG, @Const(Q), @Const(row_ptr),
                                                @Const(col_idx), @Const(Δ),
                                                @Const(inv_n), Td, Tq, C)
    k, u, j = @index(Global, NTuple)
    acc = zero(eltype(dG))
    @inbounds begin
        dest = u + (j - 1) * Td
        a = Int(row_ptr[dest]) + 1
        stop = Int(row_ptr[dest + 1])
        for p in a:stop
            s = Int(col_idx[p])
            t = ((s - 1) % Tq) + 1
            rem = (s - 1) ÷ Tq
            c = (rem % C) + 1
            b = (rem ÷ C) + 1
            acc += (Δ[c, b] * inv_n[b]) * Q[k, t, b]
        end
        dG[k, u, j] = acc
    end
end
