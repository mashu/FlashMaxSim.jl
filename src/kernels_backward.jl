# kernels_backward.jl — Sparse ∇Q gather / ∇D scatter on-device (paper Eqs. 2–3).

@kernel function gather_pair_kernel!(dq, @Const(d), @Const(qmask),
                                     @Const(argmax_u), δ, Td)
    k, t = @index(Global, NTuple)
    @inbounds if qmask[t]
        u = Int(argmax_u[t])
        if 1 <= u <= Td
            dq[k, t] = δ * d[k, u]
        end
    end
end

@kernel function scatter_pair_atomic_kernel!(dd, @Const(q), @Const(qmask),
                                             @Const(argmax_u), δ, Td)
    k, t = @index(Global, NTuple)
    @inbounds if qmask[t]
        u = Int(argmax_u[t])
        if 1 <= u <= Td
            @atomic dd[k, u] += δ * q[k, t]
        end
    end
end

@kernel function scatter_pair_invgrid_kernel!(dd, @Const(q), @Const(qmask),
                                              @Const(argmax_u), δ, Tq)
    k, u = @index(Global, NTuple)
    acc = zero(eltype(dd))
    @inbounds for t in 1:Tq
        if qmask[t] && Int(argmax_u[t]) == u
            acc += δ * q[k, t]
        end
    end
    @inbounds dd[k, u] = acc
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

@kernel function scatter_paired_invgrid_kernel!(dD, @Const(Q), @Const(qmask),
                                                @Const(args), @Const(Δ),
                                                @Const(inv_n), Tq)
    k, u, b = @index(Global, NTuple)
    acc = zero(eltype(dD))
    @inbounds for t in 1:Tq
        if qmask[t, b] && Int(args[t, b]) == u
            acc += (Δ[b] * inv_n[b]) * Q[k, t, b]
        end
    end
    @inbounds dD[k, u, b] = acc
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

@kernel function scatter_inbatch_invgrid_kernel!(dD, @Const(Q), @Const(qmask),
                                                 @Const(args), @Const(Δ),
                                                 @Const(inv_n), Tq, Bq)
    k, u, j = @index(Global, NTuple)
    acc = zero(eltype(dD))
    @inbounds for i in 1:Bq
        δ = Δ[j, i] * inv_n[i]
        for t in 1:Tq
            if qmask[t, i] && Int(args[t, j, i]) == u
                acc += δ * Q[k, t, i]
            end
        end
    end
    @inbounds dD[k, u, j] = acc
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

@kernel function scatter_candidates_invgrid_kernel!(dG, @Const(Q), @Const(idxs),
                                                    @Const(qmask), @Const(args),
                                                    @Const(Δ), @Const(inv_n),
                                                    Tq, C, B)
    k, u, j = @index(Global, NTuple)
    acc = zero(eltype(dG))
    @inbounds for b in 1:B, c in 1:C
        Int(idxs[c, b]) == j || continue
        δ = Δ[c, b] * inv_n[b]
        for t in 1:Tq
            if qmask[t, b] && Int(args[t, c, b]) == u
                acc += δ * Q[k, t, b]
            end
        end
    end
    @inbounds dG[k, u, j] = acc
end
