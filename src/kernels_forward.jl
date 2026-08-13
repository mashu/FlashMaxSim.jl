# kernels_forward.jl — Fused MaxSim token kernels (paper Alg. 1), on-device.

@kernel function pair_token_kernel!(argmax_out, partial,
                                    @Const(q), @Const(d),
                                    @Const(qmask), @Const(dmask), neg)
    t = @index(Global)
    dim = size(q, 1)
    Td = size(d, 2)
    Tq = size(q, 2)
    if t <= Tq
        mx = neg
        arg = Int32(0)
        if @inbounds qmask[t]
            @inbounds for u in 1:Td
                dmask[u] || continue
                s = zero(eltype(q))
                for k in 1:dim
                    s += q[k, t] * d[k, u]
                end
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
        # No valid doc token → contribute 0 (not `neg`) so length-norm ≠ −1e4.
        @inbounds partial[t] = arg == Int32(0) ? zero(eltype(q)) : mx
        @inbounds argmax_out[t] = arg
    end
end

@kernel function paired_token_kernel!(argmax_out, partial,
                                      @Const(Q), @Const(D),
                                      @Const(qmask), @Const(dmask), neg)
    t, b = @index(Global, NTuple)
    dim = size(Q, 1)
    Td = size(D, 2)
    mx = neg
    arg = Int32(0)
    if @inbounds qmask[t, b]
        @inbounds for u in 1:Td
            dmask[u, b] || continue
            s = zero(eltype(Q))
            for k in 1:dim
                s += Q[k, t, b] * D[k, u, b]
            end
            if arg == Int32(0) || s > mx
                mx = s
                arg = Int32(u)
            end
        end
    end
    @inbounds partial[t, b] = arg == Int32(0) ? zero(eltype(Q)) : mx
    @inbounds argmax_out[t, b] = arg
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

@kernel function candidates_token_kernel!(argmax_out, partial,
                                          @Const(Q), @Const(gallery), @Const(idxs),
                                          @Const(qmask), @Const(dmask), neg, N)
    t, c, b = @index(Global, NTuple)
    j = Int(idxs[c, b])
    if 1 <= j <= N
        dim = size(Q, 1)
        Td = size(gallery, 2)
        mx = neg
        arg = Int32(0)
        if @inbounds qmask[t, b]
            @inbounds for u in 1:Td
                dmask[u, j] || continue
                s = zero(eltype(Q))
                for k in 1:dim
                    s += Q[k, t, b] * gallery[k, u, j]
                end
                if arg == Int32(0) || s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
        end
        @inbounds partial[t, c, b] = arg == Int32(0) ? zero(eltype(Q)) : mx
        @inbounds argmax_out[t, c, b] = arg
    else
        @inbounds partial[t, c, b] = zero(eltype(Q))
        @inbounds argmax_out[t, c, b] = Int32(0)
    end
end
