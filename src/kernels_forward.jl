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
                if s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
            @inbounds partial[t] = mx
        else
            @inbounds partial[t] = zero(eltype(q))
        end
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
            if s > mx
                mx = s
                arg = Int32(u)
            end
        end
        @inbounds partial[t, b] = mx
    else
        @inbounds partial[t, b] = zero(eltype(Q))
    end
    @inbounds argmax_out[t, b] = arg
end

@kernel function inbatch_token_kernel!(argmax_out, partial,
                                       @Const(Q), @Const(D),
                                       @Const(qmask), @Const(dmask), neg)
    t, j, i = @index(Global, NTuple)
    dim = size(Q, 1)
    Td = size(D, 2)
    mx = neg
    arg = Int32(0)
    if @inbounds qmask[t, i]
        @inbounds for u in 1:Td
            dmask[u, j] || continue
            s = zero(eltype(Q))
            for k in 1:dim
                s += Q[k, t, i] * D[k, u, j]
            end
            if s > mx
                mx = s
                arg = Int32(u)
            end
        end
        @inbounds partial[t, j, i] = mx
    else
        @inbounds partial[t, j, i] = zero(eltype(Q))
    end
    @inbounds argmax_out[t, j, i] = arg
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
                if s > mx
                    mx = s
                    arg = Int32(u)
                end
            end
            @inbounds partial[t, c, b] = mx
        else
            @inbounds partial[t, c, b] = zero(eltype(Q))
        end
        @inbounds argmax_out[t, c, b] = arg
    else
        @inbounds partial[t, c, b] = zero(eltype(Q))
        @inbounds argmax_out[t, c, b] = Int32(0)
    end
end
