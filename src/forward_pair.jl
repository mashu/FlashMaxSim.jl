# forward_pair.jl — Fused single-pair MaxSim (paper Alg. 1).
#
# Host path tiles document tokens (`DOC_TILE`). KA path: one work-item per
# query token. Contract: `q`, `d`, masks colocated.

const DOC_TILE = 64

function pair_forward_host(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                           qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
                           neg::T) where {T<:AbstractFloat}
    dim, Tq = size(q)
    Td = size(d, 2)
    size(d, 1) == dim || throw(DimensionMismatch("feature dim"))
    length(qmask) == Tq || throw(DimensionMismatch("qmask"))
    length(dmask) == Td || throw(DimensionMismatch("dmask"))
    argmax_u = zeros(Int32, Tq)
    score = zero(T)
    @inbounds for t in 1:Tq
        qmask[t] || continue
        mx = neg
        arg = Int32(0)
        u = 1
        while u <= Td
            u_end = min(u + DOC_TILE - 1, Td)
            for uu in u:u_end
                dmask[uu] || continue
                s = zero(T)
                @simd for k in 1:dim
                    s += q[k, t] * d[k, uu]
                end
                if s > mx
                    mx = s
                    arg = Int32(uu)
                end
            end
            u = u_end + 1
        end
        argmax_u[t] = arg
        score += mx
    end
    score, argmax_u
end

@kernel function pair_token_kernel!(argmax_out, partial,
                                    q, d, qmask, dmask, neg, dim, Td)
    t = @index(Global)
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

function pair_forward_ka(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                         qmask::AbstractVector{Bool},
                         dmask::AbstractVector{Bool},
                         neg::T) where {T<:AbstractFloat}
    require_colocated(q, d, qmask, dmask)
    backend = get_backend(q)
    dim, Tq = size(q)
    Td = size(d, 2)
    length(qmask) == Tq || throw(DimensionMismatch("qmask"))
    length(dmask) == Td || throw(DimensionMismatch("dmask"))
    size(d, 1) == dim || throw(DimensionMismatch("feature dim"))
    argmax_u = KernelAbstractions.zeros(backend, Int32, Tq)
    partial = KernelAbstractions.zeros(backend, T, Tq)
    kernel! = pair_token_kernel!(backend)
    kernel!(argmax_u, partial, q, d, qmask, dmask, neg, dim, Td; ndrange = Tq)
    KernelAbstractions.synchronize(backend)
    score = sum(Array(partial))
    score, Array(argmax_u)
end

pair_forward(q::Array{T}, d::Array{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_host(q, d, qmask, dmask, neg)

pair_forward(q::AbstractMatrix{T}, d::AbstractMatrix{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_ka(q, d, qmask, dmask, neg)
