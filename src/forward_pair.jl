# forward_pair.jl — Fused single-pair MaxSim (paper Alg. 1).
#
# `Array` uses BLAS GEMM over document tiles (`DOC_TILE × Tq` scratch).
# Other backends keep argmax and partials on-device via KernelAbstractions.

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
    mx = fill(neg, Tq)
    tile = min(DOC_TILE, max(Td, 1))
    Stile = Matrix{T}(undef, tile, Tq)
    u = 1
    while u <= Td
        u_end = min(u + DOC_TILE - 1, Td)
        w = u_end - u + 1
        mul!(view(Stile, 1:w, :), transpose(view(d, :, u:u_end)), q)
        @inbounds for t in 1:Tq
            qmask[t] || continue
            for uu in 1:w
                dmask[u + uu - 1] || continue
                s = Stile[uu, t]
                if s > mx[t]
                    mx[t] = s
                    argmax_u[t] = Int32(u + uu - 1)
                end
            end
        end
        u = u_end + 1
    end
    score = zero(T)
    @inbounds for t in 1:Tq
        # Empty doc (no dmask hit) leaves argmax 0 — do not add `neg`.
        qmask[t] && argmax_u[t] > 0 && (score += mx[t])
    end
    score, argmax_u
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
    argmax_u = zeros_like(q, Int32, Tq)
    partial = zeros_like(q, T, Tq)
    launch!(pair_token_kernel!, backend, Tq, argmax_u, partial, q, d, qmask, dmask, neg)
    sum(partial), argmax_u
end

pair_forward(q::Array{T}, d::Array{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_host(q, d, qmask, dmask, neg)

pair_forward(q::AbstractMatrix{T}, d::AbstractMatrix{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_ka(q, d, qmask, dmask, neg)
