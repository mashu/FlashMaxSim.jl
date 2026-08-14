# forward_pair.jl — Fused single-pair MaxSim (paper Alg. 1).
#
# Host backend uses BLAS GEMM over document tiles (`DOC_TILE × Tq` scratch).
# Other backends keep argmax and partials on-device via KernelAbstractions.
# `neg` is unused in the pair scan — it is a candidate-index sentinel only.
#
# Dispatch is on the *backend*, not on the concrete array type: a `view`,
# `Adjoint` or `PermutedDimsArray` over host memory must take the tiled BLAS
# path, not the scalar KA loop.

const DOC_TILE = 64

function require_pair_shapes(q, d, qmask, dmask)
    dim, Tq = size(q)
    Td = size(d, 2)
    size(d, 1) == dim || throw(DimensionMismatch("feature dim"))
    length(qmask) == Tq || throw(DimensionMismatch("qmask"))
    length(dmask) == Td || throw(DimensionMismatch("dmask"))
    dim, Tq, Td
end

"""Allocate host scratch for tiled pair GEMM (`DOC_TILE × Tq`, clipped to `Td`)."""
function pair_host_scratch(::Type{T}, Tq::Int, Td::Int) where {T}
    tile = min(DOC_TILE, max(Td, 1))
    Matrix{T}(undef, tile, Tq), zeros(T, Tq), zeros(Int32, Tq)
end

function pair_forward_host(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                           qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
                           neg::T) where {T<:AbstractFloat}
    _, Tq, Td = require_pair_shapes(q, d, qmask, dmask)
    Stile, mx, argmax_u = pair_host_scratch(T, Tq, Td)
    pair_forward_host!(argmax_u, mx, Stile, q, d, qmask, dmask, neg)
end

"""In-place host pair forward using caller-owned `Stile`, `mx`, `argmax_u` scratch."""
function pair_forward_host!(argmax_u::AbstractVector{Int32}, mx::AbstractVector{T},
                            Stile::AbstractMatrix{T},
                            q::AbstractMatrix{T}, d::AbstractMatrix{T},
                            qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
                            ::T) where {T<:AbstractFloat}
    _, Tq, Td = require_pair_shapes(q, d, qmask, dmask)
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u"))
    length(mx) == Tq || throw(DimensionMismatch("mx"))
    size(Stile, 2) == Tq || throw(DimensionMismatch("Stile"))
    fill!(argmax_u, Int32(0))
    fill!(mx, zero(T))
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
                if argmax_u[t] == Int32(0) || s > mx[t]
                    mx[t] = s
                    argmax_u[t] = Int32(u + uu - 1)
                end
            end
        end
        u = u_end + 1
    end
    score = zero(T)
    @inbounds for t in 1:Tq
        qmask[t] && argmax_u[t] > 0 && (score += mx[t])
    end
    score, argmax_u
end

function pair_forward_ka(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                         qmask::AbstractVector{Bool},
                         dmask::AbstractVector{Bool},
                         ::T) where {T<:AbstractFloat}
    _, Tq, _ = require_pair_shapes(q, d, qmask, dmask)
    backend = array_backend(q)
    argmax_u = zeros_like(q, Int32, Tq)
    partial = zeros_like(q, T, Tq)
    launch!(pair_token_kernel!, backend, Tq, argmax_u, partial, q, d, qmask, dmask)
    sync!(backend)   # host-visible reduction
    sum(partial), argmax_u
end

function pair_forward(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                      qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
                      neg::T) where {T<:AbstractFloat}
    require_colocated(q, d, qmask, dmask)
    pair_forward(array_backend(q), q, d, qmask, dmask, neg)
end

pair_forward(::CPU, q::AbstractMatrix{T}, d::AbstractMatrix{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_host(q, d, qmask, dmask, neg)

pair_forward(::Backend, q::AbstractMatrix{T}, d::AbstractMatrix{T},
             qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool},
             neg::T) where {T<:AbstractFloat} =
    pair_forward_ka(q, d, qmask, dmask, neg)
