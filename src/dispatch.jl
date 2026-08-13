# dispatch.jl — Public `maxsim` (one verb; layout via methods) + `MaxSim` callable.

# ---- pair -----------------------------------------------------------------

maxsim(q::AbstractMatrix{T}, d::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat} =
    maxsim(q, d, true_mask(q, size(q, 2)), true_mask(d, size(d, 2)); kwargs...)

function maxsim(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), q, d, qmask, dmask)
end

function maxsim(cfg::MaxSim{T}, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                qmask::AbstractVector{Bool},
                dmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    require_colocated(q, d, qmask, dmask)
    score, _ = pair_forward(q, d, qmask, dmask, cfg.neg)
    cfg.normalize || return score
    score * (one(T) / T(query_count(qmask)))
end

(cfg::MaxSim)(q::AbstractMatrix, d::AbstractMatrix) =
    maxsim(cfg, q, d, true_mask(q, size(q, 2)), true_mask(d, size(d, 2)))
(cfg::MaxSim)(q::AbstractMatrix, d::AbstractMatrix,
              qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool}) =
    maxsim(cfg, q, d, qmask, dmask)

# ---- paired batch ---------------------------------------------------------

maxsim(Q::AbstractArray{T,3}, D::AbstractArray{T,3}; kwargs...) where {T<:AbstractFloat} =
    maxsim(Q, D, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(D, size(D, 2), size(D, 3)); kwargs...)

function maxsim(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), Q, D, qmask, dmask)
end

function maxsim(cfg::MaxSim{T}, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                qmask::AbstractMatrix{Bool},
                dmask::AbstractMatrix{Bool}) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    scores, _ = paired_forward(Q, D, qmask, dmask, cfg.neg)
    cfg.normalize ? length_normalize(scores, qmask) : scores
end

(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, D::AbstractArray{<:Any,3}) =
    maxsim(cfg, Q, D, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(D, size(D, 2), size(D, 3)))
(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, D::AbstractArray{<:Any,3},
              qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool}) =
    maxsim(cfg, Q, D, qmask, dmask)

# ---- in-batch -------------------------------------------------------------

maxsim(Q::AbstractArray{T,3}, D::AbstractArray{T,3}, ::InBatch;
       kwargs...) where {T<:AbstractFloat} =
    maxsim(Q, D, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(D, size(D, 2), size(D, 3)), InBatch(); kwargs...)

function maxsim(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool},
                ::InBatch; neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), Q, D, qmask, dmask, InBatch())
end

function maxsim(cfg::MaxSim{T}, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool},
                ::InBatch) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    S, _ = inbatch_forward(Q, D, qmask, dmask, cfg.neg)
    cfg.normalize ? length_normalize(S, qmask) : S
end

(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, D::AbstractArray{<:Any,3}, ::InBatch) =
    maxsim(cfg, Q, D, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(D, size(D, 2), size(D, 3)), InBatch())
(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, D::AbstractArray{<:Any,3},
              qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool},
              ::InBatch) =
    maxsim(cfg, Q, D, qmask, dmask, InBatch())

# ---- candidates -----------------------------------------------------------

maxsim(Q::AbstractArray{T,3}, gallery::AbstractArray{T,3},
       idxs::AbstractMatrix{<:Integer}; kwargs...) where {T<:AbstractFloat} =
    maxsim(Q, gallery, idxs, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(gallery, size(gallery, 2), size(gallery, 3)); kwargs...)

function maxsim(Q::AbstractArray{T,3}, gallery::AbstractArray{T,3},
                idxs::AbstractMatrix{<:Integer},
                qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), Q, gallery, idxs, qmask, dmask)
end

function maxsim(cfg::MaxSim{T}, Q::AbstractArray{T,3}, gallery::AbstractArray{T,3},
                idxs::AbstractMatrix{<:Integer},
                qmask::AbstractMatrix{Bool},
                dmask::AbstractMatrix{Bool}) where {T<:AbstractFloat}
    require_colocated(Q, gallery, qmask, dmask)
    S, _ = candidates_forward(Q, gallery, idxs, qmask, dmask, cfg.neg)
    cfg.normalize ?
        length_normalize_candidates(S, qmask, idxs, size(gallery, 3), cfg.neg) : S
end

(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, gallery::AbstractArray{<:Any,3},
              idxs::AbstractMatrix{<:Integer}) =
    maxsim(cfg, Q, gallery, idxs, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(gallery, size(gallery, 2), size(gallery, 3)))
(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, gallery::AbstractArray{<:Any,3},
              idxs::AbstractMatrix{<:Integer},
              qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool}) =
    maxsim(cfg, Q, gallery, idxs, qmask, dmask)
