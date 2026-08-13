# rrule.jl — Sparse ChainRules (argmax taped; no dense S).
#
# Cotangents stay on the primal score's backend. AbstractZero / Real /
# AbstractArray, then one unthunk hop — never recurse on ZeroTangent.

as_array_cotangent(::Type{T}, Δ::AbstractArray, prototype::AbstractArray) where {T} =
    copyto!(similar(prototype, T), Δ)
as_array_cotangent(::Type{T}, Δ::Real, prototype::AbstractArray) where {T} =
    fill!(similar(prototype, T), T(Δ))
as_array_cotangent(::Type{T}, ::ChainRulesCore.AbstractZero, prototype::AbstractArray) where {T} =
    fill!(similar(prototype, T), zero(T))
function as_array_cotangent(::Type{T}, Δ, prototype::AbstractArray) where {T}
    u = ChainRulesCore.unthunk(Δ)
    u === Δ && throw(ArgumentError("unsupported MaxSim cotangent type $(typeof(Δ))"))
    as_array_cotangent(T, u, prototype)
end

as_scalar_cotangent(::Type{T}, Δ::Real) where {T} = T(Δ)
as_scalar_cotangent(::Type{T}, ::ChainRulesCore.AbstractZero) where {T} = zero(T)
function as_scalar_cotangent(::Type{T}, Δ) where {T}
    u = ChainRulesCore.unthunk(Δ)
    u === Δ && throw(ArgumentError("unsupported MaxSim cotangent type $(typeof(Δ))"))
    as_scalar_cotangent(T, u)
end

function ChainRulesCore.rrule(::typeof(maxsim), cfg::MaxSim{T},
                              q::AbstractMatrix{T}, d::AbstractMatrix{T},
                              qmask::AbstractVector{Bool},
                              dmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    require_colocated(q, d, qmask, dmask)
    score, argmax_u = pair_forward(q, d, qmask, dmask, cfg.neg)
    inv_n = cfg.normalize ? (one(T) / T(query_count(qmask))) : one(T)
    score_out = score * inv_n
    function pullback(Δ)
        δ = as_scalar_cotangent(T, Δ) * inv_n
        dq, dd = pair_pullback(δ, q, d, qmask, argmax_u, cfg.backward)
        (NoTangent(), NoTangent(), dq, dd, NoTangent(), NoTangent())
    end
    score_out, pullback
end

function ChainRulesCore.rrule(::typeof(maxsim), cfg::MaxSim{T},
                              Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                              qmask::AbstractMatrix{Bool},
                              dmask::AbstractMatrix{Bool}) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    scores, args = paired_forward(Q, D, qmask, dmask, cfg.neg)
    inv_n = inv_token_counts(qmask, T, cfg.normalize)
    scores_n = cfg.normalize ? length_normalize(scores, qmask) : scores
    function pullback(Δ)
        Δh = as_array_cotangent(T, Δ, scores_n)
        dQ, dD = paired_pullback(Δh, Q, D, qmask, args, inv_n, cfg.backward)
        (NoTangent(), NoTangent(), dQ, dD, NoTangent(), NoTangent())
    end
    scores_n, pullback
end

function ChainRulesCore.rrule(::typeof(maxsim), cfg::MaxSim{T},
                              Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                              qmask::AbstractMatrix{Bool},
                              dmask::AbstractMatrix{Bool},
                              ::InBatch) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    S, args = inbatch_forward(Q, D, qmask, dmask, cfg.neg)
    inv_n = inv_token_counts(qmask, T, cfg.normalize)
    Sn = cfg.normalize ? length_normalize(S, qmask) : S
    function pullback(Δ)
        Δh = as_array_cotangent(T, Δ, Sn)
        dQ, dD = inbatch_pullback(Δh, Q, D, qmask, args, inv_n, cfg.backward)
        (NoTangent(), NoTangent(), dQ, dD, NoTangent(), NoTangent(), NoTangent())
    end
    Sn, pullback
end

function ChainRulesCore.rrule(::typeof(maxsim), cfg::MaxSim{T},
                              Q::AbstractArray{T,3}, gallery::AbstractArray{T,3},
                              idxs::AbstractMatrix{<:Integer},
                              qmask::AbstractMatrix{Bool},
                              dmask::AbstractMatrix{Bool}) where {T<:AbstractFloat}
    require_colocated(Q, gallery, qmask, dmask)
    S, args = candidates_forward(Q, gallery, idxs, qmask, dmask, cfg.neg)
    inv_n = inv_token_counts(qmask, T, cfg.normalize)
    Sn = cfg.normalize ?
         length_normalize_candidates(S, qmask, idxs, size(gallery, 3), cfg.neg) : S
    function pullback(Δ)
        Δh = as_array_cotangent(T, Δ, Sn)
        dQ, dG = candidates_pullback(Δh, Q, gallery, idxs, qmask, args, inv_n, cfg.backward)
        (NoTangent(), NoTangent(), dQ, dG, NoTangent(), NoTangent(), NoTangent())
    end
    Sn, pullback
end
