# rrule.jl — Sparse ChainRules (argmax taped; no dense S).

as_array_cotangent(::Type{T}, Δ::AbstractArray, sz) where {T} =
    convert(Array{T}, Array(Δ))
as_array_cotangent(::Type{T}, Δ::Real, sz) where {T} = fill(T(Δ), sz...)
as_array_cotangent(::Type{T}, Δ, sz) where {T} =
    as_array_cotangent(T, ChainRulesCore.unthunk(Δ), sz)

function ChainRulesCore.rrule(::typeof(maxsim), cfg::MaxSim{T},
                              q::AbstractMatrix{T}, d::AbstractMatrix{T},
                              qmask::AbstractVector{Bool},
                              dmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    require_colocated(q, d, qmask, dmask)
    score, argmax_u = pair_forward(q, d, qmask, dmask, cfg.neg)
    inv_n = cfg.normalize ? (one(T) / T(max(count(host_bool(qmask)), 1))) : one(T)
    score_out = score * inv_n
    function pullback(Δ)
        δ = T(ChainRulesCore.unthunk(Δ)) * inv_n
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
    qmh = host_bool(qmask)
    inv_n = inv_token_counts(qmh, T, cfg.normalize)
    scores_n = cfg.normalize ? length_normalize(scores, qmh) : scores
    Sout = match_storage(Q, scores_n)
    function pullback(Δ)
        Δh = as_array_cotangent(T, Δ, size(scores))
        dQ, dD = paired_pullback(vec(Δh), Q, D, qmh, args, inv_n, cfg.backward)
        (NoTangent(), NoTangent(), dQ, dD, NoTangent(), NoTangent())
    end
    Sout, pullback
end

function ChainRulesCore.rrule(::typeof(maxsim), cfg::MaxSim{T},
                              Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                              qmask::AbstractMatrix{Bool},
                              dmask::AbstractMatrix{Bool},
                              ::InBatch) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    S, args = inbatch_forward(Q, D, qmask, dmask, cfg.neg)
    qmh = host_bool(qmask)
    inv_n = inv_token_counts(qmh, T, cfg.normalize)
    Sn = cfg.normalize ? length_normalize(S, qmh) : S
    Sout = match_storage(Q, Sn)
    function pullback(Δ)
        Δh = as_array_cotangent(T, Δ, size(S))
        dQ, dD = inbatch_pullback(Δh, Q, D, qmh, args, inv_n, cfg.backward)
        (NoTangent(), NoTangent(), dQ, dD, NoTangent(), NoTangent(), NoTangent())
    end
    Sout, pullback
end

function ChainRulesCore.rrule(::typeof(maxsim), cfg::MaxSim{T},
                              Q::AbstractArray{T,3}, gallery::AbstractArray{T,3},
                              idxs::AbstractMatrix{<:Integer},
                              qmask::AbstractMatrix{Bool},
                              dmask::AbstractMatrix{Bool}) where {T<:AbstractFloat}
    require_colocated(Q, gallery, qmask, dmask)
    S, args = candidates_forward(Q, gallery, idxs, qmask, dmask, cfg.neg)
    qmh = host_bool(qmask)
    inv_n = inv_token_counts(qmh, T, cfg.normalize)
    Sn = cfg.normalize ? length_normalize(S, qmh) : S
    Sout = match_storage(Q, Sn)
    function pullback(Δ)
        Δh = as_array_cotangent(T, Δ, size(S))
        dQ, dG = candidates_pullback(Δh, Q, gallery, idxs, qmh, args, inv_n, cfg.backward)
        (NoTangent(), NoTangent(), dQ, dG, NoTangent(), NoTangent(), NoTangent())
    end
    Sout, pullback
end
