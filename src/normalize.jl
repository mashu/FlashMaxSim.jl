# normalize.jl — Query-length normalization on the feature backend.

"""Per-batch inverse token counts (`1` when `normalize=false`)."""
function inv_token_counts(qmask::AbstractMatrix{Bool}, ::Type{T},
                          normalize::Bool) where {T}
    B = size(qmask, 2)
    normalize || return fill!(similar(qmask, T, B), one(T))
    n = T.(vec(sum(qmask; dims = 1)))
    one(T) ./ max.(n, one(T))
end

"""Return length-normalized scores (does not mutate `scores`)."""
function length_normalize(scores::AbstractVector{T},
                          qmask::AbstractMatrix{Bool}) where {T}
    scores .* inv_token_counts(qmask, T, true)
end

function length_normalize(scores::AbstractMatrix{T},
                          qmask::AbstractMatrix{Bool}) where {T}
    scores .* reshape(inv_token_counts(qmask, T, true), 1, :)
end

"""Length-normalize candidate scores; invalid `idxs` keep `neg` (not `neg / n_q`)."""
function length_normalize_candidates(scores::AbstractMatrix{T},
                                     qmask::AbstractMatrix{Bool},
                                     idxs::AbstractMatrix{<:Integer},
                                     n_gallery::Integer,
                                     neg::T) where {T}
    out = length_normalize(scores, qmask)
    idx = indices_on(scores, idxs)
    valid = (idx .>= 1) .& (idx .<= n_gallery)
    ifelse.(valid, out, neg)
end
