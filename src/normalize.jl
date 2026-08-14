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
    length_normalize(scores, inv_token_counts(qmask, T, true))
end

function length_normalize(scores::AbstractMatrix{T},
                          qmask::AbstractMatrix{Bool}) where {T}
    length_normalize(scores, inv_token_counts(qmask, T, true))
end

length_normalize(scores::AbstractVector, inv_n::AbstractVector) = scores .* inv_n
length_normalize(scores::AbstractMatrix, inv_n::AbstractVector) =
    scores .* reshape(inv_n, 1, :)

"""Length-normalize candidate scores; invalid `idxs` keep `neg` (not `neg / n_q`)."""
function length_normalize_candidates(scores::AbstractMatrix{T},
                                     qmask::AbstractMatrix{Bool},
                                     idxs::AbstractMatrix{<:Integer},
                                     n_gallery::Integer,
                                     neg::T) where {T}
    length_normalize_candidates(scores, inv_token_counts(qmask, T, true),
                                idxs, n_gallery, neg)
end

function length_normalize_candidates(scores::AbstractMatrix{T},
                                     inv_n::AbstractVector{T},
                                     idxs::AbstractMatrix{<:Integer},
                                     n_gallery::Integer,
                                     neg::T) where {T}
    out = length_normalize(scores, inv_n)
    idx = indices_on(scores, idxs)
    valid = (idx .>= 1) .& (idx .<= n_gallery)
    ifelse.(valid, out, neg)
end
