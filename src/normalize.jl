# normalize.jl — Query-length normalization (pure; Zygote-safe).

"""Per-batch inverse token counts (`1` when `normalize=false`)."""
function inv_token_counts(qmask::AbstractMatrix{Bool}, ::Type{T},
                          normalize::Bool) where {T}
    B = size(qmask, 2)
    normalize || return ones(T, B)
    inv = Vector{T}(undef, B)
    @inbounds for b in 1:B
        inv[b] = one(T) / T(max(count(view(qmask, :, b)), 1))
    end
    inv
end

"""Return length-normalized scores (does not mutate `scores`)."""
function length_normalize(scores::AbstractVector{T},
                          qmask::AbstractMatrix{Bool}) where {T}
    out = similar(scores)
    @inbounds for b in eachindex(scores)
        out[b] = scores[b] / T(max(count(view(qmask, :, b)), 1))
    end
    out
end

function length_normalize(scores::AbstractMatrix{T},
                          qmask::AbstractMatrix{Bool}) where {T}
    out = similar(scores)
    @inbounds for b in 1:size(scores, 2)
        n = T(max(count(view(qmask, :, b)), 1))
        @inbounds for c in 1:size(scores, 1)
            out[c, b] = scores[c, b] / n
        end
    end
    out
end
