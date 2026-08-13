# dense.jl — Materializing reference (tests / correctness only).

"""
    maxsim_dense(q, d, qmask, dmask; neg, normalize) -> score

Reference MaxSim that builds the full `(Td, Tq)` similarity matrix.
Used to assert Flash-MaxSim exactness (paper Prop. 1).
"""
function maxsim_dense(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                      qmask::AbstractVector{Bool} = true_mask(q, size(q, 2)),
                      dmask::AbstractVector{Bool} = true_mask(d, size(d, 2));
                      neg::T = T(-1.0f4),
                      normalize::Bool = false) where {T<:AbstractFloat}
    S = Matrix{T}(d' * q)  # (Td, Tq) — intentionally materialized
    Td, Tq = size(S)
    score = zero(T)
    nq = 0
    @inbounds for t in 1:Tq
        qmask[t] || continue
        mx = neg
        for u in 1:Td
            dmask[u] || continue
            mx = max(mx, S[u, t])
        end
        score += mx
        nq += 1
    end
    normalize ? score / T(max(nq, 1)) : score
end

function maxsim_dense(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                      qmask::AbstractMatrix{Bool},
                      dmask::AbstractMatrix{Bool};
                      neg::T = T(-1.0f4),
                      normalize::Bool = false) where {T<:AbstractFloat}
    B = size(Q, 3)
    out = Vector{T}(undef, B)
    @inbounds for b in 1:B
        out[b] = maxsim_dense(view(Q, :, :, b), view(D, :, :, b),
                              view(qmask, :, b), view(dmask, :, b);
                              neg, normalize)
    end
    out
end

function maxsim_dense(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                      qmask::AbstractMatrix{Bool},
                      dmask::AbstractMatrix{Bool},
                      ::InBatch;
                      neg::T = T(-1.0f4),
                      normalize::Bool = false) where {T<:AbstractFloat}
    Bq, Bd = size(Q, 3), size(D, 3)
    S = Matrix{T}(undef, Bd, Bq)
    @inbounds for j in 1:Bd, i in 1:Bq
        S[j, i] = maxsim_dense(view(Q, :, :, i), view(D, :, :, j),
                               view(qmask, :, i), view(dmask, :, j);
                               neg, normalize)
    end
    S
end

function maxsim_dense(Q::AbstractArray{T,3}, gallery::AbstractArray{T,3},
                      idxs::AbstractMatrix{<:Integer},
                      qmask::AbstractMatrix{Bool},
                      dmask::AbstractMatrix{Bool};
                      neg::T = T(-1.0f4),
                      normalize::Bool = false) where {T<:AbstractFloat}
    C, B = size(idxs)
    N = size(gallery, 3)
    S = fill(neg, C, B)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        S[c, b] = maxsim_dense(view(Q, :, :, b), view(gallery, :, :, j),
                               view(qmask, :, b), view(dmask, :, j);
                               neg, normalize)
    end
    S
end
