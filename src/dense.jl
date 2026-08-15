# dense.jl — Materializing reference (tests / correctness only).

"""Accumulate type for the dense oracle. Float16 matmul uses FP32 so the
oracle is at least as accurate as the fused path (`dot_accum`)."""
dense_accum(::Type{Float16}) = Float32
dense_accum(::Type{T}) where {T<:AbstractFloat} = T

dense_similarity(q::AbstractMatrix{A}, d::AbstractMatrix{A}, ::Type{A}) where {A} = d' * q
dense_similarity(q::AbstractMatrix, d::AbstractMatrix, ::Type{A}) where {A} =
    convert(Matrix{A}, d)' * convert(Matrix{A}, q)

"""
    maxsim_dense(q, d, qmask, dmask; neg, normalize, accum) -> score

Reference MaxSim that builds the full `(Td, Tq)` similarity matrix.
Used to assert Flash-MaxSim exactness (paper Prop. 1).

Takes the true max over valid document tokens. `neg` is only the fill
value for invalid candidate indices (see the index-matrix method); it is
not a similarity clamp and is ignored by the pair / paired / in-batch
methods.

`accum` is the GEMM / reduction eltype (default [`dense_accum`](@ref)`(T)`).
Pass `accum = Float64` in tests that need a tight Float16 tolerance.
"""
function maxsim_dense(q::AbstractMatrix{T}, d::AbstractMatrix{T},
                      qmask::AbstractVector{Bool} = true_mask(q, size(q, 2)),
                      dmask::AbstractVector{Bool} = true_mask(d, size(d, 2));
                      neg::T = T(-1.0f4),
                      normalize::Bool = false,
                      accum::Type = dense_accum(T)) where {T<:AbstractFloat}
    qh, dh = Array(q), Array(d)
    qmh, dmh = Array(qmask), Array(dmask)
    S = dense_similarity(qh, dh, accum)   # (Td, Tq)
    Td, Tq = size(S)
    score = zero(accum)
    nq = 0
    @inbounds for t in 1:Tq
        qmh[t] || continue
        mx = zero(accum)
        hit = false
        for u in 1:Td
            dmh[u] || continue
            s = S[u, t]
            if !hit || s > mx
                mx = s
                hit = true
            end
        end
        hit || continue
        score += mx
        nq += 1
    end
    out = normalize ? score / accum(max(nq, 1)) : score
    out
end

function maxsim_dense(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                      qmask::AbstractMatrix{Bool},
                      dmask::AbstractMatrix{Bool};
                      neg::T = T(-1.0f4),
                      normalize::Bool = false,
                      accum::Type = dense_accum(T)) where {T<:AbstractFloat}
    B = size(Q, 3)
    out = Vector{accum}(undef, B)
    @inbounds for b in 1:B
        out[b] = maxsim_dense(view(Q, :, :, b), view(D, :, :, b),
                              view(qmask, :, b), view(dmask, :, b);
                              neg, normalize, accum)
    end
    out
end

function maxsim_dense(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                      qmask::AbstractMatrix{Bool},
                      dmask::AbstractMatrix{Bool},
                      ::InBatch;
                      neg::T = T(-1.0f4),
                      normalize::Bool = false,
                      accum::Type = dense_accum(T)) where {T<:AbstractFloat}
    Bq, Bd = size(Q, 3), size(D, 3)
    S = Matrix{accum}(undef, Bd, Bq)
    @inbounds for j in 1:Bd, i in 1:Bq
        S[j, i] = maxsim_dense(view(Q, :, :, i), view(D, :, :, j),
                               view(qmask, :, i), view(dmask, :, j);
                               neg, normalize, accum)
    end
    S
end

function maxsim_dense(Q::AbstractArray{T,3}, gallery::AbstractArray{T,3},
                      idxs::AbstractMatrix{<:Integer},
                      qmask::AbstractMatrix{Bool},
                      dmask::AbstractMatrix{Bool};
                      neg::T = T(-1.0f4),
                      normalize::Bool = false,
                      accum::Type = dense_accum(T)) where {T<:AbstractFloat}
    C, B = size(idxs)
    N = size(gallery, 3)
    S = fill(convert_scores(accum, neg), C, B)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        S[c, b] = maxsim_dense(view(Q, :, :, b), view(gallery, :, :, j),
                               view(qmask, :, b), view(dmask, :, j);
                               neg, normalize, accum)
    end
    S
end
