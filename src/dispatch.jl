# dispatch.jl — Public `maxsim` (one verb; layout via methods) + `MaxSim` callable.

# ---- pair -----------------------------------------------------------------

"""
    maxsim(q, d; neg, normalize, backward) -> score
    maxsim(q, d, qmask, dmask; ...)
    maxsim(cfg::MaxSim, q, d, qmask, dmask)

Single-pair ColBERT MaxSim:

```text
MaxSim(q, d) = Σ_t  max_u  ⟨q_t, d_u⟩
```

over tokens where `qmask[t]` and `dmask[u]` are true. Fuses the per-token
argmax (paper Algorithm 1) without storing a dense `Tq × Td` similarity
matrix (`Array` uses BLAS-tiled GEMM; other KA backends use a token kernel).

# Keyword arguments

- `neg`: fill value for **invalid candidate indices** in the candidate
  layout. Ignored here — not a similarity clamp.
- `normalize`: divide by the number of valid query tokens.
- `backward`: [`AtomicUnified`](@ref) or [`InvGrid`](@ref) ∇D strategy.
"""
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
    score, _ = pair_forward(q, d, qmask, dmask, cfg.neg)
    pair_finalize(score, qmask, cfg.normalize)[1]
end

(cfg::MaxSim)(q::AbstractMatrix, d::AbstractMatrix) =
    maxsim(cfg, q, d, true_mask(q, size(q, 2)), true_mask(d, size(d, 2)))
(cfg::MaxSim)(q::AbstractMatrix, d::AbstractMatrix,
              qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool}) =
    maxsim(cfg, q, d, qmask, dmask)

# ---- paired batch ---------------------------------------------------------

"""
    maxsim(Q, D; ...) -> Vector
    maxsim(Q, D, qmask, dmask; ...)
    maxsim(cfg::MaxSim, Q, D, qmask, dmask)

Paired-batch MaxSim. `Q` and `D` are `(dim, T, B)`; returns a length-`B`
vector with `out[b] = MaxSim(Q[:,:,b], D[:,:,b])`. Same fused argmax as the
pair layout — no dense similarity tensor.
"""
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

"""
    maxsim(Q, D, InBatch(); ...) -> Matrix
    maxsim(Q, D, qmask, dmask, InBatch(); ...)
    maxsim(cfg::MaxSim, Q, D, qmask, dmask, InBatch())

In-batch contrastive MaxSim. Returns `(Bd, Bq)` with
`S[j,i] = MaxSim(Q[:,:,i], D[:,:,j])`.

Unlike pair / paired / candidate layouts, this path computes tiled GEMM
chunks of `D'Q` (about 64 MiB per tile, scaled by `sizeof(eltype)`) then a
light argmax accumulate — it does materialize similarity tiles, not a fused
`Tq×Td` kernel.
"""
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

"""
    maxsim(Q, gallery, idxs; ...) -> Matrix
    maxsim(Q, gallery, idxs, qmask, dmask; ...)
    maxsim(cfg::MaxSim, Q, gallery, idxs, qmask, dmask)

Candidate-set MaxSim. `idxs` is `(C, B)` into `gallery` (`N` docs).
Returns `(C, B)` with `S[c,b] = MaxSim(Q[:,:,b], gallery[:,:,idxs[c,b]])`.
Indices outside `1:N` are filled with `neg` and contribute no gradient.
"""
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

# ---- packed cu_seqlens (one query vs B ragged docs) --------------------------

"""
    maxsim(q, packed, cu; ...) -> Vector
    maxsim(q, packed, cu, qmask; ...)
    maxsim(cfg::MaxSim, q, packed, cu, qmask)

Packed-document MaxSim. `packed` is `(dim, sum Ld_i)` and `cu` is 1-based CSR
of length `B+1` (`pack_docs`). Returns `(B,)` with
`out[b] = MaxSim(q, packed[:, cu[b]:(cu[b+1]-1)])`. Python 0-based
`cu_seqlens` converts with `cu .+ 1`. GPU `InvGrid` is not supported.
"""
maxsim(q::AbstractMatrix{T}, packed::AbstractMatrix{T},
       cu::AbstractVector{<:Integer}; kwargs...) where {T<:AbstractFloat} =
    maxsim(q, packed, cu, true_mask(q, size(q, 2)); kwargs...)

function maxsim(q::AbstractMatrix{T}, packed::AbstractMatrix{T},
                cu::AbstractVector{<:Integer}, qmask::AbstractVector{Bool};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), q, packed, cu, qmask)
end

function maxsim(cfg::MaxSim{T}, q::AbstractMatrix{T}, packed::AbstractMatrix{T},
                cu::AbstractVector{<:Integer},
                qmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    scores, _ = packed_forward(q, packed, cu, qmask, cfg.neg)
    scores_n, _ = packed_finalize(scores, qmask, cfg.normalize)
    scores_n
end

(cfg::MaxSim)(q::AbstractMatrix, packed::AbstractMatrix, cu::AbstractVector{<:Integer}) =
    maxsim(cfg, q, packed, cu, true_mask(q, size(q, 2)))
(cfg::MaxSim)(q::AbstractMatrix, packed::AbstractMatrix, cu::AbstractVector{<:Integer},
              qmask::AbstractVector{Bool}) =
    maxsim(cfg, q, packed, cu, qmask)

# ---- varlen pairs (ragged query and document per pair) -----------------------

"""
    maxsim(Qp, Dp, cu_q, cu_d; ...) -> Vector
    maxsim(cfg::MaxSim, Qp, Dp, cu_q, cu_d)

Variable-length paired MaxSim (`pack_pairs`). `cu_q` / `cu_d` are 1-based CSR.
Returns `(N,)` with `out[n] = MaxSim(Qp[:, cu_q[n]:(cu_q[n+1]-1)],
Dp[:, cu_d[n]:(cu_d[n+1]-1)])`. GPU `InvGrid` is not supported.
"""
function maxsim(Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
                cu_q::AbstractVector{<:Integer}, cu_d::AbstractVector{<:Integer};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), Qp, Dp, cu_q, cu_d)
end

function maxsim(cfg::MaxSim{T}, Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
                cu_q::AbstractVector{<:Integer},
                cu_d::AbstractVector{<:Integer}) where {T<:AbstractFloat}
    scores, _ = varlen_forward(Qp, Dp, cu_q, cu_d, cfg.neg)
    scores_n, _ = varlen_finalize(scores, cu_q, cfg.normalize)
    scores_n
end

(cfg::MaxSim)(Qp::AbstractMatrix, Dp::AbstractMatrix,
              cu_q::AbstractVector{<:Integer}, cu_d::AbstractVector{<:Integer}) =
    maxsim(cfg, Qp, Dp, cu_q, cu_d)

# ---- INT8 index (forward-only deferred dequant) ------------------------------

"""
    maxsim(q, d::Int8Index; ...)
    maxsim(Q, D::Int8Index; ...)

INT8 MaxSim with on-the-fly query quantization and deferred dequant
(`⟨q,d⟩ = qscale * dscale * ⟨qcode, dcode⟩_Int32`). Forward-only.
"""
maxsim(q::AbstractMatrix{T}, d::Int8Index{<:AbstractMatrix{Int8}};
       kwargs...) where {T<:AbstractFloat} =
    maxsim(q, d, true_mask(q, size(q, 2)), true_mask(d.codes, size(d.codes, 2)); kwargs...)

function maxsim(q::AbstractMatrix{T}, d::Int8Index{<:AbstractMatrix{Int8}},
                qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), q, d, qmask, dmask)
end

function maxsim(cfg::MaxSim{T}, q::AbstractMatrix{T}, d::Int8Index{<:AbstractMatrix{Int8}},
                qmask::AbstractVector{Bool},
                dmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    score, _ = int8_pair_forward(q, d, qmask, dmask)
    pair_finalize(score, qmask, cfg.normalize)[1]
end

(cfg::MaxSim)(q::AbstractMatrix, d::Int8Index{<:AbstractMatrix{Int8}}) =
    maxsim(cfg, q, d, true_mask(q, size(q, 2)), true_mask(d.codes, size(d.codes, 2)))
(cfg::MaxSim)(q::AbstractMatrix, d::Int8Index{<:AbstractMatrix{Int8}},
              qmask::AbstractVector{Bool}, dmask::AbstractVector{Bool}) =
    maxsim(cfg, q, d, qmask, dmask)

maxsim(Q::AbstractArray{T,3}, D::Int8Index{<:AbstractArray{Int8,3}};
       kwargs...) where {T<:AbstractFloat} =
    maxsim(Q, D, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(D.codes, size(D.codes, 2), size(D.codes, 3)); kwargs...)

function maxsim(Q::AbstractArray{T,3}, D::Int8Index{<:AbstractArray{Int8,3}},
                qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), Q, D, qmask, dmask)
end

function maxsim(cfg::MaxSim{T}, Q::AbstractArray{T,3},
                D::Int8Index{<:AbstractArray{Int8,3}},
                qmask::AbstractMatrix{Bool},
                dmask::AbstractMatrix{Bool}) where {T<:AbstractFloat}
    scores, _ = int8_paired_forward(Q, D, qmask, dmask)
    cfg.normalize ? length_normalize(scores, qmask) : scores
end

(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, D::Int8Index{<:AbstractArray{Int8,3}}) =
    maxsim(cfg, Q, D, true_mask(Q, size(Q, 2), size(Q, 3)),
           true_mask(D.codes, size(D.codes, 2), size(D.codes, 3)))
(cfg::MaxSim)(Q::AbstractArray{<:Any,3}, D::Int8Index{<:AbstractArray{Int8,3}},
              qmask::AbstractMatrix{Bool}, dmask::AbstractMatrix{Bool}) =
    maxsim(cfg, Q, D, qmask, dmask)
