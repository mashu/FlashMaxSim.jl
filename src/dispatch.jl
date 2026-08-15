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

# ---- packed cu_seqlens (`PackedSeq`) -----------------------------------------

"""
    maxsim(q, D::PackedSeq; ...) -> Vector
    maxsim(q, D::PackedSeq, qmask; ...)
    maxsim(cfg::MaxSim, q, D::PackedSeq, qmask)

One query against B ragged documents (`pack_docs`). Returns `(B,)` with
`out[b] = MaxSim(q, D.tokens[:, D.cu[b]:(D.cu[b+1]-1)])`.
"""
maxsim(q::AbstractMatrix{T}, D::PackedSeq{<:AbstractMatrix{T}};
       kwargs...) where {T<:AbstractFloat} =
    maxsim(q, D, true_mask(q, size(q, 2)); kwargs...)

function maxsim(q::AbstractMatrix{T}, D::PackedSeq{<:AbstractMatrix{T}},
                qmask::AbstractVector{Bool};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), q, D, qmask)
end

function maxsim(cfg::MaxSim{T}, q::AbstractMatrix{T}, D::PackedSeq{<:AbstractMatrix{T}},
                qmask::AbstractVector{Bool}) where {T<:AbstractFloat}
    scores, _ = packed_forward(q, D, qmask, cfg.neg)
    scores_n, _ = packed_finalize(scores, qmask, cfg.normalize)
    scores_n
end

(cfg::MaxSim{T})(q::AbstractMatrix{T}, D::PackedSeq{<:AbstractMatrix{T}}) where {T} =
    maxsim(cfg, q, D, true_mask(q, size(q, 2)))
(cfg::MaxSim{T})(q::AbstractMatrix{T}, D::PackedSeq{<:AbstractMatrix{T}},
                 qmask::AbstractVector{Bool}) where {T} =
    maxsim(cfg, q, D, qmask)

# ---- varlen pairs (two PackedSeq) --------------------------------------------

"""
    maxsim(Q::PackedSeq, D::PackedSeq; ...) -> Vector
    maxsim(cfg::MaxSim, Q::PackedSeq, D::PackedSeq)

Variable-length paired MaxSim (`pack_pairs`). Returns `(N,)` with
`out[n] = MaxSim(Q[n], D[n])`.
"""
function maxsim(Q::PackedSeq{<:AbstractMatrix{T}}, D::PackedSeq{<:AbstractMatrix{T}};
                neg::Real = T(-1.0f4), normalize::Bool = false,
                backward = AtomicUnified()) where {T<:AbstractFloat}
    maxsim(MaxSim{T}(neg, normalize, backward), Q, D)
end

function maxsim(cfg::MaxSim{T}, Q::PackedSeq{<:AbstractMatrix{T}},
                D::PackedSeq{<:AbstractMatrix{T}}) where {T<:AbstractFloat}
    scores, _ = varlen_forward(Q, D, cfg.neg)
    scores_n, _ = varlen_finalize(scores, Q, cfg.normalize)
    scores_n
end

(cfg::MaxSim{T})(Q::PackedSeq{<:AbstractMatrix{T}},
                 D::PackedSeq{<:AbstractMatrix{T}}) where {T} =
    maxsim(cfg, Q, D)

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
