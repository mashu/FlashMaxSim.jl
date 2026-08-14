# forward_layouts.jl — Paired / in-batch / candidate MaxSim.
#
# Dispatch is on the *backend* (`array_backend(Q)`), not `::Array`: a host
# `view` / `Adjoint` / `PermutedDimsArray` must take the BLAS path. Named
# `*_forward_ka` entry points exercise device kernels on `CPU()` in CI.
#
# In-batch: chunked doc GEMM (`D'Q`) + light argmax accumulate — never a
# `Tq×Bd×Bq` fused similarity kernel (pathological at train batch sizes).

"""Target bytes for one in-batch GEMM tile (`Td × C × Tq × Bq`)."""
const INBATCH_TILE_BYTES = 64 * 1024 * 1024

function inbatch_doc_chunk(::Type{T}, Td::Int, Tq::Int, Bq::Int, Bd::Int) where {T}
    Bd <= 0 && return 1
    denom = sizeof(T) * max(Td, 1) * max(Tq, 1) * max(Bq, 1)
    c = max(1, INBATCH_TILE_BYTES ÷ denom)
    min(Bd, c)
end

function require_paired_shapes(Q, D, qmask, dmask)
    dim, Tq, B = size(Q)
    Td = size(D, 2)
    size(D, 1) == dim || throw(DimensionMismatch("feature dim"))
    size(D, 3) == B || throw(DimensionMismatch("batch"))
    size(qmask) == (Tq, B) || throw(DimensionMismatch("qmask"))
    size(dmask) == (Td, B) || throw(DimensionMismatch("dmask"))
    dim, Tq, Td, B
end

function require_inbatch_shapes(Q, D, qmask, dmask)
    dim, Tq, Bq = size(Q)
    Td, Bd = size(D, 2), size(D, 3)
    size(D, 1) == dim || throw(DimensionMismatch("feature dim"))
    size(qmask) == (Tq, Bq) || throw(DimensionMismatch("qmask"))
    size(dmask) == (Td, Bd) || throw(DimensionMismatch("dmask"))
    dim, Tq, Bq, Td, Bd
end

function require_candidates_shapes(Q, gallery, idxs, qmask, dmask)
    dim, Tq, B = size(Q)
    Td, N = size(gallery, 2), size(gallery, 3)
    size(gallery, 1) == dim || throw(DimensionMismatch("feature dim"))
    size(idxs, 2) == B || throw(DimensionMismatch("idxs"))
    size(qmask) == (Tq, B) || throw(DimensionMismatch("qmask"))
    size(dmask) == (Td, N) || throw(DimensionMismatch("dmask"))
    dim, Tq, B, Td, N
end

function inbatch_accumulate_host!(S::AbstractMatrix{T},
                                  args::AbstractArray{Int32,3},
                                  M4::AbstractArray{T,4},
                                  qmask::AbstractMatrix{Bool},
                                  dm::AbstractMatrix{Bool},
                                  j0::Int) where {T<:AbstractFloat}
    Td, C, Tq, Bq = size(M4)
    @inbounds for i in 1:Bq, c in 1:C, t in 1:Tq
        qmask[t, i] || continue
        mx = zero(T)
        arg = Int32(0)
        for u in 1:Td
            dm[u, c] || continue
            s = M4[u, c, t, i]
            if arg == Int32(0) || s > mx
                mx = s
                arg = Int32(u)
            end
        end
        arg == Int32(0) && continue
        j = j0 + c - 1
        args[t, j, i] = arg
        S[j, i] += mx
    end
    nothing
end

# ---- paired ------------------------------------------------------------------

function paired_forward(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                        qmask::AbstractMatrix{Bool},
                        dmask::AbstractMatrix{Bool},
                        neg::T) where {T<:AbstractFloat}
    paired_forward(array_backend(Q), Q, D, qmask, dmask, neg)
end

function paired_forward(::CPU, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                        qmask::AbstractMatrix{Bool},
                        dmask::AbstractMatrix{Bool},
                        neg::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    _, Tq, _, B = require_paired_shapes(Q, D, qmask, dmask)
    scores = zeros(T, B)
    args = zeros(Int32, Tq, B)
    @inbounds for b in 1:B
        s, au = pair_forward_host(view(Q, :, :, b), view(D, :, :, b),
                                  view(qmask, :, b), view(dmask, :, b), neg)
        scores[b] = s
        args[:, b] .= au
    end
    scores, args
end

function paired_forward(backend::Backend, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                        qmask::AbstractMatrix{Bool},
                        dmask::AbstractMatrix{Bool},
                        neg::T) where {T<:AbstractFloat}
    paired_forward_ka(backend, Q, D, qmask, dmask, neg)
end

function paired_forward_ka(backend, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                           qmask::AbstractMatrix{Bool},
                           dmask::AbstractMatrix{Bool},
                           ::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    _, Tq, _, B = require_paired_shapes(Q, D, qmask, dmask)
    args = zeros_like(Q, Int32, Tq, B)
    partial = zeros_like(Q, T, Tq, B)
    launch!(paired_token_kernel!, backend, (Tq, B),
            args, partial, Q, D, qmask, dmask)
    sync!(backend)   # host-visible reduction
    vec(sum(partial; dims = 1)), args
end

# ---- in-batch ----------------------------------------------------------------

function inbatch_forward(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                         qmask::AbstractMatrix{Bool},
                         dmask::AbstractMatrix{Bool},
                         neg::T) where {T<:AbstractFloat}
    inbatch_forward(array_backend(Q), Q, D, qmask, dmask, neg)
end

function inbatch_forward(::CPU, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                         qmask::AbstractMatrix{Bool},
                         dmask::AbstractMatrix{Bool},
                         ::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    dim, Tq, Bq, Td, Bd = require_inbatch_shapes(Q, D, qmask, dmask)
    S = zeros(T, Bd, Bq)
    args = zeros(Int32, Tq, Bd, Bq)
    (Bd == 0 || Bq == 0 || Tq == 0 || Td == 0) && return S, args
    Qmat = reshape(Q, dim, Tq * Bq)
    chunk = inbatch_doc_chunk(T, Td, Tq, Bq, Bd)
    for j0 in 1:chunk:Bd
        j1 = min(j0 + chunk - 1, Bd)
        C = j1 - j0 + 1
        Dc = D[:, :, j0:j1]
        A = reshape(Dc, dim, Td * C)
        M = A' * Qmat
        M4 = reshape(M, Td, C, Tq, Bq)
        dm = dmask[:, j0:j1]
        inbatch_accumulate_host!(S, args, M4, qmask, dm, j0)
    end
    S, args
end

function inbatch_forward(backend::Backend, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                         qmask::AbstractMatrix{Bool},
                         dmask::AbstractMatrix{Bool},
                         neg::T) where {T<:AbstractFloat}
    inbatch_forward_ka(backend, Q, D, qmask, dmask, neg)
end

function inbatch_forward_ka(backend, Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                            qmask::AbstractMatrix{Bool},
                            dmask::AbstractMatrix{Bool},
                            ::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    dim, Tq, Bq, Td, Bd = require_inbatch_shapes(Q, D, qmask, dmask)
    S = zeros_like(Q, T, Bd, Bq)
    args = zeros_like(Q, Int32, Tq, Bd, Bq)
    (Bd == 0 || Bq == 0 || Tq == 0 || Td == 0) && return S, args
    Qmat = reshape(Q, dim, Tq * Bq)
    chunk = inbatch_doc_chunk(T, Td, Tq, Bq, Bd)
    for j0 in 1:chunk:Bd
        j1 = min(j0 + chunk - 1, Bd)
        C = j1 - j0 + 1
        Dc = D[:, :, j0:j1]
        A = reshape(Dc, dim, Td * C)
        M = A' * Qmat
        M4 = reshape(M, Td, C, Tq, Bq)
        dm = dmask[:, j0:j1]
        launch!(inbatch_accumulate_kernel!, backend, (Tq, C, Bq),
                S, args, M4, qmask, dm, Int32(j0))
    end
    sync!(backend)
    S, args
end

# ---- candidates --------------------------------------------------------------

function candidates_forward(Q::AbstractArray{T,3},
                            gallery::AbstractArray{T,3},
                            idxs::AbstractMatrix{<:Integer},
                            qmask::AbstractMatrix{Bool},
                            dmask::AbstractMatrix{Bool},
                            neg::T) where {T<:AbstractFloat}
    candidates_forward(array_backend(Q), Q, gallery, idxs, qmask, dmask, neg)
end

function candidates_forward(::CPU, Q::AbstractArray{T,3},
                            gallery::AbstractArray{T,3},
                            idxs::AbstractMatrix{<:Integer},
                            qmask::AbstractMatrix{Bool},
                            dmask::AbstractMatrix{Bool},
                            neg::T) where {T<:AbstractFloat}
    require_colocated(Q, gallery, qmask, dmask)
    _, Tq, B, _, N = require_candidates_shapes(Q, gallery, idxs, qmask, dmask)
    C = size(idxs, 1)
    S = fill(neg, C, B)
    args = zeros(Int32, Tq, C, B)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        s, au = pair_forward_host(view(Q, :, :, b), view(gallery, :, :, j),
                                  view(qmask, :, b), view(dmask, :, j), neg)
        S[c, b] = s
        args[:, c, b] .= au
    end
    S, args
end

function candidates_forward(backend::Backend, Q::AbstractArray{T,3},
                            gallery::AbstractArray{T,3},
                            idxs::AbstractMatrix{<:Integer},
                            qmask::AbstractMatrix{Bool},
                            dmask::AbstractMatrix{Bool},
                            neg::T) where {T<:AbstractFloat}
    candidates_forward_ka(backend, Q, gallery, idxs, qmask, dmask, neg)
end

function candidates_forward_ka(backend, Q::AbstractArray{T,3},
                               gallery::AbstractArray{T,3},
                               idxs::AbstractMatrix{<:Integer},
                               qmask::AbstractMatrix{Bool},
                               dmask::AbstractMatrix{Bool},
                               neg::T) where {T<:AbstractFloat}
    require_colocated(Q, gallery, qmask, dmask)
    _, Tq, B, _, N = require_candidates_shapes(Q, gallery, idxs, qmask, dmask)
    idx = indices_on(Q, idxs)
    C = size(idx, 1)
    args = zeros_like(Q, Int32, Tq, C, B)
    partial = zeros_like(Q, T, Tq, C, B)
    launch!(candidates_token_kernel!, backend, (Tq, C, B),
            args, partial, Q, gallery, idx, qmask, dmask, N)
    sync!(backend)   # host-visible reduction
    S = dropdims(sum(partial; dims = 1); dims = 1)
    valid = (idx .>= 1) .& (idx .<= N)
    ifelse.(valid, S, neg), args
end
