# packed.jl — Ragged sequences (1-based cu_seqlens) and MaxSim over them.
#
# `PackedSeq` owns tokens, CSR, and `max_len` (known at pack time — no host
# scan of `cu` on the scoring path). Document `b` is
# `tokens[:, cu[b]:(cu[b+1]-1)]`. Python 0-based `cu_seqlens` converts with
# `PackedSeq(tokens, cu .+ 1)`.

"""Ragged token matrix plus 1-based CSR. `nseq(p)` sequences, max length `max_len`."""
struct PackedSeq{A <: AbstractMatrix, C <: AbstractVector{<:Integer}}
    tokens::A
    cu::C
    max_len::Int
    function PackedSeq(tokens::A, cu::C, max_len::Integer) where {
            A <: AbstractMatrix, C <: AbstractVector{<:Integer}}
        array_backend(tokens) === array_backend(cu) || throw(ArgumentError(
            "PackedSeq requires tokens and cu on the same KernelAbstractions backend"))
        length(cu) >= 1 || throw(ArgumentError("PackedSeq cu must have length B+1"))
        max_len >= 0 || throw(ArgumentError("PackedSeq max_len must be ≥ 0"))
        new{A, C}(tokens, cu, Int(max_len))
    end
end

nseq(p::PackedSeq) = length(p.cu) - 1

function packed_max_len(cu::Vector{<:Integer})
    n = length(cu) - 1
    n <= 0 && return 0
    m = 0
    @inbounds for b in 1:n
        m = max(m, Int(cu[b + 1]) - Int(cu[b]))
    end
    m
end

function packed_max_len(cu::AbstractVector{<:Integer})
    n = length(cu) - 1
    n <= 0 && return 0
    Int(maximum(view(cu, 2:(n + 1)) .- view(cu, 1:n)))
end

function require_host_csr(tokens::AbstractMatrix, cu::Vector{<:Integer})
    length(cu) >= 1 || throw(ArgumentError("PackedSeq cu must have length B+1"))
    Int(cu[1]) == 1 || throw(ArgumentError("PackedSeq cu must start at 1"))
    Int(cu[end]) - 1 == size(tokens, 2) ||
        throw(ArgumentError("PackedSeq cu does not cover tokens"))
    @inbounds for b in 1:(length(cu) - 1)
        Int(cu[b + 1]) >= Int(cu[b]) ||
            throw(ArgumentError("PackedSeq cu must be nondecreasing"))
    end
    nothing
end

"""
    PackedSeq(tokens, cu) -> PackedSeq

Wrap an existing packed matrix and 1-based CSR. `max_len` is computed once.
Prefer [`pack_docs`](@ref) / [`pack_pairs`](@ref) when concatenating sequences.
"""
PackedSeq(tokens::AbstractMatrix, cu::AbstractVector{<:Integer}) =
    PackedSeq(tokens, cu, packed_max_len(cu))

function PackedSeq(tokens::AbstractMatrix, cu::Vector{<:Integer})
    require_host_csr(tokens, cu)
    PackedSeq(tokens, cu, packed_max_len(cu))
end

Adapt.adapt_structure(to, p::PackedSeq) =
    PackedSeq(adapt(to, p.tokens), adapt(to, p.cu), p.max_len)

"""
    pack_docs(docs) -> PackedSeq

Concatenate variable-length document matrices `(dim, Ld_i)` along tokens.
"""
function pack_docs(docs::AbstractVector{<:AbstractMatrix{T}}) where {T<:AbstractFloat}
    B = length(docs)
    B == 0 && throw(ArgumentError("pack_docs: empty document list"))
    proto = first(docs)
    dim = size(proto, 1)
    lens = Vector{Int32}(undef, B)
    max_len = 0
    @inbounds for b in 1:B
        size(docs[b], 1) == dim || throw(DimensionMismatch("pack_docs: feature dim"))
        array_backend(docs[b]) === array_backend(proto) || throw(ArgumentError(
            "pack_docs: documents must share a KernelAbstractions backend"))
        n = Int(size(docs[b], 2))
        lens[b] = Int32(n)
        max_len = max(max_len, n)
    end
    cu_h = Vector{Int32}(undef, B + 1)
    cu_h[1] = 1
    @inbounds for b in 1:B
        cu_h[b + 1] = cu_h[b] + lens[b]
    end
    total = Int(cu_h[end]) - 1
    tokens = similar(proto, dim, total)
    off = 1
    @inbounds for b in 1:B
        n = Int(lens[b])
        copyto!(view(tokens, :, off:off + n - 1), docs[b])
        off += n
    end
    cu = similar(proto, Int32, B + 1)
    copyto!(cu, cu_h)
    PackedSeq(tokens, cu, max_len)
end

"""
    pack_pairs(qs, ds) -> (Q::PackedSeq, D::PackedSeq)

Pack variable-length `(query, document)` pairs.
"""
function pack_pairs(qs::AbstractVector{<:AbstractMatrix{T}},
                    ds::AbstractVector{<:AbstractMatrix{T}}) where {T<:AbstractFloat}
    length(qs) == length(ds) || throw(DimensionMismatch("pack_pairs: pair count"))
    pack_docs(qs), pack_docs(ds)
end

ChainRulesCore.@non_differentiable pack_docs(::AbstractVector)
ChainRulesCore.@non_differentiable pack_pairs(::AbstractVector, ::AbstractVector)
ChainRulesCore.@non_differentiable PackedSeq(::AbstractMatrix, ::AbstractVector)
ChainRulesCore.@non_differentiable PackedSeq(::AbstractMatrix, ::AbstractVector, ::Integer)

function packed_tangent(p::PackedSeq, dtokens)
    Tangent{typeof(p)}(; tokens = dtokens, cu = NoTangent(), max_len = NoTangent())
end

function packed_forward_host(q::AbstractMatrix{T}, P::PackedSeq,
                             qmask::AbstractVector{Bool},
                             ::T) where {T<:AbstractFloat}
    packed = P.tokens
    cu = P.cu
    Tq = size(q, 2)
    B = nseq(P)
    scores = zeros(T, max(B, 0))
    args = zeros(Int32, Tq, max(B, 0))
    B == 0 && return scores, args
    size(packed, 1) == size(q, 1) || throw(DimensionMismatch("feature dim"))
    length(qmask) == Tq || throw(DimensionMismatch("qmask"))
    max_td = P.max_len
    Stile, mx, argmax_u = pair_host_scratch(T, Tq, max_td)
    dmask = trues(max(max_td, 0))
    @inbounds for b in 1:B
        a = Int(cu[b])
        z = Int(cu[b + 1]) - 1
        Td = max(z - a + 1, 0)
        s, au = pair_forward_host!(argmax_u, mx, Stile,
                                   q, view(packed, :, a:z), qmask, view(dmask, 1:Td), zero(T))
        scores[b] = s
        args[:, b] .= au
    end
    scores, args
end

function packed_forward_ka(backend::Backend, q::AbstractMatrix{T}, P::PackedSeq,
                           qmask::AbstractVector{Bool},
                           ::T) where {T<:AbstractFloat}
    Tq = size(q, 2)
    B = nseq(P)
    args = zeros_like(q, Int32, Tq, B)
    partial = zeros_like(q, T, Tq, B)
    launch!(packed_token_kernel!, backend, (Tq, B), args, partial, q, P.tokens, P.cu, qmask)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function packed_forward_ka(backend::GPU, q::AbstractMatrix{T}, P::PackedSeq,
                           qmask::AbstractVector{Bool},
                           ::T) where {T<:AbstractFloat}
    Tq = size(q, 2)
    B = nseq(P)
    args = zeros_like(q, Int32, Tq, B)
    partial = zeros_like(q, T, Tq, B)
    launch_packed_scan!(backend, args, partial, q, P.tokens, P.cu, qmask)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function packed_forward(q::AbstractMatrix{T}, P::PackedSeq,
                        qmask::AbstractVector{Bool},
                        neg::T) where {T<:AbstractFloat}
    require_colocated(q, P.tokens, P.cu, qmask)
    packed_forward(array_backend(q), q, P, qmask, neg)
end

packed_forward(::CPU, q::AbstractMatrix{T}, P::PackedSeq,
               qmask::AbstractVector{Bool}, neg::T) where {T<:AbstractFloat} =
    packed_forward_host(q, P, qmask, neg)

packed_forward(::Backend, q::AbstractMatrix{T}, P::PackedSeq,
               qmask::AbstractVector{Bool}, neg::T) where {T<:AbstractFloat} =
    packed_forward_ka(array_backend(q), q, P, qmask, neg)

launch_packed_scan!(backend::GPU, args, partial, q, packed, cu, qmask) =
    launch_grouped!(packed_tile_kernel!, backend, query_tile_group((size(q, 2), length(cu) - 1)),
                    (size(q, 2), length(cu) - 1), args, partial, q, packed, cu, qmask)

function varlen_forward_host(Q::PackedSeq, D::PackedSeq, ::T) where {T<:AbstractFloat}
    N = nseq(Q)
    N == nseq(D) || throw(DimensionMismatch("varlen pair count"))
    max_q = Q.max_len
    scores = zeros(T, max(N, 0))
    args = zeros(Int32, max_q, max(N, 0))
    N == 0 && return scores, args
    Qp, Dp, cu_q, cu_d = Q.tokens, D.tokens, Q.cu, D.cu
    size(Qp, 1) == size(Dp, 1) || throw(DimensionMismatch("feature dim"))
    max_d = D.max_len
    Stile, mx, argmax_u = pair_host_scratch(T, max_q, max_d)
    qmask_buf = trues(max(max_q, 0))
    dmask_buf = trues(max(max_d, 0))
    @inbounds for n in 1:N
        qa, qz = Int(cu_q[n]), Int(cu_q[n + 1]) - 1
        da, dz = Int(cu_d[n]), Int(cu_d[n + 1]) - 1
        Tq = qz - qa + 1
        Td = max(dz - da + 1, 0)
        s, au = pair_forward_host!(view(argmax_u, 1:Tq), view(mx, 1:Tq),
                                   view(Stile, :, 1:Tq),
                                   view(Qp, :, qa:qz), view(Dp, :, da:dz),
                                   view(qmask_buf, 1:Tq), view(dmask_buf, 1:Td), zero(T))
        scores[n] = s
        args[1:Tq, n] .= au[1:Tq]
    end
    scores, args
end

function varlen_forward_ka(backend::Backend, Q::PackedSeq, D::PackedSeq,
                           ::T) where {T<:AbstractFloat}
    N = nseq(Q)
    max_q = Q.max_len
    args = zeros_like(Q.tokens, Int32, max_q, N)
    partial = zeros_like(Q.tokens, T, max_q, N)
    launch!(varlen_token_kernel!, backend, (max_q, N), args, partial,
            Q.tokens, D.tokens, Q.cu, D.cu)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function varlen_forward_ka(backend::GPU, Q::PackedSeq, D::PackedSeq,
                           ::T) where {T<:AbstractFloat}
    N = nseq(Q)
    max_q = Q.max_len
    args = zeros_like(Q.tokens, Int32, max_q, N)
    partial = zeros_like(Q.tokens, T, max_q, N)
    launch_varlen_scan!(backend, args, partial, Q.tokens, D.tokens, Q.cu, D.cu)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function varlen_forward(Q::PackedSeq, D::PackedSeq, neg::T) where {T<:AbstractFloat}
    nseq(Q) == nseq(D) || throw(DimensionMismatch("varlen pair count"))
    require_colocated(Q.tokens, D.tokens, Q.cu, D.cu)
    varlen_forward(array_backend(Q.tokens), Q, D, neg)
end

varlen_forward(::CPU, Q::PackedSeq, D::PackedSeq, neg::T) where {T<:AbstractFloat} =
    varlen_forward_host(Q, D, neg)

varlen_forward(::Backend, Q::PackedSeq, D::PackedSeq, neg::T) where {T<:AbstractFloat} =
    varlen_forward_ka(array_backend(Q.tokens), Q, D, neg)

launch_varlen_scan!(backend::GPU, args, partial, Qp, Dp, cu_q, cu_d) =
    launch_grouped!(varlen_tile_kernel!, backend,
                    query_tile_group((size(args, 1), size(args, 2))),
                    (size(args, 1), size(args, 2)), args, partial, Qp, Dp, cu_q, cu_d)

function packed_inv_n(qmask::AbstractVector{Bool}, ::Type{T}, normalize::Bool,
                      prototype::AbstractArray) where {T}
    normalize || return ones_like(prototype, T, 1)
    backend = array_backend(prototype)
    n = count_true_length1(backend, qmask, T)
    finish!(backend)
    one(T) ./ n
end

function apply_length1_scale(scores::AbstractVector{T}, inv_n::AbstractVector{T}) where {T}
    backend = array_backend(scores)
    out = zeros_like(scores)
    launch!(mul_length1_kernel!, backend, length(scores), out, scores, inv_n)
    finish!(backend)
    out
end

function packed_finalize(scores::AbstractVector{T}, qmask::AbstractVector{Bool},
                         normalize::Bool) where {T}
    inv_n = packed_inv_n(qmask, T, normalize, scores)
    normalize ? (apply_length1_scale(scores, inv_n), inv_n) : (scores, inv_n)
end

function varlen_inv_n(Q::PackedSeq, ::Type{T}, normalize::Bool,
                      prototype::AbstractArray) where {T}
    N = nseq(Q)
    inv_n = zeros_like(prototype, T, max(N, 0))
    N == 0 && return inv_n
    normalize || return fill!(inv_n, one(T))
    backend = array_backend(prototype)
    launch!(varlen_inv_n_kernel!, backend, N, inv_n, Q.cu)
    finish!(backend)
    inv_n
end

function varlen_finalize(scores::AbstractVector{T}, Q::PackedSeq, normalize::Bool) where {T}
    inv_n = varlen_inv_n(Q, T, normalize, scores)
    normalize ? (scores .* inv_n, inv_n) : (scores, inv_n)
end

function packed_pullback(Δ::AbstractVector{T}, q, P::PackedSeq, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    packed_pullback(array_backend(q), Δ, q, P, qmask, args, inv_n, mode)
end

function packed_pullback(::CPU, Δ::AbstractVector{T}, q, P::PackedSeq, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    packed = P.tokens
    cu = P.cu
    dq = zeros_like(q)
    dP = zeros_like(packed)
    B = nseq(P)
    @inbounds for b in 1:B
        a = Int(cu[b])
        z = Int(cu[b + 1]) - 1
        δ = T(Δ[b]) * T(inv_n[1])
        dqi, ddi = pair_pullback(δ, q, view(packed, :, a:z), qmask, view(args, :, b), mode)
        dq .+= dqi
        dP[:, a:z] .+= ddi
    end
    dq, dP
end

function packed_pullback(backend::Backend, Δ::AbstractVector{T}, q, P::PackedSeq,
                         qmask, args, inv_n, mode::BackwardStrategy) where {T}
    packed_apply_pullback!(mode, backend, Δ, q, P, qmask, args, inv_n)
end

function packed_apply_pullback!(::AtomicUnified, backend, Δ, q, P, qmask, args, inv_n)
    packed_pullback_ka(backend, Δ, q, P, qmask, args, inv_n)
end

function packed_apply_pullback!(::InvGrid, backend, Δ, q, P, qmask, args, inv_n)
    packed_pullback_invgrid(backend, Δ, q, P, qmask, args, inv_n)
end

function packed_pullback_ka(backend, Δ::AbstractVector{T}, q, P::PackedSeq, qmask, args,
                            inv_n) where {T}
    dq = zeros_like(q)
    dP = zeros_like(P.tokens)
    dim, Tq = size(q)
    launch!(unified_packed_atomic_kernel!, backend, (dim, Tq),
            dq, dP, q, P.tokens, P.cu, qmask, args, Δ, inv_n)
    finish!(backend)
    dq, dP
end

function packed_pullback_invgrid(backend, Δ::AbstractVector{T}, q, P::PackedSeq, qmask,
                                 args, inv_n) where {T}
    dq = zeros_like(q)
    dP = zeros_like(P.tokens)
    dim, Tq = size(q)
    n_dest = size(P.tokens, 2)
    B = nseq(P)
    launch!(gather_packed_kernel!, backend, (dim, Tq),
            dq, P.tokens, P.cu, qmask, args, Δ, inv_n)
    row_ptr, col_idx = build_packed_csr(backend, q, P.cu, args, qmask, n_dest, Tq, B)
    launch!(scatter_packed_csr_kernel!, backend, (dim, n_dest),
            dP, q, row_ptr, col_idx, Δ, inv_n, Tq)
    finish!(backend)
    dq, dP
end

function varlen_pullback(Δ::AbstractVector{T}, Q::PackedSeq, D::PackedSeq, args, inv_n,
                         mode::BackwardStrategy) where {T}
    varlen_pullback(array_backend(Q.tokens), Δ, Q, D, args, inv_n, mode)
end

function varlen_pullback(::CPU, Δ::AbstractVector{T}, Q::PackedSeq, D::PackedSeq, args, inv_n,
                         mode::BackwardStrategy) where {T}
    Qp, Dp, cu_q, cu_d = Q.tokens, D.tokens, Q.cu, D.cu
    dQ = zeros_like(Qp)
    dD = zeros_like(Dp)
    N = nseq(Q)
    qmask_buf = trues(max(Q.max_len, 0))
    @inbounds for n in 1:N
        qa, qz = Int(cu_q[n]), Int(cu_q[n + 1]) - 1
        da, dz = Int(cu_d[n]), Int(cu_d[n + 1]) - 1
        Tq = qz - qa + 1
        δ = T(Δ[n]) * T(inv_n[n])
        dqi, ddi = pair_pullback(δ, view(Qp, :, qa:qz), view(Dp, :, da:dz),
                                 view(qmask_buf, 1:Tq), view(args, 1:Tq, n), mode)
        dQ[:, qa:qz] .+= dqi
        dD[:, da:dz] .+= ddi
    end
    dQ, dD
end

function varlen_pullback(backend::Backend, Δ::AbstractVector{T}, Q::PackedSeq, D::PackedSeq,
                         args, inv_n, mode::BackwardStrategy) where {T}
    varlen_apply_pullback!(mode, backend, Δ, Q, D, args, inv_n)
end

function varlen_apply_pullback!(::AtomicUnified, backend, Δ, Q, D, args, inv_n)
    varlen_pullback_ka(backend, Δ, Q, D, args, inv_n)
end

function varlen_apply_pullback!(::InvGrid, backend, Δ, Q, D, args, inv_n)
    varlen_pullback_invgrid(backend, Δ, Q, D, args, inv_n)
end

function varlen_pullback_ka(backend, Δ::AbstractVector{T}, Q::PackedSeq, D::PackedSeq, args,
                            inv_n) where {T}
    dQ = zeros_like(Q.tokens)
    dD = zeros_like(D.tokens)
    dim = size(Q.tokens, 1)
    max_q = Q.max_len
    N = nseq(Q)
    launch!(unified_varlen_atomic_kernel!, backend, (dim, max_q, N),
            dQ, dD, Q.tokens, D.tokens, Q.cu, D.cu, args, Δ, inv_n)
    finish!(backend)
    dQ, dD
end

function varlen_pullback_invgrid(backend, Δ::AbstractVector{T}, Q::PackedSeq, D::PackedSeq,
                                 args, inv_n) where {T}
    dQ = zeros_like(Q.tokens)
    dD = zeros_like(D.tokens)
    dim = size(Q.tokens, 1)
    max_q = Q.max_len
    N = nseq(Q)
    n_dest = size(D.tokens, 2)
    launch!(gather_varlen_kernel!, backend, (dim, max_q, N),
            dQ, D.tokens, Q.cu, D.cu, args, Δ, inv_n)
    row_ptr, col_idx = build_varlen_csr(backend, Q.tokens, Q.cu, D.cu, args, n_dest, max_q, N)
    launch!(scatter_varlen_csr_kernel!, backend, (dim, n_dest),
            dD, Q.tokens, Q.cu, row_ptr, col_idx, Δ, inv_n, max_q)
    finish!(backend)
    dQ, dD
end
