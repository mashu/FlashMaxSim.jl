# packed.jl — cu_seqlens packing and MaxSim over ragged documents.
#
# `cu` is length `B+1`, 1-based CSR: document `b` is `packed[:, cu[b]:(cu[b+1]-1)]`.
# Python 0-based `cu_seqlens` converts with `cu .+ 1`.

"""Maximum document / query length encoded by a CSR `cu` vector."""
function packed_max_len(cu::AbstractVector{<:Integer})
    n = length(cu) - 1
    n <= 0 && return 0
    m = 0
    @inbounds for b in 1:n
        m = max(m, Int(cu[b + 1]) - Int(cu[b]))
    end
    m
end

"""
    pack_docs(docs) -> (packed, cu)

Concatenate variable-length document matrices `(dim, Ld_i)` along tokens.
`cu` is 1-based CSR of length `B+1` on the same backend as `docs[1]`.
"""
function pack_docs(docs::AbstractVector{<:AbstractMatrix{T}}) where {T<:AbstractFloat}
    B = length(docs)
    B == 0 && throw(ArgumentError("pack_docs: empty document list"))
    proto = first(docs)
    dim = size(proto, 1)
    lens = Vector{Int32}(undef, B)
    @inbounds for b in 1:B
        size(docs[b], 1) == dim || throw(DimensionMismatch("pack_docs: feature dim"))
        array_backend(docs[b]) === array_backend(proto) || throw(ArgumentError(
            "pack_docs: documents must share a KernelAbstractions backend"))
        lens[b] = Int32(size(docs[b], 2))
    end
    cu_h = Vector{Int32}(undef, B + 1)
    cu_h[1] = 1
    @inbounds for b in 1:B
        cu_h[b + 1] = cu_h[b] + lens[b]
    end
    total = Int(cu_h[end]) - 1
    packed = similar(proto, dim, total)
    off = 1
    @inbounds for b in 1:B
        n = Int(lens[b])
        copyto!(view(packed, :, off:off + n - 1), docs[b])
        off += n
    end
    cu = similar(proto, Int32, B + 1)
    copyto!(cu, cu_h)
    packed, cu
end

"""
    pack_pairs(qs, ds) -> (Q_packed, D_packed, cu_q, cu_d)

Pack variable-length `(query, document)` pairs. `cu_q` / `cu_d` are 1-based CSR.
"""
function pack_pairs(qs::AbstractVector{<:AbstractMatrix{T}},
                    ds::AbstractVector{<:AbstractMatrix{T}}) where {T<:AbstractFloat}
    length(qs) == length(ds) || throw(DimensionMismatch("pack_pairs: pair count"))
    Qp, cu_q = pack_docs(qs)
    Dp, cu_d = pack_docs(ds)
    Qp, Dp, cu_q, cu_d
end

function packed_forward_host(q::AbstractMatrix{T}, packed::AbstractMatrix{T},
                             cu::AbstractVector{<:Integer},
                             qmask::AbstractVector{Bool},
                             ::T) where {T<:AbstractFloat}
    Tq = size(q, 2)
    B = length(cu) - 1
    scores = zeros(T, max(B, 0))
    args = zeros(Int32, Tq, max(B, 0))
    B == 0 && return scores, args
    size(packed, 1) == size(q, 1) || throw(DimensionMismatch("feature dim"))
    length(qmask) == Tq || throw(DimensionMismatch("qmask"))
    Stile, mx, argmax_u = pair_host_scratch(T, Tq, packed_max_len(cu))
    @inbounds for b in 1:B
        a = Int(cu[b])
        z = Int(cu[b + 1]) - 1
        dmask = trues(max(z - a + 1, 0))
        s, au = pair_forward_host!(argmax_u, mx, Stile,
                                   q, view(packed, :, a:z), qmask, dmask, zero(T))
        scores[b] = s
        args[:, b] .= au
    end
    scores, args
end

function packed_forward_ka(backend::Backend, q::AbstractMatrix{T},
                           packed::AbstractMatrix{T},
                           cu::AbstractVector{<:Integer},
                           qmask::AbstractVector{Bool},
                           ::T) where {T<:AbstractFloat}
    Tq = size(q, 2)
    B = length(cu) - 1
    args = zeros_like(q, Int32, Tq, B)
    partial = zeros_like(q, T, Tq, B)
    launch!(packed_token_kernel!, backend, (Tq, B), args, partial, q, packed, cu, qmask)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function packed_forward_ka(backend::GPU, q::AbstractMatrix{T},
                           packed::AbstractMatrix{T},
                           cu::AbstractVector{<:Integer},
                           qmask::AbstractVector{Bool},
                           ::T) where {T<:AbstractFloat}
    Tq = size(q, 2)
    B = length(cu) - 1
    args = zeros_like(q, Int32, Tq, B)
    partial = zeros_like(q, T, Tq, B)
    nd = (Tq, B)
    launch_grouped!(packed_tile_kernel!, backend, query_tile_group(nd), nd,
                    args, partial, q, packed, cu, qmask)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function packed_forward(q::AbstractMatrix{T}, packed::AbstractMatrix{T},
                        cu::AbstractVector{<:Integer},
                        qmask::AbstractVector{Bool},
                        neg::T) where {T<:AbstractFloat}
    require_colocated(q, packed, qmask)
    array_backend(cu) === array_backend(q) || throw(ArgumentError(
        "FlashMaxSim requires cu on the same backend as features"))
    packed_forward(array_backend(q), q, packed, cu, qmask, neg)
end

packed_forward(::CPU, q::AbstractMatrix{T}, packed::AbstractMatrix{T},
               cu::AbstractVector{<:Integer}, qmask::AbstractVector{Bool},
               neg::T) where {T<:AbstractFloat} =
    packed_forward_host(q, packed, cu, qmask, neg)

packed_forward(::Backend, q::AbstractMatrix{T}, packed::AbstractMatrix{T},
               cu::AbstractVector{<:Integer}, qmask::AbstractVector{Bool},
               neg::T) where {T<:AbstractFloat} =
    packed_forward_ka(array_backend(q), q, packed, cu, qmask, neg)

function varlen_forward_host(Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
                             cu_q::AbstractVector{<:Integer},
                             cu_d::AbstractVector{<:Integer},
                             ::T) where {T<:AbstractFloat}
    N = length(cu_q) - 1
    N == length(cu_d) - 1 || throw(DimensionMismatch("varlen pair count"))
    max_q = packed_max_len(cu_q)
    scores = zeros(T, max(N, 0))
    args = zeros(Int32, max_q, max(N, 0))
    N == 0 && return scores, args
    size(Qp, 1) == size(Dp, 1) || throw(DimensionMismatch("feature dim"))
    Stile, mx, argmax_u = pair_host_scratch(T, max_q, packed_max_len(cu_d))
    @inbounds for n in 1:N
        qa, qz = Int(cu_q[n]), Int(cu_q[n + 1]) - 1
        da, dz = Int(cu_d[n]), Int(cu_d[n + 1]) - 1
        Tq = qz - qa + 1
        qmask = trues(Tq)
        dmask = trues(max(dz - da + 1, 0))
        s, au = pair_forward_host!(view(argmax_u, 1:Tq), view(mx, 1:Tq),
                                   view(Stile, :, 1:Tq),
                                   view(Qp, :, qa:qz), view(Dp, :, da:dz),
                                   qmask, dmask, zero(T))
        scores[n] = s
        args[1:Tq, n] .= au[1:Tq]
    end
    scores, args
end

function varlen_forward_ka(backend::Backend, Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
                           cu_q::AbstractVector{<:Integer},
                           cu_d::AbstractVector{<:Integer},
                           ::T) where {T<:AbstractFloat}
    N = length(cu_q) - 1
    max_q = packed_max_len(Array(cu_q))
    args = zeros_like(Qp, Int32, max_q, N)
    partial = zeros_like(Qp, T, max_q, N)
    launch!(varlen_token_kernel!, backend, (max_q, N), args, partial, Qp, Dp, cu_q, cu_d)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function varlen_forward_ka(backend::GPU, Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
                           cu_q::AbstractVector{<:Integer},
                           cu_d::AbstractVector{<:Integer},
                           ::T) where {T<:AbstractFloat}
    N = length(cu_q) - 1
    max_q = packed_max_len(Array(cu_q))
    args = zeros_like(Qp, Int32, max_q, N)
    partial = zeros_like(Qp, T, max_q, N)
    nd = (max_q, N)
    launch_grouped!(varlen_tile_kernel!, backend, query_tile_group(nd), nd,
                    args, partial, Qp, Dp, cu_q, cu_d)
    scores = vec(sum(partial; dims = 1))
    finish!(backend)
    scores, args
end

function varlen_forward(Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
                        cu_q::AbstractVector{<:Integer},
                        cu_d::AbstractVector{<:Integer},
                        neg::T) where {T<:AbstractFloat}
    require_colocated(Qp, Dp)
    array_backend(cu_q) === array_backend(Qp) || throw(ArgumentError(
        "FlashMaxSim requires cu_q on the same backend as features"))
    array_backend(cu_d) === array_backend(Qp) || throw(ArgumentError(
        "FlashMaxSim requires cu_d on the same backend as features"))
    varlen_forward(array_backend(Qp), Qp, Dp, cu_q, cu_d, neg)
end

varlen_forward(::CPU, Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
               cu_q::AbstractVector{<:Integer}, cu_d::AbstractVector{<:Integer},
               neg::T) where {T<:AbstractFloat} =
    varlen_forward_host(Qp, Dp, cu_q, cu_d, neg)

varlen_forward(::Backend, Qp::AbstractMatrix{T}, Dp::AbstractMatrix{T},
               cu_q::AbstractVector{<:Integer}, cu_d::AbstractVector{<:Integer},
               neg::T) where {T<:AbstractFloat} =
    varlen_forward_ka(array_backend(Qp), Qp, Dp, cu_q, cu_d, neg)

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

function varlen_inv_n(cu_q::AbstractVector{<:Integer}, ::Type{T}, normalize::Bool,
                      prototype::AbstractArray) where {T}
    N = length(cu_q) - 1
    inv_n = zeros_like(prototype, T, max(N, 0))
    N == 0 && return inv_n
    normalize || return fill!(inv_n, one(T))
    backend = array_backend(prototype)
    launch!(varlen_inv_n_kernel!, backend, N, inv_n, cu_q)
    finish!(backend)
    inv_n
end

function varlen_finalize(scores::AbstractVector{T}, cu_q, normalize::Bool) where {T}
    inv_n = varlen_inv_n(cu_q, T, normalize, scores)
    normalize ? (scores .* inv_n, inv_n) : (scores, inv_n)
end

function packed_pullback(Δ::AbstractVector{T}, q, packed, cu, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    packed_pullback(array_backend(q), Δ, q, packed, cu, qmask, args, inv_n, mode)
end

function packed_pullback(::CPU, Δ::AbstractVector{T}, q, packed, cu, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    dq = zeros_like(q)
    dP = zeros_like(packed)
    B = length(cu) - 1
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

function packed_pullback(backend::Backend, Δ::AbstractVector{T}, q, packed, cu,
                         qmask, args, inv_n, mode::BackwardStrategy) where {T}
    packed_apply_pullback!(mode, backend, Δ, q, packed, cu, qmask, args, inv_n)
end

function packed_apply_pullback!(::AtomicUnified, backend, Δ, q, packed, cu, qmask,
                                args, inv_n)
    packed_pullback_ka(backend, Δ, q, packed, cu, qmask, args, inv_n)
end

function packed_apply_pullback!(::InvGrid, backend, Δ, q, packed, cu, qmask, args, inv_n)
    throw(ArgumentError("packed MaxSim InvGrid is host-only; use AtomicUnified()"))
end

function packed_pullback_ka(backend, Δ::AbstractVector{T}, q, packed, cu, qmask, args,
                            inv_n) where {T}
    dq = zeros_like(q)
    dP = zeros_like(packed)
    dim, Tq = size(q)
    B = length(cu) - 1
    launch!(unified_packed_atomic_kernel!, backend, (dim, Tq, B),
            dq, dP, q, packed, cu, qmask, args, Δ, inv_n)
    finish!(backend)
    dq, dP
end

function varlen_pullback(Δ::AbstractVector{T}, Qp, Dp, cu_q, cu_d, args, inv_n,
                         mode::BackwardStrategy) where {T}
    varlen_pullback(array_backend(Qp), Δ, Qp, Dp, cu_q, cu_d, args, inv_n, mode)
end

function varlen_pullback(::CPU, Δ::AbstractVector{T}, Qp, Dp, cu_q, cu_d, args, inv_n,
                         mode::BackwardStrategy) where {T}
    dQ = zeros_like(Qp)
    dD = zeros_like(Dp)
    N = length(cu_q) - 1
    @inbounds for n in 1:N
        qa, qz = Int(cu_q[n]), Int(cu_q[n + 1]) - 1
        da, dz = Int(cu_d[n]), Int(cu_d[n + 1]) - 1
        Tq = qz - qa + 1
        δ = T(Δ[n]) * T(inv_n[n])
        dqi, ddi = pair_pullback(δ, view(Qp, :, qa:qz), view(Dp, :, da:dz),
                                 trues(Tq), view(args, 1:Tq, n), mode)
        dQ[:, qa:qz] .+= dqi
        dD[:, da:dz] .+= ddi
    end
    dQ, dD
end

function varlen_pullback(backend::Backend, Δ::AbstractVector{T}, Qp, Dp, cu_q, cu_d,
                         args, inv_n, mode::BackwardStrategy) where {T}
    varlen_apply_pullback!(mode, backend, Δ, Qp, Dp, cu_q, cu_d, args, inv_n)
end

function varlen_apply_pullback!(::AtomicUnified, backend, Δ, Qp, Dp, cu_q, cu_d, args, inv_n)
    varlen_pullback_ka(backend, Δ, Qp, Dp, cu_q, cu_d, args, inv_n)
end

function varlen_apply_pullback!(::InvGrid, backend, Δ, Qp, Dp, cu_q, cu_d, args, inv_n)
    throw(ArgumentError("varlen MaxSim InvGrid is host-only; use AtomicUnified()"))
end

function varlen_pullback_ka(backend, Δ::AbstractVector{T}, Qp, Dp, cu_q, cu_d, args,
                            inv_n) where {T}
    dQ = zeros_like(Qp)
    dD = zeros_like(Dp)
    dim = size(Qp, 1)
    max_q = size(args, 1)
    N = length(cu_q) - 1
    launch!(unified_varlen_atomic_kernel!, backend, (dim, max_q, N),
            dQ, dD, Qp, Dp, cu_q, cu_d, args, Δ, inv_n)
    finish!(backend)
    dQ, dD
end
