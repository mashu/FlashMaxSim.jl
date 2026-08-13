# forward_layouts.jl — Paired / in-batch / candidate MaxSim.
#
# `Array` composes tiled pair GEMM over views (no slice copies). Other
# backends launch one fused kernel over the layout; scores and argmax stay
# on the feature backend.

function paired_forward(Q::Array{T,3}, D::Array{T,3},
                        qmask::AbstractMatrix{Bool},
                        dmask::AbstractMatrix{Bool},
                        neg::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    B = size(Q, 3)
    Tq = size(Q, 2)
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

function paired_forward(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                        qmask::AbstractMatrix{Bool},
                        dmask::AbstractMatrix{Bool},
                        neg::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    B = size(Q, 3)
    Tq = size(Q, 2)
    backend = get_backend(Q)
    args = zeros_like(Q, Int32, Tq, B)
    partial = zeros_like(Q, T, Tq, B)
    launch!(paired_token_kernel!, backend, (Tq, B),
            args, partial, Q, D, qmask, dmask, neg)
    vec(sum(partial; dims = 1)), args
end

function inbatch_forward(Q::Array{T,3}, D::Array{T,3},
                         qmask::AbstractMatrix{Bool},
                         dmask::AbstractMatrix{Bool},
                         neg::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    Bq, Bd = size(Q, 3), size(D, 3)
    Tq = size(Q, 2)
    S = zeros(T, Bd, Bq)
    args = zeros(Int32, Tq, Bd, Bq)
    @inbounds for j in 1:Bd, i in 1:Bq
        s, au = pair_forward_host(view(Q, :, :, i), view(D, :, :, j),
                                  view(qmask, :, i), view(dmask, :, j), neg)
        S[j, i] = s
        args[:, j, i] .= au
    end
    S, args
end

function inbatch_forward(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                         qmask::AbstractMatrix{Bool},
                         dmask::AbstractMatrix{Bool},
                         neg::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    Bq, Bd = size(Q, 3), size(D, 3)
    Tq = size(Q, 2)
    backend = get_backend(Q)
    args = zeros_like(Q, Int32, Tq, Bd, Bq)
    partial = zeros_like(Q, T, Tq, Bd, Bq)
    launch!(inbatch_token_kernel!, backend, (Tq, Bd, Bq),
            args, partial, Q, D, qmask, dmask, neg)
    dropdims(sum(partial; dims = 1); dims = 1), args
end

function candidates_forward(Q::Array{T,3},
                            gallery::Array{T,3},
                            idxs::AbstractMatrix{<:Integer},
                            qmask::AbstractMatrix{Bool},
                            dmask::AbstractMatrix{Bool},
                            neg::T) where {T<:AbstractFloat}
    require_colocated(Q, gallery, qmask, dmask)
    C, B = size(idxs, 1), size(idxs, 2)
    Tq, N = size(Q, 2), size(gallery, 3)
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

function candidates_forward(Q::AbstractArray{T,3},
                            gallery::AbstractArray{T,3},
                            idxs::AbstractMatrix{<:Integer},
                            qmask::AbstractMatrix{Bool},
                            dmask::AbstractMatrix{Bool},
                            neg::T) where {T<:AbstractFloat}
    require_colocated(Q, gallery, qmask, dmask)
    idx = indices_on(Q, idxs)
    C, B = size(idx, 1), size(idx, 2)
    Tq, N = size(Q, 2), size(gallery, 3)
    backend = get_backend(Q)
    args = zeros_like(Q, Int32, Tq, C, B)
    partial = zeros_like(Q, T, Tq, C, B)
    launch!(candidates_token_kernel!, backend, (Tq, C, B),
            args, partial, Q, gallery, idx, qmask, dmask, neg, N)
    S = dropdims(sum(partial; dims = 1); dims = 1)
    valid = (idx .>= 1) .& (idx .<= N)
    ifelse.(valid, S, neg), args
end
