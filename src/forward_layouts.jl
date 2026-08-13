# forward_layouts.jl — Paired / in-batch / candidate MaxSim over pairs.
#
# Batch layouts compose [`pair_forward`](@ref). On GPU this is sequential
# pair launches (host orchestration); not a fused multi-pair kernel yet.

function paired_forward(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                        qmask::AbstractMatrix{Bool},
                        dmask::AbstractMatrix{Bool},
                        neg::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    B = size(Q, 3)
    Tq = size(Q, 2)
    scores = zeros(T, B)
    args = zeros(Int32, Tq, B)
    @inbounds for b in 1:B
        s, au = pair_forward(Q[:, :, b], D[:, :, b],
                             qmask[:, b], dmask[:, b], neg)
        scores[b] = s
        args[:, b] = au
    end
    scores, args
end

function inbatch_forward(Q::AbstractArray{T,3}, D::AbstractArray{T,3},
                         qmask::AbstractMatrix{Bool},
                         dmask::AbstractMatrix{Bool},
                         neg::T) where {T<:AbstractFloat}
    require_colocated(Q, D, qmask, dmask)
    Bq, Bd = size(Q, 3), size(D, 3)
    Tq = size(Q, 2)
    S = zeros(T, Bd, Bq)
    args = zeros(Int32, Tq, Bd, Bq)
    @inbounds for j in 1:Bd, i in 1:Bq
        s, au = pair_forward(Q[:, :, i], D[:, :, j],
                             qmask[:, i], dmask[:, j], neg)
        S[j, i] = s
        args[:, j, i] = au
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
    C, B = size(idxs, 1), size(idxs, 2)
    Tq, N = size(Q, 2), size(gallery, 3)
    S = fill(neg, C, B)
    args = zeros(Int32, Tq, C, B)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        s, au = pair_forward(Q[:, :, b], gallery[:, :, j],
                             qmask[:, b], dmask[:, j], neg)
        S[c, b] = s
        args[:, c, b] = au
    end
    S, args
end
