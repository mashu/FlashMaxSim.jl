# backward.jl — Sparse MaxSim pullback (paper §4.2, Eqs. 2–3).
#
# Host aggregation today: downloads features, scatters on CPU, uploads
# cotangents. Strategy chosen by [`BackwardStrategy`](@ref) dispatch.

"""
``∇Q`` gather (Eq. 2) + ``∇D`` via [`accumulate_doc!`](@ref).
"""
function pair_pullback(δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{Int32},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    qh, dh = Array(q), Array(d)
    qmh = Vector{Bool}(Array(qmask))
    dq = zeros(T, size(qh))
    dd = zeros(T, size(dh))
    (!isfinite(δ) || δ == zero(T)) && return upload_like(q, dq), upload_like(d, dd)
    dim, Tq = size(qh)
    Td = size(dh, 2)
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    δ_src = zeros(T, Tq)
    @inbounds for t in 1:Tq
        qmh[t] || continue
        u = Int(argmax_u[t])
        (1 <= u <= Td) || continue
        δ_src[t] = δ
        @simd for k in 1:dim
            dq[k, t] += δ * dh[k, u]
        end
    end
    accumulate_doc!(mode, dd, qh, δ_src, argmax_u)
    upload_like(q, dq), upload_like(d, dd)
end

function paired_pullback(Δ::AbstractVector{T}, Q, D, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    Qh, Dh = Array(Q), Array(D)
    dQ = zeros(T, size(Qh))
    dD = zeros(T, size(Dh))
    qmh = host_bool(qmask)
    B = size(Qh, 3)
    @inbounds for b in 1:B
        δ = T(Δ[b]) * inv_n[b]
        dq, dd = pair_pullback(δ, view(Qh, :, :, b), view(Dh, :, :, b),
                               view(qmh, :, b), view(args, :, b), mode)
        dQ[:, :, b] .+= Array(dq)
        dD[:, :, b] .+= Array(dd)
    end
    upload_like(Q, dQ), upload_like(D, dD)
end

function inbatch_pullback(Δ::AbstractMatrix{T}, Q, D, qmask, args, inv_n,
                          mode::BackwardStrategy) where {T}
    Qh, Dh = Array(Q), Array(D)
    qmh = host_bool(qmask)
    dQ = zeros(T, size(Qh))
    dD = zeros(T, size(Dh))
    Bd, Bq = size(Δ)
    dim, Tq, _ = size(Qh)
    Td = size(Dh, 2)
    @inbounds for j in 1:Bd, i in 1:Bq
        δ = T(Δ[j, i]) * inv_n[i]
        (!isfinite(δ) || δ == zero(T)) && continue
        δ_src = zeros(T, Tq)
        for t in 1:Tq
            qmh[t, i] || continue
            u = Int(args[t, j, i])
            (1 <= u <= Td) || continue
            δ_src[t] = δ
            for k in 1:dim
                dQ[k, t, i] += δ * Dh[k, u, j]
            end
        end
        accumulate_doc!(mode, view(dD, :, :, j), view(Qh, :, :, i),
                        δ_src, view(args, :, j, i))
    end
    upload_like(Q, dQ), upload_like(D, dD)
end

function candidates_pullback(Δ::AbstractMatrix{T}, Q, gallery, idxs, qmask, args,
                             inv_n, mode::BackwardStrategy) where {T}
    Qh, Gh = Array(Q), Array(gallery)
    qmh = host_bool(qmask)
    dQ = zeros(T, size(Qh))
    dG = zeros(T, size(Gh))
    C, B = size(Δ)
    dim, Tq, _ = size(Qh)
    Td = size(Gh, 2)
    N = size(Gh, 3)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        δ = T(Δ[c, b]) * inv_n[b]
        (!isfinite(δ) || δ == zero(T)) && continue
        δ_src = zeros(T, Tq)
        for t in 1:Tq
            qmh[t, b] || continue
            u = Int(args[t, c, b])
            (1 <= u <= Td) || continue
            δ_src[t] = δ
            for k in 1:dim
                dQ[k, t, b] += δ * Gh[k, u, j]
            end
        end
        accumulate_doc!(mode, view(dG, :, :, j), view(Qh, :, :, b),
                        δ_src, view(args, :, c, b))
    end
    upload_like(Q, dQ), upload_like(gallery, dG)
end
