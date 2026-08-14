# backward.jl — Sparse MaxSim pullback (paper §4.2, Eqs. 2–3).
#
# CPU: sequential gather + [`accumulate_doc!`](@ref) (BitArray-safe).
# GPU and other KA backends: gather/scatter kernels; no host copies of Q/D.

function pair_pullback(δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{<:Integer},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    pair_pullback(array_backend(q), δ, q, d, qmask, argmax_u, mode)
end

function pair_pullback(::CPU, δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{<:Integer},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    dq = zeros_like(q)
    dd = zeros_like(d)
    δ == zero(T) && return dq, dd
    dim, Tq = size(q)
    Td = size(d, 2)
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    δ_src = zeros(T, Tq)
    @inbounds for t in 1:Tq
        qmask[t] || continue
        u = Int(argmax_u[t])
        (1 <= u <= Td) || continue
        δ_src[t] = δ
        @simd for k in 1:dim
            dq[k, t] += δ * d[k, u]
        end
    end
    accumulate_doc!(mode, dd, q, δ_src, argmax_u)
    dq, dd
end

function pair_pullback(backend, δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{<:Integer},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    pair_pullback_ka(backend, δ, q, d, qmask, argmax_u, mode)
end

function pair_pullback_ka(backend, δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                          qmask::AbstractVector{Bool},
                          argmax_u::AbstractVector{<:Integer},
                          mode::BackwardStrategy) where {T<:AbstractFloat}
    dq = zeros_like(q)
    dd = zeros_like(d)
    δ == zero(T) && return dq, dd
    dim, Tq = size(q)
    Td = size(d, 2)
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    launch!(gather_pair_kernel!, backend, (dim, Tq), dq, d, qmask, argmax_u, δ, Td)
    scatter_pair!(mode, backend, dd, q, qmask, argmax_u, δ)
    dq, dd
end

scatter_pair!(::AtomicUnified, backend, dd, q, qmask, argmax_u, δ) =
    launch!(scatter_pair_atomic_kernel!, backend, (size(q, 1), size(q, 2)),
            dd, q, qmask, argmax_u, δ, size(dd, 2))

scatter_pair!(::InvGrid, backend, dd, q, qmask, argmax_u, δ) =
    launch!(scatter_pair_invgrid_kernel!, backend, (size(dd, 1), size(dd, 2)),
            dd, q, qmask, argmax_u, δ, size(q, 2))

function paired_pullback(Δ::AbstractVector{T}, Q, D, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    paired_pullback(array_backend(Q), Δ, Q, D, qmask, args, inv_n, mode)
end

function paired_pullback(::CPU, Δ::AbstractVector{T}, Q, D, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dD = zeros_like(D)
    B = size(Q, 3)
    @inbounds for b in 1:B
        δ = T(Δ[b]) * inv_n[b]
        dq, dd = pair_pullback(δ, view(Q, :, :, b), view(D, :, :, b),
                               view(qmask, :, b), view(args, :, b), mode)
        dQ[:, :, b] .+= dq
        dD[:, :, b] .+= dd
    end
    dQ, dD
end

function paired_pullback(backend, Δ::AbstractVector{T}, Q, D, qmask, args, inv_n,
                         mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dD = zeros_like(D)
    dim, Tq, B = size(Q)
    Td = size(D, 2)
    launch!(gather_paired_kernel!, backend, (dim, Tq, B),
            dQ, D, qmask, args, Δ, inv_n, Td)
    scatter_paired!(mode, backend, dD, Q, qmask, args, Δ, inv_n)
    dQ, dD
end

scatter_paired!(::AtomicUnified, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_paired_atomic_kernel!, backend, (size(Q, 1), size(Q, 2), size(Q, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(dD, 2))

scatter_paired!(::InvGrid, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_paired_invgrid_kernel!, backend, (size(dD, 1), size(dD, 2), size(dD, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(Q, 2))

function inbatch_pullback(Δ::AbstractMatrix{T}, Q, D, qmask, args, inv_n,
                          mode::BackwardStrategy) where {T}
    inbatch_pullback(array_backend(Q), Δ, Q, D, qmask, args, inv_n, mode)
end

function inbatch_pullback(::CPU, Δ::AbstractMatrix{T}, Q, D, qmask, args, inv_n,
                          mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dD = zeros_like(D)
    Bd, Bq = size(Δ)
    dim, Tq, _ = size(Q)
    Td = size(D, 2)
    @inbounds for j in 1:Bd, i in 1:Bq
        δ = T(Δ[j, i]) * inv_n[i]
        δ == zero(T) && continue
        δ_src = zeros(T, Tq)
        for t in 1:Tq
            qmask[t, i] || continue
            u = Int(args[t, j, i])
            (1 <= u <= Td) || continue
            δ_src[t] = δ
            for k in 1:dim
                dQ[k, t, i] += δ * D[k, u, j]
            end
        end
        accumulate_doc!(mode, view(dD, :, :, j), view(Q, :, :, i),
                        δ_src, view(args, :, j, i))
    end
    dQ, dD
end

function inbatch_pullback(backend, Δ::AbstractMatrix{T}, Q, D, qmask, args, inv_n,
                          mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dD = zeros_like(D)
    dim, Tq, Bq = size(Q)
    Td, Bd = size(D, 2), size(D, 3)
    launch!(gather_inbatch_kernel!, backend, (dim, Tq, Bq),
            dQ, D, qmask, args, Δ, inv_n, Td, Bd)
    scatter_inbatch!(mode, backend, dD, Q, qmask, args, Δ, inv_n)
    dQ, dD
end

scatter_inbatch!(::AtomicUnified, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_inbatch_atomic_kernel!, backend, (size(Q, 1), size(Q, 2), size(Q, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(dD, 2), size(dD, 3))

scatter_inbatch!(::InvGrid, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_inbatch_invgrid_kernel!, backend, (size(dD, 1), size(dD, 2), size(dD, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(Q, 2), size(Q, 3))

function candidates_pullback(Δ::AbstractMatrix{T}, Q, gallery, idxs, qmask, args,
                             inv_n, mode::BackwardStrategy) where {T}
    candidates_pullback(array_backend(Q), Δ, Q, gallery, idxs, qmask, args, inv_n, mode)
end

function candidates_pullback(::CPU, Δ::AbstractMatrix{T}, Q, gallery, idxs, qmask, args,
                             inv_n, mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dG = zeros_like(gallery)
    C, B = size(Δ)
    dim, Tq, _ = size(Q)
    Td = size(gallery, 2)
    N = size(gallery, 3)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        δ = T(Δ[c, b]) * inv_n[b]
        δ == zero(T) && continue
        δ_src = zeros(T, Tq)
        for t in 1:Tq
            qmask[t, b] || continue
            u = Int(args[t, c, b])
            (1 <= u <= Td) || continue
            δ_src[t] = δ
            for k in 1:dim
                dQ[k, t, b] += δ * gallery[k, u, j]
            end
        end
        accumulate_doc!(mode, view(dG, :, :, j), view(Q, :, :, b),
                        δ_src, view(args, :, c, b))
    end
    dQ, dG
end

function candidates_pullback(backend, Δ::AbstractMatrix{T}, Q, gallery, idxs, qmask, args,
                             inv_n, mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dG = zeros_like(gallery)
    idx = indices_on(Q, idxs)
    dim, Tq, B = size(Q)
    Td = size(gallery, 2)
    C, N = size(idx, 1), size(gallery, 3)
    launch!(gather_candidates_kernel!, backend, (dim, Tq, B),
            dQ, gallery, idx, qmask, args, Δ, inv_n, Td, C, N)
    scatter_candidates!(mode, backend, dG, Q, idx, qmask, args, Δ, inv_n)
    dQ, dG
end

scatter_candidates!(::AtomicUnified, backend, dG, Q, idxs, qmask, args, Δ, inv_n) =
    launch!(scatter_candidates_atomic_kernel!, backend,
            (size(Q, 1), size(Q, 2), size(Q, 3)),
            dG, Q, idxs, qmask, args, Δ, inv_n, size(dG, 2), size(idxs, 1), size(dG, 3))

scatter_candidates!(::InvGrid, backend, dG, Q, idxs, qmask, args, Δ, inv_n) =
    launch!(scatter_candidates_invgrid_kernel!, backend,
            (size(dG, 1), size(dG, 2), size(dG, 3)),
            dG, Q, idxs, qmask, args, Δ, inv_n, size(Q, 2), size(idxs, 1), size(Q, 3))
