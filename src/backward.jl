# backward.jl — Sparse MaxSim pullback (paper §4.2, Eqs. 2–3).
#
# CPU: sequential gather + [`accumulate_doc!`](@ref) (BitArray-safe).
# GPU and other KA backends: gather/scatter kernels; no host copies of Q/D.
#
# InvGrid on every KA backend builds a real inverse-grid CSR (Alg. 3) via the
# count → prefix → fill kernels in `kernels_backward.jl`, then accumulates.
#
# Every layout exposes its KA implementation as a named `*_pullback_ka`
# function so the device code paths can be exercised on the CPU backend in CI
# (backend dispatch alone can never select them on a host array).

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
    length(qmask) == Tq || throw(DimensionMismatch("qmask vs query tokens"))
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

function pair_pullback(backend::Backend, δ::T, q::AbstractMatrix{T},
                       d::AbstractMatrix{T},
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
    length(qmask) == Tq || throw(DimensionMismatch("qmask vs query tokens"))
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    launch!(gather_pair_kernel!, backend, (dim, Tq), dq, d, qmask, argmax_u, δ, Td)
    scatter_pair!(mode, backend, dd, q, qmask, argmax_u, δ)
    sync!(backend)
    dq, dd
end

scatter_pair!(::AtomicUnified, backend, dd, q, qmask, argmax_u, δ) =
    launch!(scatter_pair_atomic_kernel!, backend, (size(q, 1), size(q, 2)),
            dd, q, qmask, argmax_u, δ, size(dd, 2))

function scatter_pair!(::InvGrid, backend, dd, q, qmask, argmax_u, δ)
    Td = size(dd, 2)
    Tq = size(q, 2)
    row_ptr, col_idx = build_pair_csr(backend, q, argmax_u, Td, Tq)
    launch!(scatter_pair_csr_kernel!, backend, (size(dd, 1), Td),
            dd, q, row_ptr, col_idx, δ)
    nothing
end

"""Device CSR for one pair: `argmax_u[t] → u` (Alg. 3 counting sort)."""
function build_pair_csr(backend, prototype, argmax_u, Td, Tq)
    row_ptr = zeros_like(prototype, Int32, Td + 1)
    launch!(csr_count_pair_kernel!, backend, Tq, row_ptr, argmax_u, Td)
    launch!(csr_prefix_pair_kernel!, backend, 1, row_ptr, Td)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq)
    launch!(csr_fill_pair_kernel!, backend, Tq, col_idx, cursor, argmax_u, Td)
    row_ptr, col_idx
end

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

function paired_pullback(backend::Backend, Δ::AbstractVector{T}, Q, D, qmask,
                         args, inv_n, mode::BackwardStrategy) where {T}
    paired_pullback_ka(backend, Δ, Q, D, qmask, args, inv_n, mode)
end

function paired_pullback_ka(backend, Δ::AbstractVector{T}, Q, D, qmask, args,
                            inv_n, mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dD = zeros_like(D)
    dim, Tq, B = size(Q)
    Td = size(D, 2)
    launch!(gather_paired_kernel!, backend, (dim, Tq, B),
            dQ, D, qmask, args, Δ, inv_n, Td)
    scatter_paired!(mode, backend, dD, Q, qmask, args, Δ, inv_n)
    sync!(backend)
    dQ, dD
end

scatter_paired!(::AtomicUnified, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_paired_atomic_kernel!, backend, (size(Q, 1), size(Q, 2), size(Q, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(dD, 2))

function scatter_paired!(::InvGrid, backend, dD, Q, qmask, args, Δ, inv_n)
    Td = size(dD, 2)
    Tq, B = size(Q, 2), size(Q, 3)
    row_ptr, col_idx = build_paired_csr(backend, Q, args, Td, Tq, B)
    launch!(scatter_paired_csr_kernel!, backend, (size(dD, 1), Td, B),
            dD, Q, row_ptr, col_idx, Δ, inv_n, Tq)
    nothing
end

function build_paired_csr(backend, prototype, args, Td, Tq, B)
    row_ptr = zeros_like(prototype, Int32, Td + 1, B)
    launch!(csr_count_paired_kernel!, backend, (Tq, B), row_ptr, args, Td)
    launch!(csr_prefix_paired_kernel!, backend, B, row_ptr, Td)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq * B)
    launch!(csr_fill_paired_kernel!, backend, (Tq, B), col_idx, cursor, args, Td, Tq)
    row_ptr, col_idx
end

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
    δ_src = zeros(T, Tq)
    @inbounds for j in 1:Bd, i in 1:Bq
        δ = T(Δ[j, i]) * inv_n[i]
        δ == zero(T) && continue
        fill!(δ_src, zero(T))
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

function inbatch_pullback(backend::Backend, Δ::AbstractMatrix{T}, Q, D, qmask,
                          args, inv_n, mode::BackwardStrategy) where {T}
    inbatch_pullback_ka(backend, Δ, Q, D, qmask, args, inv_n, mode)
end

function inbatch_pullback_ka(backend, Δ::AbstractMatrix{T}, Q, D, qmask, args,
                             inv_n, mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dD = zeros_like(D)
    dim, Tq, Bq = size(Q)
    Td, Bd = size(D, 2), size(D, 3)
    launch!(gather_inbatch_kernel!, backend, (dim, Tq, Bq),
            dQ, D, qmask, args, Δ, inv_n, Td, Bd)
    scatter_inbatch!(mode, backend, dD, Q, qmask, args, Δ, inv_n)
    sync!(backend)
    dQ, dD
end

scatter_inbatch!(::AtomicUnified, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_inbatch_atomic_kernel!, backend, (size(Q, 1), size(Q, 2), size(Q, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(dD, 2), size(dD, 3))

function scatter_inbatch!(::InvGrid, backend, dD, Q, qmask, args, Δ, inv_n)
    Td, Bd = size(dD, 2), size(dD, 3)
    Tq, Bq = size(Q, 2), size(Q, 3)
    row_ptr, col_idx = build_inbatch_csr(backend, Q, args, Td, Tq, Bq, Bd)
    launch!(scatter_inbatch_csr_kernel!, backend, (size(dD, 1), Td, Bd),
            dD, Q, row_ptr, col_idx, Δ, inv_n, Tq, Bq)
    nothing
end

function build_inbatch_csr(backend, prototype, args, Td, Tq, Bq, Bd)
    row_ptr = zeros_like(prototype, Int32, Td + 1, Bd)
    launch!(csr_count_inbatch_kernel!, backend, (Tq, Bd, Bq), row_ptr, args, Td)
    launch!(csr_prefix_inbatch_kernel!, backend, Bd, row_ptr, Td)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq * Bq * Bd)
    launch!(csr_fill_inbatch_kernel!, backend, (Tq, Bd, Bq),
            col_idx, cursor, args, Td, Tq, Bq)
    row_ptr, col_idx
end

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
    δ_src = zeros(T, Tq)
    @inbounds for b in 1:B, c in 1:C
        j = Int(idxs[c, b])
        (1 <= j <= N) || continue
        δ = T(Δ[c, b]) * inv_n[b]
        δ == zero(T) && continue
        fill!(δ_src, zero(T))
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

function candidates_pullback(backend::Backend, Δ::AbstractMatrix{T}, Q, gallery,
                             idxs, qmask, args, inv_n,
                             mode::BackwardStrategy) where {T}
    candidates_pullback_ka(backend, Δ, Q, gallery, idxs, qmask, args, inv_n, mode)
end

function candidates_pullback_ka(backend, Δ::AbstractMatrix{T}, Q, gallery, idxs,
                                qmask, args, inv_n, mode::BackwardStrategy) where {T}
    dQ = zeros_like(Q)
    dG = zeros_like(gallery)
    idx = indices_on(Q, idxs)
    dim, Tq, B = size(Q)
    Td = size(gallery, 2)
    C, N = size(idx, 1), size(gallery, 3)
    launch!(gather_candidates_kernel!, backend, (dim, Tq, B),
            dQ, gallery, idx, qmask, args, Δ, inv_n, Td, C, N)
    scatter_candidates!(mode, backend, dG, Q, idx, qmask, args, Δ, inv_n)
    sync!(backend)
    dQ, dG
end

scatter_candidates!(::AtomicUnified, backend, dG, Q, idxs, qmask, args, Δ, inv_n) =
    launch!(scatter_candidates_atomic_kernel!, backend,
            (size(Q, 1), size(Q, 2), size(Q, 3)),
            dG, Q, idxs, qmask, args, Δ, inv_n, size(dG, 2), size(idxs, 1), size(dG, 3))

function scatter_candidates!(::InvGrid, backend, dG, Q, idxs, qmask, args, Δ, inv_n)
    Td, N = size(dG, 2), size(dG, 3)
    Tq, B = size(Q, 2), size(Q, 3)
    C = size(idxs, 1)
    row_ptr, col_idx = build_candidates_csr(backend, Q, idxs, args, Td, N, Tq, C, B)
    launch!(scatter_candidates_csr_kernel!, backend, (size(dG, 1), Td, N),
            dG, Q, row_ptr, col_idx, Δ, inv_n, Td, Tq, C)
    nothing
end

function build_candidates_csr(backend, prototype, idxs, args, Td, N, Tq, C, B)
    n_dest = Td * N
    row_ptr = zeros_like(prototype, Int32, n_dest + 1)
    launch!(csr_count_candidates_kernel!, backend, (Tq, C, B),
            row_ptr, idxs, args, Td, N)
    launch!(csr_prefix_candidates_kernel!, backend, 1, row_ptr, n_dest)
    cursor = copy(row_ptr)
    col_idx = zeros_like(prototype, Int32, Tq * C * B)
    launch!(csr_fill_candidates_kernel!, backend, (Tq, C, B),
            col_idx, cursor, idxs, args, Td, N, Tq, C)
    row_ptr, col_idx
end
