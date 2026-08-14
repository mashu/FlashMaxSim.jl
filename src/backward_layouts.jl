# backward_layouts.jl — Paired / in-batch / candidate sparse pullbacks.
#
# Same contract as `backward_pair.jl`: CPU sequential paths, KA gather/scatter,
# InvGrid via device CSR. Named `*_pullback_ka` entry points for CI on CPU().

# ---- paired ------------------------------------------------------------------

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
    finish!(backend)
    dQ, dD
end

scatter_paired!(::AtomicUnified, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_paired_atomic_kernel!, backend, (size(Q, 1), size(Q, 2), size(Q, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(dD, 2))

function scatter_paired!(::InvGrid, backend, dD, Q, qmask, args, Δ, inv_n)
    Td = size(dD, 2)
    Tq, B = size(Q, 2), size(Q, 3)
    row_ptr, col_idx = build_paired_csr(backend, Q, args, qmask, Td, Tq, B)
    launch!(scatter_paired_csr_kernel!, backend, (size(dD, 1), Td, B),
            dD, Q, row_ptr, col_idx, Δ, inv_n, Tq)
    nothing
end

# ---- in-batch ----------------------------------------------------------------

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
    finish!(backend)
    dQ, dD
end

scatter_inbatch!(::AtomicUnified, backend, dD, Q, qmask, args, Δ, inv_n) =
    launch!(scatter_inbatch_atomic_kernel!, backend, (size(Q, 1), size(Q, 2), size(Q, 3)),
            dD, Q, qmask, args, Δ, inv_n, size(dD, 2), size(dD, 3))

function scatter_inbatch!(::InvGrid, backend, dD, Q, qmask, args, Δ, inv_n)
    Td, Bd = size(dD, 2), size(dD, 3)
    Tq, Bq = size(Q, 2), size(Q, 3)
    row_ptr, col_idx = build_inbatch_csr(backend, Q, args, qmask, Td, Tq, Bq, Bd)
    launch!(scatter_inbatch_csr_kernel!, backend, (size(dD, 1), Td, Bd),
            dD, Q, row_ptr, col_idx, Δ, inv_n, Tq, Bq)
    nothing
end

# ---- candidates --------------------------------------------------------------

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
    finish!(backend)
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
    row_ptr, col_idx = build_candidates_csr(backend, Q, idxs, args, qmask, Td, N, Tq, C, B)
    launch!(scatter_candidates_csr_kernel!, backend, (size(dG, 1), Td, N),
            dG, Q, row_ptr, col_idx, Δ, inv_n, Td, Tq, C)
    nothing
end
