# invgrid.jl — Inverse-grid CSR for atomic-free ∇D (paper §4.2.2 / Alg. 3).

"""
Build CSR `(row_ptr, col_idx)` inverting `argmax[source] → dest`.

`argmax` length `n_src`; values in `1:n_dest` (0 = inactive source, dropped).
`row_ptr` is length `n_dest + 1` with **0-based exclusive ends** (CSR prefix
sums): destination `u` owns `col_idx[(row_ptr[u] + 1):row_ptr[u + 1]]`.

Counting sort: one pass to count, prefix-sum `row_ptr`, one pass to scatter
sources. The host builder keeps sources that share a destination in source
order. Device CSR builders (`csr_fill_*`) obtain slots from `@atomic` and
do **not** preserve source order — floating-point scatter sums may differ
run-to-run on GPU.
"""
function build_inverse_csr(arg_u::AbstractVector{<:Integer}, n_dest::Integer)
    n_src = length(arg_u)
    nd = Int(n_dest)
    row_ptr = zeros(Int32, nd + 1)
    @inbounds for s in 1:n_src
        u = Int(arg_u[s])
        (1 <= u <= nd) || continue
        row_ptr[u + 1] += Int32(1)
    end
    @inbounds for u in 1:nd
        row_ptr[u + 1] += row_ptr[u]
    end
    nnz = Int(row_ptr[nd + 1])
    col_idx = Vector{Int32}(undef, nnz)
    cursor = copy(row_ptr)
    @inbounds for s in 1:n_src
        u = Int(arg_u[s])
        (1 <= u <= nd) || continue
        cursor[u] += Int32(1)
        col_idx[Int(cursor[u])] = Int32(s)
    end
    row_ptr, col_idx
end

"""
Destination-owned ``∇D`` from CSR: for each doc token `u`,
``dd[:,u] += Σ_{t: argmax[t]=u} δ_t * q[:,t]``.

`δ_per_src[t]` is the upstream scale for query token `t` (usually `δ` or 0).
"""
function invgrid_accumulate_doc!(dd::AbstractMatrix{T},
                                 q::AbstractMatrix{T},
                                 δ_per_src::AbstractVector{T},
                                 row_ptr::AbstractVector{Int32},
                                 col_idx::AbstractVector{Int32}) where {T}
    dim, Td = size(dd)
    @inbounds for u in 1:Td
        a = Int(row_ptr[u]) + 1
        b = Int(row_ptr[u + 1])
        for k in a:b
            t = Int(col_idx[k])
            δt = δ_per_src[t]
            δt == zero(T) && continue
            @simd for d in 1:dim
                dd[d, u] += δt * q[d, t]
            end
        end
    end
    dd
end
