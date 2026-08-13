# invgrid.jl — Inverse-grid CSR for atomic-free ∇D (paper §4.2.2 / Alg. 2).

"""
Build CSR `(row_ptr, col_idx)` inverting `argmax[source] → dest`.

`argmax` length `n_src`; values in `1:n_dest` (0 = inactive source, dropped).
`row_ptr` is length `n_dest + 1` with **0-based exclusive ends** (CSR prefix
sums): destination `u` owns `col_idx[(row_ptr[u] + 1):row_ptr[u + 1]]`.
"""
function build_inverse_csr(argmax::AbstractVector{<:Integer}, n_dest::Integer)
    n_src = length(argmax)
    ah = Int.(Array(argmax))
    srcs = Int[]
    dests = Int[]
    sizehint!(srcs, n_src)
    sizehint!(dests, n_src)
    @inbounds for s in 1:n_src
        u = ah[s]
        (1 <= u <= n_dest) || continue
        push!(srcs, s)
        push!(dests, u)
    end
    nnz = length(srcs)
    order = sortperm(dests; alg = Base.Sort.DEFAULT_STABLE)
    col_idx = Vector{Int32}(undef, nnz)
    sorted_dest = Vector{Int}(undef, nnz)
    @inbounds for k in 1:nnz
        col_idx[k] = Int32(srcs[order[k]])
        sorted_dest[k] = dests[order[k]]
    end
    row_ptr = zeros(Int32, Int(n_dest) + 1)
    @inbounds for k in 1:nnz
        row_ptr[sorted_dest[k] + 1] += Int32(1)
    end
    @inbounds for u in 1:n_dest
        row_ptr[u + 1] += row_ptr[u]
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
