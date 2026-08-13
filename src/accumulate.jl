# accumulate.jl — ∇D aggregation by [`BackwardStrategy`](@ref) dispatch.

function scatter_query_length(dd, q, δ_per_src, argmax_u)
    size(q, 1) == size(dd, 1) || throw(DimensionMismatch("q and ∇D feature dim"))
    Tq = size(q, 2)
    length(δ_per_src) == Tq || throw(DimensionMismatch("δ_per_src vs query tokens"))
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    Tq
end

"""Scatter ``∇D`` for one pair (Eq. 3): host sequential atomic-unified."""
function accumulate_doc!(::AtomicUnified, dd::AbstractMatrix{T},
                         q::AbstractMatrix{T},
                         δ_per_src::AbstractVector{T},
                         argmax_u::AbstractVector{<:Integer}) where {T}
    dim = size(q, 1)
    Tq = scatter_query_length(dd, q, δ_per_src, argmax_u)
    Td = size(dd, 2)
    @inbounds for t in 1:Tq
        δt = δ_per_src[t]
        δt == zero(T) && continue
        u = Int(argmax_u[t])
        (1 <= u <= Td) || continue
        @simd for k in 1:dim
            dd[k, u] += δt * q[k, t]
        end
    end
    dd
end

"""Scatter ``∇D`` via inverse-grid CSR (paper Alg. 2)."""
function accumulate_doc!(::InvGrid, dd::AbstractMatrix{T},
                         q::AbstractMatrix{T},
                         δ_per_src::AbstractVector{T},
                         argmax_u::AbstractVector{<:Integer}) where {T}
    scatter_query_length(dd, q, δ_per_src, argmax_u)
    row_ptr, col_idx = build_inverse_csr(argmax_u, size(dd, 2))
    invgrid_accumulate_doc!(dd, q, δ_per_src, row_ptr, col_idx)
end
