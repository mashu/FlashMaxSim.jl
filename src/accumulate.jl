# accumulate.jl — ∇D aggregation by [`BackwardStrategy`](@ref) dispatch.

"""Scatter ``∇D`` for one pair (Eq. 3): host sequential atomic-unified."""
function accumulate_doc!(::AtomicUnified, dd::AbstractMatrix{T},
                         q::AbstractMatrix{T},
                         δ_per_src::AbstractVector{T},
                         argmax_u::AbstractVector{<:Integer}) where {T}
    dim, Tq = size(q)
    @inbounds for t in 1:Tq
        δt = δ_per_src[t]
        δt == zero(T) && continue
        u = Int(argmax_u[t])
        u < 1 && continue
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
    row_ptr, col_idx = build_inverse_csr(argmax_u, size(dd, 2))
    invgrid_accumulate_doc!(dd, q, δ_per_src, row_ptr, col_idx)
end
