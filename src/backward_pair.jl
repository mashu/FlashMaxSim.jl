# backward_pair.jl — Single-pair sparse MaxSim pullback (paper §4.2, Eqs. 2–3).
#
# CPU: sequential gather + [`accumulate_doc!`](@ref).
# Other KA backends: fused atomic-unified ∇Q+∇D, or gather + InvGrid CSR.
# so the scale never forces a host round-trip. Named `pair_pullback_ka` for CI.

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
    δv = fill!(zeros_like(q, T, 1), δ)
    pair_pullback_ka(backend, δv, q, d, qmask, argmax_u, mode)
end

function pair_pullback(backend::Backend, δ::AbstractVector{T}, q::AbstractMatrix{T},
                       d::AbstractMatrix{T},
                       qmask::AbstractVector{Bool},
                       argmax_u::AbstractVector{<:Integer},
                       mode::BackwardStrategy) where {T<:AbstractFloat}
    pair_pullback_ka(backend, δ, q, d, qmask, argmax_u, mode)
end

function pair_pullback_ka(backend, δ::AbstractVector{T}, q::AbstractMatrix{T},
                          d::AbstractMatrix{T},
                          qmask::AbstractVector{Bool},
                          argmax_u::AbstractVector{<:Integer},
                          mode::BackwardStrategy) where {T<:AbstractFloat}
    dq = zeros_like(q)
    dd = zeros_like(d)
    length(δ) == 1 || throw(DimensionMismatch("pair δ must be length 1"))
    dim, Tq = size(q)
    Td = size(d, 2)
    length(qmask) == Tq || throw(DimensionMismatch("qmask vs query tokens"))
    length(argmax_u) == Tq || throw(DimensionMismatch("argmax_u vs query tokens"))
    pair_apply_pullback!(mode, backend, dq, dd, q, d, qmask, argmax_u, δ, dim, Tq, Td)
    finish!(backend)
    dq, dd
end

function pair_apply_pullback!(::AtomicUnified, backend, dq, dd, q, d, qmask,
                              argmax_u, δ, dim, Tq, Td)
    launch!(unified_pair_atomic_kernel!, backend, (dim, Tq),
            dq, dd, q, d, qmask, argmax_u, δ, Td)
    nothing
end

function pair_apply_pullback!(::InvGrid, backend, dq, dd, q, d, qmask,
                              argmax_u, δ, dim, Tq, Td)
    launch!(gather_pair_kernel!, backend, (dim, Tq), dq, d, qmask, argmax_u, δ, Td)
    row_ptr, col_idx = build_pair_csr(backend, q, argmax_u, qmask, Td, Tq)
    launch!(scatter_pair_csr_kernel!, backend, (size(dd, 1), Td),
            dd, q, row_ptr, col_idx, δ)
    nothing
end

# Scalar convenience for tests / CPU-KA callers
function pair_pullback_ka(backend, δ::T, q::AbstractMatrix{T}, d::AbstractMatrix{T},
                          qmask::AbstractVector{Bool},
                          argmax_u::AbstractVector{<:Integer},
                          mode::BackwardStrategy) where {T<:AbstractFloat}
    pair_pullback_ka(backend, fill!(zeros_like(q, T, 1), δ), q, d, qmask, argmax_u, mode)
end

# Scalar convenience for tests / CPU-KA callers
